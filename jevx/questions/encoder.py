# Copyright (c) 2026 ysnrfd
"""Question encoder ``E_Q`` (Definition 4 of the math spec).

Two text paths, selected at construction time:

* ``method="backbone"`` — tokenize with a real tokenizer, run a (optionally
  frozen) HF backbone under ``torch.no_grad()`` when ``frozen=True``, pool the
  resulting sequence with the shared mask-aware operators, then project
  ``backbone_hidden -> hidden_size``. This is the shared-backbone default of
  the spec (``θ_q`` may equal ``θ``).
* ``method="projection"`` (default) — hashed bag-of-tokens embedding: for
  tokens ``w_1..w_T`` of the question text,
  ``h_Q = (1/T) Σ_t E[hash(w_t) mod N_buckets]`` with a learned embedding
  table ``E``. This is the fallback for *non-text* backbones (a ViT or
  wav2vec2 state model has no tokenizer or text tower): deterministic per
  string, trainable, and deliberately independent of any HF checkpoint.
  When ``tokenizer`` is ``None`` the tokenizer is a plain whitespace split.

:meth:`QuestionEncoder.forward` is the trainable residual transform applied
to already-encoded questions: ``h' = h + MLP(h)``.
"""

from __future__ import annotations

import zlib
from contextlib import nullcontext
from typing import Any, List, Optional, Sequence

import torch
import torch.nn as nn

from jevx.representations.pooling import pool
from jevx.representations.projection import Projection
from jevx.utils.shapes import assert_shape, check_finite

__all__ = ["QuestionEncoder", "QUESTION_METHODS", "TYPE_TAGS"]

QUESTION_METHODS: tuple[str, ...] = ("projection", "backbone")

#: Role tags a shared encoder can carry (``JevXConfig.shared_question_schema_encoder``).
#: The index of each name in this tuple is the row of
#: :attr:`QuestionEncoder.type_embedding`.
TYPE_TAGS: tuple[str, ...] = ("question", "option")


def _validated_texts(texts: Any) -> List[str]:
    """Normalize ``texts`` to a list of non-empty, non-blank strings."""
    if isinstance(texts, str):
        items: Sequence[Any] = [texts]
    elif isinstance(texts, (list, tuple)):
        items = texts
    else:
        raise TypeError(
            f"texts: expected a str or list of str, got {type(texts).__name__}."
        )
    if len(items) == 0:
        raise ValueError("texts: expected at least one question, got an empty list.")
    out: List[str] = []
    for i, item in enumerate(items):
        if not isinstance(item, str):
            raise TypeError(f"texts[{i}]: expected a str, got {type(item).__name__}.")
        if not item.strip():
            raise ValueError(
                f"texts[{i}]: question is empty or whitespace-only — E_Q is undefined for an "
                "empty string; pass the question's actual text."
            )
        out.append(item)
    return out


class QuestionEncoder(nn.Module):
    """Encode question strings to ``H_Q ∈ R^{B x hidden_size}``.

    Args:
        hidden_size: Output width ``D_q`` (``JevXConfig.question_hidden_size``).
        backbone_hidden: Hidden width of the backbone used by
            ``method="backbone"``; required for that method (its forward-time
            mismatch with the actual backbone output is detected with a
            descriptive error).
        method: ``"projection"`` (hashed bag-of-tokens fallback; default) or
            ``"backbone"`` (tokenize + backbone + pool + project).
        num_buckets: Vocabulary size ``N`` of the hash embedding table
            (``method="projection"`` only).
        type_tags: Allocate a learnable 2-row TYPE-TAG embedding (the
            ``is_question`` / ``is_option`` rows of :data:`TYPE_TAGS`) and
            let :meth:`encode_texts` add the selected row to the encoded
            output. It exists so ONE encoder can serve both the question and
            the schema path (``JevXConfig.shared_question_schema_encoder``)
            without conflating the two roles. Default False allocates nothing,
            so a two-encoder model is bit for bit today's model.

    The tag is ADDED to the encoded rows (after the hashed mean, or after the
    backbone's pooling + projection), not prepended to the token sequence: an
    additive offset is a pure role signal and leaves the token statistics
    exactly untouched, while a prepended ``[TAG]`` token would be averaged
    into the bag of tokens and rescale the text signal by ``T/(T+1)``
    (silently length-dependent, and length is not what distinguishes a
    question from an option).

    Raises:
        ValueError: Unknown ``method``, a non-positive width, or
            ``method="backbone"`` without ``backbone_hidden``.
    """

    def __init__(
        self,
        hidden_size: int,
        backbone_hidden: Optional[int] = None,
        method: str = "projection",
        num_buckets: int = 4096,
        type_tags: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if not isinstance(hidden_size, int) or hidden_size < 1:
            raise ValueError(f"hidden_size: expected a positive int, got {hidden_size!r}.")
        if method not in QUESTION_METHODS:
            raise ValueError(
                f"method: unknown question-encoding method {method!r}; expected one of "
                f"{list(QUESTION_METHODS)}."
            )
        if method == "backbone" and (
            not isinstance(backbone_hidden, int) or backbone_hidden < 1
        ):
            raise ValueError(
                "method='backbone' requires backbone_hidden=<int> so E_Q can project the "
                "backbone's pooled width to hidden_size."
            )
        if not isinstance(num_buckets, int) or num_buckets < 1:
            raise ValueError(f"num_buckets: expected a positive int, got {num_buckets!r}.")
        if not isinstance(type_tags, bool):
            raise TypeError(f"type_tags: expected a bool, got {type(type_tags).__name__}.")
        self.hidden_size = hidden_size
        self.backbone_hidden = backbone_hidden
        self.method = method
        self.num_buckets = num_buckets
        # Trainable residual transform: h' = h + MLP(h).
        self.transform = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        if method == "projection":
            self.hash_embedding = nn.Embedding(num_buckets, hidden_size)
            self.project: Optional[Projection] = None
        else:
            self.hash_embedding = None
            assert backbone_hidden is not None  # checked above
            self.project = Projection(backbone_hidden, hidden_size)
        # Shared-encoder role tag: 2 rows, added (not prepended) to the encoded
        # rows. Allocated ONLY when asked for, so the default two-encoder stack
        # keeps today's state_dict keys and parameter count.
        self.type_embedding = (
            nn.Embedding(len(TYPE_TAGS), hidden_size) if type_tags else None
        )

    # -- text paths ---------------------------------------------------------

    def _hash_bucket(self, token: str) -> int:
        """Stable (process-independent) bucket index for one token."""
        return zlib.crc32(token.encode("utf-8")) % self.num_buckets

    def _encode_hashed(self, texts: List[str], tokenizer: Any) -> torch.Tensor:
        """Hashed bag-of-tokens: mean of token-hash embeddings, ``[B, D_q]``."""
        assert self.hash_embedding is not None
        rows: List[torch.Tensor] = []
        device = self.hash_embedding.weight.device
        for i, text in enumerate(texts):
            if tokenizer is not None:
                tokens = tokenizer.tokenize(text)
            else:
                tokens = text.split()
            if not tokens:
                raise ValueError(
                    f"texts[{i}]: tokenizer produced no tokens for {text!r} — E_Q needs at "
                    "least one token (check the tokenizer's pre-tokenizer)."
                )
            buckets = torch.tensor(
                [self._hash_bucket(tok) for tok in tokens], dtype=torch.long, device=device
            )
            rows.append(self.hash_embedding(buckets).mean(dim=0))
        return torch.stack(rows, dim=0)

    def _add_type_tag(
        self, encoded: torch.Tensor, type_tag: Optional[str]
    ) -> torch.Tensor:
        """Add the shared encoder's role vector to ``encoded`` rows.

        Returns ``encoded`` unchanged when ``type_tag`` is ``None`` (the
        default two-encoder path, bit for bit today's computation).
        """
        if type_tag is None:
            return encoded
        if self.type_embedding is None:
            raise ValueError(
                f"type_tag={type_tag!r} was requested but this encoder was built with "
                "type_tags=False — construct QuestionEncoder(type_tags=True) to let one "
                "encoder serve both the question and the schema path "
                "(JevXConfig.shared_question_schema_encoder=True)." 
            )
        if type_tag not in TYPE_TAGS:
            raise ValueError(
                f"type_tag: {type_tag!r} is not a known role tag; expected one of "
                f"{list(TYPE_TAGS)}."
            )
        tag = self.type_embedding.weight[TYPE_TAGS.index(type_tag)]  # [D_q]
        return encoded + tag

    @staticmethod
    def _sequence_from_backbone_output(outputs: Any) -> torch.Tensor:
        """Extract a ``[B, T, d]`` sequence from a backbone forward output."""
        seq: Any = None
        if isinstance(outputs, dict):
            seq = outputs.get("last_hidden_state")
            if seq is None:
                for value in outputs.values():
                    if isinstance(value, torch.Tensor) and value.dim() == 3:
                        seq = value
                        break
        elif isinstance(outputs, torch.Tensor):
            seq = outputs if outputs.dim() == 3 else None
        elif isinstance(outputs, (tuple, list)) and outputs:
            seq = outputs[0] if isinstance(outputs[0], torch.Tensor) and outputs[0].dim() == 3 else None
        if not isinstance(seq, torch.Tensor) or seq.dim() != 3:
            raise ValueError(
                "backbone: no sequence states (rank-3 tensor) in the backbone output — pass a "
                "base/AutoModel class that returns last_hidden_state, not a For*LM head that "
                "returns logits."
            )
        return seq

    def encode_texts(
        self,
        texts: List[str],
        *,
        tokenizer: Any,
        backbone: Any = None,
        strategy: str = "masked_mean",
        frozen: bool = True,
        type_tag: Optional[str] = None,
    ) -> torch.Tensor:
        """Encode a batch of question strings to ``H_Q [B, hidden_size]``.

        Args:
            texts: Question strings (a plain ``str`` is accepted and treated
                as a batch of one).
            tokenizer: Tokenizer for the ``"backbone"`` path (a real HF
                tokenizer); ``None`` is allowed for ``method="projection"``
                where a whitespace split is used instead.
            backbone: The text backbone for ``method="backbone"``.
            strategy: Pooling strategy applied to the backbone sequence
                (ignored by the hashed path).
            frozen: When True (default), run the backbone under
                ``torch.no_grad()`` so no gradients are ever produced for it;
                when False the graph is kept and gradients flow.
            type_tag: Role tag added to the encoded rows (see the class
                docstring); one of :data:`TYPE_TAGS` (''"question"'' /
                ''"option"''), or ``None`` (default) for no tag. Only
                meaningful on an encoder built with ``type_tags=True``.

        Returns:
            ``H_Q`` of shape ``[B, hidden_size]``.

        Raises:
            TypeError: ``texts`` holds non-strings.
            ValueError: Empty/whitespace-only question, empty batch, a
                method/backbone/tokenizer mismatch, a
                ``backbone_hidden`` mismatch with the backbone's output, or a
                ``type_tag`` the encoder was not built for.
        """
        items = _validated_texts(texts)
        if self.method == "projection":
            return self._add_type_tag(self._encode_hashed(items, tokenizer), type_tag)

        # -- method == "backbone" ------------------------------------------
        if backbone is None:
            raise ValueError(
                "encode_texts: method='backbone' requires backbone=<HF model>; construct "
                "QuestionEncoder(method='projection') for the tokenizer-only fallback."
            )
        if tokenizer is None:
            raise ValueError(
                "encode_texts: method='backbone' requires tokenizer= to turn questions into "
                "input_ids (text-only families resolve through AutoTokenizer, audit Q8)."
            )
        batch = dict(tokenizer(items, return_tensors="pt", padding=True))
        # Only forward what the backbone declares (control flags allowed);
        # mirrors the adapter input contract without importing it.
        import inspect

        try:
            sig = inspect.signature(backbone.forward)
            explicit = {
                name
                for name, p in sig.parameters.items()
                if name != "self"
                and p.kind
                in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
            }
            allowed = explicit | {"output_hidden_states", "output_attentions", "return_dict"}
            forward_kwargs = {k: v for k, v in batch.items() if k in allowed}
        except (TypeError, ValueError):  # pragma: no cover - exotic callables
            forward_kwargs = dict(batch)
        context = torch.no_grad() if frozen else nullcontext()
        with context:
            outputs = backbone(**forward_kwargs)
        seq = self._sequence_from_backbone_output(outputs)
        if self.backbone_hidden is not None and seq.shape[-1] != self.backbone_hidden:
            raise ValueError(
                f"backbone_hidden: declared {self.backbone_hidden} but the backbone emits "
                f"hidden width {seq.shape[-1]} — rebuild QuestionEncoder with the correct "
                "backbone_hidden."
            )
        mask = batch.get("attention_mask")
        if mask is not None and tuple(mask.shape) != tuple(seq.shape[:2]):
            mask = None  # never pool with a length-mismatched mask
        pooled = pool(seq, mask, strategy)  # [B, backbone_hidden]
        assert self.project is not None
        return self._add_type_tag(self.project(pooled), type_tag)

    # -- trainable transform ------------------------------------------------

    def forward(self, question_repr: torch.Tensor) -> torch.Tensor:
        """Apply the trainable residual transform ``h' = h + MLP(h)``.

        Args:
            question_repr: Encoded questions ``[B, hidden_size]``.

        Returns:
            Transformed questions ``[B, hidden_size]`` (same shape).
        """
        assert_shape(question_repr, (None, self.hidden_size), "question_repr")
        check_finite(question_repr, "question_repr")
        return question_repr + self.transform(question_repr)

    def extra_repr(self) -> str:
        tags = "none" if self.type_embedding is None else "question/option"
        return (
            f"hidden_size={self.hidden_size}, method={self.method!r}, "
            f"type_tags={tags}"
        )
