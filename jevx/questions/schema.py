# Copyright (c) 2026 ysnrfd
"""Schema encoder ``E_C`` for variable-``K`` option sets (Definitions 3/4).

Each option ``O_k = (key, label, description, metadata)`` is encoded
*independently* through the same text path as :class:`QuestionEncoder`
(shared code, own parameters — so ``E_C`` never couples to ``E_Q``), padded
to ``K_max`` across the batch, and returned with a bool validity mask that
aligns with ``DecisionHead.option_mask`` (leading valid entries, ``True`` =
real option).

Documented semantics:

* ``K = 0`` is rejected (axiom D3.1: softmax/argmax over an empty schema is
  undefined). ``K = 1`` is allowed (Regression/Threshold degenerate schemas).
* Option *duplicates* are accepted as-is: ``encode`` performs no key
  uniqueness check (D3.2 belongs to schema validation upstream); identical
  option texts encode to identical rows by construction.
* ``Option.text()`` concatenates label and description
  (``"label: description"``, either side alone when the other is empty) and
  falls back to ``str(key)`` when both are empty so ``E_C`` never sees an
  empty string.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from jevx.utils.shapes import assert_shape

from .encoder import QuestionEncoder

__all__ = ["Option", "SchemaEncoder"]


@dataclass
class Option:
    """One schema element ``O_k`` (Definition 3).

    Args:
        key: Identity of the option within the schema. ``str`` for named
            options; ``int``/``float`` for ordinal levels, which
            ``JevXForDecision._normalize_option`` builds as
            ``Option(key=option, label=str(option))``
            (``jevx/modeling_jevx.py:367-369``) and which
            ``project_to_schema`` then returns verbatim - so a non-``str``
            key is a documented, supported path, not a type error.
        label: Display string (may be empty).
        description: Longer tokenizable text (may be empty).
        metadata: JSON-serializable payload; never read by any JevX map
            (axiom D2.2).
    """

    key: str | int | float
    label: str = ""
    description: str = ""
    metadata: dict = field(default_factory=dict)

    def text(self) -> str:
        """Tokenizable text: ``"label: description"`` / either side / key."""
        label = self.label or ""
        description = self.description or ""
        if label and description:
            return f"{label}: {description}"
        if label:
            return label
        if description:
            return description
        return str(self.key)


class SchemaEncoder(nn.Module):
    """Encode a batch of variable-size schemas to ``H_C`` plus a bool mask.

    Args:
        hidden_size: Option width ``D_c`` (``JevXConfig.schema_hidden_size``).
        backbone_hidden: Back-bone width for ``method="backbone"``.
        method: Text path — ``"projection"`` (hashed fallback, default) or
            ``"backbone"``; forwarded to the internal
            :class:`QuestionEncoder`.
        num_buckets: Hash-bucket count for the projection path.
        text_encoder: Use THIS :class:`QuestionEncoder` instance instead of
            building a private one, i.e. the shared-encoder layout behind
            ``JevXConfig.shared_question_schema_encoder``. The instance must
            carry role tags (built with ``type_tags=True``, so this encoder
            tags every option row with the ``"option"`` role) and its
            ``hidden_size`` must equal this encoder's. ``None`` (default)
            keeps the shipped two-encoder stack, in which this encoder owns
            independent parameters (the "independent per-option projection"
            contract; every option row goes through it independently, with no
            cross-option interaction).

    Raises:
        ValueError: A ``text_encoder`` that is not a ``QuestionEncoder`, has
            a different ``hidden_size`, or carries no role tags.
    """

    def __init__(
        self,
        hidden_size: int,
        backbone_hidden: Optional[int] = None,
        method: str = "projection",
        num_buckets: int = 4096,
        text_encoder: Optional[QuestionEncoder] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if text_encoder is None:
            self.text_encoder = QuestionEncoder(
                hidden_size,
                backbone_hidden=backbone_hidden,
                method=method,
                num_buckets=num_buckets,
            )
        else:
            if not isinstance(text_encoder, QuestionEncoder):
                raise TypeError(
                    "text_encoder: expected a QuestionEncoder or None, got "
                    f"{type(text_encoder).__name__}."
                )
            if text_encoder.hidden_size != hidden_size:
                raise ValueError(
                    "text_encoder: the shared encoder has hidden_size "
                    f"{text_encoder.hidden_size} but this schema encoder was asked for "
                    f"{hidden_size} — a shared encoder has ONE width, so build both paths "
                    "at the same width (JevXConfig.question_hidden_size == "
                    "schema_hidden_size)."
                )
            if text_encoder.type_embedding is None:
                raise ValueError(
                    "text_encoder: a shared encoder must be built with type_tags=True so "
                    "the option rows can carry the 'option' role tag — rebuild it with "
                    "QuestionEncoder(..., type_tags=True)."
                )
            self.text_encoder = text_encoder
        self.hidden_size = hidden_size
        # Role tag applied to every option row when the encoder is shared with
        # the question path; None keeps the shipped untagged computation.
        self.type_tag: Optional[str] = (
            "option" if self.text_encoder.type_embedding is not None else None
        )

    def encode(
        self,
        batch_of_options: Sequence[Sequence[Option]],
        *,
        tokenizer: Any = None,
        backbone: Any = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode ``B`` schemas with per-row ``K_b`` options.

        Args:
            batch_of_options: ``List[List[Option]]`` — row ``b`` holds the
                schema ``C_b`` in canonical schema order (row ``k`` of the
                output encodes ``O_k``).
            tokenizer: Tokenizer for ``method="backbone"``; ``None`` allowed
                for the hashed path.
            backbone: Backbone for ``method="backbone"``.

        Returns:
            ``(H_C, option_mask)`` with ``H_C [B, K_max, D]`` and a bool
            ``option_mask [B, K_max]`` where ``mask[b, k]`` is True iff
            ``k < K_b`` — the exact layout ``DecisionHead.validate_schema``
            expects.

        Raises:
            TypeError: Not a sequence of sequences of :class:`Option`.
            ValueError: Empty batch, or any row with ``K = 0`` (axiom D3.1).
        """
        if not isinstance(batch_of_options, (list, tuple)) or len(batch_of_options) == 0:
            raise ValueError(
                "SchemaEncoder.encode: batch_of_options must be a non-empty list of schemas "
                f"(one per question), got {type(batch_of_options).__name__} of length "
                f"{len(batch_of_options) if hasattr(batch_of_options, '__len__') else '?'}."
            )
        rows: List[List[Option]] = []
        for b, row in enumerate(batch_of_options):
            if not isinstance(row, (list, tuple)):
                raise TypeError(
                    f"batch_of_options[{b}]: expected a list of Option, got "
                    f"{type(row).__name__}."
                )
            if len(row) == 0:
                raise ValueError(
                    f"batch_of_options[{b}]: schema has K=0 options — a decision schema must "
                    "contain at least one option (axiom D3.1); softmax/argmax over an empty "
                    "schema is undefined."
                )
            for k, option in enumerate(row):
                if not isinstance(option, Option):
                    raise TypeError(
                        f"batch_of_options[{b}][{k}]: expected an Option, got "
                        f"{type(option).__name__}."
                    )
            rows.append(list(row))

        flat_texts: List[str] = [option.text() for row in rows for option in row]
        flat_embeddings = self.text_encoder.encode_texts(
            flat_texts, tokenizer=tokenizer, backbone=backbone, type_tag=self.type_tag
        )  # [sum K_b, D]
        assert_shape(flat_embeddings, (None, self.hidden_size), "flat option embeddings")

        batch_size = len(rows)
        k_max = max(len(row) for row in rows)
        device, dtype = flat_embeddings.device, flat_embeddings.dtype
        out = torch.zeros(batch_size, k_max, self.hidden_size, device=device, dtype=dtype)
        mask = torch.zeros(batch_size, k_max, dtype=torch.bool, device=device)
        cursor = 0
        for b, row in enumerate(rows):
            k = len(row)
            out[b, :k] = flat_embeddings[cursor : cursor + k]
            mask[b, :k] = True
            cursor += k
        return out, mask

    def extra_repr(self) -> str:
        shared = "shared" if self.type_tag is not None else "private"
        return f"hidden_size={self.hidden_size}, text_encoder={shared}"
