# Copyright (c) 2026 ysnrfd
"""Adapter base contract: :class:`Representation` and :class:`BaseAdapter`.

An adapter turns raw user *state* ``S`` into model kwargs (``prepare_inputs``)
and extracts the state representation ``H_S`` from the model's output
(``extract_representation``), implementing the InputAdapter ``R_mu`` and the
state half of ``E_M`` from Definitions 1/4 of the mathematical
specification.

The shared extraction logic is *introspection-driven* — output fields are
read by name in a fixed priority order, never "the first tensor":

1. ``pooler_output`` (native trained pooler; strategy ``None``/``"native"``)
2. semantic fields ``sentence_embedding`` / ``text_embeds`` / ``image_embeds``
3. ``last_hidden_state``
4. ``encoder_last_hidden_state``
5. decoder state when ``mode="decoder_state"`` (tuple outputs)
6. ``hidden_states[-1]`` (only when ``output_hidden_states=True`` was asked —
   audit Q4: collection is hook-based and opt-in; for vision/audio models this
   element is *pre-final-norm*, audit §1.3/Q4)

No forward pass ever runs during :meth:`BaseAdapter.describe_capabilities`;
:meth:`BaseAdapter.validate` is the one cautious ``no_grad`` execution.
"""

from __future__ import annotations

import inspect
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

from jevx.capabilities import (
    CapabilityReport,
    JevXCapabilityDetector,
    UnsupportedRepresentationError,
)
from jevx.representations.pooling import pool
from jevx.utils.shapes import assert_batch_alignment

__all__ = [
    "FORWARD_CONTROL_FIELDS",
    "SEMANTIC_FIELDS",
    "BaseAdapter",
    "Representation",
    "call_backbone",
    "forward_kwargs",
    "inject_no_cache",
    "observed_output_fields",
]

#: Output-control kwargs consumed by the v5 ``capture_outputs`` /
#: ``can_return_tuple`` decorators instead of explicit forward parameters
#: (audit §1.4/Q2 — they must survive keyword filtering even though no model
#: declares them in its signature).
FORWARD_CONTROL_FIELDS = frozenset(
    {"output_hidden_states", "output_attentions", "return_dict"}
)

#: Contrastive/semantic representation fields (priority 2), in selection
#: order — for joint contrastive outputs the text embedding wins (matrix
#: "post-projection for cross-modal comparability" note).
SEMANTIC_FIELDS = ("sentence_embedding", "text_embeds", "image_embeds")


@dataclass
class Representation:
    """One extracted state representation.

    Args:
        sequence_output: Unpooled states ``[B, T, D]`` (kept for
            sequence-level fusions, Definition 5e) or ``None``.
        pooled: Per-row vector ``[B, D]`` — the ``H_S`` consumed by fusion,
            or ``None`` only when extraction failed (never on success).
        attention_mask: Validity mask ``[B, T]`` aligned with
            ``sequence_output`` (``None`` when no mask is available).
        raw_output: The model's raw forward output, passed through untouched
            (modality sub-features of multimodal models stay here).
        field: Name of the output field the representation came from
            (``"pooler_output"``, ``"last_hidden_state"``,
            ``"encoder_last_hidden_state"``, ``"hidden_states[-1]"``, ...).
        layout: ``"btd"`` when ``sequence_output`` is present, else ``"bd"``.

    Raises:
        ValueError: ``layout`` is not ``"btd"``/``"bd"``, disagrees with
            ``sequence_output``'s presence, or the present tensors disagree
            on the batch size.
    """

    sequence_output: Optional[torch.Tensor]
    pooled: Optional[torch.Tensor]
    attention_mask: Optional[torch.Tensor]
    raw_output: Any
    field: str
    layout: str

    def __post_init__(self) -> None:
        if self.layout not in ("btd", "bd"):
            raise ValueError(
                f"Representation.layout: expected 'btd' or 'bd', got {self.layout!r}."
            )
        if (self.layout == "btd") != (self.sequence_output is not None):
            raise ValueError(
                f"Representation: layout={self.layout!r} disagrees with sequence_output="
                f"{'None' if self.sequence_output is None else 'present'} — layout 'btd' "
                "requires a [B, T, D] sequence and 'bd' requires none."
            )
        assert_batch_alignment(
            sequence_output=self.sequence_output,
            pooled=self.pooled,
            attention_mask=self.attention_mask,
        )


def _forward_signature(model: Any) -> Optional[inspect.Signature]:
    try:
        return inspect.signature(model.forward)
    except (TypeError, ValueError):  # pragma: no cover - exotic callables
        return None


def forward_kwargs(model: Any, inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Filter ``inputs`` to what ``model.forward`` actually accepts.

    Keep-set = explicitly declared forward parameters ∪
    :data:`FORWARD_CONTROL_FIELDS`. Undeclared extras are dropped instead of
    raising deep ``TypeError``s (audit: every audited family declares its
    modality/mask inputs explicitly; control flags flow through ``**kwargs``
    into the v5 decorators).

    Falls back to the unfiltered mapping when the signature is uninspectable.
    """
    sig = _forward_signature(model)
    if sig is None:
        return dict(inputs)
    explicit = {
        name
        for name, p in sig.parameters.items()
        if name != "self"
        and p.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    allowed = explicit | FORWARD_CONTROL_FIELDS
    return {k: v for k, v in inputs.items() if k in allowed}


def _forward_signature_parts(model: Any) -> tuple[set[str], bool]:
    """`(declared keyword names, has `**kwargs`)`; `(set(), False)` if uninspectable."""
    sig = _forward_signature(model)
    if sig is None:
        return set(), False
    explicit = {
        name
        for name, p in sig.parameters.items()
        if name != "self"
        and p.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    has_var_kwargs = any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    return explicit, has_var_kwargs


def inject_no_cache(inputs: Dict[str, Any], model: Any) -> Dict[str, Any]:
    """Filter `inputs` for `model.forward` and settle `use_cache`.

    The ONE shared implementation of the extraction keyword contract used by
    the base/generic/decoder and encoder-decoder paths (audit Q-014: the
    kwargs-only gate must not be re-implemented per adapter):
    :func:`forward_kwargs` plus the cache rule — `use_cache` is injected
    as `False` whenever the forward DECLARES it or accepts `**kwargs` (v5
    `TransformersKwargs` wrappers such as `Qwen3_5Model` declare it only
    via `**kwargs`; without injection every extraction built a fresh
    `DynamicCache` from the config default — audit §5/§8.7). Extraction
    never needs a KV cache; a caller-supplied `use_cache` in `inputs`
    passes through untouched (a forward that can receive neither form gets no
    `use_cache` key at all, so an undeclared flag can never surface as a
    deep `TypeError`).

    Args:
        inputs: Prepared model kwargs (from `prepare_inputs`).
        model: Model whose `forward` signature is consulted.

    Returns:
        A NEW dict; `inputs` is never mutated.
    """
    explicit, has_var_kwargs = _forward_signature_parts(model)
    accepts = "use_cache" in explicit or has_var_kwargs
    fwd = forward_kwargs(model, inputs)
    if accepts:
        fwd["use_cache"] = inputs["use_cache"] if "use_cache" in inputs else False
    return fwd


#: Modality backbone inputs whose presence marks the "partial processor
#: output" failure class (audit Q-013: companion vision/audio metadata such
#: as `image_grid_thw` missing -> cryptic upstream `AttributeError`).
_MODALITY_INPUT_KEYS = frozenset(
    {"pixel_values", "pixel_values_videos", "input_features"}
)

#: Signature of Python's "the caller passed a kwarg I do not take" rejection
#: (`forward() got an unexpected keyword argument 'pixel_values'`).
_UNEXPECTED_KWARG_RE = re.compile(r"unexpected keyword argument\s+['\"](\w+)['\"]")

#: `repr(None)` attribute failure — the documented missing-companion-
#: metadata signature (`'NoneType' object has no attribute 'device'`).
_NONETYPE_ATTR_RE = re.compile(r"['\"]NoneType['\"] object has no attribute")


def _is_input_contract_error(exc: BaseException, inputs: Dict[str, Any]) -> bool:
    """Narrow match rule for re-wrapping backbone failures (audit Q-013).

    Returns True iff BOTH hold:

    1. `exc` is a `TypeError` / `AttributeError` / `ValueError`; and
    2. its message matches exactly one of two input-contract signatures:
       a. `unexpected keyword argument '<name>'` where `<name>` is a key
          JevX actually forwarded (the backbone rejected a kwarg JevX passed
          through), or
       b. `'NoneType' object has no attribute ...` while the call carried a
          modality input (`pixel_values` / `pixel_values_videos` /
          `input_features`) — the documented partial-processor-output
          failure (companion vision metadata such as `image_grid_thw`
          missing).

    Everything else — shape/dtype/NaN/CUDA failures and any error whose
    message does not name a forwarded kwarg — is a genuine model error and is
    re-raised untouched, so this wrapper never swallows real bugs.
    """
    if not isinstance(exc, (TypeError, AttributeError, ValueError)):
        return False
    message = str(exc)
    kwarg = _UNEXPECTED_KWARG_RE.search(message)
    if kwarg is not None:
        return kwarg.group(1) in inputs
    if _NONETYPE_ATTR_RE.search(message) is not None:
        return bool(_MODALITY_INPUT_KEYS.intersection(inputs))
    return False


def call_backbone(model: Any, inputs: Dict[str, Any]) -> Any:
    """Run `model(**inputs)`, re-wrapping input-contract failures (audit Q-013).

    The backbone call is the one place a caller's modality mistake leaves
    JevX: without this boundary it surfaces as a raw upstream traceback
    (observed: `AttributeError: 'NoneType' object has no attribute 'device'`
    from `input_ids` + `pixel_values` without the processor's
    `image_grid_thw`). Failures matching :func:`_is_input_contract_error`
    become a `ValueError` that names the model class, the forwarded keys and
    the processor/modality fix — the original exception is preserved as
    `__cause__`.

    Args:
        model: The backbone (or any `**kwargs`-callable).
        inputs: Prepared forward kwargs.

    Returns:
        The model's output.

    Raises:
        ValueError: An input-contract failure (see the match rule); the
            message carries JevX guidance, `__cause__` the original.
        Others: re-raised untouched when the narrow rule does not match.
    """
    try:
        return model(**inputs)
    except (TypeError, AttributeError, ValueError) as exc:
        if not _is_input_contract_error(exc, inputs):
            raise
        model_cls = type(model).__name__
        keys = sorted(inputs)
        raise ValueError(
            f"JevX could not call the backbone {model_cls} with the prepared inputs "
            f"(keys: {keys}). Upstream {type(exc).__name__}: {exc}\n"
            "This is an INPUT/MODALITY mismatch, not a model bug: text inputs "
            "(input_ids/attention_mask) combined with vision pixel_values, or "
            "pixel_values / input_features passed without the companion metadata "
            "their processor emits (e.g. image_grid_thw for a vision-language "
            "backbone), or a modality routed under the wrong key. Pass the FULL "
            f"processor output for this modality — processor(text=..., "
            "images=..., return_tensors='pt') — or a text-only state; JevX "
            "forwards declared backbone kwargs unchanged and never invents "
            "vision/audio metadata."
        ) from exc


def observed_output_fields(outputs: Any) -> List[str]:
    """Best-effort list of non-``None`` field names on a forward output."""
    if isinstance(outputs, dict):  # covers ModelOutput (OrderedDict subclass)
        return [k for k, v in outputs.items() if v is not None]
    if isinstance(outputs, torch.Tensor):
        return ["<tensor>"]
    if isinstance(outputs, (tuple, list)):
        return [f"<tuple len={len(outputs)}>"]
    return [f"<{type(outputs).__name__}>"]


def _get_output_field(outputs: Any, name: str) -> Any:
    """None-safe field access for dict/ModelOutput/object outputs."""
    if isinstance(outputs, dict):
        return outputs.get(name)
    return getattr(outputs, name, None)


class BaseAdapter(ABC):
    """Adapter interface shared by every model family.

    Class attributes:

    * ``name`` — registry key (``"generic"``, ``"encoder"``, ...).
    * ``default_pooling`` — strategy used when ``extract_representation``
      receives ``strategy=None`` and no native pooled field exists.

    Contract deviations from the pinned interface are documented on the
    individual methods; all deviations are *additive keyword defaults* that
    keep pinned call sites source-compatible.
    """

    name: str = "base"
    default_pooling: str = "masked_mean"

    @abstractmethod
    def can_handle(self, model: Any) -> bool:
        """True iff this adapter should be selected for ``model`` (no forward)."""

    @abstractmethod
    def prepare_inputs(
        self,
        state: Any,
        *,
        tokenizer: Any = None,
        processor: Any = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Turn raw user state into model kwargs (see :class:`GenericAdapter`)."""

    def describe_capabilities(self, model: Any) -> CapabilityReport:
        """Introspection-only capability report for ``model``."""
        return JevXCapabilityDetector.detect(model)

    def _postprocess_forward_kwargs(
        self, model: Any, forward_inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Hook: adjust forward kwargs right before the call (default: none)."""
        return forward_inputs

    def _normalize_sequence(
        self, sequence: torch.Tensor, mask: Optional[torch.Tensor]
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Hook: bring a found sequence into the ``[B, T, D]`` pooling layout.

        The default is the identity: every text/frame/frame-sequence family
        already returns a rank-3 ``[B, T, D]`` tensor. Adapters whose families
        can return a different rank override this (see
        :meth:`jevx.adapters.vision.VisionAdapter._normalize_sequence` for the
        rank-4 spatial feature maps of the CNN / dense-prediction vision
        backbones).

        Args:
            sequence: The sequence tensor just selected from the output.
            mask: The mask resolved for it, or ``None``.

        Returns:
            The ``(sequence, mask)`` pair to pool and to store in
            :class:`Representation`.
        """
        return sequence, mask

    # -- shared extraction ---------------------------------------------------

    def extract_representation(
        self,
        model: Any,
        inputs: Dict[str, Any],
        *,
        strategy: Optional[str] = None,
        output_hidden_states: bool = False,
        mode: Optional[str] = None,
    ) -> Representation:
        """Run ``model(**inputs)`` and extract ``H_S`` by field priority.

        Args:
            model: The HF model (or model-like object).
            inputs: Prepared model kwargs (see :meth:`prepare_inputs`).
            strategy: Pooling strategy for the sequence path (one of
                ``jevx.representations.pooling.available_pooling_strategies``);
                ``None`` adapts — a native pooled field is preferred when
                present, otherwise ``self.default_pooling`` is applied.
                ``"native"`` *requires* a native field and raises otherwise.
            output_hidden_states: Request per-layer states (hook machinery,
                audit Q4) — only ever passed to the model when True.
            mode: Optional extraction mode; ``"decoder_state"`` additionally
                accepts decoder states from tuple outputs (step 5 of the
                priority list). Known modes are validated by subclasses that
                define more (encoder-decoder: ``"encoder_state"`` /
                ``"decoder_state"`` / ``"encoder_decoder_fusion"``).

        Returns:
            :class:`Representation` with ``pooled`` always populated.

        Raises:
            UnsupportedRepresentationError: No priority step yielded a
                representation; the error carries observed fields, attempted
                and failed strategies, and an actionable suggestion.
        """
        if not isinstance(inputs, dict):
            raise TypeError(
                f"inputs: expected a dict of model kwargs, got {type(inputs).__name__}."
            )
        report = self.describe_capabilities(model)
        # Single shared keyword contract: forward_kwargs + the use_cache gate
        # (audit Q-014) — base/generic/decoder and encoder-decoder paths all
        # route through inject_no_cache, so the kwargs-only rule cannot drift.
        fwd = inject_no_cache(inputs, model)
        if output_hidden_states:
            fwd["output_hidden_states"] = True
        fwd = self._postprocess_forward_kwargs(model, fwd)
        # Call boundary with JevX context on input-contract failures (Q-013).
        outputs = call_backbone(model, fwd)

        attempted: List[str] = []
        failed: List[str] = []

        def note(label: str, ok: bool, reason: str) -> None:
            attempted.append(label)
            if not ok:
                failed.append(f"{label}: {reason}")

        observed = observed_output_fields(outputs)
        native_allowed = strategy in (None, "native")
        selected: Optional[str] = None
        pooled: Optional[torch.Tensor] = None
        sequence: Optional[torch.Tensor] = None

        # Priority 1 — native pooler.
        if native_allowed:
            pooler_output = _get_output_field(outputs, "pooler_output")
            note("pooler_output", pooler_output is not None, f"absent from output {observed}")
            if pooler_output is not None:
                selected, pooled = "pooler_output", pooler_output
            # Priority 2 — explicit semantic fields.
            if selected is None:
                found = False
                for field_name in SEMANTIC_FIELDS:
                    value = _get_output_field(outputs, field_name)
                    if value is not None:
                        selected, pooled, found = field_name, value, True
                        note(f"semantic field {field_name!r}", True, "")
                        break
                if not found:
                    note(
                        f"semantic fields {list(SEMANTIC_FIELDS)}",
                        False,
                        f"none present in output {observed}",
                    )

        if selected is None and strategy == "native":
            self._raise_unsupported(
                model,
                report,
                observed,
                attempted=[
                    *attempted,
                    "explicit sequence pooling (skipped: strategy='native')",
                ],
                failed=[
                    *failed,
                    "native: strategy='native' was requested but the model exposes no "
                    "pooler_output/semantic field",
                ],
                suggestion=(
                    "this backbone has no trained pooler (audit Q7: llama/gpt2/t5/distilbert/"
                    "wav2vec2 ship none) — pass strategy='masked_mean'/'cls'/'last_valid' "
                    "instead of 'native'."
                ),
            )

        # Priorities 3/4 — sequence fields. Runs even when a native pooled
        # field was selected so the unpooled sequence (kept for
        # sequence-level fusions) is attached to the Representation too.
        if sequence is None:
            for field_name in ("last_hidden_state", "encoder_last_hidden_state"):
                value = _get_output_field(outputs, field_name)
                if value is not None:
                    sequence = value
                    if selected is None:
                        selected = field_name
                        note(field_name, True, "")
                    break
            else:
                note(
                    "sequence fields (last_hidden_state, encoder_last_hidden_state)",
                    selected is not None,
                    f"absent from output {observed}",
                )

        # Priority 5 — decoder state from tuple outputs when requested.
        if selected is None and mode == "decoder_state":
            ok = (
                isinstance(outputs, (tuple, list))
                and len(outputs) > 0
                and isinstance(outputs[0], torch.Tensor)
                and outputs[0].dim() == 3
            )
            note("decoder state (tuple output element 0)", ok, f"no rank-3 element in {observed}")
            if ok:
                selected, sequence = "decoder_state", outputs[0]

        # Priority 6 — last layer of collected hidden states (opt-in, audit Q4).
        if selected is None:
            hidden_states = _get_output_field(outputs, "hidden_states")
            ok = isinstance(hidden_states, (tuple, list)) and len(hidden_states) > 0
            note("hidden_states[-1]", ok, f"hidden_states not collected (output fields {observed})")
            if ok:
                selected, sequence = "hidden_states[-1]", hidden_states[-1]

        if selected is None or (pooled is None and sequence is None):
            self._raise_unsupported(
                model,
                report,
                observed,
                attempted=attempted,
                failed=failed,
                suggestion=(
                    "no standard representation field is exposed; retry with "
                    "output_hidden_states=True (audit Q4: collection is hook-based and "
                    "opt-in) or provide a model returning pooler_output/"
                    "last_hidden_state/encoder_last_hidden_state."
                ),
            )

        mask = self._resolve_mask(model, inputs, sequence, report, selected)
        if sequence is not None:
            # Bring the sequence into the [B, T, D] layout the pooling contract
            # requires; adapters that change the rank drop the stale mask here.
            sequence, mask = self._normalize_sequence(sequence, mask)
        if pooled is None:
            assert sequence is not None  # guarded above
            effective = strategy if strategy not in (None, "native") else self.default_pooling
            pooled = pool(sequence, mask, effective)

        return Representation(
            sequence_output=sequence,
            pooled=pooled,
            attention_mask=mask,
            raw_output=outputs,
            field=selected,
            layout="btd" if sequence is not None else "bd",
        )

    @staticmethod
    def _resolve_mask(
        model: Any,
        inputs: Dict[str, Any],
        sequence: Optional[torch.Tensor],
        report: CapabilityReport,
        selected_field: str,
    ) -> Optional[torch.Tensor]:
        """Pick the mask aligned with ``sequence`` (``None`` when unusable).

        For encoder-decoder outputs the decoder-side mask is preferred when
        the selected field is the decoder sequence. Audio frame masks whose
        length differs from the sequence (wav2vec2 downsamples the time axis
        ~320x, audit "wav2vec2" row) are re-aligned via the family's own
        ``_get_feature_vector_attention_mask`` hook
        (``modeling_wav2vec2.py:964``); if no hook can align them the mask
        is dropped rather than passed length-mismatched to ``pool``.
        """
        if sequence is None:
            return None
        decoder_side = report.encoder_decoder and selected_field in (
            "last_hidden_state",
            "hidden_states[-1]",
        )
        candidates = (
            ("decoder_attention_mask", "attention_mask")
            if decoder_side
            else ("attention_mask",)
        )
        mask: Optional[torch.Tensor] = None
        for key in candidates:
            value = inputs.get(key)
            if isinstance(value, torch.Tensor):
                mask = value
                break
        if mask is None:
            return None
        if tuple(mask.shape) == tuple(sequence.shape[:2]):
            return mask
        hook = getattr(model, "_get_feature_vector_attention_mask", None)
        if callable(hook):
            try:
                aligned = hook(sequence.shape[1], mask)
                if isinstance(aligned, torch.Tensor) and tuple(aligned.shape) == tuple(
                    sequence.shape[:2]
                ):
                    return aligned
            except Exception:  # noqa: BLE001 - best-effort realignment
                return None
        return None

    # -- validation ----------------------------------------------------------

    def validate(self, model: Any, sample_state: Any) -> None:
        """No-``grad`` dry run: prepare + extract, raising structured errors.

        ``sample_state`` should be raw state *or* an already-prepared input
        dict; raw ``str`` states cannot be tokenized here (no tokenizer is
        attached to the contract) and raise a structured error telling the
        caller to pass ``adapter.prepare_inputs(state, tokenizer=...)`` output
        instead.

        Raises:
            UnsupportedRepresentationError: Anything in prepare/extract fails,
                or the resulting representation has neither sequence nor
                pooled tensor; all structured fields are populated.
        """
        report = self.describe_capabilities(model)
        try:
            inputs = self.prepare_inputs(sample_state, model=model)
        except Exception as exc:  # noqa: BLE001 - converted to structured error
            self._raise_unsupported(
                model,
                report,
                detected_outputs=[report.output_class or f"<{type(model).__name__}>"],
                attempted=["prepare_inputs"],
                failed=[f"prepare_inputs({type(sample_state).__name__}): {exc}"],
                suggestion=(
                    "pass a prepared input dict (output of "
                    "adapter.prepare_inputs(state, tokenizer=...)) as sample_state, or give "
                    "raw str state a tokenizer before validate()."
                ),
                cause=exc,
            )
        was_training = bool(getattr(model, "training", False))
        try:
            if was_training:
                model.eval()
            with torch.no_grad():
                representation = self.extract_representation(model, inputs)
        except UnsupportedRepresentationError:
            raise
        except Exception as exc:  # noqa: BLE001 - converted to structured error
            self._raise_unsupported(
                model,
                report,
                detected_outputs=[report.output_class or f"<{type(model).__name__}>"],
                attempted=["extract_representation"],
                failed=[f"extract_representation: {type(exc).__name__}: {exc}"],
                suggestion=(
                    "inspect model(**inputs) manually — the forward call itself failed; "
                    "check input dtypes/shapes against the model card."
                ),
                cause=exc,
            )
        finally:
            if was_training:
                model.train()
        if representation.pooled is None and representation.sequence_output is None:
            self._raise_unsupported(
                model,
                report,
                detected_outputs=[representation.field],
                attempted=["extract_representation"],
                failed=["extract_representation: returned neither pooled nor sequence tensor"],
                suggestion="file an adapter bug — success requires at least one tensor.",
            )

    # -- structured error -----------------------------------------------------

    @staticmethod
    def _model_name(model: Any) -> str:
        config = getattr(model, "config", None)
        name = getattr(config, "_name_or_path", None)
        return name or type(model).__name__

    def _raise_unsupported(
        self,
        model: Any,
        report: CapabilityReport,
        detected_outputs: List[str],
        *,
        attempted: List[str],
        failed: List[str],
        suggestion: str,
        message: Optional[str] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        error = UnsupportedRepresentationError(
            model_name=self._model_name(model),
            architecture=report.architecture,
            detected_outputs=detected_outputs,
            attempted_strategies=attempted,
            failed_strategies=failed,
            suggestion=suggestion,
            message=message,
        )
        if cause is not None:
            raise error from cause
        raise error
