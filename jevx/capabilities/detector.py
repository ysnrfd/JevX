# Copyright (c) 2026 ysnrfd
"""Capability detection for HF models: introspection only, no forward pass.

Design constraints verified against transformers v5.16.1
(``reports/TRANSFORMERS_SOURCE_AUDIT.md``):

* Output class is read from the ``forward`` return annotation (v5 annotates
  real ``ModelOutput`` dataclasses; unions like ``tuple | BaseModelOutput...``
  are unwrapped, ``Q1``/``Q2`` of the audit).
* ``config.can_generate()`` does NOT exist on v5 ``PreTrainedConfig`` —
  generation is detected via ``model.can_generate()`` (a ``PreTrainedModel``
  classmethod, ``modeling_utils.py:1561``) with a config-level fallback for
  configs that do define it.
* Decoder-only detection cannot rely on ``config.is_decoder`` (absent on
  llama/gpt2 configs). Causality is read from four independent structural
  signals, strongest first, none of them name-based: the ``is_decoder`` flag
  when present (gpt_neox, enc-dec sub-configs); a module whose ``is_causal``
  attribute is True (``causal_attention_present`` — every causal
  self-attention in v5 is tagged); a recurrent/SSM decoding-state output field
  (:data:`RECURRENT_STATE_FIELDS` — rwkv ``state``, mamba ``cache_params``,
  consumable only by a left-to-right stack); and, as a legacy fallback, a
  ``past_key_values``-bearing output with neither a pooler nor an ``encoder``
  submodule. A submodule merely *named* ``decoder`` is not evidence: funnel's
  post-pooling half is ``FunnelDecoder`` and is still bidirectional.
* ``pooler`` presence via ``hasattr`` / ``get_submodule`` / a
  ``named_modules`` scan (catches ``model.pooler`` and ``base.pooler``).
* ``text``/``vision``/``audio`` from the forward input signature
  (``input_ids`` / ``pixel_values`` / ``input_values``/``input_features``)
  plus nested config keys, per audit §Q8.
* hybrid attention structure from ``config.layer_types`` (or the nested
  ``config.text_config.layer_types``, Qwen3.5) → ``attention_layout`` —
  parsed and reported only, never a support gate (record, don't gate:
  ``docs/HYBRID_ATTENTION_ANALYSIS.md`` §7.1).
"""

from __future__ import annotations

import dataclasses
import inspect
import types
import typing
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from transformers import PreTrainedModel
from transformers.utils import ModelOutput

__all__ = [
    "CapabilityReport",
    "JevXCapabilityDetector",
    "NATIVE_REPRESENTATION_FIELDS",
    "RECURRENT_STATE_FIELDS",
    "causal_attention_present",
    "pooler_present",
]

#: Output fields that constitute a *native* (zero-glue) representation route
#: — a dedicated pooled/semantic vector, or the enc-dec encoder field
#: (compatibility-matrix adapter column "native").
NATIVE_REPRESENTATION_FIELDS = frozenset(
    {
        "pooler_output",
        "encoder_last_hidden_state",
        "text_embeds",
        "image_embeds",
        "sentence_embedding",
    }
)

#: Output fields carrying a **recurrent / state-space decoding state** rather
#: than a KV cache. transformers documents each of them identically -- "The
#: state of the model at the last time step. Can be used in a forward method
#: with the next ``input_ids`` to avoid providing the old ``input_ids``"
#: (``modeling_mamba.py:576-581``, ``modeling_mamba2.py:681-686``,
#: ``modeling_rwkv.py:451-457``) -- only a strictly left-to-right (causal)
#: stack can consume such a state, so its presence is proof of a decoder-only
#: direction exactly where a ``past_key_values``-only rule is silent. Measured
#: by the breadth sweep: rwkv and mamba are decoder-only causal LMs that the
#: ``past_key_values`` rule reads as bidirectional encoders.
RECURRENT_STATE_FIELDS = frozenset(
    {"state", "cache_params", "conv_states", "recurrent_states", "ssm_states"}
)

#: Forward parameters excluded from ``input_fields``: control/cache flags that
#: are not representation inputs (audit §5: ``use_cache``/``past_key_values``
#: are generation machinery, ``output_*``/``return_dict`` are output controls).
_EXCLUDED_INPUT_FIELDS = frozenset(
    {
        "self",
        "kwargs",
        "output_attentions",
        "output_hidden_states",
        "return_dict",
        "use_cache",
        "past_key_values",
        "cache_position",
    }
)


@dataclass
class CapabilityReport:
    """Structured capability summary for one model.

    Field semantics:

    * ``text``/``vision``/``audio``: modalities accepted by the forward
      signature (a multimodal model has several ``True``).
    * ``encoder``/``decoder``/``encoder_decoder``: architectural directions —
      ``encoder_decoder`` mirrors ``config.is_encoder_decoder``; ``decoder``
      is True for causal stacks (output exposes ``past_key_values``, the
      ``is_decoder`` flag is set, or a ``decoder`` submodule exists);
      ``encoder`` is True for anything that is not decoder-only.
    * ``hidden_states``: per-layer states obtainable via
      ``output_hidden_states=True`` (hook machinery, audit §1.4/Q4).
    * ``pooler``: a ``pooler`` module exists somewhere in the module tree.
    * ``generation``: ``model.can_generate()`` (v5 configs have no
      ``can_generate``; a config-level callable is used when present).
    * ``native_representation``: output class exposes a field in
      :data:`NATIVE_REPRESENTATION_FIELDS` (or a pooler module exists).
    * ``fallback_representation``: sequence states
      (``last_hidden_state``/``hidden_states``) are available for generic
      pooling.
    * ``jevx_conversion``: ``"supported"`` — a standard extraction route is
      introspectable; ``"experimental"`` — route exists but semantics need a
      pinned convention (interleaved multimodal, audit level-2 rows);
      ``"unsupported"`` — no representation field could be identified.
    * ``representation_field``: preferred output field name, if any.
    * ``output_class``: ``ModelOutput`` class name from the forward return
      annotation (unions unwrapped; ``None`` if unresolvable).
    * ``input_fields``: declared forward inputs minus control flags.
    * ``processor_required``: a processor/feature extractor (not just a
      tokenizer) is needed to build inputs (vision/audio/multimodal).
    * ``attention_layout``: summary of the declared hybrid-attention
      structure ``config.layer_types`` — also read from a nested
      ``config.text_config.layer_types`` (Qwen3.5 keeps it nested; see
      ``vendor/reports/QWEN35_RECON.md``). Values: ``None`` when the config
      declares no layer types (parse-only — never *inferred*, so a plain
      ``BertConfig`` reports ``None``, not ``"full"``), ``"full"`` when
      every entry is ``full_attention``, ``"linear"`` when every entry is
      ``linear_attention``, otherwise the count form
      ``"hybrid:<count>x<type>+<count>x<type>"`` with per-type counts
      aggregated in CONFIG ORDER (first-seen order, actual config strings).
      Purely informational: it never gates ``jevx_conversion`` or any other
      verdict; mixed (hybrid) layouts additionally append one human-readable
      line to ``reasons``.
    * ``reasons``: human-readable justifications, one per decision; probe
      status is appended by :meth:`JevXCapabilityDetector.probe`.
    """

    text: bool
    vision: bool
    audio: bool
    encoder: bool
    decoder: bool
    encoder_decoder: bool
    hidden_states: bool
    pooler: bool
    generation: bool
    native_representation: bool
    fallback_representation: bool
    jevx_conversion: str
    representation_field: Optional[str]
    output_class: Optional[str]
    model_type: Optional[str]
    architecture: Optional[str]
    input_fields: List[str]
    processor_required: bool
    attention_layout: Optional[str] = None
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Return a plain-``dict`` copy (JSON-serializable field values)."""
        return dataclasses.asdict(self)


def _forward_signature(model: Any) -> Optional[inspect.Signature]:
    """Signature of ``model.forward`` or ``None`` when uninspectable."""
    try:
        return inspect.signature(model.forward)
    except (TypeError, ValueError):  # pragma: no cover - exotic callables
        return None


def _resolve_output_class(annotation: Any, model: Any = None) -> Optional[type]:
    """Unwrap a forward return annotation to its ``ModelOutput`` class.

    v5 annotates unions such as ``tuple[torch.Tensor] | BaseModelOutput...``
    (verified on ``BertModel``); the ``ModelOutput`` member is selected and
    plain ``tuple``/``dict``/``None`` members are skipped.

    String annotations (PEP 563 — ``from __future__ import annotations``,
    common in remote-code models) are resolved through
    ``typing.get_type_hints(model.forward)`` when a model is supplied.
    """
    if isinstance(annotation, str) and model is not None:
        try:
            annotation = typing.get_type_hints(model.forward).get("return", annotation)
        except Exception:  # noqa: BLE001 - unresolvable names → give up gracefully
            return None
    if inspect.isclass(annotation) and issubclass(annotation, ModelOutput):
        return annotation
    origin = typing.get_origin(annotation)
    if origin is typing.Union or origin is types.UnionType:
        for arg in typing.get_args(annotation):
            if inspect.isclass(arg) and issubclass(arg, ModelOutput):
                return arg
    return None


def _output_field_names(output_class: Optional[type]) -> Optional[frozenset]:
    """Dataclass field names of an output class, or ``None`` if unknown."""
    if output_class is None or not dataclasses.is_dataclass(output_class):
        return None
    return frozenset(f.name for f in dataclasses.fields(output_class))


def _attention_layout(config: Any) -> tuple[Optional[str], Optional[str]]:
    """Summarize the declared hybrid-attention structure ``layer_types``.

    Lookup order (Qwen3.5's list lives in the nested ``text_config`` —
    ``vendor/reports/QWEN35_RECON.md``; the top-level ``Qwen3_5Config``
    exposes no ``layer_types`` attribute):

    1. ``config.layer_types`` (list[str]);
    2. ``config.text_config.layer_types``.

    Parse-only: nothing is inferred from architecture, interval fields or
    model names (a config that declares no ``layer_types`` — e.g. plain
    ``BertConfig`` — yields ``None``).

    Returns ``(attention_layout, hybrid_reason)``:

    * ``attention_layout``: ``None`` (absent/empty/non-str entries);
      ``"full"`` when every entry is ``"full_attention"``; ``"linear"``
      when every entry is ``"linear_attention"``; otherwise
      ``"hybrid:<count>x<type>+..."`` — per-type counts aggregated in
      CONFIG ORDER (first-seen order preserved, actual config strings used).
    * ``hybrid_reason``: one human-readable line when more than one type is
      declared (a hybrid/mixed layout), else ``None``.
    """
    if config is None:
        return None, None
    layer_types = getattr(config, "layer_types", None)
    if not isinstance(layer_types, list) or not layer_types:
        text_config = getattr(config, "text_config", None)
        layer_types = getattr(text_config, "layer_types", None)
    if not isinstance(layer_types, list) or not layer_types:
        return None, None
    if not all(isinstance(entry, str) for entry in layer_types):
        return None, None
    # Config-order aggregation: first-seen order preserved, counts per type.
    counts: Dict[str, int] = {}
    for entry in layer_types:
        counts[entry] = counts.get(entry, 0) + 1
    if list(counts) == ["full_attention"]:
        layout = "full"
    elif list(counts) == ["linear_attention"]:
        layout = "linear"
    else:
        layout = "hybrid:" + "+".join(f"{n}x{t}" for t, n in counts.items())
    reason: Optional[str] = None
    if len(counts) > 1:
        human = " + ".join(f"{n} {t}" for t, n in counts.items())
        reason = f"hybrid attention layout: {human} layers (config.layer_types)"
    return layout, reason


def causal_attention_present(model: Any) -> bool:
    """True iff any attention module in the tree is marked causal.

    transformers v5 tags every causal self-attention with an ``is_causal``
    instance attribute and leaves it ``False`` on bidirectional ones, so a
    module-tree scan reads the attention direction directly, with no model
    name involved. Evidence in the vendored clone:

    * ``True`` on causal stacks -- ``models/opt/modeling_opt.py:121``,
      ``models/llama/modeling_llama.py:228``,
      ``models/falcon/modeling_falcon.py:223``,
      ``models/phi3/modeling_phi3.py:212``,
      ``models/stablelm/modeling_stablelm.py:242``,
      ``models/olmo/modeling_olmo.py:215``,
      ``models/qwen3/modeling_qwen3.py:223``,
      ``models/jamba/modeling_jamba.py:162``,
      ``models/gpt2/modeling_gpt2.py:111``,
      ``models/gemma2/modeling_gemma2.py:231``;
    * ``False`` on bidirectional stacks -- ``models/vit/modeling_vit.py:200``,
      ``models/wav2vec2/modeling_wav2vec2.py:496``,
      ``models/bert/modeling_bert.py:161,227``,
      ``models/t5/modeling_t5.py:379``.

    Recurrent / state-space stacks (rwkv, mamba) own no attention module at
    all; they are covered by :data:`RECURRENT_STATE_FIELDS` instead.

    Args:
        model: Any module tree (typically a ``PreTrainedModel``).

    Returns:
        True iff at least one submodule has ``is_causal is True``.
    """
    named_modules = getattr(model, "named_modules", None)
    if not callable(named_modules):
        return False
    try:
        return any(
            getattr(module, "is_causal", None) is True
            for _, module in named_modules()
        )
    except Exception:  # pragma: no cover - defensive
        return False


def pooler_present(model: Any) -> bool:
    """True iff a ``pooler`` module exists on ``model`` or a submodule.

    Checks, in order: the attribute (``model.pooler is not None``), a
    ``get_submodule("pooler")`` lookup, and a full ``named_modules`` scan so
    wrappers such as ``XForSequenceClassification`` are covered.
    """
    pooler = getattr(model, "pooler", None)
    if pooler is not None:
        return True
    get_submodule = getattr(model, "get_submodule", None)
    if callable(get_submodule):
        try:
            if get_submodule("pooler") is not None:
                return True
        except AttributeError:
            pass
    named_modules = getattr(model, "named_modules", None)
    if callable(named_modules):
        try:
            return any(
                name == "pooler" or name.endswith(".pooler")
                for name, _ in named_modules()
            )
        except Exception:  # pragma: no cover - defensive
            return False
    return False


def _can_generate(model: Any, config: Any) -> bool:
    """Generation capability.

    v5 ``PreTrainedConfig`` does not define ``can_generate`` (verified:
    ``hasattr(BertConfig(), "can_generate") is False``); the pinned
    ``config.can_generate()`` call is attempted first for configs that do
    define it, then ``model.can_generate()`` (the v5 mechanism,
    ``modeling_utils.py:1561``).
    """
    fn = getattr(config, "can_generate", None)
    if callable(fn):
        try:
            return bool(fn())
        except TypeError:  # bound-classmethod-style callables
            return bool(fn(model))
    fn = getattr(model, "can_generate", None)
    if callable(fn):
        try:
            return bool(fn())
        except TypeError:  # classmethod accessed oddly
            return False
    return False


class JevXCapabilityDetector:
    """Static detector producing :class:`CapabilityReport` values.

    ``detect`` performs pure introspection (signatures, config, module tree)
    and never runs the model. ``probe`` optionally runs a single cautious
    forward pass to confirm the introspected contract, marking its status in
    ``reasons``.
    """

    @staticmethod
    def detect(model: Any, config: Any = None) -> CapabilityReport:
        """Inspect ``model`` and return its capability report.

        Args:
            model: Any model-like object (typically ``PreTrainedModel``).
            config: Explicit config; defaults to ``model.config``.

        Returns:
            A :class:`CapabilityReport` — never raises for odd models;
            unidentifiable routes are reported as ``jevx_conversion ==
            "unsupported"`` with populated ``reasons``.
        """
        reasons: List[str] = []
        if config is None:
            config = getattr(model, "config", None)
        if config is None:
            reasons.append("no config found on model; falling back to signature-only detection")

        sig = _forward_signature(model)
        param_names = list(sig.parameters) if sig is not None else []
        has_var_kwargs = (
            sig is not None
            and any(
                p.kind is inspect.Parameter.VAR_KEYWORD
                for p in sig.parameters.values()
            )
        )
        if sig is None:
            reasons.append("forward signature uninspectable; input detection limited")
        input_fields = [n for n in param_names if n not in _EXCLUDED_INPUT_FIELDS]

        # --- modality detection (audit Q8: input signature + config keys) ---
        text = "input_ids" in param_names or getattr(model, "main_input_name", None) == "input_ids"
        vision = "pixel_values" in param_names or getattr(model, "main_input_name", None) == "pixel_values"
        if not vision and config is not None and hasattr(config, "vision_config"):
            vision = True
        audio = (
            "input_values" in param_names
            or "input_features" in param_names
            or getattr(model, "main_input_name", None) in {"input_values", "input_features"}
        )
        reasons.append(
            f"modalities from forward signature: text={text}, vision={vision}, audio={audio}"
        )

        # --- architectural directions -------------------------------------
        is_enc_dec = bool(getattr(config, "is_encoder_decoder", False)) if config is not None else False
        is_decoder_flag = getattr(config, "is_decoder", None) if config is not None else None

        output_class = (
            _resolve_output_class(sig.return_annotation, model) if sig is not None else None
        )
        out_fields = _output_field_names(output_class)
        if output_class is not None:
            reasons.append(f"output class from forward annotation: {output_class.__name__}")
        else:
            reasons.append("no ModelOutput class resolvable from forward annotation")

        has_encoder_sub = False
        has_decoder_sub = False
        get_submodule = getattr(model, "get_submodule", None)
        if callable(get_submodule):
            for sub, flag in (("encoder", "encoder"), ("decoder", "decoder")):
                try:
                    if get_submodule(sub) is not None:
                        if flag == "encoder":
                            has_encoder_sub = True
                        else:
                            has_decoder_sub = True
                except AttributeError:
                    pass

        cache_capable = out_fields is not None and "past_key_values" in out_fields
        recurrent_state = out_fields is not None and bool(
            out_fields & RECURRENT_STATE_FIELDS
        )
        causal_attention = causal_attention_present(model)
        pooler = pooler_present(model)
        has_pooler_field = out_fields is not None and "pooler_output" in out_fields
        # Decoder-only signature, strongest evidence first:
        #   (a) `config.is_decoder` -- gpt_neox, enc-dec sub-configs;
        #   (b) a module marked `is_causal is True` -- opt/llama/falcon/phi3/
        #       stablelm/olmo/qwen3/jamba/gpt2/gemma2 (see
        #       causal_attention_present for the source citations);
        #   (c) a RECURRENT_STATE_FIELDS output field -- rwkv `state`
        #       (modeling_rwkv.py:457), mamba/mamba2 `cache_params`
        #       (modeling_mamba.py:584), which no bidirectional encoder emits;
        #   (d) the legacy cache heuristic: a cache-capable output with no
        #       pooler (bloom/mpt/codegen, which build their causal mask
        #       without tagging a module) AND no `encoder` submodule.
        # BERT-family outputs ALSO declare past_key_values
        # (BaseModelOutputWithPoolingAndCrossAttentions, for cross-attention
        # generation) while carrying a pooler, so a pooler field/module rules
        # *out* decoder-only (audit Q7: llama/gpt2 return
        # BaseModelOutputWithPast[...], no pooler anywhere). The added
        # `encoder` submodule guard covers the pooler-free encoder stacks that
        # keep the field: nystromformer builds a bidirectional mask
        # (modeling_nystromformer.py:478) yet its output class is
        # BaseModelOutputWithPastAndCrossAttentions.
        decoder = (
            bool(is_decoder_flag)
            or causal_attention
            or recurrent_state
            or (
                cache_capable
                and not is_enc_dec
                and not has_pooler_field
                and not pooler
                and not has_encoder_sub
            )
        )
        # A submodule merely NAMED `decoder` is not evidence of causality: a
        # non-encoder-decoder model uses the name for its second stack
        # (funnel calls the post-pooling half `FunnelDecoder`,
        # modeling_funnel.py:814, and keeps a bidirectional mask; vits names its
        # vocoder stack `decoder`). Such a name only counts when the causality
        # itself is corroborated by (a)/(b)/(c) or by the cache heuristic.
        if (
            has_decoder_sub
            and not is_enc_dec
            and not decoder
        ):
            has_decoder_sub = False
        if is_enc_dec:
            encoder = True
            decoder = True
        elif decoder:
            encoder = False
        else:
            # Not decoder-only ⇒ bidirectional encoder stack (bert/distilbert/
            # vit/wav2vec2 all reach here; audit Q7 families).
            encoder = True
        reasons.append(
            f"directions: is_encoder_decoder={is_enc_dec}, is_decoder flag={is_decoder_flag}, "
            f"encoder submodule={has_encoder_sub}, decoder submodule={has_decoder_sub}, "
            f"causal attention module={causal_attention}, "
            f"output exposes past_key_values={cache_capable}, "
            f"recurrent state field={recurrent_state}, pooler field/module="
            f"{has_pooler_field}/{pooler} → encoder={encoder}, decoder={decoder}"
        )

        # --- hidden states / generation ------------------------------------
        hidden_states = (
            (config is not None and hasattr(config, "output_hidden_states"))
            or hasattr(model, "_can_record_outputs")
            or "output_hidden_states" in param_names
            or has_var_kwargs
        )
        generation = _can_generate(model, config)
        reasons.append(
            f"hidden_states support={hidden_states}, pooler present={pooler}, "
            f"generation={generation}"
        )

        # --- representation route ------------------------------------------
        model_type = getattr(config, "model_type", None) if config is not None else None
        architectures = getattr(config, "architectures", None) if config is not None else None
        architecture: Optional[str] = architectures[0] if architectures else type(model).__name__

        native = pooler or bool(out_fields is not None and (out_fields & NATIVE_REPRESENTATION_FIELDS))
        has_sequence_field = out_fields is not None and "last_hidden_state" in out_fields
        # `hidden_states` alone is the OPT-IN per-layer collection (audit Q4),
        # not a sequence field: record it as such instead of naming a
        # `last_hidden_state` the output class does not declare. vits is the
        # measured case (VitsModelOutput declares only waveform /
        # sequence_lengths / spectrogram / hidden_states, modeling_vits.py:45-60).
        only_hidden_states = (
            out_fields is not None
            and "hidden_states" in out_fields
            and not has_sequence_field
            and "encoder_last_hidden_state" not in out_fields
        )
        fallback = has_sequence_field or only_hidden_states

        representation_field: Optional[str] = None
        if pooler and (out_fields is None or "pooler_output" in out_fields):
            representation_field = "pooler_output"
        elif out_fields is not None and "pooler_output" in out_fields:
            representation_field = "pooler_output"
        elif is_enc_dec and (out_fields is None or "encoder_last_hidden_state" in out_fields):
            representation_field = "encoder_last_hidden_state"
        elif out_fields is not None and (out_fields & {"text_embeds", "image_embeds", "sentence_embedding"}):
            # Same priority as adapters.base.SEMANTIC_FIELDS (sentence wins, then
            # text over image — "post-projection for cross-modal comparability",
            # matrix clip/siglip rows) so capability_report.representation_field
            # always names the field extract_representation will actually pick.
            representation_field = next(
                field
                for field in ("sentence_embedding", "text_embeds", "image_embeds")
                if field in out_fields
            )
        elif has_sequence_field:
            representation_field = "last_hidden_state"
        elif only_hidden_states:
            # Priority 6 of the extraction list: the opt-in collection's last
            # element. Name it as such, and say it costs
            # output_hidden_states=True (audit Q4).
            representation_field = "hidden_states[-1]"
            reasons.append(
                "no pooled/sequence output field; the only route is the opt-in "
                "per-layer collection hidden_states[-1] (output_hidden_states=True, "
                "audit Q4)"
            )

        multimodal = text and vision
        if representation_field is None:
            jevx_conversion = "unsupported"
            reasons.append("no representation field identifiable → unsupported")
        elif config is None and not isinstance(model, PreTrainedModel):
            jevx_conversion = "unsupported"
            reasons.append("neither a config nor a PreTrainedModel → unsupported")
        elif representation_field in NATIVE_REPRESENTATION_FIELDS and not multimodal:
            jevx_conversion = "supported"
            reasons.append(f"native field {representation_field!r} → supported")
        elif multimodal and representation_field == "last_hidden_state":
            # Interleaved image/text tokens: pooling must pin a token-selection
            # convention (compatibility matrix level-2 rows: llava_next).
            jevx_conversion = "experimental"
            reasons.append(
                "multimodal interleaved states pool lossily → experimental "
                "(pin a token-selection convention)"
            )
        else:
            jevx_conversion = "supported"
            reasons.append(f"standard field {representation_field!r} → supported")

        processor_required = bool(vision or audio)
        if processor_required:
            reasons.append("vision/audio inputs require a processor or feature extractor")

        if architectures:
            reasons.append(f"architecture from config.architectures: {architecture}")

        # --- hybrid attention layout (informational; never gates verdicts) ---
        # layer_types determines transformers' per-layer mask/cache dispatch
        # (docs/HYBRID_ATTENTION_ANALYSIS.md §7.1: record it, never gate on it).
        # Only MIXED (hybrid) layouts add a reasons line — the field itself
        # carries the summary for every other case.
        attention_layout, hybrid_reason = _attention_layout(config)
        if hybrid_reason is not None:
            reasons.append(hybrid_reason)

        return CapabilityReport(
            text=text,
            vision=vision,
            audio=audio,
            encoder=encoder,
            decoder=decoder,
            encoder_decoder=is_enc_dec,
            hidden_states=hidden_states,
            pooler=pooler,
            generation=generation,
            native_representation=native,
            fallback_representation=fallback,
            jevx_conversion=jevx_conversion,
            representation_field=representation_field,
            output_class=output_class.__name__ if output_class is not None else None,
            model_type=model_type,
            architecture=architecture,
            input_fields=input_fields,
            processor_required=processor_required,
            attention_layout=attention_layout,
            reasons=reasons,
        )

    @staticmethod
    def probe(model: Any, dummy_inputs: dict) -> CapabilityReport:
        """Detect, then confirm with a single cautious forward pass.

        Runs ``model.eval()`` under ``torch.no_grad()`` exactly once with
        ``dummy_inputs``; the model's previous training state is restored
        afterwards. Probe status (success with observed output fields, or the
        exception) is appended to ``reasons``; on success the report's
        ``output_class``/``representation_field``/``native_representation``
        are refined from the *observed* output keys.

        Args:
            model: Model to probe.
            dummy_inputs: Keyword mapping passed to ``model(**dummy_inputs)``.

        Returns:
            The (possibly refined) :class:`CapabilityReport`. Probe failures
            do not raise — they are recorded in ``reasons``.
        """
        report = JevXCapabilityDetector.detect(model)
        was_training = bool(getattr(model, "training", False))
        try:
            model.eval()
            with torch.no_grad():
                output = model(**dummy_inputs)
        except Exception as exc:  # noqa: BLE001 - probe must never raise
            report.reasons.append(
                f"probe: forward FAILED with {type(exc).__name__}: {exc}"
            )
            return report
        finally:
            if was_training:
                model.train()

        if isinstance(output, ModelOutput):
            keys = [k for k, v in output.items() if v is not None]
            report.output_class = type(output).__name__
        elif isinstance(output, dict):
            keys = [k for k, v in output.items() if v is not None]
        elif isinstance(output, torch.Tensor):
            keys = ["<tensor>"]
        elif isinstance(output, (tuple, list)):
            keys = [f"<tuple len={len(output)}>"]
        else:
            keys = [f"<{type(output).__name__}>"]
        report.reasons.append(f"probe: forward ok; observed output fields {keys}")

        observed = set(keys)
        native_observed = observed & NATIVE_REPRESENTATION_FIELDS
        fallback_observed = observed & {"last_hidden_state", "hidden_states"}
        if native_observed:
            report.native_representation = True
            if report.representation_field is None or report.representation_field == "last_hidden_state":
                report.representation_field = sorted(native_observed)[0]
        if fallback_observed:
            report.fallback_representation = True
            if report.representation_field is None:
                report.representation_field = "last_hidden_state"
        if report.representation_field is None:
            report.jevx_conversion = "unsupported"
            report.reasons.append("probe: output exposes no known representation field → unsupported")
        elif report.jevx_conversion == "unsupported":
            report.jevx_conversion = "supported"
            report.reasons.append(
                f"probe: representation field {report.representation_field!r} confirmed → supported"
            )
        return report
