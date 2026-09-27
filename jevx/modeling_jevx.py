# Copyright (c) 2026 ysnrfd
"""``JevXForDecision``: the typed decision wrapper over an HF backbone.

Pipeline implemented here (``docs/MATHEMATICAL_SPECIFICATION.md``)::

    H_S = E_M(S)          adapter.prepare_inputs + extract_representation
    H_Q = E_Q(Q)          QuestionEncoder (batched over all questions)
    H_C = E_C(C)          SchemaEncoder  (variable K, bool mask)
    H   = F(H_S, H_Q, H_C)    build_fusion(config.fusion_type, D)
    P   = G_type(H, ...)      decision heads (choice / score / noul)
    P'  = Cal(P)              CalibrationEngine (logits scale, see below)
    D   = Project(P', C)      project_to_schema + per-question abstention

Execution flow of one ``forward`` call (m questions, B states, n = B*m
pairs) — the shared-state property (Def. 11 / T7):

1. the state batch is featurized and passed through the backbone
   **exactly once** (one ``extract_representation`` call) → ``H_S [B, D]``;
2. all m questions are encoded in **one** ``encode_texts`` batch → ``H_Q``
   (the question path uses the hashed bag-of-tokens ``E_Q`` by default and
   never invokes the shared backbone);
3. schemas of the group's questions are encoded with variable ``K`` →
   ``H_C [g, K_max, D_c]`` + bool mask;
4. fusion runs per (state, question) row; rows never interact — no
   operation reduces over the question axis (A7.3);
5. heads run **per decision-type group** (mixed types in one call are
   partitioned first — probability spaces are never concatenated);
6. calibration, when configured (``calibration_method != "none"`` — applied
   over VALID option columns only for variable-K batches, audit Q-012) AND
   fitted, is applied to the head **logits** before the probability map:
   ``z' = Cal(z)``, ``P' = G_type(z')``. For temperature/vector this is
   family-equivalent to calibrating probabilities (softmax is invariant to
   the additive constant), and it is the only well-defined form for the
   K == 1 noul sigmoid; isotonic returns a ``log p'`` representative that
   ``G_type`` exponentiates back exactly. Calibration runs BEFORE
   confidence (confidence sees ``P'``) and BEFORE projection;
7. ``ConfidenceEngine`` computes per-question confidence from ``P'``;
8. abstention: ``confidence < confidence_threshold`` sets
   ``abstained=True`` and clears ``predicted_index`` / ``predicted_value`` /
   ``expected_score`` for that question only — probabilities are untouched;
9. outputs are assembled into :class:`JevXOutput`.

``decisions`` layout: a **flat list of B*m single-row**
:class:`DecisionOutput` objects, **state-major**, i.e. entry
``b * m + j`` holds state ``b`` answered with question ``j``. Each entry
keeps head batch semantics with a leading dim of 1 (``logits [1, K]`` ...).

Save/load strategy (§43): the backbone is embedded as the child module
``model`` inside ONE self-contained checkpoint — ``save_pretrained``
writes ``config.json`` (incl. the nested ``backbone_config`` dict),
``model.safetensors`` (child ties declared through ``post_init``
aggregation, so safetensors shared-tensor removal drops aliases such as a
causal LM's ``lm_head.weight`` and the tie is restored on load), plus
``jevx_metadata.json`` (capability report + adapter identity, JSON) and
the tokenizer/processor files when attached. Calibration and confidence
fit state live in the regular ``state_dict`` (submodule ``calibrator`` /
``confidence``) — fit-data-shaped calibrator buffers (isotonic) are
reallocated in :meth:`JevXForDecision.load_state_dict`. Reload goes
through the standard ``PreTrainedModel.from_pretrained`` with an
explicit ``JevXConfig`` (no Auto-class registration required).

``base_forward(*args, **kw)`` (§31) forwards to the untouched backbone;
in eval mode it is bit-identical to loading the base checkpoint directly
(tested: ``max_abs_error`` reported per checkpoint in the test suite).

Deviations / conventions worth knowing:

* ``decision_type`` defaults to ``None`` → resolved as
  explicit argument → ``config.decision_type`` → ``"choice"`` (a superset
  of a plain ``"choice"`` default). It is a declared
  ``JevXConfig`` field: validated against the head registry at
  construction and round-tripped through save/load.
* ``config.conditioning_mode`` is the source of truth for whether the
  OPTION features condition a call; ``config.conditioning_enabled`` is its
  derived alias, resolved per call by `_resolve_condition`. ``"off"``
  (the DEFAULT) is bitwise ``forward(condition_options=False)`` and is
  state-only in BOTH conditioning paths: the option residual is exactly zero
  AND the question embedding entering fusion is exactly zero, so no part of
  the call reads the question or the schema. ``"constant"`` memoizes
  ``H_Q``/``H_C`` on their input content (inference only, and WEIGHT-keyed —
  see `_condition_cache_enabled` / `_condition_cache_sync`).
* ``decision_type`` accepts a list (one entry per question — lengths 1, m
  or B*m) so mixed types fit one call.
* ``adapter.extract_representation(strategy=...)`` receives
  ``config.pooling_method`` — in the adapter contract ``strategy`` is the
  *pooling* strategy (``None`` = adapter default: native pooler when
  present, ``cls``/``last_valid``/``encoder_pool`` otherwise);
  ``config.representation_strategy`` is metadata only.
* ``state_representation`` holds the shared projected ``H_S [B, D]``;
  the per-row fused ``H = F(H_S, H_Q, H_C)`` is internal to ``forward`` and
  is **not** exposed on the output (``capability_metadata`` carries no fused
  vectors — see docs/API.md); ``question_representations`` is ``[B, m, D]``;
  ``schema_representations`` is ``[B*m, K_max, D]``.
"""

from __future__ import annotations

import dataclasses
import json
import warnings
from dataclasses import fields as _config_dataclass_fields
from dataclasses import replace as _dataclass_replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoProcessor,
    AutoTokenizer,
    PreTrainedConfig,
    PreTrainedModel,
)
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

from .adapters import resolve_adapter
from .calibration import CalibrationEngine
from .capabilities import JevXCapabilityDetector, UnsupportedRepresentationError
from .configuration_jevx import JevXConfig, validate_decision_type
from .confidence import ConfidenceEngine
from .decisions import available_decision_heads, build_decision_head, project_to_schema
from .modeling_outputs import DecisionOutput, JevXOutput
from .questions import Option, QuestionEncoder, SchemaEncoder
from .representations import Projection, build_fusion
from .training.freeze import freeze_module, unfreeze_last_n_layers as _unfreeze_layers
from .utils.shapes import check_finite
from .validation import ValidationEngine

__all__ = ["JevX", "JevXForDecision", "coverage_at_risk", "risk_coverage_curve"]

__version__ = "0.1.0"  # mirrored by jevx.__init__ (kept local to avoid a cycle)

#: ``model_type`` of a wrapped checkpoint's base config, and the hidden-width
#: keys probed in order across the audited families (BERT ``hidden_size``,
#: T5 ``d_model``, GPT-2 ``n_embd``, ...).
_HIDDEN_WIDTH_KEYS: Tuple[str, ...] = (
    "hidden_size",
    "d_model",
    "n_embd",
    "hidden_dim",
    "d_embed",
    "embed_dim",
    "model_dim",
    "dim",
)

#: Config fields that declare a per-stage width **LIST** instead of a scalar.
#: A convolutional backbone carries one width per stage, and the value a head
#: consumes is the width of the LAST stage the model actually builds:
#:
#: * ``hidden_sizes`` -- convnext (``self.classifier = nn.Linear(
#:   config.hidden_sizes[-1], config.num_labels)``, modeling_convnext.py:304),
#:   resnet (``nn.Sequential(nn.Flatten(), nn.Linear(config.hidden_sizes[-1],
#:   config.num_labels))``, modeling_resnet.py:346-348), poolformer
#:   (``nn.Linear(config.hidden_sizes[-1], config.num_labels)``,
#:   modeling_poolformer.py:326) and segformer (``SegformerStage`` emits
#:   ``hidden_size=config.hidden_sizes[stage_idx]``,
#:   modeling_segformer.py:321).
_STAGE_WIDTH_LIST_KEYS: Tuple[str, ...] = (
    "hidden_sizes",
)

#: Config fields that declare how many of those stages the model ACTUALLY
#: builds, when that is fewer than the declared list is long. ``SegformerModel``
#: builds exactly ``config.num_encoder_blocks`` stages
#: (modeling_segformer.py:384-386) and returns the last one (
#: modeling_segformer.py:399-401), so a list longer than the block count is
#: indexed at the block count, NOT at ``-1``.
_STAGE_COUNT_KEYS: Tuple[str, ...] = ("num_encoder_blocks", "num_stages")

#: Keyword arguments consumed by the *backbone* loader in the wrap path of
#: :meth:`JevXForDecision.from_pretrained` (audit: ``dtype`` is canonical,
#: ``torch_dtype`` BC; the rest are hub/attention controls).
_BACKBONE_LOAD_KWARGS: Tuple[str, ...] = (
    "cache_dir",
    "force_download",
    "local_files_only",
    "revision",
    "token",
    "subfolder",
    "proxies",
    "resume_download",
    "dtype",
    "torch_dtype",
    "attn_implementation",
    "ignore_mismatched_sizes",
    "use_safetensors",
    "variant",
    "device_map",
)

#: Hub kwargs forwarded when merely *peeking* a remote/local config.
_HUB_PEEK_KWARGS: Tuple[str, ...] = (
    "cache_dir",
    "force_download",
    "local_files_only",
    "revision",
    "token",
    "subfolder",
)

#: Auto classes tried (in order) to rebuild a saved backbone from its nested
#: ``backbone_config``; the first whose model class matches the saved
#: ``architectures`` entry wins, else the first candidate is used.
_REBUILD_AUTO_CLASSES: Tuple[Any, ...] = (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
)

_METADATA_FILENAME = "jevx_metadata.json"
_METADATA_FORMAT = "jevx-metadata-v1"
_CALIBRATION_FILENAME = "jevx_calibration.pt"
_CALIBRATION_FORMAT = "jevx-calibration-v1"
_VALID_DECISION_TYPES_FROZEN = ("choice", "score", "noul")


# ---------------------------------------------------------------------------
# Abstention / risk-coverage utilities (§62)
# ---------------------------------------------------------------------------


def _validate_risk_inputs(
    confidences: Any, correct_mask: Any
) -> tuple[torch.Tensor, torch.Tensor]:
    """Coerce and validate ``(confidences, correct_mask)`` for §62 curves."""
    if not isinstance(confidences, torch.Tensor):
        confidences = torch.as_tensor(confidences)
    if not isinstance(correct_mask, torch.Tensor):
        correct_mask = torch.as_tensor(correct_mask)
    if confidences.ndim != 1:
        raise ValueError(
            f"confidences: expected a 1-D tensor of per-question confidences, got shape "
            f"{tuple(confidences.shape)}."
        )
    if correct_mask.ndim != 1:
        raise ValueError(
            f"correct_mask: expected a 1-D tensor, got shape {tuple(correct_mask.shape)}."
        )
    if confidences.shape[0] != correct_mask.shape[0]:
        raise ValueError(
            f"correct_mask: expected one entry per confidence — confidences has "
            f"{confidences.shape[0]} entries but correct_mask has {correct_mask.shape[0]}."
        )
    if not confidences.is_floating_point():
        confidences = confidences.float()
    check_finite(confidences, "confidences")
    return confidences, correct_mask.bool()


def risk_coverage_curve(
    confidences: Any, correct_mask: Any
) -> tuple[torch.Tensor, torch.Tensor]:
    """Risk-coverage curve (§62, Eqs. 10.2–10.3) by confidence ranking.

    Sorts questions by confidence (descending, stable) and walks the
    resulting ranking: with ``n`` questions, coverage ``γ_i = (i+1)/n`` is
    the fraction of highest-confidence decisions *covered* up to rank ``i``,
    and risk ``ρ_i`` is the empirical error rate ``mean(1 - correct)`` among
    them.

    Args:
        confidences: Per-question confidences ``[n]`` (any dtype; finite).
        correct_mask: ``[n]`` correctness flags (bool, or 0/1 convertible).

    Returns:
        ``(coverage, risk)`` — two ``[n]`` float32 tensors (``n >= 1``);
        for ``n == 0`` two empty tensors. Confidence ties are broken by a
        stable sort (original order), so the curve is deterministic.

    Raises:
        ValueError: Rank/dtype/length problems; the message names the
            offending argument.
    """
    conf, correct = _validate_risk_inputs(confidences, correct_mask)
    n = conf.shape[0]
    if n == 0:
        empty = torch.zeros(0, dtype=torch.float32)
        return empty, empty.clone()
    order = torch.argsort(conf, descending=True, stable=True)
    errors = (~correct[order]).to(torch.float32)
    risk = errors.cumsum(dim=0) / torch.arange(
        1, n + 1, dtype=torch.float32, device=errors.device
    )
    coverage = torch.arange(1, n + 1, dtype=torch.float32, device=errors.device) / n
    return coverage, risk


def coverage_at_risk(confidences: Any, correct_mask: Any, max_risk: float) -> float:
    """Maximum coverage achievable at risk ``<= max_risk`` (§62, Eq. 10.4).

    Args:
        confidences: Per-question confidences ``[n]``.
        correct_mask: ``[n]`` correctness flags.
        max_risk: Risk budget in ``[0, 1]``.

    Returns:
        ``rho*(max_risk)`` in ``[0, 1]`` — the largest coverage on the
        confidence ranking whose empirical error rate stays within budget
        (``0.0`` when even the single most confident decision exceeds it;
        ``1.0`` when the full set does). ``n == 0`` returns ``0.0``.

    Raises:
        ValueError: ``max_risk`` outside ``[0, 1]``, or the tensor problems
            of :func:`risk_coverage_curve`.
    """
    if isinstance(max_risk, bool) or not isinstance(max_risk, (int, float)):
        raise TypeError(f"max_risk: expected a float in [0, 1], got {type(max_risk).__name__}.")
    if not 0.0 <= float(max_risk) <= 1.0:
        raise ValueError(f"max_risk: expected a value in [0, 1], got {max_risk!r}.")
    coverage, risk = risk_coverage_curve(confidences, correct_mask)
    if coverage.numel() == 0:
        return 0.0
    ok = risk <= float(max_risk)
    if not bool(ok.any()):
        return 0.0
    return float(coverage[ok].max())


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _declares(config: Any, key: str) -> bool:
    """Is ``key`` part of the config CLASS schema rather than a stray kwarg?

    ``PreTrainedConfig.__init__`` copies unknown constructor kwargs onto the
    instance, so an attribute can exist without belonging to the width schema
    of the family. ``MobileNetV2Config`` declares no width field at all
    (configuration_mobilenet_v2.py:60-76), yet a caller may still pass
    ``hidden_sizes=[8, 16]``; the attribute then reads back as if it were the
    backbone per-stage widths, while the real output width is
    ``conv_1x1.convolution.out_channels`` (modeling_mobilenet_v2.py:313-318,
    381). The ``hidden_sizes`` of the families that DO declare it are
    dataclass fields of the config class (configuration_convnext.py:49,
    configuration_resnet.py:57, configuration_segformer.py:63,
    configuration_poolformer.py), so the declared-field test separates the two
    without naming a family. Dicts carry no schema and are accepted as-is.

    Args:
        config: A ``PreTrainedConfig`` or a plain dict.
        key: The config attribute to test.

    Returns:
        ``True`` when the key is declared by the config class, or when the
        config is a dict and no schema is available to check.
    """
    if isinstance(config, dict):
        return True
    cls = type(config)
    if dataclasses.is_dataclass(cls):
        return key in {field.name for field in dataclasses.fields(cls)}
    return hasattr(cls, key)


def _stage_width(config: Any) -> Optional[int]:
    """Width of the last stage the model actually builds.

    Rule 4 of :func:`_hidden_width`: a convolutional config declares a
    per-stage width list and no scalar, and the value a classification or
    segmentation head consumes is the FINAL stage width. "Final" means the
    last stage the model really builds, which is not always ``[-1]``:
    ``SegformerModel`` builds exactly ``config.num_encoder_blocks`` stages
    (modeling_segformer.py:384-386) and returns the last one (
    modeling_segformer.py:399-401), so a ``hidden_sizes`` list longer than the
    block count is indexed at the block count. convnext/resnet/poolformer build
    one stage per ``depths`` entry (configuration_convnext.py:50,61;
    configuration_resnet.py:57-58) and their lists are the same length, so
    ``[-1]`` already is the last active stage for them.

    Args:
        config: A ``PreTrainedConfig`` or its ``to_diff_dict()`` form.

    Returns:
        The final active stage width, or ``None`` when the config declares no
        per-stage width list.
    """
    getter = config.get if isinstance(config, dict) else lambda k: getattr(config, k, None)
    for key in _STAGE_WIDTH_LIST_KEYS:
        if not _declares(config, key):
            # A value the config class never declared is not a width schema.
            continue
        stages = getter(key)
        if not isinstance(stages, (list, tuple)) or not stages:
            continue
        count = len(stages)
        for count_key in _STAGE_COUNT_KEYS:
            declared = getter(count_key)
            if (
                isinstance(declared, int)
                and not isinstance(declared, bool)
                and 0 < declared < count
            ):
                count = declared
                break
        value = stages[count - 1]
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return None


def _last_conv_width(model: Any) -> Optional[int]:
    """Output channels of the last registered ``nn.Conv2d`` of the backbone.

    Rule 5 of :func:`_hidden_width`: the last resort, for configs that carry
    no width field at all. ``MobileNetV2Config`` declares neither
    ``hidden_size`` nor ``hidden_sizes`` (configuration_mobilenet_v2.py:60-76)
    and the classification head reads its input width straight off the final
    convolution -- ``last_hidden_size =
    self.mobilenet_v2.conv_1x1.convolution.out_channels``
    (modeling_mobilenet_v2.py:381) -- which is the last module the feature
    extractor builds (modeling_mobilenet_v2.py:313-318). The head input width
    and the last registered conv ``out_channels`` are therefore the same
    number by construction, not by coincidence.

    Deliberately AFTER :func:`_stage_width`: a stage Mix-FFN widens to
    ``int(hidden_size * mlp_ratio)`` (modeling_segformer.py:285) and its
    depth-wise conv is ``nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)``
    (modeling_segformer.py:211), so for segformer the last registered conv is
    the MLP INTERNAL width (measured: 32 for a stage width of 16), not the
    stage output.

    Args:
        model: The constructed backbone.

    Returns:
        The last registered conv ``out_channels``, or ``None`` when the
        backbone holds no convolution.
    """
    width: Optional[int] = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            channels = getattr(module, "out_channels", None)
            if isinstance(channels, int) and not isinstance(channels, bool) and channels > 0:
                width = channels
    return width


def _hidden_width(config: Any, model: Any = None) -> Optional[int]:
    """Width of the pooled state vector for a config/dict (and its model).

    Order (first positive int wins):

    1. :data:`_HIDDEN_WIDTH_KEYS` on the config itself (plain families);
    2. ``projection_dim`` — composite contrastive configs (clip) where the
       adapter extracts the POST-projection semantic field
       (``text_embeds``, adapters/base SEMANTIC_FIELDS), so the pooled
       width is the projection width, not the encoder hidden size;
    3. nested ``text_config`` / ``vision_config`` (llava_next-style
       composites whose pooled vector comes from interleaved
       ``last_hidden_state`` tokens of the language tower);
    4. :func:`_stage_width` — the CNN families, whose configs declare a
       per-stage width LIST and no scalar (convnext/resnet/segformer, and
       poolformer, which declares the same field);
    5. :func:`_last_conv_width` — only when ``model`` is supplied, for the
       configs that carry no width field at all (mobilenet_v2).

    Rules 1-3 are unchanged and still run first, so every family that
    resolved before resolves to the same value; 4-5 can only fire where this
    function used to return ``None``. Config and model are always built from
    the same config object, so the picked width agrees with what extraction
    actually returns (probe-verified for clip/siglip/llava_next in the
    real-model matrix and for all 70 breadth-sweep families: the width this
    returns equals the extracted ``[B, D]`` width on every one).

    Args:
        config: The backbone config (or its ``to_diff_dict()`` form).
        model: The constructed backbone, when available. Only rule 5 uses it.

    Returns:
        The pooled-state width, or ``None`` when no rule resolves one.
    """
    getter = config.get if isinstance(config, dict) else lambda k: getattr(config, k, None)
    for key in _HIDDEN_WIDTH_KEYS:
        value = getter(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    projection = getter("projection_dim")
    if isinstance(projection, int) and not isinstance(projection, bool) and projection > 0:
        return projection
    for nested_key in ("text_config", "vision_config"):
        nested = getter(nested_key)
        if isinstance(nested, (dict, PreTrainedConfig)):
            width = _hidden_width(nested)
            if width is not None:
                return width
    stage = _stage_width(config)
    if stage is not None:
        return stage
    if model is not None:
        return _last_conv_width(model)
    return None


def _question_text(question: Any, index: int) -> str:
    """Extract the text of question ``index`` (str, or dict with text key)."""
    if isinstance(question, str):
        return question
    if isinstance(question, dict):
        for key in ("text", "question"):
            value = question.get(key)
            if isinstance(value, str):
                return value
        raise ValueError(
            f"questions[{index}]: dict questions need a 'text' (or 'question') string key — "
            f"got keys {sorted(question)}. Per-question decision types are passed via "
            "decision_type=[...] (one entry per question), not inside the question dict."
        )
    raise TypeError(
        f"questions[{index}]: expected a str or dict, got {type(question).__name__}."
    )


def _coerce_option(option: Any, question_index: int, option_index: int) -> Option:
    """Normalize one schema entry to :class:`Option` with an actionable error."""
    if isinstance(option, Option):
        return option
    if isinstance(option, str):
        return Option(key=option, label=option)
    if isinstance(option, dict):
        allowed = {"key", "label", "description", "metadata"}
        unknown = sorted(set(option) - allowed)
        if unknown or "key" not in option:
            raise ValueError(
                f"options[{question_index}][{option_index}]: option dicts need a 'key' and may "
                f"only carry {sorted(allowed)} fields; got unknown {unknown} "
                f"{'(missing key)' if 'key' not in option else ''}."
            )
        return Option(**option)
    if isinstance(option, bool):
        raise TypeError(
            f"options[{question_index}][{option_index}]: bool options are ambiguous — pass a "
            "str/Option/dict."
        )
    if isinstance(option, (int, float)):
        # Ordinal levels: key preserved as given, text falls back to str(key).
        return Option(key=option, label=str(option))
    raise TypeError(
        f"options[{question_index}][{option_index}]: expected an Option, str, dict or number, "
        f"got {type(option).__name__}."
    )


def _normalize_adapter(adapter: Any) -> Any:
    """Accept an adapter instance or class; classes are instantiated."""
    if isinstance(adapter, type):
        return adapter()
    return adapter


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------


class JevXForDecision(PreTrainedModel):
    """Typed probabilistic decision layer wrapping one HF backbone.

    Args:
        config: The :class:`JevXConfig`.
        backbone: A pre-loaded backbone to embed (factory path). When
            ``None`` the backbone is rebuilt from ``config.backbone_config``
            through the Auto classes (checkpoint-reload path — happens on
            ``meta`` device inside ``from_pretrained``).
        adapter: Adapter instance; ``None`` resolves one via
            ``jevx.adapters.resolve_adapter``.
        tokenizer / processor: Attached featurizers for raw text / modality
            states (saved alongside the checkpoint by
            ``save_pretrained``).

    Child modules: ``model`` (backbone, per the audit wrapper convention —
    also what ``jevx.training.freeze`` resolution expects), ``state_proj``,
    ``question_encoder``/``question_proj``, ``schema_encoder``/`schema_proj``,
    ``fusion``, ``heads`` (ModuleDict), ``confidence``, ``calibrator``.
    ``config.schema_encoder_width`` narrows ``schema_encoder`` +
    ``schema_proj``; ``config.shared_question_schema_encoder`` is RESERVED
    and rejected by ``JevXConfig``, so there are always two encoders.
    ``get_input_embeddings``/``set_input_embeddings`` are overridden to
    delegate to the backbone (audit risk #2: the default name heuristics
    raise ``NotImplementedError`` for most children).

    ``trainable_parameters()`` yields the *structural* JevX-side parameters
    (everything not under ``model.``) — its complement is the backbone,
    which is exactly the contract ``jevx.training.freeze.freeze_backbone``
    / ``unfreeze_backbone`` rely on. For Mode B optimization over the
    backbone, iterate ``model.parameters()``.
    """

    config_class = JevXConfig
    base_model_prefix = "model"

    def __init__(
        self,
        config: JevXConfig,
        *,
        backbone: Optional[nn.Module] = None,
        adapter: Any = None,
        tokenizer: Any = None,
        processor: Any = None,
        **_: Any,
    ) -> None:
        super().__init__(config)
        if backbone is None:
            backbone = self._build_backbone(config)
        else:
            # Pre-loaded (materialized) backbone: mark it initialized so
            # post_init()/init_weights never re-draw pretrained values
            # (audit: wrappers must preserve base weights bit-for-bit).
            for module in backbone.modules():
                module._is_hf_initialized = True  # noqa: SLF001 - HF contract
        self.model = backbone
        # Failure-mode contract (sec 90 row 6): learned absolute position tables
        # hard-fail past num_embeddings with a cryptic kernel error - detect
        # the limit eagerly so forward can raise a descriptive ValueError.
        self._learned_position_limit = self._learned_position_table_limit(backbone)
        if not getattr(config, "backbone_config", None):
            # A directly constructed wrapper must still embed the base config:
            # save_pretrained's config.json is documented to include the nested
            # backbone_config so the checkpoint reloads without the base id.
            # architectures pins the child CLASS: the checkpoint's key layout
            # (model.transformer.* vs model.*) only matches the class we saved,
            # which is not always the AutoModel default (GPT2LMHeadModel etc.).
            config.backbone_config = self.model.config.to_diff_dict()
        if not (config.backbone_config.get("architectures") or [None])[0]:
            config.backbone_config["architectures"] = [type(self.model).__name__]

        self.capability_report = JevXCapabilityDetector.detect(self.model, self.model.config)
        self.adapter = _normalize_adapter(
            adapter if adapter is not None else resolve_adapter(self.model)
        )
        self.tokenizer = tokenizer
        self.processor = processor
        self.jevx_metadata: Optional[Dict[str, Any]] = None

        d_model = _hidden_width(self.model.config, self.model)
        if d_model is None and isinstance(getattr(config, "backbone_config", None), dict):
            d_model = _hidden_width(config.backbone_config, self.model)
        if d_model is None:
            raise ValueError(
                f"cannot determine the backbone's hidden width from config keys "
                f"{list(_HIDDEN_WIDTH_KEYS)} — pass decision_hidden_size explicitly or wrap "
                "via JevX.from_pretrained so backbone_config carries the base config."
            )
        hidden = config.decision_hidden_size

        self.state_proj = Projection(d_model, hidden)
        # G0.1: the option encoder is BUILT from the config, so
        # schema_encoder_width changes the module tree instead of being
        # accepted and then ignored. The default reproduces today's stack
        # exactly. (shared_question_schema_encoder stays False: it is
        # RESERVED and JevXConfig rejects True, see that field.)
        if config.shared_question_schema_encoder:  # pragma: no cover - config rejects it
            raise ValueError(
                "shared_question_schema_encoder=True is RESERVED (not implemented); "
                "JevXConfig rejects it at construction."
            )
        self.question_encoder = QuestionEncoder(hidden_size=config.question_hidden_size)
        self.question_proj = Projection(config.question_hidden_size, hidden)
        # schema_encoder_width narrows E_C (and schema_proj's input) without
        # touching E_Q; None = schema_hidden_size, i.e. today's widths.
        schema_width = (
            config.schema_hidden_size
            if config.schema_encoder_width is None
            else config.schema_encoder_width
        )
        self.schema_encoder = SchemaEncoder(hidden_size=schema_width)
        self.schema_proj = Projection(schema_width, hidden)
        self.fusion = build_fusion(config.fusion_type, hidden)
        # G0.1 conditioning_mode="constant": a per-model, NON-PERSISTENT,
        # content-addressed memo of H_Q / H_C. A plain dict attribute (never a
        # buffer), so it never enters state_dict or save_pretrained.
        self._condition_cache: Dict[Tuple[Any, ...], Any] = {}
        # The WEIGHT the memo was computed from: one tensor version counter per
        # parameter/buffer tensor of the two memoized encoders (question and
        # schema). `_condition_cache_sync` compares it on every lookup that is
        # allowed to consult the memo, so an in-place parameter update (an
        # optimizer step, `load_state_dict`, a manual `param.data = ...`) drops
        # the memo instead of serving encodings computed from the OLD weights.
        # None = "nothing memoized since the last invalidation".
        self._condition_cache_versions: Optional[Tuple[int, ...]] = None
        # Spec Def 5.e (audit Q-006): only the cross_attention kind consumes
        # the unpooled state sequence — forward plumbs H_S^seq for this kind
        # alone so every other fusion path stays byte-identical.
        self._fusion_needs_sequence = getattr(self.fusion, "kind", "") == "cross_attention"

        self.heads = nn.ModuleDict()
        pathway_kwargs = self._state_pathway_kwargs(config)
        for name in config.decision_primitives:
            try:
                self.heads[name] = build_decision_head(
                    name, hidden_size=hidden, **pathway_kwargs
                )
            except TypeError as exc:
                raise ValueError(
                    f"decision head {name!r} cannot be built from decision_primitives alone — "
                    f"it needs extra constructor arguments ({exc}). Remove it from "
                    "decision_primitives and construct/register it manually."
                ) from exc

        try:
            self.confidence = ConfidenceEngine.from_config(config)
        except TypeError:
            # e.g. confidence_method="learned" needs fused_dim.
            self.confidence = ConfidenceEngine(
                getattr(config, "confidence_method", "max_probability"), fused_dim=hidden
            )
        engine = CalibrationEngine.from_config(config)
        # Registered as a submodule so fit state rides the checkpoint's
        # state_dict; the plain engine facade shares the same object.
        self.calibrator = engine.calibrator
        self.calibration_engine = engine
        # Audit Q-003 / docs ARCHITECTURE L5: runtime guardrail component --
        # stateless invariant checks run from forward behind
        # config.runtime_validation. Energy models allow the documented
        # pre-fit unbounded confidence (docs/CONFIDENCE.md section 1).
        self.validator = ValidationEngine(
            allow_unbounded_confidence=(
                getattr(config, "confidence_method", "max_probability") == "energy"
            )
        )

        # Audit risk #5: wrapper parameters follow the backbone's
        # device/dtype (no-op when constructed inside from_pretrained's
        # local_torch_dtype/meta context, which already aligns them).
        reference = next(
            (p for p in self.model.parameters() if p.is_floating_point()), None
        )
        # The loop below aligns wrapper params with a PRE-LOADED materialized
        # backbone (wrap path). On the reload path transformers constructs the
        # whole tree under the meta device and materializes afterwards; torch
        # refuses to move meta tensors (NotImplementedError: "Cannot copy out
        # of meta tensor") and some backbones (wav2vec2) even escape the meta
        # context so reference lands on CPU while these modules are still
        # meta. Skip alignment whenever either side is meta — checkpoint
        # weights for these modules are loaded (or re-initialized for truly
        # missing keys) by transformers' own finalization, and the calibrator
        # is re-materialized by _materialize_calibrator().
        def _has_meta_tensors(module: nn.Module) -> bool:
            return any(p.is_meta for p in module.parameters()) or any(
                buffer is not None and buffer.is_meta for buffer in module.buffers()
            )

        if reference is not None and not reference.is_meta and not _has_meta_tensors(self.model):
            for module in (
                self.state_proj,
                self.question_encoder,
                self.question_proj,
                self.schema_encoder,
                self.schema_proj,
                self.fusion,
                self.heads,
                self.confidence,
                self.calibrator,
            ):
                if _has_meta_tensors(module):
                    continue  # reload path: transformers materializes later
                module.to(device=reference.device, dtype=reference.dtype)

        self.post_init()

        if getattr(config, "freeze_base_model", True):
            freeze_module(self.model)

    # -- construction helpers ------------------------------------------------

    @staticmethod
    def _state_pathway_kwargs(config: JevXConfig) -> Dict[str, Any]:
        """Config -> head-constructor kwargs for the state pathway / conditioning.

        Returns an EMPTY dict while every field sits at its shipped default, so a
        default model builds exactly the pre-flag heads (bit-identical logits and
        state_dict).

        The ``conditioning_scale_init`` mapping is DELIBERATE, not an oversight.
        :attr:`JevXConfig.conditioning_scale_init` defaults to 0.0 but the heads'
        own default lambda is 1.0 (the pre-flag scoring function
        ``z_k = state_k + r_k``), so a config of 0.0 is read as "unspecified"
        and mapped to 1.0. A FRESH :class:`JevXForDecision` therefore starts at
        ``lambda == 1.0`` even though its config says 0.0; a caller that wants a
        different start must set a non-zero ``conditioning_scale_init`` (that is
        what a curriculum does, and it also sets the scale explicitly at every
        step). 0.0 remains reachable at RUNTIME through
        ``set_interaction_scale(0.0)`` / ``forward(condition_options=False)``,
        which is the state-only path. Forcing the constructor value to 0.0
        instead would silently change what a PRE-EXISTING checkpoint computes:
        ``interaction_scale`` is a persistent buffer, so a checkpoint written
        before it existed loads with ``strict=False``, keeps the CONSTRUCTOR
        value, and would silently lose its residual. Pinned by
        ``tests/unit/test_head_next.py`` (test_default_config_passes_no_head_kwargs,
        test_default_model_builds_the_pre_flag_heads,
        test_pre_flag_state_dict_loads_and_reproduces_todays_numbers) and by
        ``tests/unit/test_state_only_path.py``.
        """
        if (
            config.state_pathway == "linear"
            and config.state_hidden_size is None
            and config.state_activation == "gelu"
            and config.conditioning_enabled
            and config.conditioning_scale_init == 0.0
        ):
            return {}
        return {
            "state_pathway": config.state_pathway,
            "state_hidden_size": config.state_hidden_size,
            "state_activation": config.state_activation,
            "conditioning_scale_init": (
                1.0 if config.conditioning_scale_init == 0.0 else config.conditioning_scale_init
            ),
            "conditioning_enabled": config.conditioning_enabled,
        }

    # -- conditioning: config mode -> per-call flag ---------------------------

    def _resolve_condition(self, condition_options: Optional[bool]) -> bool:
        """Resolve the per-call option-conditioning flag.

        Args:
            condition_options: The per-call switch from :meth:``forward``;
                ``None`` defers to the config, a bool is taken verbatim.

        Returns:
            Whether the OPTION features may condition this call.

        ``JevXConfig.conditioning_mode`` is the source of truth:

        * ``"off"`` (the DEFAULT) -> ``False``, i.e. exactly the shipped
          ``forward(condition_options=False)`` path — and ``False`` now means
          "no conditioning AT ALL": forward also feeds fusion an EXACT-ZERO
          question embedding, so the fused state does not depend on the
          question (or the schema) either;
        * ``"constant"`` -> ``True``; whether the encoders are actually
          memoized is decided per call by :meth:``_condition_cache_enabled``;
        * ``"full"`` -> ``bool(config.conditioning_enabled)``, read
          on EVERY call so a runtime flip (e.g.
          ``jevx.training.conditioning_gate.apply_gate``) still takes effect.
          A user who wants question conditioning sets this mode explicitly:
          ``JevXConfig(conditioning_mode="full")``.
        """
        if isinstance(condition_options, bool):
            return condition_options
        mode = self.config.conditioning_mode
        if mode == "off":
            return False
        if mode == "constant":
            return True
        return bool(self.config.conditioning_enabled)

    #: Upper bound on the entries of the ``"constant"`` conditioning memo.
    #: Bounded so a caller that sweeps distinct content per call cannot grow
    #: the process's memory without limit; eviction is FIFO (oldest key first).
    _CONDITION_CACHE_LIMIT: int = 8

    def _condition_cache_enabled(self) -> bool:
        """Whether this call may read/write the ``"constant"`` conditioning memo.

        The memo holds DETACHED tensors, so it is served only when no autograd
        graph is wanted: handing it to a grad-enabled call would silently drop
        the encoder gradients, and replaying a live graph would double-backward.
        ``self.training`` also disables it, because the weights move between
        steps there.

        On the one path that IS allowed to consult the memo, this is also where
        the memo is checked against the weights it was computed from
        (:meth:`_condition_cache_sync`): the memo is content-keyed, so on its
        own it cannot notice that the encoder parameters moved. This method is
        called exactly once per :meth:`forward`, before the first lookup, so the
        version counters are read once per call and not per entry.
        """
        if (
            self.config.conditioning_mode != "constant"
            or self.training
            or torch.is_grad_enabled()
        ):
            return False
        self._condition_cache_sync()
        return True

    @staticmethod
    def _module_tensors(module: nn.Module) -> Tuple[torch.Tensor, ...]:
        """Every parameter and buffer TENSOR owned by ``module`` and its children.

        A hand-rolled walk over ``nn.Module``'s storage dicts. It is exactly
        ``module.parameters() + module.buffers()`` without the generator and
        de-duplication memo that ``nn.Module.named_parameters`` builds — which
        is pure overhead for a 2-3 module encoder and which measured 110 us of
        the 161 us of the first version of this check (see
        ``_condition_cache_signature``).
        """
        found: List[torch.Tensor] = []
        stack: List[nn.Module] = [module]
        while stack:
            current = stack.pop()
            found.extend(t for t in current._parameters.values() if t is not None)  # noqa: SLF001 - nn.Module storage
            found.extend(t for t in current._buffers.values() if t is not None)  # noqa: SLF001 - nn.Module storage
            stack.extend(current.children())
        return tuple(found)

    def _condition_cache_signature(self) -> Tuple[int, ...]:
        """One version counter per TENSOR of the two memoized encoders.

        The memo stores exactly two things: the output of ``question_encoder``
        (``H_Q``) and the output of ``schema_encoder.encode`` (``H_C``). Only
        those two modules' parameters and buffers can change what a memoized
        value IS, so they are the whole dependency set — ``question_proj`` /
        ``schema_proj`` run AFTER the lookup and are deliberately not tracked.

        ``torch.Tensor._version`` is the autograd version counter: it is
        incremented by every in-place mutation of the tensor, which is exactly
        what an optimizer step (``param.add_``), ``load_state_dict``
        (``param.copy_``) and a manual ``param.data[...] = ...`` are. The cost is
        therefore one integer read per TENSOR — never a walk over scalars —
        because the conditioning half of a JevX stack is 80-90 % of ~1.3 M
        scalars but only a few dozen tensors.

        Measured on this repo's ``tiny_bert`` fixture (10 tracked tensors,
        266 368 scalars, 20 000 iterations, contended 4-core box,
        ``torch.set_num_threads(1)``):

        | form | us per call |
        |---|---|
        | ``parameters()``/``buffers()`` version counters | 161.3 |
        | ... of which the module-tree walk | 110.3 |
        | ... of which the ``_version`` reads | 5.1 |
        | **this hand-rolled walk + version reads** | **83.5** |
        | (a cached tensor list would be 3.3, but it can go stale on parameter
        |  REPLACEMENT, so the walk is redone — correctness before speed) |

        The in-suite measurement (``tests/unit/test_condition_cache_invalidation.py``)
        printed 57.7 / 72.2 / 89.0 us on three runs of the same code on the same
        box, i.e. ~0.5-0.9 % of the 7-12 ms batch-1 forward it protects
        (reports/MULTI_QUESTION.md §2.3) and ~2-6 % of the 1.4-3.6 ms a memo hit
        saves. The memo stays worth having.
        """
        return tuple(
            tensor._version  # noqa: SLF001 - documented torch autograd counter
            for module in (self.question_encoder, self.schema_encoder)
            for tensor in self._module_tensors(module)
        )

    def _condition_cache_sync(self) -> None:
        """Drop the memo whenever an encoder tensor it was computed from changed.

        This is the fix for the content-keyed-staleness defect: an
        ``eval -> optimizer step -> eval`` sequence that does not call
        ``clear_condition_cache()`` used to serve ``H_Q``/``H_C`` computed from
        the PREVIOUS weights (measured max abs logit diff 4.1556e+04 in
        ``reports/MULTI_QUESTION.md`` §2.2). Comparing the encoder tensors'
        version counters catches that — and every other in-place weight update —
        without a hand-written invalidation call.

        Conservative by design: ANY change to ANY tracked tensor drops the WHOLE
        memo, including entries a single optimizer step could not have changed.
        That can only cost a re-encode (the miss is always a correct
        recomputation), never correctness, and during training the memo is off
        anyway.
        """
        signature = self._condition_cache_signature()
        if signature != self._condition_cache_versions:
            self._condition_cache.clear()
            self._condition_cache_versions = signature

    def _condition_cache_get(self, key: Optional[Tuple[Any, ...]]) -> Any:
        """Memoized value for ``key``, or ``None`` on a miss.

        A miss is also returned when the memo was filled before a
        ``model.to(...)``/``.half()``: the entry's device/dtype no longer
        matches the conditioning parameters, so it is dropped rather than fed
        to a projection that would raise a device error.
        """
        if key is None:
            return None
        entry = self._condition_cache.get(key)
        if entry is None:
            return None
        reference = next(self.question_encoder.parameters())
        tensors = entry if isinstance(entry, tuple) else (entry,)
        if any(
            tensor.device != reference.device
            # the option mask is bool by construction; only the float tensors
            # have to track the conditioning parameters' dtype
            or (tensor.is_floating_point() and tensor.dtype != reference.dtype)
            for tensor in tensors
        ):
            return None
        return entry

    def _condition_cache_put(self, key: Optional[Tuple[Any, ...]], value: Any) -> None:
        """Memoize ``value`` under ``key`` (FIFO eviction at the size limit)."""
        if key is None:
            return
        if key in self._condition_cache:
            del self._condition_cache[key]
        elif len(self._condition_cache) >= self._CONDITION_CACHE_LIMIT:
            self._condition_cache.pop(next(iter(self._condition_cache)))
        self._condition_cache[key] = value

    def clear_condition_cache(self) -> None:
        """Drop the ``conditioning_mode="constant"`` memo of ``H_Q``/``H_C``.

        Called automatically by :meth:``load_state_dict``. It is now only a
        HINT, not a correctness requirement: :meth:`_condition_cache_sync`
        compares the memoized encoders' tensor version counters on every
        memo-eligible call, so an in-place parameter update (an optimizer step,
        ``load_state_dict``, a manual ``param.data[...] = ...``) invalidates the
        memo on its own. Calling this by hand is still supported and still
        correct; it simply makes the next lookup a guaranteed miss.

        The residual case a version counter cannot see is a whole-object
        REPLACEMENT of an encoder parameter (state_dict loading with
        assign=True, or assigning a brand-new Parameter to a submodule), which
        can hand the new tensor a version counter equal to the recorded one.
        ``load_state_dict`` clears the memo explicitly for that reason;
        hand-assigned parameters need the same one-line call.
        """
        self._condition_cache.clear()
        self._condition_cache_versions = None

    @staticmethod
    def _build_backbone(config: JevXConfig) -> nn.Module:
        """Rebuild the backbone from ``config.backbone_config`` (reload path).

        Resolution: rebuild through Auto classes and prefer the candidate
        whose model class matches the saved ``architectures`` entry (a
        causal-LM wrapper like ``GPT2LMHeadModel`` needs
        ``AutoModelForCausalLM``; base models resolve via ``AutoModel``).
        ``AutoModel`` is the primary/default candidate.
        """
        backbone_config = getattr(config, "backbone_config", None)
        if not isinstance(backbone_config, dict) or not backbone_config.get("model_type"):
            if config.base_model_name_or_path:
                return AutoModel.from_pretrained(config.base_model_name_or_path)
            raise ValueError(
                "JevXConfig is missing `backbone_config` (the nested base config dict) and "
                "`base_model_name_or_path` — this checkpoint cannot rebuild its backbone. "
                "Load it through JevX.from_pretrained / JevXForDecision.from_pretrained on a "
                "complete save, or construct the wrapper with backbone=<model>."
            )
        model_type = backbone_config["model_type"]
        expected = (backbone_config.get("architectures") or [None])[0]
        base_cfg = None
        if model_type in CONFIG_MAPPING:
            base_cfg = CONFIG_MAPPING[model_type].from_dict(backbone_config)
        elif config.base_model_name_or_path:
            base_cfg = AutoConfig.from_pretrained(config.base_model_name_or_path)
        else:
            raise ValueError(
                f"backbone_config model_type {model_type!r} is not registered with transformers "
                "and base_model_name_or_path is unset — cannot rebuild the backbone "
                "(remote-code backbones need trust_remote_code at wrap time)."
            )
        first = None
        errors: List[str] = []
        for auto_cls in _REBUILD_AUTO_CLASSES:
            try:
                candidate = auto_cls.from_config(base_cfg)
            except Exception as exc:  # noqa: BLE001 - candidate probing
                errors.append(f"{auto_cls.__name__}: {type(exc).__name__}: {exc}")
                continue
            if first is None:
                first = candidate
            if expected is None or type(candidate).__name__ == expected:
                return candidate
        if first is not None:
            return first
        raise ValueError(
            f"could not rebuild backbone {expected or model_type!r} from backbone_config via "
            f"{[c.__name__ for c in _REBUILD_AUTO_CLASSES]}; attempts: {errors}"
        )

    @staticmethod
    def _learned_position_table_limit(model: nn.Module) -> Optional[int]:
        """Smallest learned (absolute) position-Embedding table size, or None.

        Scans ``model`` for Embedding modules whose qualified name ends with
        ``position_embeddings`` (bert/roberta/vit/albert/distilbert) or
        ``.wpe`` (gpt2) - the tables that hard-fail with a kernel-level
        IndexError/IndexOutOfRange past their size. Relative/RoPE backbones
        have no such table (None) and are never pre-checked.
        """
        limit: Optional[int] = None
        for name, module in model.named_modules():
            if isinstance(module, nn.Embedding):
                lname = name.lower()
                if (
                    lname.endswith("position_embeddings")
                    or lname == "wpe"  # GPT2Model registers its table as `wpe`
                    or lname.endswith(".wpe")
                ):
                    size = int(module.num_embeddings)
                    limit = size if limit is None else min(limit, size)
        return limit
    # -- HF embedding accessors (audit risk #2) ------------------------------

    def get_input_embeddings(self) -> Optional[nn.Module]:
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.model.set_input_embeddings(value)

    # -- freeze / Mode A-B API ----------------------------------------------

    def trainable_parameters(self):
        """Yield JevX-side parameters (everything not under ``model.``).

        Structural, not flag-based: the complement of this set is exactly
        the backbone, which is the resolution contract used by
        ``jevx.training.freeze.freeze_backbone``/``unfreeze_backbone``.
        """
        for name, parameter in self.named_parameters():
            if not name.startswith("model."):
                yield parameter

    def unfreeze_last_n_layers(self, n: int) -> "JevXForDecision":
        """Mode B helper: unfreeze the last ``n`` backbone layers (§29).

        Layer discovery is by convention (``encoder.layer`` / ``h`` /
        ``layers`` under ``model.``) — see
        :func:`jevx.training.freeze.unfreeze_last_n_layers`.
        """
        _unfreeze_layers(self, n)
        return self

    # -- preservation (§31) ---------------------------------------------------

    def base_forward(self, *args: Any, **kwargs: Any) -> Any:
        """Run the raw backbone untouched (preserve-base-model, §31).

        Pass-through of every argument — the call is exactly
        ``self.model(*args, **kwargs)``, bit-identical to loading the base
        checkpoint directly (verified in eval + no_grad by the test suite;
        reported as ``max_abs_error`` per checkpoint).
        """
        return self.model(*args, **kwargs)

    # -- save / load (§43) ----------------------------------------------------

    def save_pretrained(self, save_directory: Union[str, Path], *args: Any, **kwargs: Any) -> Any:
        """Save config + weights (safetensors) + capability metadata + featurizers.

        Writes, into ``save_directory``:

        * ``config.json`` — the ``JevXConfig`` including the nested
          ``backbone_config`` dict (so the backbone can be rebuilt without
          re-downloading the base checkpoint) and any ``decision_type`` pin;
        * ``model.safetensors`` — wrapper + embedded backbone; declared
          child ties (e.g. causal ``lm_head.weight``) are removed before
          writing and restored on load (safetensors has no aliasing);
        * ``jevx_metadata.json`` — capability report, adapter identity,
          pooling/fusion choices, JevX version (documented choice: JSON
          sidecar, not config fields, so config stays a pure schema);
        * tokenizer/processor files when attached.

        Returns:
            The ``super().save_pretrained`` result.
        """
        result = super().save_pretrained(save_directory, *args, **kwargs)
        directory = Path(save_directory)
        # Calibrator state travels in a sidecar: fit-data-shaped buffers
        # (isotonic breakpoints, vector scale/bias) are dropped by v5's load
        # path on shape mismatch (core_model_loading.set_param_for_module),
        # so they cannot ride the safetensors state_dict. state_dict() is
        # filtered symmetrically — see the state_dict override.
        if self.calibration_engine.method != "none":
            torch.save(
                {
                    "format": _CALIBRATION_FORMAT,
                    "method": self.calibration_engine.method,
                    "state_dict": self.calibration_engine.state_dict(),
                },
                directory / _CALIBRATION_FILENAME,
            )
        metadata = {
            "format": _METADATA_FORMAT,
            "jevx_version": __version__,
            "adapter": {
                "class": type(self.adapter).__qualname__,
                "name": getattr(self.adapter, "name", None),
            },
            "capability": self.capability_report.to_dict(),
            "pooling_method": self.config.pooling_method,
            "fusion_type": self.config.fusion_type,
            "base_model_name_or_path": self.config.base_model_name_or_path,
            "decision_type": self.config.decision_type,
        }
        (directory / _METADATA_FILENAME).write_text(
            json.dumps(metadata, indent=2, default=str), encoding="utf-8"
        )
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(directory)
        if self.processor is not None:
            self.processor.save_pretrained(directory)
        return result

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *args: Any,
        decision_type: Optional[str] = None,
        trust_remote_code: bool = False,
        tokenizer: Any = None,
        processor: Any = None,
        adapter: Any = None,
        **kwargs: Any,
    ):
        """Factory + standard reload in one entry point (§45/§63).

        Extra keyword-only arguments (beyond the ``PreTrainedModel``
        signature):

        Args:
            decision_type: Pin the default decision type
                (``config.decision_type``; resolved at forward time as
                argument → config pin → ``"choice"``).
            trust_remote_code: Passed through to config/backbone loading —
                never defaults to ``True`` (Windows has no SIGALRM prompt,
                audit §2).
            tokenizer / processor: Override featurizer resolution.
            adapter: Override adapter resolution (instance or class).

        Behavior:

        * ``model_type == "jevx"`` → standard checkpoint reload (config is
          read directly, so Auto-class registration is NOT required);
        * otherwise the base checkpoint is wrapped: AutoConfig → backbone
          (AutoModel primary; AutoModelForCausalLM → AutoModelForSeq2SeqLM
          fallback) → tokenizer/processor (text-only AutoTokenizer, else
          AutoProcessor, fallback AutoFeatureExtractor) → capability
          detection → adapter resolution → ``JevXConfig`` → instantiate.

        Remaining kwargs are forwarded to the backbone loader (wrap path) or
        to ``PreTrainedModel.from_pretrained`` (reload path); unknown JevX
        config field names (``pooling_method``, ``fusion_type``, ...) are
        consumed as config fields on the wrap path, and anything else raises
        with the list of accepted names.
        """
        if args:
            raise TypeError(
                "JevXForDecision.from_pretrained takes only keyword arguments beyond the model "
                f"id; got {len(args)} extra positional argument(s)."
            )
        user_config = kwargs.get("config")
        if isinstance(user_config, JevXConfig):
            jevx_config, is_jevx = user_config, True
        else:
            jevx_config, is_jevx = cls._peek_config(
                pretrained_model_name_or_path, trust_remote_code, kwargs
            )
        if is_jevx:
            if decision_type is not None:
                # the kwarg overrides the config's pin; validate it (G17)
                jevx_config.decision_type = validate_decision_type(decision_type)
            kwargs["config"] = jevx_config
            model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
            model._attach_reload_assets(
                pretrained_model_name_or_path,
                tokenizer=tokenizer,
                processor=processor,
                adapter=adapter,
            )
            return model
        return cls._wrap_base_model(
            pretrained_model_name_or_path,
            decision_type=decision_type,
            trust_remote_code=trust_remote_code,
            tokenizer=tokenizer,
            processor=processor,
            adapter=adapter,
            **kwargs,
        )

    # -- from_pretrained internals -------------------------------------------

    @staticmethod
    def _peek_config(path: Any, trust_remote_code: bool, kwargs: Dict[str, Any]):
        """Classify ``path``: ``(config, is_jevx)`` without consuming kwargs."""
        peek = {k: kwargs[k] for k in _HUB_PEEK_KWARGS if k in kwargs}
        try:
            config = AutoConfig.from_pretrained(
                path, trust_remote_code=trust_remote_code, **peek
            )
            return config, config.model_type == "jevx"
        except Exception as auto_exc:  # noqa: BLE001 - classification probe
            # AutoConfig refuses unregistered model types (a saved JevX
            # checkpoint before AutoConfig.register ran); retry directly
            # with our own config class — but only when the RAW config
            # dict actually declares model_type "jevx". Inspecting the
            # parsed JevXConfig is not enough: its class-level model_type
            # is always "jevx", so any config.json that AutoConfig rejects
            # for lacking a model_type key (e.g. prajjwal1/bert-tiny) was
            # misclassified as a JEVX checkpoint and failed later with a
            # misleading "missing backbone_config" error instead of
            # AutoConfig’s actionable one.
            try:
                raw_config, _ = PreTrainedConfig.get_config_dict(path, **peek)
            except Exception:  # noqa: BLE001 - config unreadable at source
                raise auto_exc from None
            if raw_config.get("model_type") != "jevx":
                raise auto_exc from None
            try:
                config = JevXConfig.from_pretrained(path, **peek)
            except Exception:  # noqa: BLE001 - jevx type but broken config
                raise auto_exc from None
            return config, True

    @classmethod
    def _wrap_base_model(
        cls,
        model_id: Any,
        *,
        decision_type: Optional[str],
        trust_remote_code: bool,
        tokenizer: Any,
        processor: Any,
        adapter: Any,
        **kwargs: Any,
    ) -> "JevXForDecision":
        """Wrap a base (non-JevX) checkpoint — the §45/§63 factory path."""
        backbone_load_kwargs: Dict[str, Any] = {}
        for key in _BACKBONE_LOAD_KWARGS:
            if key in kwargs:
                backbone_load_kwargs[key] = kwargs.pop(key)
        # dtype/torch_dtype BC merge: only one spelling should reach the loader.
        if "dtype" in backbone_load_kwargs and "torch_dtype" in backbone_load_kwargs:
            backbone_load_kwargs.pop("torch_dtype")
        backbone_load_kwargs["trust_remote_code"] = trust_remote_code

        backbone: Any = None
        load_errors: List[str] = []
        for auto_cls in (AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM):
            try:
                backbone = auto_cls.from_pretrained(model_id, **backbone_load_kwargs)
                break
            except Exception as exc:  # noqa: BLE001 - documented fallback chain
                load_errors.append(f"{auto_cls.__name__}: {type(exc).__name__}: {exc}")
        if backbone is None:
            raise ValueError(
                f"JevX.from_pretrained: could not load {model_id!r} as a base model via the "
                f"documented chain AutoModel → AutoModelForCausalLM → AutoModelForSeq2SeqLM. "
                f"Attempts: {load_errors}"
            )

        report = JevXCapabilityDetector.detect(backbone, backbone.config)

        # Featurizers: text-only → AutoTokenizer; else AutoProcessor; then
        # AutoFeatureExtractor fallback (audit §4/Q8: AST has no processor
        # mapping, text families have no processor at all).
        if tokenizer is None and processor is None:
            tokenizer, processor = cls._resolve_featurizers(
                model_id, report, trust_remote_code
            )
        if tokenizer is not None and getattr(tokenizer, "pad_token_id", None) is None:
            # Batches of variable-length text states need padding (GPT-2
            # ships without a pad token): reuse EOS, then UNK.
            fallback = getattr(tokenizer, "eos_token", None) or getattr(
                tokenizer, "unk_token", None
            )
            if fallback is not None:
                tokenizer.pad_token = fallback

        # JevXConfig fields supplied as kwargs; everything else is rejected.
        config_fields = {f.name for f in _config_dataclass_fields(JevXConfig)}
        config_overrides = {
            key: kwargs.pop(key) for key in list(kwargs) if key in config_fields
        }
        if kwargs:
            raise TypeError(
                f"JevX.from_pretrained: unexpected keyword argument(s) {sorted(kwargs)}. "
                f"JevX config fields: {sorted(config_fields)}; backbone loader fields: "
                f"{list(_BACKBONE_LOAD_KWARGS)}; factory fields: "
                "['decision_type', 'trust_remote_code', 'tokenizer', 'processor', 'adapter']."
            )

        width = _hidden_width(backbone.config, backbone)
        if width is None:
            raise ValueError(
                f"cannot determine hidden width of {type(backbone).__name__} from config keys "
                f"{list(_HIDDEN_WIDTH_KEYS)}; pass decision_hidden_size= explicitly."
            )
        base_model_type = getattr(backbone.config, "model_type", None)
        # architectures must name the class actually embedded: the saved key
        # layout follows the child (AutoModel may load T5Model where the hub
        # config says T5ForConditionalGeneration), and _build_backbone matches
        # this entry to pick the rebuild Auto class.
        backbone_config = backbone.config.to_diff_dict()
        backbone_config["architectures"] = [type(backbone).__name__]
        config_kwargs: Dict[str, Any] = {
            "base_model_name_or_path": str(model_id),
            "base_model_type": base_model_type,
            "backbone_config": backbone_config,
            "decision_hidden_size": width,
            "question_hidden_size": width,
            "schema_hidden_size": width,
        }
        if decision_type is not None:
            # passed through the constructor so JevXConfig.__post_init__
            # validates it against the head registry (G17)
            config_overrides = {**config_overrides, "decision_type": decision_type}
        config_kwargs.update(config_overrides)
        config = JevXConfig(**config_kwargs)

        model = cls(
            config,
            backbone=backbone,
            adapter=adapter if adapter is not None else None,
            tokenizer=tokenizer,
            processor=processor,
        )
        # Decision models are inference-oriented by default (dropout off);
        # call .train() explicitly for Mode A/B training.
        model.eval()
        return model

    @staticmethod
    def _resolve_featurizers(model_id: Any, report: Any, trust_remote_code: bool):
        """(tokenizer, processor) resolution per audit §4/Q8. Never raises.

        Text-only families resolve through ``AutoTokenizer`` first (they
        have no processor mapping); everything else (or a tokenizer
        failure) falls to ``AutoProcessor``, then ``AutoFeatureExtractor``
        (AST-like repos ship only feature extractors).
        """
        tokenizer: Any = None
        processor: Any = None
        if bool(report.text) and not report.vision and not report.audio:
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_id, trust_remote_code=trust_remote_code
                )
            except Exception:  # noqa: BLE001 - documented fallback ladder
                tokenizer = None
        if tokenizer is None:
            try:
                processor = AutoProcessor.from_pretrained(
                    model_id, trust_remote_code=trust_remote_code
                )
            except Exception:  # noqa: BLE001 - fallback
                processor = None
        if tokenizer is None and processor is None:
            try:
                processor = AutoFeatureExtractor.from_pretrained(
                    model_id, trust_remote_code=trust_remote_code
                )
            except Exception:  # noqa: BLE001 - last resort: no featurizer
                processor = None
        if tokenizer is None and processor is not None:
            # Symmetry with the reload path (_attach_reload_assets attaches
            # AutoTokenizer whenever tokenizer files were saved — and they
            # ARE saved: processor.save_pretrained writes them). Question and
            # schema encoding are tokenizer-sensitive, so a processor-only
            # wrap followed by a reload with an attached tokenizer produced
            # different decisions at identical weights (real-model matrix:
            # CLIP 6.8e-02, wav2vec2 1.7e-03 save/load drift).
            embedded = getattr(processor, "tokenizer", None)
            if embedded is not None:
                tokenizer = embedded
        return tokenizer, processor

    def _attach_reload_assets(
        self,
        path: Any,
        *,
        tokenizer: Any,
        processor: Any,
        adapter: Any,
    ) -> None:
        """Re-attach metadata/featurizers/adapter after a checkpoint reload."""
        local_metadata = None
        candidate = Path(str(path))
        metadata_file = candidate / _METADATA_FILENAME
        if metadata_file.is_file():
            try:
                local_metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                local_metadata = None
        self.jevx_metadata = local_metadata

        if adapter is not None:
            self.adapter = _normalize_adapter(adapter)
        if tokenizer is None and self.tokenizer is None:
            try:
                tokenizer = AutoTokenizer.from_pretrained(path)
            except Exception:  # noqa: BLE001 - tokenizer is optional until used
                tokenizer = None
        if tokenizer is not None:
            self.tokenizer = tokenizer
            if getattr(tokenizer, "pad_token_id", None) is None:
                fallback = getattr(tokenizer, "eos_token", None) or getattr(
                    tokenizer, "unk_token", None
                )
                if fallback is not None:
                    tokenizer.pad_token = fallback
        if processor is not None:
            self.processor = processor
        elif self.processor is None and local_metadata:
            needs_processor = bool(
                (local_metadata.get("capability") or {}).get("processor_required")
            )
            if needs_processor:
                try:
                    self.processor = AutoProcessor.from_pretrained(path)
                except Exception:  # noqa: BLE001 - user must pass processor=
                    self.processor = None
        self._materialize_calibrator()
        self._restore_calibration_sidecar(path)

    def _materialize_calibrator(self) -> None:
        """Move calibrator tensors off the meta device after a reload.

        ``state_dict()`` excludes ``calibrator.*`` keys (see the sidecar
        design), so ``from_pretrained`` never loads them and they would
        remain meta tensors created during construction. Re-materialize
        them with their constructed shapes/dtypes (unfitted defaults); the
        sidecar, when present, then overwrites them with fitted state.
        """
        calibrator = getattr(self, "calibrator", None)
        if calibrator is None:
            return
        reference = next(
            (p for p in self.state_proj.parameters() if not p.is_meta), None
        )
        device = reference.device if reference is not None else torch.device("cpu")
        dtype = reference.dtype if reference is not None else torch.float32
        for name, buffer in list(calibrator._buffers.items()):  # noqa: SLF001
            if buffer is None or buffer.is_meta:
                # Float buffers follow the REFERENCE dtype (the wrap path casts
                # the whole calibrator module to the backbone dtype in __init__);
                # re-creating them at their constructed dtype (float64 for
                # log_temperature) made a fitted sidecar round-trip disagree with
                # the wrap-path value at exp() rounding (rel ~1e-8, above the 1e-12
                # the serialization contract asserts). Non-float buffers (_fitted
                # bool) keep their dtype.
                buf_dtype = dtype if buffer.is_floating_point() else buffer.dtype
                calibrator._buffers[name] = torch.zeros(  # noqa: SLF001
                    buffer.shape, dtype=buf_dtype, device=device
                )
        for name, param in list(calibrator._parameters.items()):  # noqa: SLF001
            if param is not None and param.is_meta:
                calibrator._parameters[name] = torch.nn.Parameter(  # noqa: SLF001
                    torch.zeros(param.shape, dtype=dtype, device=device),
                    requires_grad=param.requires_grad,
                )

    def _restore_calibration_sidecar(self, path: Any) -> None:
        """Load ``jevx_calibration.pt`` (local dir or hub) if present."""
        calibration_file: Optional[Path] = Path(str(path)) / _CALIBRATION_FILENAME
        if not calibration_file.is_file():
            calibration_file = None
            try:
                from transformers.utils.hub import cached_file

                resolved = cached_file(str(path), _CALIBRATION_FILENAME, _raise_exceptions=False)
                if resolved is not None:
                    calibration_file = Path(resolved)
            except Exception:  # noqa: BLE001 - sidecar is optional
                calibration_file = None
        if calibration_file is None:
            # Q-028: a sidecar that cannot be resolved must be LOUD. The
            # calibrator silently staying unfitted turns every calibrated
            # probability into an uncalibrated one with no signal at all.
            method = self.calibration_engine.method
            if method != "none":
                warnings.warn(
                    f"jevx calibration sidecar {method!r} not found under "
                    f"{path!r}; the calibrator stays UNFITTED and calibration is "
                    "a no-op",
                    RuntimeWarning,
                    stacklevel=2,
                )
            return
        payload = torch.load(calibration_file, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or payload.get("format") != _CALIBRATION_FORMAT:
            raise ValueError(
                f"{calibration_file}: not a JevX calibration sidecar (missing format marker "
                f"{_CALIBRATION_FORMAT!r})."
            )
        if payload.get("method") != self.calibration_engine.method:
            raise ValueError(
                f"{calibration_file}: calibration method {payload.get('method')!r} disagrees with "
                f"config.calibration_method={self.calibration_engine.method!r} — reload with the "
                "same config the checkpoint was saved with."
            )
        self.calibration_engine.load_state_dict(payload["state_dict"])

    # -- state_dict filtering + sidecar calibration persistence ---------------

    def state_dict(self, *args: Any, **kwargs: Any) -> Dict[str, torch.Tensor]:
        """Model state WITHOUT ``calibrator.*`` keys (see save sidecar).

        Fit-data-shaped calibrator buffers (isotonic breakpoints, vector
        scale/bias) cannot survive safetensors round trips: v5's loader
        *drops* checkpoint tensors whose shape differs from the freshly
        constructed buffer (``core_model_loading.set_param_for_module``)
        and then raises unless ``ignore_mismatched_sizes=True`` (which
        would discard the values anyway). Calibrator state is therefore
        persisted in ``jevx_calibration.pt`` by ``save_pretrained`` and
        restored by ``from_pretrained`` — the filtering here keeps
        ``expected_keys``/checkpoint keys symmetric so no missing/unexpected
        key noise appears. Confidence fit state (scalar ``energy_lo/hi``)
        has fixed shapes and DOES ride the regular state_dict.
        """
        state = super().state_dict(*args, **kwargs)
        return {
            key: value
            for key, value in state.items()
            if not (key.startswith("calibrator.") or ".calibrator." in key)
        }

    def load_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
        strict: bool = True,
        assign: bool = False,
    ) -> Any:
        """Load weights; reallocates fit-shaped calibrator buffers if present.

        ``from_pretrained`` filters calibrator keys out via :meth:`state_dict`
        symmetry, so its path never sees them. A state dict that *does*
        contain ``calibrator.*`` keys (e.g. produced by merging a sidecar, or
        an external full-dict save) is accepted: data-dependent buffers are
        resized to the checkpoint's shapes first, mirroring
        ``BaseCalibrator.load_state_dict``.

        Also drops the ``conditioning_mode="constant"`` memo: it is keyed on
        content, not on weights, so pre-load ``H_Q``/``H_C`` must not survive.
        (With ``assign=True`` a whole new Parameter object replaces the old one
        and its version counter can coincide with the recorded one, so this
        explicit clear is still required — see ``clear_condition_cache``.)
        """
        state_dict = dict(state_dict)
        calibrator = getattr(self, "calibrator", None)
        if calibrator is not None:
            for key, value in state_dict.items():
                if not key.startswith("calibrator.") or not isinstance(value, torch.Tensor):
                    continue
                name = key[len("calibrator."):]
                buffer = calibrator._buffers.get(name)  # noqa: SLF001 - HF-style hook
                param = calibrator._parameters.get(name) if hasattr(calibrator, "_parameters") else None  # noqa: SLF001
                if buffer is not None and buffer.shape != value.shape:
                    calibrator._buffers[name] = torch.empty_like(  # noqa: SLF001
                        value, device=buffer.device
                    )
                elif param is not None and param.shape != value.shape:
                    calibrator._parameters[name] = torch.nn.Parameter(  # noqa: SLF001
                        torch.empty_like(value, device=param.device),
                        requires_grad=param.requires_grad,
                    )
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        # New weights invalidate a conditioning memo keyed on content alone, so
        # a load must not leave pre-load H_Q / H_C behind.
        self.clear_condition_cache()
        return result

    # -- forward ---------------------------------------------------------------

    def forward(
        self,
        state: Any = None,
        questions: Any = None,
        decision_type: Union[str, Sequence[str], None] = None,
        options: Optional[Sequence[Any]] = None,
        confidence_threshold: Optional[float] = None,
        return_base_output: bool = False,
        *,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        condition_options: Optional[bool] = None,
        **kwargs: Any,
    ) -> Union[JevXOutput, Tuple[Any, ...]]:
        """Answer m questions about B states in one call.

        Args:
            state: Raw state — ``str``, ``list[str]`` (batch of B states),
                pre-tokenized ``dict`` of tensors, or a raw input tensor
                (mapped to ``model.main_input_name``). Mutually exclusive
                with ``input_ids=``.
            questions: ``str`` / ``list[str | dict]`` (dicts need a
                ``"text"`` key) of length m, shared across the state batch;
                OR a pre-encoded tensor ``[m, D_q]`` (or ``[B, m, D_q]`` for
                per-state encodings), interpreted as raw question
                representations and passed through ``question_proj``.
            decision_type: ``"choice" | "score" | "noul"`` (or a list with
                1, m or B*m entries for mixed calls — types are grouped and
                never share a probability space). ``None`` resolves as
                argument → ``config.decision_type`` → ``"choice"``.
            options: One schema per question — a list of lists (length 1, m
                or B*m), or a single flat schema when there is exactly one
                (state, question) pair. Entries: :class:`Option`, ``str``,
                ``dict`` or number. Required for ``choice``/``score``;
                forbidden for ``noul``.
            confidence_threshold: If given, questions with
                ``confidence < threshold`` come back ``abstained=True``
                with cleared predicted fields (probabilities untouched).
            return_base_output: Include the raw backbone output in
                ``base_output``.
            input_ids / attention_mask / token_type_ids: Pre-tokenized
                state inputs (alternative to ``state=``); extra tensor
                kwargs (``pixel_values`` ...) merge into the state inputs.
            output_hidden_states / output_attentions: Per transformers
                conventions; ``None`` falls back to the config flags
                (attentions require ``attn_implementation="eager"``,
                audit §8.6).
            return_dict: ``False`` returns the tuple form of
                :class:`JevXOutput` (``None`` → ``config.return_dict``).
            condition_options: Whether the OPTION features may condition this
                call. A bool is taken verbatim; ``None`` defers to the config
                through ``_resolve_condition(None)``, i.e. to
                ``config.conditioning_mode``: ``"off"`` → False,
                ``"constant"`` → True, ``"full"`` (default) →
                ``bool(config.conditioning_enabled)``, re-read on every call
                so a runtime flip still takes effect. ``False``
                makes the call STATE-ONLY in both halves of the scoring
                identity ``z_k = state_k + lambda * r_k``:

                1. the option interaction is switched off in the head for this
                   call (the residual is exactly zero — it is not even
                   computed — so no gradient can reach the option rows), and
                2. the schema rows are removed BEFORE fusion: fusion receives
                   ``options=None`` and returns the state-only context, while
                   the head receives exact zeros of the original ``[B, K, D]``
                   shape so the ``K``-column contract (mask and schema
                   metadata) is unchanged.

                The head is also told, via ``duplicate_tie=False``, that those
                zeros are NOT a schema. The duplicate-option tie is a
                statement about the schema's option rows, and exact zeros are
                "identical" for every schema, so letting the tie fire
                here would average the K columns of the pure state term
                ``state_k = s^T W[k, :] + b[k]`` (per-class BY CONSTRUCTION)
                into one value. The K state-only logits would then all be
                equal "(measured spread 0.0)", the softmax would be uniform, and because the
                tie gives every logit in the group the same value, the
                gradient into every state-pathway parameter is analytically
                ZERO (only float32 round-off, ~3e-07, survives) — so a state-only call could
                never be trained out of the uniform distribution. With
                ``condition_options=True`` (or ``None`` deferring to the
                config's resolved mode) the tie is left ON, so
                genuinely duplicated options still score identically.

                The fused state is byte-identical to the conditioned call's
                (the option branch never feeds back into ``fused``), so the
                only difference is the residual. ``True`` is the shipped
                behavior; the flag is per call and never mutates the heads.

        Returns:
            :class:`JevXOutput` (or its tuple form) with ``decisions`` a
            flat ``B*m`` list ordered ``b*m + j`` (state-major).

        Raises:
            ValueError / TypeError: Any input contract violation — unknown
                decision type (message lists the model's primitives),
                missing/forbidden options, batch-length mismatches, bad
                threshold — each message names the offending argument.
            UnsupportedRepresentationError: The adapter cannot extract a
                representation (structured fields: detected outputs,
                attempted/failed strategies, suggestion).
            ValueError: Also raised by the runtime ValidationEngine (audit
                Q-003, behind ``config.runtime_validation``) when a
                representation or decision violates its invariants; the
                message names the tensor and the failed check.
        """
        # ---- runtime validation gate (audit Q-003 / JevXConfig.runtime_validation)
        runtime_validation = bool(getattr(self.config, "runtime_validation", True))

        # ---- option conditioning flag (per call; never mutates the heads) -----
        if not (condition_options is None or isinstance(condition_options, bool)):
            raise TypeError(
                "condition_options: expected a bool or None (fall back to "
                f"config.conditioning_mode={self.config.conditioning_mode!r}), got "
                f"{type(condition_options).__name__}."
            )
        condition = self._resolve_condition(condition_options)
        # conditioning_mode="constant": may this call use the H_Q/H_C memo?
        cache_conditioning = self._condition_cache_enabled()

        # ---- state inputs -------------------------------------------------
        extra_tensors = {
            key: value for key, value in kwargs.items() if isinstance(value, torch.Tensor)
        }
        stragglers = sorted(key for key, value in kwargs.items() if not isinstance(value, torch.Tensor))
        if stragglers:
            raise TypeError(
                f"forward: unexpected keyword argument(s) {stragglers} — tensor model inputs "
                "are accepted as kwargs; control flags are the explicit parameters "
                "(output_hidden_states, output_attentions, return_dict)."
            )
        if state is None:
            if input_ids is None:
                raise ValueError(
                    "forward: missing state — pass `state=` (str/list[str]/dict) or "
                    "pre-tokenized `input_ids=`."
                )
            state_payload: Any = {
                key: value
                for key, value in (
                    ("input_ids", input_ids),
                    ("attention_mask", attention_mask),
                    ("token_type_ids", token_type_ids),
                )
                if value is not None
            }
            state_payload.update(extra_tensors)
        else:
            if input_ids is not None:
                raise ValueError(
                    "forward: pass either `state=` or pre-tokenized `input_ids=`, not both."
                )
            state_payload = state

        inputs = self.adapter.prepare_inputs(
            state_payload,
            tokenizer=self.tokenizer,
            processor=self.processor,
            model=self.model,
        )
        for key, value in extra_tensors.items():
            inputs[key] = value
        batch_tensor = inputs.get("input_ids")
        if isinstance(batch_tensor, torch.Tensor) and batch_tensor.ndim >= 1:
            batch_size = int(batch_tensor.shape[0])
        else:
            batch_size = next(
                (
                    int(value.shape[0])
                    for value in inputs.values()
                    if isinstance(value, torch.Tensor) and value.ndim >= 1
                ),
                None,
            )
            if batch_size is None:
                raise ValueError(
                    "forward: could not infer a batch dimension from the prepared state inputs "
                    f"{sorted(inputs)} — provide input_ids=[B, T] (or a batched modality tensor)."
                )
        mask = inputs.get("attention_mask")
        if isinstance(mask, torch.Tensor):
            if mask.shape[0] != batch_size:
                raise ValueError(
                    f"attention_mask: batch {int(mask.shape[0])} does not match the state "
                    f"batch {batch_size}."
                )
            # Input-contract validation (sec 90 row 8): token-validity masks are
            # bool/integer {0, 1}. Float or negative values are almost always
            # caller bugs and would otherwise flow into the backbone silently.
            if mask.dtype == torch.bool:
                pass
            elif mask.is_floating_point():
                raise ValueError(
                    f"attention_mask: expected a bool or integer tensor with values in "
                    f"{{0, 1}}, got floating dtype {mask.dtype} — binarize the mask and pass "
                    "`.long()`/`.bool()` (fractional token-validity weights are not "
                    "supported; the backbone consumes a validity mask, not soft weights)."
                )
            elif mask.dtype not in (
                torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8
            ):
                raise TypeError(
                    f"attention_mask: expected a bool or integer/float tensor, got "
                    f"{mask.dtype}."
                )
            else:
                invalid = ~((mask == 0) | (mask == 1))
                if bool(invalid.any()):
                    bad = torch.nonzero(invalid, as_tuple=False)[0]
                    first = tuple(int(v) for v in bad.tolist())
                    value = mask[tuple(int(v) for v in bad.tolist())].item()
                    count = int(invalid.sum())
                    raise ValueError(
                        f"attention_mask: expected values in {{0, 1}}, found {count} "
                        f"invalid entries — first at index {first} with value {value!r}; "
                        "negative or fractional mask values are not valid token-validity "
                        "indicators — binarize with `(mask != 0).long()` before calling "
                        "JevX."
                    )
        # Position-table contract (sec 90 row 6 / sec 72): learned absolute
        # position embeddings hard-fail past their table size with a cryptic
        # kernel error; JevX pre-checks token-sequence states and raises a
        # descriptive error naming both lengths (never silently truncates).
        state_ids = inputs.get("input_ids")
        if (
            self._learned_position_limit is not None
            and isinstance(state_ids, torch.Tensor)
            and state_ids.ndim == 2
        ):
            seq_len = int(state_ids.shape[-1])
            if seq_len > self._learned_position_limit:
                cfg = getattr(self.model, "config", None)
                limit_name = next(
                    (
                        key
                        for key in ("max_position_embeddings", "n_positions", "max_seq_len")
                        if getattr(cfg, key, None) == self._learned_position_limit
                    ),
                    "position embedding table",
                )
                raise ValueError(
                    f"forward: state token length {seq_len} exceeds the backbone's "
                    f"learned position limit {self._learned_position_limit} ({limit_name}) "
                    "— truncate the state before calling JevX (shorter input, or tokenizer "
                    "truncation) or wrap a backbone with a larger position table; JevX never "
                    "silently truncates a state."
                )

        # ---- questions -------------------------------------------------------
        if questions is None:
            raise ValueError(
                "forward: `questions` is required — pass a str, a list of str/dict (one per "
                "question), or a pre-encoded [m, D_q] tensor."
            )
        question_texts: Optional[List[str]] = None
        per_state_questions: Optional[torch.Tensor] = None
        encoded_questions: Optional[torch.Tensor] = None
        if isinstance(questions, torch.Tensor):
            if questions.ndim == 2:
                encoded_questions = questions
                num_questions = int(questions.shape[0])
            elif questions.ndim == 3:
                if int(questions.shape[0]) != batch_size:
                    raise ValueError(
                        f"questions: pre-encoded [B, m, D_q] tensor has B={int(questions.shape[0])} "
                        f"but the state batch is {batch_size}."
                    )
                per_state_questions = questions
                encoded_questions = questions
                num_questions = int(questions.shape[1])
            else:
                raise ValueError(
                    f"questions: a tensor must be pre-encoded [m, D_q] or [B, m, D_q], got "
                    f"shape {tuple(questions.shape)}."
                )
            if num_questions < 1:
                raise ValueError("questions: expected at least one question (m >= 1).")
        else:
            if isinstance(questions, str):
                question_list: List[Any] = [questions]
            elif isinstance(questions, (list, tuple)):
                question_list = list(questions)
            else:
                raise TypeError(
                    f"questions: expected a str, list of str/dict or torch.Tensor, got "
                    f"{type(questions).__name__}."
                )
            if len(question_list) == 0:
                raise ValueError("questions: expected at least one question (m >= 1).")
            question_texts = [_question_text(q, i) for i, q in enumerate(question_list)]
            num_questions = len(question_texts)
        num_pairs = batch_size * num_questions

        # ---- decision types --------------------------------------------------
        resolved_type = decision_type
        if resolved_type is None:
            resolved_type = getattr(self.config, "decision_type", None) or "choice"
        if isinstance(resolved_type, str):
            flat_types = [resolved_type] * num_pairs
        elif isinstance(resolved_type, (list, tuple)):
            if len(resolved_type) == 0:
                raise ValueError("decision_type: got an empty list.")
            if len(resolved_type) == 1:
                flat_types = [resolved_type[0]] * num_pairs
            elif len(resolved_type) == num_questions:
                flat_types = list(resolved_type) * batch_size
            elif len(resolved_type) == num_pairs:
                flat_types = list(resolved_type)
            else:
                raise ValueError(
                    f"decision_type: expected 1, m={num_questions} or B*m={num_pairs} entries, "
                    f"got {len(resolved_type)}."
                )
        else:
            raise TypeError(
                f"decision_type: expected a str or list of str, got "
                f"{type(resolved_type).__name__}."
            )
        for entry in dict.fromkeys(flat_types):
            if entry not in self.heads:
                raise ValueError(
                    f"unknown decision type {entry!r} — this model was built with "
                    f"decision_primitives={list(self.config.decision_primitives)}; registered "
                    f"heads are {available_decision_heads()}."
                )

        # ---- options ----------------------------------------------------------
        flat_options = self._normalize_options(options, batch_size, num_questions, num_pairs)

        # ---- per-pair contract checks (fail fast, actionable) ------------------
        for pair, dtype in enumerate(flat_types):
            has_schema = flat_options[pair] is not None
            if dtype == "noul" and has_schema:
                raise ValueError(
                    f"decision_type 'noul' is binary and takes no options — pair {pair} received "
                    f"{len(flat_options[pair])} option(s). Drop options= for noul questions or "
                    "use decision_type 'choice'."
                )
            if dtype in ("choice", "score") and not has_schema:
                raise ValueError(
                    f"decision_type {dtype!r} requires a schema: pair {pair} has no options — "
                    "pass options=[...] with one option list per question (a "
                    f"{dtype} decision needs its K options/levels; axiom D3.1)."
                )

        # ---- threshold ---------------------------------------------------------
        threshold: Optional[float] = None
        if confidence_threshold is not None:
            if isinstance(confidence_threshold, bool) or not isinstance(
                confidence_threshold, (int, float)
            ):
                raise TypeError(
                    "confidence_threshold: expected a float in [0, 1], got "
                    f"{type(confidence_threshold).__name__}."
                )
            if not 0.0 <= float(confidence_threshold) <= 1.0:
                raise ValueError(
                    f"confidence_threshold: expected a value in [0, 1], got "
                    f"{confidence_threshold!r}."
                )
            threshold = float(confidence_threshold)

        # ---- 1. state: ONE backbone call --------------------------------------
        want_hidden = (
            self.config.output_hidden_states if output_hidden_states is None else bool(output_hidden_states)
        )
        want_attentions = (
            self.config.output_attentions if output_attentions is None else bool(output_attentions)
        )
        if want_attentions:
            inputs["output_attentions"] = True
        try:
            representation = self.adapter.extract_representation(
                self.model,
                inputs,
                strategy=self.config.pooling_method,
                output_hidden_states=want_hidden,
            )
        except UnsupportedRepresentationError:
            if want_hidden:
                raise
            # LM-head-only wrappers (ForCausalLM/ForMaskedLM) expose just
            # `logits`; per audit Q4, collection is opt-in — retry once with
            # output_hidden_states=True and pool hidden_states[-1] (cost:
            # the L+1 activation references stay alive for this call).
            representation = self.adapter.extract_representation(
                self.model,
                inputs,
                strategy=self.config.pooling_method,
                output_hidden_states=True,
            )
        pooled = representation.pooled
        if pooled is not None and pooled.dim() > 2:
            # Contract repair, not a new convention. Representation.pooled is
            # documented as the per-row vector [B, D] (adapters/base.py:76), but a
            # native spatial pooler returns a feature MAP: ResNetModel pools with
            # nn.AdaptiveAvgPool2d((1, 1)) and hands it back unflattened
            # (modeling_resnet.py:297, 322-329), so pooler_output is [B, C, 1, 1]
            # -- while its sibling in the same audit group flattens the very same
            # op (torch.flatten(self.pooler(...), start_dim=1),
            # modeling_mobilenet_v2.py:354). Merging the trailing spatial axes is
            # exactly what the own head of that family does with nn.Flatten()
            # before nn.Linear(config.hidden_sizes[-1], ...)
            # (modeling_resnet.py:346-348) and restores the [B, D] the validator
            # and state_proj consume. Only reachable on a wrong rank, which the
            # validator below rejects, so no already-passing family changes
            # shape here.
            pooled = pooled.flatten(start_dim=1)
            representation = _dataclass_replace(representation, pooled=pooled)
        if pooled is None and representation.sequence_output is not None:
            # Defensive: adapter contract promises pooled, but never silently
            # pool here with an unrecorded strategy.
            raise UnsupportedRepresentationError(
                model_name=str(
                    getattr(self.config, "base_model_name_or_path", None)
                    or type(self.model).__name__
                ),
                architecture=self.capability_report.architecture,
                detected_outputs=[representation.field],
                attempted_strategies=["adapter.extract_representation"],
                failed_strategies=["pooled is None despite a sequence_output"],
                suggestion=(
                    "the adapter returned an unpooled representation — pass "
                    "pooling_method= explicitly or upgrade the adapter."
                ),
            )
        if pooled is None:
            raise UnsupportedRepresentationError(
                model_name=str(
                    getattr(self.config, "base_model_name_or_path", None)
                    or type(self.model).__name__
                ),
                architecture=self.capability_report.architecture,
                detected_outputs=[representation.field],
                attempted_strategies=["adapter.extract_representation"],
                failed_strategies=["no pooled or sequence representation in the output"],
                suggestion=(
                    "the backbone exposes no usable representation field; check "
                    "capability_report.jevx_conversion / reasons for the detection detail."
                ),
            )
        if runtime_validation:
            # Audit Q-003: shape/finiteness/alignment invariants of the
            # extracted representation, right after extraction succeeds.
            self.validator.validate_representation(
                representation, context="forward: state extraction"
            )
        check_finite(pooled, "state pooled E_M(S)")
        state_repr = self.state_proj(pooled)  # [B, D]
        check_finite(state_repr, "state_representation H_S")

        # ---- 1b. state sequence for cross_attention fusion (Def 5.e, Q-006) --
        fusion_sequence: Optional[torch.Tensor] = None
        fusion_state_mask: Optional[torch.Tensor] = None
        if self._fusion_needs_sequence:
            seq = representation.sequence_output
            if seq is None:
                raise ValueError(
                    f"fusion_type {self.config.fusion_type!r} needs the unpooled state "
                    "sequence H_S^seq, but the adapter extracted only a pooled "
                    f"representation (field={representation.field!r}, layout="
                    f"{representation.layout!r}). Use one of the pooled-state fusion "
                    "kinds (concat/additive/gated/bilinear/dot/mlp), or wrap a "
                    "backbone whose adapter exposes a [B, T, D] sequence output."
                )
            # Same learned map as the pooled H_S (Projection acts on the last
            # axis), so sequence and pooled state live in one width-D space.
            fusion_sequence = self.state_proj(seq)  # [B, T, D]
            check_finite(fusion_sequence, "state sequence H_S^seq")
            raw_mask = representation.attention_mask
            if isinstance(raw_mask, torch.Tensor) and tuple(raw_mask.shape) == tuple(
                seq.shape[:2]
            ):
                fusion_state_mask = raw_mask.to(torch.bool)
            # Fusion runs per (state, question) row — repeat state-major, the
            # same layout as state_rows/question_rows/option batches.
            fusion_sequence = fusion_sequence.repeat_interleave(
                num_questions, dim=0
            )
            if fusion_state_mask is not None:
                fusion_state_mask = fusion_state_mask.repeat_interleave(
                    num_questions, dim=0
                )

        # ---- 2. questions: ONE batched encode ---------------------------------
        question_device = state_repr.device
        question_dtype = state_repr.dtype
        # The memo is keyed on the question TEXTS: the pre-encoded-tensor
        # branch below has no text to key on (and a caller can mutate a tensor
        # in place, which no key would catch), so it is never memoized.
        question_cache_key = (
            ("H_Q", tuple(question_texts))
            if question_texts is not None and cache_conditioning
            else None
        )
        if question_texts is not None:
            question_raw = self._condition_cache_get(question_cache_key)
            if question_raw is None:
                question_raw = self.question_encoder.encode_texts(
                    question_texts, tokenizer=self.tokenizer
                )
                # residual transform
                question_raw = self.question_encoder(question_raw)
                self._condition_cache_put(
                    question_cache_key, question_raw.detach().clone()
                )
        else:
            assert encoded_questions is not None
            question_raw = encoded_questions
            if question_raw.is_floating_point():
                question_raw = question_raw.to(device=question_device, dtype=question_dtype)
            else:
                raise TypeError(
                    "questions: the pre-encoded tensor must be floating point, got "
                    f"{question_raw.dtype}."
                )
        if not condition:
            # conditioning_mode="off" (or the per-call
            # condition_options=False): the WHOLE call is state-only, so the
            # question path is neutralized too, not just the option residual.
            # Without this, "off" still concatenated H_Q into ConcatFusion's
            # W_cat [s ; q] and left half the conditioning stack live — the
            # naming/behaviour mismatch measured in reports/MULTI_QUESTION.md
            # §5.2 (off 0.7199 vs the state-only reference 0.7294 pooled over 11
            # questions). zeros_like is a non-differentiable substitute, so NO
            # gradient reaches E_Q; question_proj still receives one because the
            # zeroed row is a real input to it. This is bitwise the WAVE-2
            # state_only reference arm, which registered the same
            # question_encoder hook returning torch.zeros_like(o).
            question_raw = torch.zeros_like(question_raw)
        question_repr = self.question_proj(question_raw)  # [m, D] (or [B, m, D])
        check_finite(question_repr, "question_representations H_Q")
        if per_state_questions is not None:
            question_rows = question_repr.reshape(num_pairs, question_repr.shape[-1])
            question_field = question_repr
        else:
            question_field = question_repr.unsqueeze(0).expand(
                batch_size, -1, -1
            )  # [B, m, D]
            question_rows = (
                question_repr.unsqueeze(0)
                .expand(batch_size, -1, -1)
                .reshape(num_pairs, -1)
            )
        state_rows = state_repr.repeat_interleave(num_questions, dim=0)  # [n, D] state-major

        # ---- group by decision type (never mix probability spaces) -------------
        groups: Dict[str, List[int]] = {}
        for pair, dtype in enumerate(flat_types):
            groups.setdefault(dtype, []).append(pair)

        calibration = self.calibration_engine
        calibrate = calibration.method != "none" and calibration.is_fitted
        calibration_metadata = (
            {"method": calibration.method, "applied": calibrate}
            if calibration.method != "none"
            else None
        )

        max_schema_k = max(
            (len(row) for row in flat_options if row is not None), default=0
        )
        schema_field = (
            torch.zeros(
                num_pairs,
                max_schema_k,
                self.schema_proj.out_dim,
                device=state_repr.device,
                dtype=state_repr.dtype,
            )
            if max_schema_k > 0
            else None
        )

        decisions: List[Optional[DecisionOutput]] = [None] * num_pairs

        # ---- 3-8: per-group encode → fuse → head → calibrate → confidence → abstain
        for dtype, indices in groups.items():
            head = self.heads[dtype]
            schema_rows = [flat_options[i] for i in indices]
            present = [row is not None for row in schema_rows]
            if any(present) and not all(present):
                raise ValueError(
                    f"decision group {dtype!r} mixes questions with and without schemas at pairs "
                    f"{indices} — every question of one type must carry options consistently."
                )
            option_features = None
            option_mask = None
            if all(present):
                assert all(row is not None for row in schema_rows)
                # Group members, in group order, with each option's tokenizable
                # text: two calls with the same group content hit the same key.
                schema_cache_key = (
                    (
                        "H_C",
                        dtype,
                        tuple(
                            tuple(option.text() for option in row) for row in schema_rows
                        ),
                    )
                    if cache_conditioning
                    else None
                )
                cached_schema = self._condition_cache_get(schema_cache_key)
                if cached_schema is None:
                    features, mask = self.schema_encoder.encode(
                        schema_rows, tokenizer=self.tokenizer
                    )
                    self._condition_cache_put(
                        schema_cache_key,
                        (features.detach().clone(), mask.detach().clone()),
                    )
                else:
                    features, mask = cached_schema
                option_features = self.schema_proj(features)  # [g, K, D]
                check_finite(option_features, "schema_representations H_C")
                option_mask = mask if not bool(mask.all()) else None
                assert schema_field is not None
                local_k = option_features.shape[1]
                schema_field[indices, :local_k] = option_features

            fused_output = self.fusion(
                state_rows[indices],
                question_rows[indices],
                option_features if condition else None,
                option_mask if condition else None,
                state_sequence=fusion_sequence,
                state_mask=fusion_state_mask,
            )
            head_options = fused_output.option_features
            if head_options is None and option_features is not None:
                # condition_options=False: keep the [g, K, D] shape contract
                # with EXACT zeros (padded columns included, matching the
                # masked-row invariant) so the head still scores K options
                # from the state alone.
                head_options = torch.zeros_like(option_features)
            head_output = head(
                fused_output.fused,
                head_options,
                option_mask=option_mask,
                conditioning_enabled=condition,
                # State-only call => the option block above is a shape
                # placeholder, not a schema, so the duplicate-option tie (a
                # statement about the schema's rows) must not fire: every K
                # column of EXACT ZEROS is trivially a duplicate and the tie
                # would average the K per-class state scores into one value,
                # zeroing the spread AND (analytically) the state-pathway
                # gradient, so the arm could never leave the uniform score.
                # Conditioned calls pass True, i.e. exactly today's behaviour.
                duplicate_tie=condition,
            )
            logits = head_output.logits
            if calibrate:
                # Audit Q-012: calibrate ONLY the question's valid columns —
                # padded columns of a variable-K batch hold arbitrary raw
                # logits and must never feed isotonic's top-label max (or any
                # K-shaped calibrator). Padded columns keep their raw values
                # and are re-masked by head.build_output below.
                calibrated_logits = calibration.apply_valid_columns(
                    logits, option_mask, "logits"
                )
                calibrated_output = head.build_output(calibrated_logits, option_mask)
                probabilities = calibrated_output.probabilities
                predicted_index = calibrated_output.predicted_index
                expected_score = calibrated_output.expected_score
            else:
                probabilities = head_output.probabilities
                predicted_index = head_output.predicted_index
                expected_score = head_output.expected_score
            check_finite(probabilities, f"probabilities[{dtype!r}]")
            confidence = self.confidence(
                probabilities,
                logits=logits,
                option_mask=option_mask,
                fused=fused_output.fused,
            )
            check_finite(confidence, "confidence")

            for local, pair in enumerate(indices):
                abstained: Optional[torch.Tensor] = None
                if threshold is not None:
                    is_abstained = bool(confidence[local].detach() < threshold)
                    abstained = torch.tensor(
                        [is_abstained], dtype=torch.bool, device=confidence.device
                    )
                else:
                    is_abstained = False

                schema_row = flat_options[pair]
                option_keys = [option.key for option in schema_row] if schema_row else None
                if is_abstained:
                    index_value = None
                    value: Any = None
                    score_value = None
                else:
                    index_value = predicted_index[local : local + 1]
                    if option_keys is not None:
                        value = project_to_schema(index_value, option_keys)[0]
                    else:
                        value = int(index_value[0])
                    score_value = (
                        expected_score[local : local + 1]
                        if expected_score is not None
                        else None
                    )
                decisions[pair] = DecisionOutput(
                    logits=logits[local : local + 1],
                    probabilities=probabilities[local : local + 1],
                    confidence=confidence[local : local + 1],
                    predicted_index=index_value,
                    expected_score=score_value,
                    predicted_value=value,
                    abstained=abstained,
                    calibration_metadata=calibration_metadata,
                    schema_metadata={
                        "decision_type": dtype,
                        "option_keys": option_keys,
                    },
                )
                if runtime_validation:
                    # Audit Q-003: per-decision invariants (choice/score
                    # simplex, noul interior, confidence range, index range)
                    # checked right after each head output is assembled.
                    self.validator.validate_decision(
                        decisions[pair],
                        context=f"forward: decision pair {pair} ({dtype})",
                    )
        assert all(entry is not None for entry in decisions)  # groups cover all pairs
        final_decisions: List[DecisionOutput] = decisions  # type: ignore[assignment]

        # ---- 9. assemble --------------------------------------------------------
        effective_return_dict = (
            self.config.return_dict if return_dict is None else bool(return_dict)
        )
        raw_output = representation.raw_output
        capability_metadata: Dict[str, Any] = {
            "model_type": self.capability_report.model_type,
            "architecture": self.capability_report.architecture,
            "representation_field": representation.field,
            "adapter": getattr(self.adapter, "name", type(self.adapter).__name__),
            "pooling_method": self.config.pooling_method,
            "fusion_type": self.config.fusion_type,
            "decision_types": list(flat_types),
            "confidence_method": self.confidence.method_name,
            "calibration": calibration_metadata,
            "num_states": batch_size,
            "questions_per_state": num_questions,
            "pair_ordering": "flat index = b*m + j (state-major)",
            "backbone_forward_calls": 1,
        }
        output = JevXOutput(
            decisions=final_decisions,
            state_representation=state_repr,
            question_representations=question_field,
            schema_representations=schema_field,
            base_output=raw_output if return_base_output else None,
            hidden_states=getattr(raw_output, "hidden_states", None) if want_hidden else None,
            attentions=getattr(raw_output, "attentions", None) if want_attentions else None,
            capability_metadata=capability_metadata,
        )
        if not effective_return_dict:
            return output.to_tuple()
        return output

    def _normalize_options(
        self,
        options: Optional[Sequence[Any]],
        batch_size: int,
        num_questions: int,
        num_pairs: int,
    ) -> List[Optional[List[Option]]]:
        """Normalize ``options`` to one schema (or None) per pair."""
        if options is None:
            return [None] * num_pairs
        if not isinstance(options, (list, tuple)):
            raise TypeError(
                f"options: expected a list of schemas (list of list) or one flat schema list, "
                f"got {type(options).__name__}."
            )
        if len(options) == 0:
            return [None] * num_pairs
        per_question = all(
            entry is None or isinstance(entry, (list, tuple)) for entry in options
        )
        if per_question:
            length = len(options)
            if length == 1:
                rows: List[Any] = list(options) * num_pairs
            elif length == num_questions:
                rows = list(options) * batch_size
            elif length == num_pairs:
                rows = list(options)
            else:
                raise ValueError(
                    f"options: expected 1, m={num_questions} or B*m={num_pairs} schemas, got "
                    f"{length} — pass one option list per question (list of lists)."
                )
        else:
            if num_pairs != 1:
                raise ValueError(
                    f"options: got a single flat schema but there are B*m={num_pairs} "
                    "(state, question) pairs — pass one schema list per question "
                    "(a list of lists)."
                )
            rows = [list(options)]
        normalized: List[Optional[List[Option]]] = []
        for pair, row in enumerate(rows):
            if row is None:
                normalized.append(None)
                continue
            coerced = [
                _coerce_option(entry, pair, option_index)
                for option_index, entry in enumerate(row)
            ]
            if len(coerced) == 0:
                raise ValueError(
                    f"options[{pair}]: schema has K=0 options — every question needs at least "
                    "one option (axiom D3.1)."
                )
            normalized.append(coerced)
        return normalized

    # -- convenience API --------------------------------------------------------

    def decide(
        self,
        state: Any,
        questions: Any,
        decision_type: Union[str, Sequence[str], None] = None,
        options: Optional[Sequence[Any]] = None,
        **kwargs: Any,
    ) -> JevXOutput:
        """Answer ``questions`` about ``state`` (see :meth:`forward`).

        Accepts ONE state (``str``) with a list of questions, or a batch of
        B states (``list[str]`` / ``[B, T]`` inputs) — the same m questions
        are answered for every state; results come back as ``B*m``
        decisions in state-major order.
        """
        return self.forward(
            state=state,
            questions=questions,
            decision_type=decision_type,
            options=options,
            **kwargs,
        )

    def choice(self, state: Any, questions: Any, options: Optional[Sequence[Any]] = None, **kwargs: Any) -> JevXOutput:
        """Categorical choice decisions (``options`` required per question)."""
        return self.forward(
            state=state, questions=questions, decision_type="choice", options=options, **kwargs
        )

    def score(self, state: Any, questions: Any, options: Optional[Sequence[Any]] = None, **kwargs: Any) -> JevXOutput:
        """Ordinal score decisions — ``options`` are the levels ``0..K-1``."""
        return self.forward(
            state=state, questions=questions, decision_type="score", options=options, **kwargs
        )

    def noul(self, state: Any, questions: Any, **kwargs: Any) -> JevXOutput:
        """Binary (yes/no) decisions — no schema, probability in [0, 1]."""
        # options is popped (not hard-coded to None) so a caller-supplied
        # schema reaches forward's validation and raises the documented
        # "noul takes no options" ValueError instead of a TypeError.
        return self.forward(
            state=state,
            questions=questions,
            decision_type="noul",
            options=kwargs.pop("options", None),
            **kwargs,
        )

    # -- class-level registry passthrough ----------------------------------------

    @staticmethod
    def register_adapter(adapter_cls: Any) -> Any:
        """Register a custom adapter class (highest resolution priority)."""
        from .adapters import register_adapter

        return register_adapter(adapter_cls)


#: §45/§63: ``JevX`` is an alias of :class:`JevXForDecision` (same class —
#: saved ``architectures`` stays ``["JevXForDecision"]`` so Auto-class
#: resolution on reload is unaffected).
JevX = JevXForDecision
