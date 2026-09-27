# Copyright (c) 2026 ysnrfd
"""JevX configuration.

Follows the transformers v5 config conventions (verified against
``vendor/transformers/src/transformers/configuration_utils.py`` and
``models/bert/configuration_bert.py``): ``PreTrainedConfig`` is a
``@dataclass`` whose subclasses are auto-annotated via ``__init_subclass__``
(kw-only fields, arbitrary kwargs accepted for backward compatibility), and
``model_type`` selects the class on load.

JevX uses *composition*, not inheritance-of-fields: the wrapped model's full
config is never duplicated here — only its ``base_model_type`` and
``base_model_name_or_path`` are stored, and the base config is loaded
separately by the modeling layer when it builds ``E_M``.
"""

from __future__ import annotations

import math
from dataclasses import field

from huggingface_hub.dataclasses import strict

from transformers.configuration_utils import PreTrainedConfig

from jevx.decisions.generic import (
    STATE_ACTIVATIONS,
    STATE_PATHWAYS,
    available_decision_heads,
)

__all__ = [
    "JevXConfig",
    "CALIBRATION_METHODS",
    "CONDITIONING_MODES",
    "validate_decision_type",
]

#: Calibration methods accepted by :attr:`JevXConfig.calibration_method`.
#: Validated eagerly so typos fail at config-construction time.
CALIBRATION_METHODS: tuple[str, ...] = ("none", "temperature", "vector", "isotonic")

#: Conditioning modes accepted by :attr:`JevXConfig.conditioning_mode`.
#: Validated eagerly so typos fail at config-construction time.
CONDITIONING_MODES: tuple[str, ...] = ("full", "constant", "off")


def validate_decision_type(name: str | None) -> str | None:
    """Validate one decision-head name against the live head registry.

    Args:
        name: A registered head name, or ``None`` (the default — "no pin";
            the per-call ``forward(decision_type=...)`` argument decides).

    Returns:
        ``name`` unchanged, or ``None``.

    Raises:
        TypeError: ``name`` is neither ``str`` nor ``None``.
        ValueError: ``name`` is not in the registry; the message lists every
            registered head and how to add one.

    The registry is read at CALL time, not at import time, so a head
    registered after this module was imported still validates.
    """
    if name is None:
        return None
    if not isinstance(name, str):
        raise TypeError(
            "decision_type: expected a registered head name (str) or None (no pin), got "
            f"{type(name).__name__}."
        )
    registered = available_decision_heads()
    if name not in registered:
        raise ValueError(
            f"decision_type: {name!r} is not a registered decision head; registered heads are "
            f"{registered}. Register a new one with @register_decision_head(name), or leave "
            "decision_type=None and pass the type per call to forward(decision_type=...)."
        )
    return name


@strict
class JevXConfig(PreTrainedConfig):
    """Configuration for the JevX typed probabilistic decision layer.

    Pipeline this config parameterizes::

        H_S = E_M(S),  H_Q = E_Q(Q),  H_C = E_C(C)
        H   = F(H_S, H_Q, H_C)             # fusion_type
        P   = G_type(H, options)           # decision_primitives
        P'  = Cal(P)                       # calibration_method
        D   = Project(P', C)               # schema mapping in the caller

    Field semantics (shapes refer to the wrapper model, not this config):

    Args:
        base_model_name_or_path: Identifier of the wrapped HF checkpoint
            (composition: the base model's own config is NOT embedded here).
        base_model_type: ``model_type`` string of the wrapped model, e.g.
            ``"bert"``; used to rebuild ``E_M`` via the Auto classes.
        representation_strategy: Free-form metadata tag recorded in the
            config and round-tripped through save/load. It is **not read by
            any computation** — it does not change extraction, fusion or
            output. The strategy that actually runs is
            :attr:`pooling_method`.
        pooling_method: Sequence-pooling strategy name for ``H_S``; one of
            :func:`jevx.representations.available_pooling_strategies`
            (``"cls"``, ``"masked_mean"``, ``"encoder_pool"``,
            ``"last_valid"``, ``"max"``, ``"mean"``,
            ``"attention_pool"``, ``"learned_pool"``, ``"native"``).
            ``None`` (the default) means **adapter default**: a native
            pooled field is used when the backbone exposes one, otherwise
            the adapter's ``default_pooling`` is applied. The registered
            name is ``"last_valid"``, not ``"last"``.
        decision_hidden_size: Width ``D`` of the fused state consumed by the
            decision heads (also the heads' ``hidden_size``).
        question_hidden_size: Width of question embeddings ``E_Q(Q)``.
        schema_hidden_size: Width of schema/option embeddings ``E_C(C)``.
        fusion_type: How ``H`` is assembled from ``H_S``/``H_Q``/``H_C``
            (e.g. ``"concat"`` (default), ``"add"``, ``"gated"``,
            ``"cross_attention"`` — spec Def 5.e; that kind also consumes the
            backbone's `[B, T, D]` state sequence, see
            `jevx.representations.CrossAttentionFusion`).
        decision_primitives: Names of registered decision heads available to
            this model (see ``jevx.decisions.register_decision_head``);
            each name must resolve in the head registry at build time.
        decision_type: Default decision type for calls that pass none.
            ``None`` (default) = no pin, and ``forward(decision_type=...)``
            falls back to ``"choice"`. A name is validated against the head
            registry at construction, round-trips through save/load, and is the
            same value the ``JevX.from_pretrained(decision_type=...)`` factory
            kwarg pins.
        calibration_method: Post-hoc logit calibration applied to ``P`` —
            one of ``"none"`` (default), ``"temperature"``, ``"vector"``,
            ``"isotonic"``.
        confidence_method: Confidence engine method, e.g.
            ``"max_probability"`` (default), ``"margin"``, ``"entropy"``,
            ``"energy"``, ``"learned"``.
        state_pathway: Kind of the PRIMARY state score of the decision heads,
            ``"linear"`` (default, the shipped linear probe
            ``state_k = s^T W[k, :] + b[k]``) or ``"mlp"``
            (``state_k = h_s^T W2[k, :] + b[k]`` with
            ``h_s = act(W1 s + b1)``). The option interaction is an additive
            residual on top of it:
            ``z_k = state_k + lambda * r_k``. The default keeps the head
            bit-identical to the pre-flag one.
        state_hidden_size: Hidden width ``H`` of the ``"mlp"`` state pathway;
            ``None`` (default) means the same as :attr:`decision_hidden_size`.
        state_activation: Elementwise activation of the ``"mlp"`` state
            pathway, one of ``["gelu", "relu", "tanh"]``.
        conditioning_scale_init: Initial ``lambda`` of the option-interaction
            residual. The DEFAULT 0.0 means "unspecified" and leaves the heads
            at their own default lambda = 1.0 (the pre-flag scoring function);
            any other value is forwarded to the heads, so a curriculum can start
            a ramp from it. A lambda of exactly 0.0 also makes the residual
            exactly zero at runtime.
        conditioning_enabled: DERIVED ALIAS of :attr:``conditioning_mode``,
            kept for callers that predate the mode field. It is never an
            independent switch — see the precedence rule under
            :attr:``conditioning_mode``. It is ``True`` at the CLASS level but
            a FRESH ``JevXConfig()`` derives it from the default mode
            ``"off"``, so ``JevXConfig().conditioning_enabled is False``.
            When it ends up False the residual is EXACTLY zero on every call,
            and the question embedding entering fusion is exactly zero too,
            making the whole call state-only. A caller that flips it at RUNTIME
            (``config.conditioning_enabled = False``, e.g.
            ``jevx.training.conditioning_gate.apply_gate``) is still honoured by
            mode ``"full"``, which reads it on every call.
        conditioning_mode: How the conditioning path is executed — the SOURCE
            OF TRUTH for :attr:``conditioning_enabled``. One of
            :data:``CONDITIONING_MODES`:

            * ``"off"`` (DEFAULT) — no conditioning at all: bitwise equal to
              the shipped ``forward(condition_options=False)`` call. The option
              branch never enters fusion, the head's residual is exactly zero,
              AND the question rows handed to fusion are exactly zero, so the
              fused state does not depend on the question either. The default
              was changed from ``"full"`` to ``"off"`` on the measured evidence
              in ``reports/MULTI_QUESTION.md`` (pooled over 11 questions on 4
              datasets x 5 seeds: ``off`` 0.7199 > ``full`` 0.7046, 0 wins in 35
              paired comparisons, the one non-overlapping comparison a -16.92 pp
              loss for ``full`` on ravdess_gender, and 80.8-89.7 % of the
              trainable parameters sitting in the conditioning half). Revert in
              one line by setting the field default back to ``"full"``; a user
              who wants question conditioning passes
              ``JevXConfig(conditioning_mode="full")``.
            * ``"full"`` — the per-call path: the question
              and schema encoders run on EVERY call.
            * ``"constant"`` — conditioning stays ON, but the encoders
              are memoized on their INPUT CONTENT by a per-model,
              NON-PERSISTENT runtime cache
              (``JevXForDecision._condition_cache``; see
              ``JevXForDecision.clear_condition_cache``). A repeated call
              with the same content reuses the memoized ``H_Q`` / ``H_C`` and pays
              only the projections; a content change is a MISS, so that call
              computes the ``"full"`` path and refreshes the memo. The
              memo is consulted ONLY while the module is in ``eval()`` AND
              ``torch.is_grad_enabled()`` is False: a memoized tensor is
              detached, so serving it to a grad-enabled call would silently
              drop the encoder gradients (and replaying a live graph would
              double-backward). The memo is also WEIGHT-keyed: on every
              memo-eligible call the two memoized encoders' tensor version
              counters are compared, and ANY change drops the memo, so an
              ``eval -> optimizer step -> eval`` loop cannot serve an encoding
              computed from the previous weights.

            Precedence is TOTAL over all six (mode, enabled) pairs (see
            :meth:``__post_init__``): ``"off"`` FORCES
            ``conditioning_enabled = False`` — INCLUDING for the pair
            ``("off", conditioning_enabled=True)``, which is a COERCION (the
            mode wins) and does not raise, because "off" and False are the same
            request; ``"full"`` / ``"constant"``
            with ``conditioning_enabled=False`` is an explicit
            contradiction and raises a ValueError naming BOTH fields; every
            other pair ends at ``conditioning_enabled = True``. There
            is no silent choice. The per-call
            ``forward(condition_options=...)`` argument still overrides the mode
            for that one call.
        schema_encoder_width: Width the OPTION encoder runs at, projected up to
            the decision width by ``schema_proj``. ``None`` (default) keeps
            today's width, i.e. :attr:``schema_hidden_size``; any other value
            must be a positive int `<= schema_hidden_size``. IMPLEMENTED in
            ``JevXForDecision.__init__``: both ``schema_encoder.hidden_size`` and
            ``schema_proj.in_dim`` become this width. The default is unchanged,
            so a default model keeps today's state_dict bit for bit.
        shared_question_schema_encoder: SPECIFIED, NOT IMPLEMENTED — RESERVED.
            Intended semantics: when True, questions and options are encoded by
            ONE shared ``QuestionEncoder`` carrying a learnable TYPE TAG
            (``is_question`` / ``is_option``, added to the encoded rows)
            instead of two independent encoders — about half the conditioning
            parameters.

            ``JevXConfig.__post_init__`` therefore REJECTS any non-default
            (True) value with a named ValueError: the config can never
            silently accept a switch that does nothing. The blocking reason is
            serialization, not the encoder: one shared module appears in
            ``state_dict()`` twice (``question_encoder.*`` and
            ``schema_encoder.text_encoder.*``), and transformers'
            ``remove_tied_weights_from_state_dict`` raises
            "The weights trying to be saved contained shared tensors" unless
            the alias is declared in the model's ``_tied_weights_keys``, which
            on this wrapper is also what carries the BACKBONE's own tied
            weights (e.g. a causal-LM ``lm_head``). A model that configures
            this flag but cannot be saved is worse than an explicit error.
            The supporting pieces DO exist and are tested on their own
            (``QuestionEncoder(type_tags=True)`, ``SchemaEncoder(
            text_encoder=...)``); what is missing is the model-level wiring
            and the tied-weight declaration.
        freeze_base_model: When True (default), ``E_M`` parameters are
            frozen and only the JevX layers train.
        runtime_validation: When True (default), ``JevXForDecision.forward``
            runs the ``jevx.validation.ValidationEngine`` invariant checks
            (representation shape/finiteness, per-question simplex, noul
            interior, confidence/index ranges) at their boundaries;
            ``False`` disables them — outputs are unchanged either way.
        output_attentions / output_hidden_states: Passthrough flags handled
            by ``PreTrainedConfig`` (inherited, not redeclared — redeclaring
            would shadow the base property and its eager-attention dispatch).
        version: JevX checkpoint format version metadata.

    Example:
        >>> from jevx.configuration_jevx import JevXConfig
        >>> cfg = JevXConfig(base_model_type="bert", decision_hidden_size=256)
        >>> cfg.model_type
        'jevx'
        >>> JevXConfig.from_dict(cfg.to_dict()).decision_hidden_size
        256
    """

    model_type = "jevx"

    # --- composition with the base model ---------------------------------
    base_model_name_or_path: str | None = None
    base_model_type: str | None = None

    # --- representations / fusion ----------------------------------------
    representation_strategy: str = "native"
    pooling_method: str | None = None
    decision_hidden_size: int = 768
    question_hidden_size: int = 768
    schema_hidden_size: int = 768
    fusion_type: str = "concat"

    # --- decision / post-processing --------------------------------------
    decision_primitives: list[str] = field(
        default_factory=lambda: ["choice", "score", "noul"]
    )
    #: Default decision type; None = no pin (forward falls back to "choice").
    #: Validated against the head registry in __post_init__.
    decision_type: str | None = None
    calibration_method: str = "none"
    confidence_method: str = "max_probability"

    # --- decision scoring: state pathway + option conditioning ---------------
    state_pathway: str = "linear"
    state_hidden_size: int | None = None
    state_activation: str = "gelu"
    conditioning_scale_init: float = 0.0
    conditioning_enabled: bool = True
    #: Conditioning-path execution mode — the SOURCE OF TRUTH for
    #: conditioning_enabled (see the class docstring).
    #:
    #: DEFAULT "off" (WAVE-2 follow-up, DECISION-3). Revert in ONE line:
    #:     conditioning_mode: str = "full"
    #: Measured basis (reports/MULTI_QUESTION.md; 11 questions, 4 datasets,
    #: 5 seeds, frozen features, every arm from bitwise-identical init):
    #:     pooled accuracy  off 0.7199  <  state_only 0.7294  <  full 0.7046
    #:     35 paired comparisons, 0 wins for ANY conditioning mode; the only
    #:     non-overlapping one is a -16.92 pp LOSS for "full"
    #:     (ravdess_gender full 0.7296+-0.0687 vs off 0.8988+-0.0535)
    #:     the conditioning half is 80.8-89.7 % of the trainable parameters
    #:     and receives EXACTLY zero gradient in the "off" arm at step 1
    #:
    #: "off" is no longer a misnomer: it neutralizes the QUESTION path in
    #: fusion as well as the option residual, so it IS the state-only
    #: reference that measured best. A user who wants question conditioning
    #: sets conditioning_mode="full" (one kwarg) and gets the per-call path.
    conditioning_mode: str = "off"
    #: Option-encoder width; None = schema_hidden_size (today's width).
    schema_encoder_width: int | None = None
    #: One shared, type-tagged encoder for questions AND options; False = the
    #: shipped two-encoder setup.
    shared_question_schema_encoder: bool = False

    # --- training / metadata ---------------------------------------------
    freeze_base_model: bool = True
    runtime_validation: bool = True
    version: str = "0.1.0"

    def __post_init__(self, **kwargs) -> None:
        """Validate JevX-specific enums, then run base-class normalization."""
        super().__post_init__(**kwargs)
        if self.calibration_method not in CALIBRATION_METHODS:
            raise ValueError(
                f"calibration_method: {self.calibration_method!r} is not a known calibration "
                f"method; expected one of {list(CALIBRATION_METHODS)}."
            )
        if not isinstance(self.runtime_validation, bool):
            raise TypeError(
                "runtime_validation: expected a bool (run the ValidationEngine "
                f"checks from forward), got {type(self.runtime_validation).__name__}."
            )
        if self.state_pathway not in STATE_PATHWAYS:
            raise ValueError(
                f"state_pathway: {self.state_pathway!r} is not a known state pathway; "
                f"expected one of {list(STATE_PATHWAYS)}."
            )
        if self.state_activation not in STATE_ACTIVATIONS:
            raise ValueError(
                f"state_activation: {self.state_activation!r} is not a known activation; "
                f"expected one of {list(STATE_ACTIVATIONS)}."
            )
        if self.state_hidden_size is not None and (
            not isinstance(self.state_hidden_size, int)
            or isinstance(self.state_hidden_size, bool)
            or self.state_hidden_size < 1
        ):
            raise ValueError(
                f"state_hidden_size: expected a positive int or None (= decision_hidden_size), "
                f"got {self.state_hidden_size!r}."
            )
        if not isinstance(self.conditioning_enabled, bool):
            raise TypeError(
                "conditioning_enabled: expected a bool (allow the option-interaction "
                f"residual), got {type(self.conditioning_enabled).__name__}."
            )
        if self.conditioning_mode not in CONDITIONING_MODES:
            raise ValueError(
                f"conditioning_mode: {self.conditioning_mode!r} is not a known conditioning "
                f"mode; expected one of {list(CONDITIONING_MODES)}."
            )
        # G0.2 — the (mode, enabled) pair must never end in two contradictory
        # values. conditioning_mode is the source of truth; conditioning_enabled
        # is its derived alias, so all 3x2 pairs end in AGREEMENT or in a
        # named construction error (never in a silent choice):
        #   off       + False  -> agree (enabled forced False)
        #   off       + True   -> agree (enabled forced False; the mode wins)
        #   full/const+ True   -> agree
        #   full/const+ False  -> ValueError naming BOTH fields
        if self.conditioning_mode == "off":
            self.conditioning_enabled = False
        elif not self.conditioning_enabled:
            raise ValueError(
                f"conditioning_mode={self.conditioning_mode!r} asks for the OPTION "
                "features to condition every call, but conditioning_enabled=False asks for "
                "the state-only path — the two fields contradict each other and JevX does "
                "not pick one silently. Set conditioning_enabled=True to keep "
                f"conditioning_mode={self.conditioning_mode!r}, or set "
                "conditioning_mode='off' for the state-only path."
            )
        else:
            self.conditioning_enabled = True
        if not isinstance(self.shared_question_schema_encoder, bool):
            raise TypeError(
                "shared_question_schema_encoder: expected a bool (one shared, "
                f"type-tagged encoder for questions and options), got "
                f"{type(self.shared_question_schema_encoder).__name__}."
            )
        if self.shared_question_schema_encoder:
            # RESERVED: a value that is accepted and then ignored is the exact
            # defect this check exists to prevent.
            raise ValueError(
                "shared_question_schema_encoder=True is RESERVED — specified in the "
                "JevX spec but NOT implemented by JevXForDecision, and therefore "
                "rejected rather than silently accepted. Sharing one type-tagged "
                "encoder would register the same module twice in state_dict() "
                "(question_encoder.* and schema_encoder.text_encoder.*), which "
                "transformers refuses to serialize unless the alias is declared in "
                "_tied_weights_keys — the slot that also carries the backbone's own "
                "tied weights. Leave it False (the shipped two-encoder stack) until "
                "the shared layout ships with that declaration."
            )
        if self.schema_encoder_width is not None and (
            not isinstance(self.schema_encoder_width, int)
            or isinstance(self.schema_encoder_width, bool)
            or self.schema_encoder_width < 1
        ):
            raise ValueError(
                "schema_encoder_width: expected a positive int or None "
                f"(= schema_hidden_size {self.schema_hidden_size}), got "
                f"{self.schema_encoder_width!r}."
            )
        if (
            self.schema_encoder_width is not None
            and self.schema_encoder_width > self.schema_hidden_size
        ):
            raise ValueError(
                f"schema_encoder_width: {self.schema_encoder_width} is wider than "
                f"schema_hidden_size {self.schema_hidden_size} - this field exists to run "
                "the option encoder at a FRACTION of the stack width; set it to None to "
                "keep the full width."
            )
        scale = self.conditioning_scale_init
        if (
            not isinstance(scale, (int, float))
            or isinstance(scale, bool)
            or not math.isfinite(float(scale))
        ):
            raise ValueError(
                f"conditioning_scale_init: expected a finite float, got {scale!r}; an "
                "inf/NaN lambda would poison every logit."
            )
        if not isinstance(self.decision_primitives, (list, tuple)):
            raise TypeError(
                "decision_primitives: expected a list of registered decision-head names, "
                f"got {type(self.decision_primitives).__name__}."
            )
        self.decision_primitives = [str(name) for name in self.decision_primitives]
        # G17: decision_type is a declared field (it used to be set by
        # from_pretrained only, so it neither round-tripped nor validated).
        self.decision_type = validate_decision_type(self.decision_type)
        for attr in (
            "decision_hidden_size",
            "question_hidden_size",
            "schema_hidden_size",
        ):
            value = getattr(self, attr)
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{attr}: expected a positive int, got {value!r}.")
