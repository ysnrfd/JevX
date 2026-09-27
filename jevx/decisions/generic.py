# Copyright (c) 2026 ysnrfd
"""Generic typed decision-head framework: base class, registry, experimental heads.

Contract shared by every head
-----------------------------
``forward(fused, option_features, ...) -> DecisionOutput`` where

* ``fused``          — ``[B, D]`` fused state ``H = F(H_S, H_Q, H_C)``;
* ``option_features``— ``[B, K, D]`` per-option embeddings ``e_k`` (or
  ``None`` for heads with a fixed number of outputs);
* ``option_mask``    — ``[B, Kmax]`` bool, ``True`` = valid option, for
  variable-``K`` batches; ``num_options`` (``[B]`` or int) is an equivalent
  count-based spelling;
* returns raw ``logits [B, K]`` (unmasked), mask-aware ``probabilities
  [B, K]`` (invalid entries exactly 0, each row sums to 1) and
  ``predicted_index [B]`` (always inside the valid set).

New primitives plug in with ``@register_decision_head("name")``; nothing in
this package is hardcoded to the built-in three.

Import layout note: the framework (this module) is dependency-free with
respect to sibling heads so ``choice``/``score``/``noul`` can import from it
without cycles.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, Callable, ClassVar, Sequence, TypeVar

import torch
import torch.nn as nn

from jevx.modeling_outputs import DecisionOutput
from jevx.utils.numerical import masked_softmax
from jevx.utils.shapes import assert_batch_alignment, assert_shape, check_finite

__all__ = [
    "DECISION_HEAD_REGISTRY",
    "DEFAULT_STATE_PATHWAY_OPTIONS",
    "MLPStatePathway",
    "STATE_ACTIVATIONS",
    "STATE_PATHWAYS",
    "DecisionHead",
    "LinearProbePathway",
    "MultiLabelHead",
    "RegressionHead",
    "available_decision_heads",
    "build_decision_head",
    "get_decision_head",
    "mask_from_num_options",
    "project_to_schema",
    "register_decision_head",
]

DECISION_HEAD_REGISTRY: dict[str, type["DecisionHead"]] = {}

T = TypeVar("T")


def register_decision_head(name: str) -> Callable[[type[T]], type[T]]:
    """Class decorator registering a :class:`DecisionHead` under ``name``.

    Example:
        >>> @register_decision_head("choice")
        ... class ChoiceHead(DecisionHead): ...
    """

    def decorator(cls: type[T]) -> type[T]:
        if name in DECISION_HEAD_REGISTRY and DECISION_HEAD_REGISTRY[name] is not cls:
            raise ValueError(
                f"decision head {name!r} is already registered to "
                f"{DECISION_HEAD_REGISTRY[name].__name__}; choose a different name."
            )
        DECISION_HEAD_REGISTRY[name] = cls
        setattr(cls, "head_name", name)
        return cls

    return decorator


def available_decision_heads() -> list[str]:
    """Return the sorted names of all registered decision heads."""
    return sorted(DECISION_HEAD_REGISTRY)


def get_decision_head(name: str) -> type["DecisionHead"]:
    """Look up a registered head class by name.

    Raises:
        ValueError: ``name`` is unknown; the message lists every registered
            name and how to register a new one.
    """
    if not isinstance(name, str):
        raise TypeError(f"decision head name must be a str, got {type(name).__name__}.")
    if name not in DECISION_HEAD_REGISTRY:
        raise ValueError(
            f"unknown decision head {name!r}; registered heads are "
            f"{available_decision_heads()}. Register new ones with "
            f"@register_decision_head({name!r})."
        )
    return DECISION_HEAD_REGISTRY[name]


def build_decision_head(name: str, **kwargs: Any) -> "DecisionHead":
    """Instantiate the head registered under ``name`` with ``kwargs``.

    Every constructor argument is forwarded verbatim, including the
    state-pathway ones — ``state_pathway`` (``"linear"``/``"mlp"``),
    ``state_hidden_size``, ``state_activation``, ``conditioning_scale_init``
    and ``conditioning_enabled`` — so a config-level recipe reaches any head
    through the registry. A head that rejects an argument raises ``TypeError``,
    which :class:``jevx.modeling_jevx.JevXForDecision`` reports as an
    actionable "needs extra constructor arguments" error.
    """
    cls = get_decision_head(name)
    return cls(**kwargs)


def mask_from_num_options(
    num_options: torch.Tensor | int, width: int, batch_size: int | None = None
) -> torch.Tensor:
    """Build a ``[B, Kmax]`` boolean mask from per-row option counts.

    Args:
        num_options: ``[B]`` long tensor of valid-option counts, or a python
            int applied to every row.
        width: ``Kmax`` — the mask's second dimension.
        batch_size: Required when ``num_options`` is an int (to know ``B``);
            ignored for tensor input.

    Returns:
        Bool tensor ``[B, width]`` where ``True`` marks a valid option
        (valid options are always the leading ``num_options[b]`` entries).
    """
    if width < 1:
        raise ValueError(f"mask_from_num_options: width must be >= 1, got {width}.")
    if isinstance(num_options, int):
        if batch_size is None:
            raise ValueError(
                "mask_from_num_options: batch_size is required when num_options is an int."
            )
        counts = torch.full((batch_size,), num_options, dtype=torch.long)
    else:
        assert_shape(num_options, (None,), "num_options")
        counts = num_options.long()
    arange = torch.arange(width, device=counts.device)
    return arange.unsqueeze(0) < counts.unsqueeze(1)


def project_to_schema(
    indices: torch.Tensor,
    option_keys: Sequence[T],
) -> list[T]:
    """Map predicted indices to schema values, validating ``y in C``.

    This is the ``Project`` step: ``D = Project(P', C)``. Every index must
    address an entry of the schema ``C = option_keys``.

    Args:
        indices: Predicted indices — 1-D ``[B]`` tensor or scalar tensor of
            ``torch.long`` (or python ints).
        option_keys: The schema ``C``: a sequence of option keys/values of
            length ``K``.

    Returns:
        List of length ``B`` holding ``option_keys[idx]`` per row.

    Raises:
        ValueError: ``option_keys`` is empty, or an index lies outside
            ``[0, K)`` — i.e. the prediction is not in ``C``.
    """
    if option_keys is None or len(option_keys) == 0:
        raise ValueError(
            "project_to_schema: option_keys (schema C) is empty; a decision schema must "
            "contain at least one option."
        )
    if isinstance(indices, torch.Tensor):
        flat = indices.reshape(-1).tolist()
    elif isinstance(indices, int):
        flat = [indices]
    else:
        flat = list(indices)
    k = len(option_keys)
    out: list[T] = []
    for row, idx in enumerate(flat):
        idx = int(idx)
        if not (0 <= idx < k):
            raise ValueError(
                f"project_to_schema: predicted index {idx} (row {row}) is not in the decision "
                f"schema C of size {k}; every prediction must satisfy y in C."
            )
        out.append(option_keys[idx])
    return out


#: Default width of the additive state-only pathway (see
#: :class:`LinearProbePathway`). It is allocated EAGERLY at construction so the
#: parameter is visible to an optimizer built before the first forward pass, and
#: sliced to the observed ``K`` at call time (variable-K safe). 128 covers every
#: schema size in the library and its benchmarks; a wider schema transparently
#: grows the parameter (see :meth:`LinearProbePathway._ensure_width`).
DEFAULT_STATE_PATHWAY_OPTIONS: int = 128

#: State-pathway kinds accepted by every :class:``DecisionHead``. ``"linear"`` is
#: the shipped linear probe (`state_k = s^T W[:, k] + b[k]`); `"mlp"` is the
#: nonlinear state path (`state_k = h_s^T W2[:, k] + b[k]` with
#: `h_s = act(W1 s + b1)`). Heads additionally accept `"none"` (head level only,
#: the pre-EXP-H ablation arm); the config enum is the two shipped kinds.
STATE_PATHWAYS: tuple[str, ...] = ("linear", "mlp")

#: Elementwise activations available to the ``"mlp"`` state pathway.
STATE_ACTIVATIONS: tuple[str, ...] = ("gelu", "relu", "tanh")


def build_activation(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return the elementwise activation called `name`.

    Args:
        name: One of :data:`STATE_ACTIVATIONS`.

    Raises:
        ValueError: `name` is not a shipped activation; the message lists them.
    """
    if name == "gelu":
        return torch.nn.functional.gelu
    if name == "relu":
        return torch.nn.functional.relu
    if name == "tanh":
        return torch.tanh
    raise ValueError(
        f"state_activation: {name!r} is not a known activation; expected one of "
        f"{list(STATE_ACTIVATIONS)}."
    )


class LinearProbePathway(nn.Module):
    """Additive state-only scoring pathway: ``z_k = s^T W[:, k] + b[k]``.

    This is the missing half of a choice head. Without it, a softmax-over-options
    head can only score ``z_k = f(s, e_k)`` through the OPTION INTERACTION, and
    for a single fixed question + fixed schema every option row ``e_k`` is the
    same vector on every batch row, so the whole head is a multiplicative
    reparameterization of a linear map whose per-class weights must be reached
    indirectly (through the schema encoder and the fusion option gate). That
    indirection is what makes short training underfit a linear-probe task.

    With this pathway the scoring function is the SUM

        z_k = s^T W[:, k] + b[k]  +  r_k(s, e_k)

    where ``r_k`` is the option-interaction residual. Setting the residual to
    zero leaves EXACTLY ``nn.Linear(hidden_size, K)`` on the fused state (the
    same map ``BertForSequenceClassification.classifier`` applies), so the head
    is a strict SUPERSET of a linear probe and can never be worse than one for
    long enough.

    Initialization is ZERO, not the kaiming-uniform default of
    ``nn.Linear`` (see :meth:`reset_parameters`). The pathway is a
    RESIDUAL added to the option-interaction term, so a freshly constructed
    head is bit-for-bit identical to the same head without the pathway; the
    additive probe path then grows from a gradient that is non-zero at
    ``W = 0`` exactly as it is for a plain probe.

    Invariants preserved: the returned logits are raw ``[B, K]`` scores, so the
    caller's mask-aware ``masked_softmax`` still normalizes each question over
    its valid options only; invalid (masked) columns keep whatever raw value the
    interaction gives them and are forced to probability exactly 0 downstream.

    Args:
        hidden_size: Input dim ``D`` of the fused state.
        num_options: Initial capacity ``K`` of the additive term. ``None`` uses
            :data:`DEFAULT_STATE_PATHWAY_OPTIONS`. The parameter is sliced to
            the observed ``K`` on every call, so variable-``K`` batches never
            reallocate.
    """

    def __init__(self, hidden_size: int, num_options: int | None = None) -> None:
        if not isinstance(hidden_size, int) or hidden_size < 1:
            raise ValueError(f"hidden_size must be a positive int, got {hidden_size!r}.")
        if num_options is None:
            num_options = DEFAULT_STATE_PATHWAY_OPTIONS
        if not isinstance(num_options, int) or num_options < 1:
            raise ValueError(f"num_options must be a positive int, got {num_options!r}.")
        super().__init__()
        self.hidden_size = hidden_size
        # Stored [K, D] so F.linear(weight) is the natural call; K is the
        # OUT dim and D the IN dim, exactly like nn.Linear(K, D).transpose.
        self.weight = nn.Parameter(torch.empty(num_options, hidden_size))
        self.bias = nn.Parameter(torch.empty(num_options))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """ZERO-initialized (the standard zero-init residual convention).

        The pathway is a RESIDUAL added to the option-interaction term, so
        starting it at zero leaves a freshly constructed head numerically
        IDENTICAL to the pre-EXP-H head (bit-for-bit on every fixture and
        golden reference), and the additive probe path grows under training
        from a gradient that is non-zero at W = 0 exactly as it is for a plain
        ``nn.Linear(D, K)`` probe. AdamW is
        scale-adaptive per coordinate, so the term reaches a probe scale
        within the benchmark budget without ever dominating the interaction
        term at step 0.
        """
        with torch.no_grad():
            self.weight.zero_()
            self.bias.zero_()

    @property
    def width(self) -> int:
        """Allocated capacity ``K`` of the additive term (weight rows)."""
        return int(self.weight.shape[0])

    def _ensure_width(self, width: int) -> None:
        """Grow the parameter to hold ``width`` options (no-op when it fits).

        Growth REPLACES the ``nn.Parameter`` objects rather than resizing their
        ``.data`` in place, so a parameter an optimizer has already stepped on
        never sees a shape mismatch: the new objects start with fresh optimizer
        state, and the previously trained columns are copied across.
        """
        if width <= self.width:
            return
        weight = torch.zeros(width, self.hidden_size)
        bias = torch.zeros(width)
        with torch.no_grad():
            weight[: self.width].copy_(self.weight)
            bias[: self.width].copy_(self.bias)
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)

    def forward(self, fused: torch.Tensor, width: int | None = None) -> torch.Tensor:
        """Additive state-only logits ``[B, width]`` for ``width`` options.

        ``width=None`` means the full allocated capacity, so a head whose output
        width is fixed at construction can call ``pathway(fused)``.
        """
        if width is None:
            width = self.width
        if not isinstance(width, int) or isinstance(width, bool) or width < 1:
            raise ValueError(f"width must be a positive int, got {width!r}.")
        self._ensure_width(width)
        return torch.nn.functional.linear(
            fused, self.weight[:width], self.bias[:width]
        )

class MLPStatePathway(nn.Module):
    """``Nonlinear state pathway: ``z_k = h_s^T W2[k, :] + b[k]`` with ``h_s = act(W1 s + b1)``.

    This is the PRIMARY scoring term of a head configured with
    ``state_pathway="mlp"``; the option interaction is an additive RESIDUAL
    scaled by lambda, never the primary signal:

        state_k = h_s^T W2[k, :] + b[k]
        z_k     = state_k + lambda * r_k(h_s, e_k)

    Where :class:`LinearProbePathway` maps the fused state with a single affine
    map, this one first pushes the state through a hidden layer of width ``H``,
    so the state score is a genuine function of the state rather than a
    reparameterization of one. The math review that motivated the flag is the
    single-shared-question regime: with ``e_k`` constant across rows the bilinear
    interaction collapses to ``z_k = s^T (W_k + A e_k) + b_k``, still LINEAR in
    ``s``, so a linear state path is all the interaction can ever supply.

    Layout note: ``W2``/``b2`` are stored ``[K_max, H]`` — the
    ``nn.Linear(out_features=K, in_features=H)`` layout, i.e. the ``[H, K]`` map
    of the equation above transposed — so the same ``[:width]`` slice as
    :class:`LinearProbePathway` keeps variable-``K`` batches exact. The module is
    therefore bit-for-bit the shape of
    ``nn.Sequential(nn.Linear(D, H), act, nn.Linear(H, K))``.

    Initialization: ``W1``/``b1`` use the standard ``nn.Linear`` scheme (the
    hidden layer starts at a well-scaled projection) while ``W2``/``b2`` are
    ZERO — the standard zero-init residual convention, so switching a head to
    ``state_pathway="mlp"`` leaves its step-0 scores bit-identical to the linear
    pathway's (also zero at init) while still receiving a non-zero gradient at
    step 0, because the gradient of the state score w.r.t. ``W2`` is
    ``(dL/dz) (x) h_s`` and ``h_s != 0``.

    Args:
        hidden_size: Input dim ``D`` of the fused state.
        num_options: Initial capacity ``K``; ``None`` uses
            :data:``DEFAULT_STATE_PATHWAY_OPTIONS`. Sliced to the observed
            ``K`` at call time (variable-``K`` safe).
        mlp_hidden_size: Hidden width ``H``; ``None`` reuses ``hidden_size``.
        activation: Elementwise activation name, one of
            :data:``STATE_ACTIVATIONS`.
    """

    def __init__(
        self,
        hidden_size: int,
        num_options: int | None = None,
        mlp_hidden_size: int | None = None,
        activation: str = "gelu",
    ) -> None:
        if not isinstance(hidden_size, int) or hidden_size < 1:
            raise ValueError(f"hidden_size must be a positive int, got {hidden_size!r}.")
        if num_options is None:
            num_options = DEFAULT_STATE_PATHWAY_OPTIONS
        if not isinstance(num_options, int) or num_options < 1:
            raise ValueError(f"num_options must be a positive int, got {num_options!r}.")
        if mlp_hidden_size is None:
            mlp_hidden_size = hidden_size
        if not isinstance(mlp_hidden_size, int) or mlp_hidden_size < 1:
            raise ValueError(
                f"state_hidden_size must be a positive int, got {mlp_hidden_size!r}."
            )
        super().__init__()
        self.hidden_size = hidden_size
        self.mlp_hidden_size = mlp_hidden_size
        self.activation_name = activation
        self.act = build_activation(activation)  # validates the name
        self.w1 = nn.Parameter(torch.empty(mlp_hidden_size, hidden_size))
        self.b1 = nn.Parameter(torch.zeros(mlp_hidden_size))
        self.w2 = nn.Parameter(torch.empty(num_options, mlp_hidden_size))
        self.b2 = nn.Parameter(torch.empty(num_options))
        self.reset_parameters()


    def reset_parameters(self) -> None:
        """Standard init on the hidden layer, ZERO on the state-scoring layer."""
        bound = 1.0 / math.sqrt(self.hidden_size)
        with torch.no_grad():
            self.w1.uniform_(-bound, bound)
            self.b1.zero_()
            self.w2.zero_()
            self.b2.zero_()

    @property
    def width(self) -> int:
        """Allocated capacity ``K`` of the state-scoring layer (``W2`` rows)."""
        return int(self.w2.shape[0])

    def _ensure_width(self, width: int) -> None:
        """Grow ``W2``/``b2`` to hold ``width`` options (no-op when it fits).

        Same contract as :meth:``LinearProbePathway._ensure_width``: the
        parameters are REPLACED, not resized in place, so an optimizer that has
        already stepped on them never sees a shape mismatch.
        """
        if width <= self.width:
            return
        weight = torch.zeros(width, self.mlp_hidden_size)
        bias = torch.zeros(width)
        with torch.no_grad():
            weight[: self.width].copy_(self.w2)
            bias[: self.width].copy_(self.b2)
        self.w2 = nn.Parameter(weight)
        self.b2 = nn.Parameter(bias)

    def hidden(self, fused: torch.Tensor) -> torch.Tensor:
        """The state hidden layer ``h_s = act(W1 s + b1)``, shape ``[B, H]``."""
        return self.act(torch.nn.functional.linear(fused, self.w1, self.b1))

    def project(
        self, hidden: torch.Tensor, width: int | None = None
    ) -> torch.Tensor:
        """Map the hidden state to option scores, sliced to ``width`` options."""
        if width is None:
            width = self.width
        if not isinstance(width, int) or isinstance(width, bool) or width < 1:
            raise ValueError(f"width must be a positive int, got {width!r}.")
        self._ensure_width(width)
        return torch.nn.functional.linear(hidden, self.w2[:width], self.b2[:width])

    def forward(self, fused: torch.Tensor, width: int | None = None) -> torch.Tensor:
        """State-pathway logits ``[B, width]`` (the MLP score, no residual)."""
        return self.project(self.hidden(fused), width)
def _interact(
    fused: torch.Tensor,
    option_features: torch.Tensor,
    mode: str,
    bilinear_weight: torch.Tensor | None = None,
    mlp: nn.Module | None = None,
) -> torch.Tensor:
    """Score every option against the fused state: ``[B, K]`` logits.

    Modes:

    * ``"dot"``     — ``z_k = <fused, e_k>``
    * ``"bilinear"``— ``z_k = fused^T W e_k``
    * ``"mlp"``     — ``z_k = MLP([fused ; e_k])``
    """
    if mode == "dot":
        return torch.einsum("bd,bkd->bk", fused, option_features)
    if mode == "bilinear":
        if bilinear_weight is None:
            raise RuntimeError("_interact: mode='bilinear' requires bilinear_weight.")
        return torch.einsum("bd,de,bke->bk", fused, bilinear_weight, option_features)
    if mode == "mlp":
        if mlp is None:
            raise RuntimeError("_interact: mode='mlp' requires an mlp module.")
        b, k, _ = option_features.shape
        paired = torch.cat(
            [fused.unsqueeze(1).expand(-1, k, -1), option_features], dim=-1
        )  # [B, K, 2D]
        return mlp(paired).squeeze(-1)
    raise ValueError(
        f"unknown interaction mode {mode!r}; expected one of ['dot', 'bilinear', 'mlp']."
    )


def _build_interaction(
    interaction: str, hidden_size: int | None
) -> tuple[nn.Parameter | None, nn.Module | None]:
    """Create (bilinear weight, mlp) for ``interaction``; validates the name."""
    if interaction == "dot":
        return None, None
    if hidden_size is None or hidden_size < 1:
        raise ValueError(
            f"interaction {interaction!r} requires a positive hidden_size (input dim D), "
            f"got {hidden_size!r}."
        )
    if interaction == "bilinear":
        weight = nn.Parameter(
            torch.randn(hidden_size, hidden_size) / math.sqrt(hidden_size)
        )
        return weight, None
    if interaction == "mlp":
        mlp = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        return None, mlp
    raise ValueError(
        f"unknown interaction mode {interaction!r}; expected one of "
        "['dot', 'bilinear', 'mlp']."
    )


class DecisionHead(nn.Module, ABC):
    """Abstract base for all JevX decision heads.

    Subclasses implement :meth:`compute_logits` (and optionally override
    :meth:`build_output` for decision-type-specific quantities such as the
    Score expectation); :meth:`forward` wires validation, masking and
    output construction.

    Args:
        hidden_size: Input dim ``D`` of ``fused`` (and of each option
            embedding). Also used to size interaction parameters.
    """

    head_name: ClassVar[str] = ""

    def __init__(
        self,
        hidden_size: int,
        *,
        state_pathway: str = "linear",
        state_hidden_size: int | None = None,
        state_activation: str = "gelu",
        conditioning_scale_init: float = 1.0,
        conditioning_enabled: bool = True,
        has_interaction: bool = True,
    ) -> None:
        """Wire the scoring contract ``z_k = state_k + lambda * r_k``.

        Subclasses implement :meth:``compute_logits`` (and optionally override
        :meth:``build_output`` for decision-type-specific quantities such as the
        Score expectation); :meth:``forward`` wires validation, masking and
        output construction.

        Args:
            hidden_size: Input dim ``D`` of ``fused`` (and of each option
                embedding). Also used to size interaction parameters.
            state_pathway: Kind of the PRIMARY state score, one of
                :data:``STATE_PATHWAYS`` (plus the head-level-only ``"none"``
                ablation arm, i.e. interaction only):

                * ``"linear"`` — ``state_k = s^T W[k, :] + b[k]`` (shipped probe)
                * ``"mlp"`` — ``state_k = h_s^T W2[k, :] + b[k]`` with
                  ``h_s = act(W1 s + b1)`` (:class:``MLPStatePathway````)
            state_hidden_size: Hidden width ``H`` of the ``"mlp"`` state
                pathway; ``None`` reuses ``hidden_size``. Ignored by the linear
                pathway, which has no hidden layer.
            state_activation: Elementwise activation of the ``"mlp"`` pathway,
                one of :data:``STATE_ACTIVATIONS`.
            conditioning_scale_init: Initial ``lambda`` of the option-
                interaction residual. The DEFAULT is 1.0 so a head built with
                default arguments is bit-identical to the pre-flag head
                (``z_k = state_k + r_k``); a curriculum that wants to RAMP the
                interaction starts it at 0.0 explicitly.
            conditioning_enabled: When False the residual is EXACTLY zero on
                every call (not merely small) and no gradient can reach the
                option rows.
            has_interaction: Internal wiring flag — whether this head has an
                option-interaction residual to scale at all. Heads whose
                scorer IS the state pathway (noul / multi_label /
                regression, and the fixed-K score schema) pass False: lambda
                is inert for them, so the buffer is registered NON-persistent
                and stays out of the head state_dict — a classifier transplant
                into such a head (convert()) then sees exactly the tensors
                that define its scoring function.

        Raises:
            ValueError: ``hidden_size``/``state_hidden_size`` is not a positive
                int, or ``state_pathway``/``state_activation`` is not a shipped
                enum member (the message lists the accepted values).
            TypeError: ``conditioning_enabled`` is not a bool.
        """
        super().__init__()
        if not isinstance(hidden_size, int) or hidden_size < 1:
            raise ValueError(f"hidden_size must be a positive int, got {hidden_size!r}.")
        pathways = (*STATE_PATHWAYS, "none")
        if not isinstance(state_pathway, str) or state_pathway not in pathways:
            raise ValueError(
                f"state_pathway: {state_pathway!r} is not a known state pathway; expected one of "
                f"{list(pathways)} ('none' is the head-level interaction-only ablation arm)."
            )
        if not isinstance(state_activation, str) or state_activation not in STATE_ACTIVATIONS:
            raise ValueError(
                f"state_activation: {state_activation!r} is not a known activation; expected one "
                f"of {list(STATE_ACTIVATIONS)}."
            )
        if state_hidden_size is None:
            state_hidden_size = hidden_size
        if not isinstance(state_hidden_size, int) or state_hidden_size < 1:
            raise ValueError(
                f"state_hidden_size: expected a positive int, got {state_hidden_size!r}."
            )
        if not isinstance(conditioning_enabled, bool):
            raise TypeError(
                "conditioning_enabled: expected a bool (turn the option-interaction residual "
                f"on/off), got {type(conditioning_enabled).__name__}."
            )
        scale = float(conditioning_scale_init)
        if not math.isfinite(scale):
            raise ValueError(
                f"conditioning_scale_init: expected a finite float, got {conditioning_scale_init!r}; "
                "an inf/NaN lambda would poison every logit."
            )
        self.hidden_size = hidden_size
        self.state_hidden_size = int(state_hidden_size)
        self.state_activation = state_activation
        self.conditioning_enabled = conditioning_enabled
        # `state_pathway` is the SELECTOR string. Heads that own a pathway
        # module shadow this attribute with that module (ChoiceHead has carried
        # a module named `state_pathway` since EXP-H — `state_pathway.weight` is
        # a shipped state_dict key), so the selector stays readable on every
        # head as `state_pathway_name`.
        self.state_pathway = state_pathway
        self.state_pathway_name = state_pathway
        # Buffer, not Parameter: lambda is a curriculum control, ramped through
        # set_interaction_scale, not a weight the optimizer may move. On a head
        # that HAS an interaction it rides the state_dict, so a checkpoint
        # restores the lambda it was run with and an older checkpoint simply
        # lacks the key (strict=False) and keeps the constructor value.
        self.register_buffer(
            "interaction_scale",
            torch.tensor(scale, dtype=torch.float32),
            persistent=has_interaction,
        )

    def _make_state_pathway(self, kind: str, num_options: int | None) -> nn.Module | None:
        """Build the pathway module for the resolved ``kind`` selector.

        Returns ``None`` for the interaction-only ablation arm. The caller
        assigns the result to ``self.state_pathway`, which is also
        the shipped state_dict prefix of the parameters.
        """
        if kind == "none":
            return None
        if kind == "linear":
            return LinearProbePathway(self.hidden_size, num_options)
        if kind == "mlp":
            return MLPStatePathway(
                self.hidden_size,
                num_options,
                self.state_hidden_size,
                self.state_activation,
            )
        raise ValueError(  # pragma: no cover - __init__ validates the selector
            f"state_pathway: {kind!r} is not a known state pathway; expected one of "
            f"{list(STATE_PATHWAYS)}."
        )

    # -- conditioning scale -------------------------------------------------
    @torch.no_grad()
    def set_interaction_scale(self, value: float) -> None:
        """Set ``lambda``, the scale of the option-interaction residual.

        This is the hook the lambda ramp drives: ``z_k = state_k + lambda * r_k``.
        A value of exactly 0.0 makes the residual exactly zero on the next
        forward (it is not computed at all).

        Raises:
            ValueError: ``value`` is not a finite float.
        """
        scale = float(value)
        if not math.isfinite(scale):
            raise ValueError(
                f"set_interaction_scale: expected a finite float, got {value!r}; an inf/NaN "
                "lambda would poison every logit."
            )
        self.interaction_scale.fill_(scale)

    @property
    def get_interaction_scale(self) -> float:
        """The current ``lambda`` of the option-interaction residual."""
        return float(self.interaction_scale)

    # -- state pathway scoring ---------------------------------------------

    def _state_module(self) -> nn.Module | None:
        """The module implementing the configured state pathway (``None`` = none).

        The default reads ``self.state_pathway``, which is the ChoiceHead's
        shipped module attribute; a head whose scorer IS its state pathway
        (``noul``/``multi_label``/``regression``, whose ``self.proj`` is the
        scorer) overrides this to return it.
        """
        module = getattr(self, "state_pathway", None)
        return module if isinstance(module, nn.Module) else None

    def _state_scores(
        self, fused: torch.Tensor, width: int
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """Evaluate the state pathway for ``width`` options.

        Returns:
            ``(state_logits, s_or_h_s)`` — the PRIMARY score and the vector the
            residual is scored against (the fused state ``s`` for the linear
            pathway, the hidden state ``h_s`` for the mlp one). ``state_logits``
            is ``None`` when the head has no state pathway.
        """
        module = self._state_module()
        if module is None:
            return None, fused
        if isinstance(module, MLPStatePathway):
            hidden = module.hidden(fused)
            # The residual is scored against h_s, but only when h_s lives in
            # the same space as the option rows (D); an H != D hidden state has
            # no bilinear/dot pairing with e_k, so the residual falls back to s.
            vector = hidden if module.mlp_hidden_size == self.hidden_size else fused
            return module.project(hidden, width), vector
        return module(fused, width), fused

    def _residual_is_active(self, conditioning_enabled: bool | None) -> bool:
        """Whether the option interaction contributes to THIS call.

        ``conditioning_enabled=None`` defers to the head's own flag; the scale
        is read from the buffer so a ramp takes effect immediately.
        """
        enabled = (
            self.conditioning_enabled
            if conditioning_enabled is None
            else bool(conditioning_enabled)
        )
        return enabled and self.get_interaction_scale != 0.0

    def _combine(
        self,
        state_logits: torch.Tensor | None,
        residual: torch.Tensor | None,
        conditioning_enabled: bool | None = None,
    ) -> torch.Tensor:
        """The scoring identity ``z_k = state_k + lambda * r_k``.

        When conditioning is off for this call (explicit flag or the head's
        own) or ``lambda`` is exactly 0, the residual is EXACTLY zero: it is not
        computed, so no gradient reaches the option rows and a non-finite
        option feature cannot become ``0 * inf = NaN``. Otherwise the residual
        is scaled by ``lambda`` and added to the state score (both may be
        ``None`` for a head that has neither).
        """
        if not self._residual_is_active(conditioning_enabled):
            if state_logits is not None:
                return state_logits
            return torch.zeros_like(residual)
        assert residual is not None  # noqa: S101 - internal contract, not input
        scaled = residual * self.interaction_scale
        return scaled if state_logits is None else state_logits + scaled

    # -- schema hooks -------------------------------------------------------

    def _expected_width(self, option_features: torch.Tensor | None) -> int | None:
        """Number of logit columns this head will emit (``None`` = unknown yet)."""
        if option_features is not None:
            return option_features.shape[1]
        return None

    def validate_schema(
        self,
        fused: torch.Tensor,
        option_features: torch.Tensor | None = None,
        option_mask: torch.Tensor | None = None,
        num_options: torch.Tensor | int | None = None,
    ) -> None:
        """Validate shapes, batch alignment and finiteness of one forward batch.

        Subclasses add their own constraints (e.g. ChoiceHead requires
        ``option_features``) and then call ``super().validate_schema(...)``.

        Raises:
            TypeError / ValueError: with a message naming the offending input
                (batch mismatch, wrong rank, non-finite values, impossible
                option counts, all-masked rows, ...).
        """
        if not isinstance(fused, torch.Tensor):
            raise TypeError(f"fused: expected a torch.Tensor, got {type(fused).__name__}.")
        assert_shape(fused, (None, None), "fused")
        check_finite(fused, "fused")

        if option_features is not None:
            assert_shape(option_features, (None, None, None), "option_features")
            check_finite(option_features, "option_features")
            if option_features.shape[-1] != fused.shape[-1]:
                raise ValueError(
                    f"option_features: hidden dim {option_features.shape[-1]} must match fused "
                    f"hidden dim {fused.shape[-1]}; scoring modes operate in one shared space."
                )

        width = self._expected_width(option_features)

        if option_mask is not None:
            if not isinstance(option_mask, torch.Tensor):
                raise TypeError(
                    f"option_mask: expected a torch.Tensor, got {type(option_mask).__name__}."
                )
            if option_mask.dtype != torch.bool:
                raise TypeError(
                    f"option_mask: expected a torch.bool tensor (True = valid option), got "
                    f"{option_mask.dtype}; convert it with `.bool()`."
                )
            if width is None:
                raise ValueError(
                    f"{type(self).__name__}: cannot validate option_mask without a known option "
                    "width — pass option_features, or construct the head with a fixed "
                    "output size (num_levels / num_labels)."
                )
            assert_shape(option_mask, (fused.shape[0], width), "option_mask")
            if not bool(option_mask.any(dim=-1).all()):
                bad = (~option_mask.any(dim=-1)).nonzero(as_tuple=False)[0]
                raise ValueError(
                    f"option_mask: row {int(bad)} has no valid option; every question must keep "
                    "at least one option in its schema slice."
                )

        if num_options is not None:
            if isinstance(num_options, int):
                counts = torch.full((fused.shape[0],), num_options, dtype=torch.long)
            elif isinstance(num_options, torch.Tensor):
                assert_shape(num_options, (fused.shape[0],), "num_options")
                counts = num_options.long()
            else:
                raise TypeError(
                    f"num_options: expected an int or a [B] tensor, got "
                    f"{type(num_options).__name__}."
                )
            if width is None:
                raise ValueError(
                    f"{type(self).__name__}: cannot validate num_options without a known option "
                    "width — pass option_features or a fixed output size."
                )
            if counts.numel() > 0:
                lo, hi = int(counts.min()), int(counts.max())
                if lo < 1:
                    raise ValueError(
                        f"num_options: every question needs >= 1 valid option, got min={lo}."
                    )
                if hi > width:
                    raise ValueError(
                        f"num_options: max count {hi} exceeds the available options {width} "
                        f"(Kmax={width})."
                    )
            if option_mask is not None:
                mask_counts = option_mask.sum(dim=-1).long()
                if not torch.equal(mask_counts, counts):
                    raise ValueError(
                        f"num_options: disagrees with option_mask — counts {counts.tolist()} vs "
                        f"mask sums {mask_counts.tolist()}."
                    )

        assert_batch_alignment(
            fused=fused,
            option_features=option_features,
            option_mask=option_mask,
            num_options=(
                num_options if isinstance(num_options, torch.Tensor) else None
            ),
        )

    def _resolve_mask(
        self,
        fused: torch.Tensor,
        option_features: torch.Tensor | None,
        option_mask: torch.Tensor | None,
        num_options: torch.Tensor | int | None,
    ) -> torch.Tensor | None:
        """Turn whichever count spelling was given into a bool mask (or ``None``)."""
        if option_mask is not None:
            return option_mask
        if num_options is None:
            return None
        width = self._expected_width(option_features)
        if width is None:  # pragma: no cover - validate_schema rejects this first
            raise ValueError(f"{type(self).__name__}: unknown option width for num_options.")
        return mask_from_num_options(num_options, width, batch_size=fused.shape[0]).to(
            fused.device
        )

    # -- core API -----------------------------------------------------------

    @abstractmethod
    def compute_logits(
        self,
        fused: torch.Tensor,
        option_features: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Map the batch to raw scores ``logits [B, K]`` (masking happens later).

        ``option_mask`` (the RESOLVED mask) is forwarded so a head can act on
        it; other unknown ``**kwargs`` are accepted and ignored (reserved for
        head-specific options); validation lives in :meth:`validate_schema`.
        """

    def _probs_and_index(
        self, logits: torch.Tensor, mask: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Mask-aware probabilities + argmax over valid options only."""
        probs = masked_softmax(logits, mask)
        if mask is None:
            idx_logits = logits
        else:
            idx_logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        return probs, idx_logits.argmax(dim=-1)

    def build_output(
        self, logits: torch.Tensor, mask: torch.Tensor | None = None
    ) -> DecisionOutput:
        """Assemble the :class:`DecisionOutput` core from raw logits."""
        probs, predicted_index = self._probs_and_index(logits, mask)
        return DecisionOutput(
            logits=logits, probabilities=probs, predicted_index=predicted_index
        )

    def forward(
        self,
        fused: torch.Tensor,
        option_features: torch.Tensor | None = None,
        *,
        option_mask: torch.Tensor | None = None,
        num_options: torch.Tensor | int | None = None,
        **kwargs: Any,
    ) -> DecisionOutput:
        """Run the head on one batch.

        Args:
            fused: ``[B, D]`` fused state ``H``.
            option_features: ``[B, K, D]`` option embeddings (required by
                ChoiceHead, optional for ScoreHead, forbidden for
                Noul/MultiLabel/Regression).
            option_mask: ``[B, K]`` bool mask, ``True`` = valid option.
            num_options: ``[B]`` counts (or int) equivalent to ``option_mask``.

        Any other keyword argument is forwarded to :meth:`compute_logits` and
        is ignored by a head that does not use it, so head-specific per-call
        switches ride this path (e.g. ChoiceHead's ``duplicate_tie``).

        Returns:
            :class:`DecisionOutput` with ``logits`` (raw, unmasked),
            ``probabilities`` (mask-aware) and ``predicted_index`` (in the
            valid set) populated.
        """
        self.validate_schema(fused, option_features, option_mask, num_options)
        mask = self._resolve_mask(fused, option_features, option_mask, num_options)
        logits = self.compute_logits(
            fused, option_features, option_mask=mask, **kwargs
        )
        assert_shape(logits, (fused.shape[0], None), f"{type(self).__name__}.logits")
        return self.build_output(logits, mask)

    def project(
        self,
        probabilities: torch.Tensor,
        option_keys: Sequence[Any] | None = None,
    ) -> torch.Tensor | list[Any]:
        """Select the decision from probabilities: ``D = Project(P', C)``.

        Args:
            probabilities: ``[B, K]`` calibrated probabilities.
            option_keys: Optional schema ``C``; when given, indices are
                validated against it and mapped to its values.

        Returns:
            ``[B]`` index tensor, or (if ``option_keys`` is given) the list
            of schema values — every one guaranteed to be in ``C``.
        """
        if probabilities is None:
            raise ValueError(
                f"{type(self).__name__}.project: probabilities is None; a decision needs a "
                "distribution to project."
            )
        assert_shape(probabilities, (None, None), "probabilities")
        indices = probabilities.argmax(dim=-1)
        if option_keys is None:
            return indices
        return project_to_schema(indices, option_keys)


# ---------------------------------------------------------------------------
# Experimental generic heads
# ---------------------------------------------------------------------------


@register_decision_head("multi_label")
class MultiLabelHead(DecisionHead):
    """EXPERIMENTAL: independent sigmoid per label (multi-label decisions).

    Contract: ``logits [B, L]`` from a linear map of ``fused``; probabilities
    are ``sigmoid(logits)`` per label (rows do NOT sum to 1 — labels are not
    mutually exclusive); ``predicted_value`` is the multi-hot decision
    ``[B, L]`` bool at threshold 0.5; ``predicted_index`` stays ``None``
    because there is no single predicted index.

    Args:
        hidden_size: Input dim ``D``.
        num_labels: Number of labels ``L`` (fixed at construction).
        state_pathway: ``"linear"`` (default) keeps ``proj`` an
            ``nn.Linear(D, L)``; ``"mlp"`` inserts the nonlinear state pathway of
            width ``state_hidden_size`` in front of the same ``L`` outputs.
        state_hidden_size: Hidden width ``H`` of the ``"mlp"`` pathway.
        state_activation: Elementwise activation of the ``"mlp"`` pathway.
        conditioning_scale_init: Accepted for a uniform recipe; inert (labels
            carry no option rows, so no residual exists).
        conditioning_enabled: Accepted for a uniform recipe; inert.
    """

    def __init__(
        self,
        hidden_size: int,
        num_labels: int,
        *,
        state_pathway: str = "linear",
        state_hidden_size: int | None = None,
        state_activation: str = "gelu",
        conditioning_scale_init: float = 1.0,
        conditioning_enabled: bool = True,
    ) -> None:
        super().__init__(
            hidden_size,
            state_pathway=state_pathway,
            state_hidden_size=state_hidden_size,
            state_activation=state_activation,
            conditioning_scale_init=conditioning_scale_init,
            conditioning_enabled=conditioning_enabled,
            has_interaction=False,  # labels carry no option rows
        )
        if not isinstance(num_labels, int) or num_labels < 1:
            raise ValueError(f"num_labels must be a positive int, got {num_labels!r}.")
        self.num_labels = num_labels
        # The state pathway IS the scorer: labels have no option rows, so there
        # is no interaction residual to scale.
        self.proj: nn.Module = (
            MLPStatePathway(hidden_size, num_labels, self.state_hidden_size, state_activation)
            if state_pathway == "mlp"
            else nn.Linear(hidden_size, num_labels)
        )

    def _state_module(self) -> nn.Module | None:
        """``proj`` is the state pathway of a label head."""
        return self.proj

    def _expected_width(self, option_features: torch.Tensor | None) -> int | None:
        return self.num_labels

    def validate_schema(
        self,
        fused: torch.Tensor,
        option_features: torch.Tensor | None = None,
        option_mask: torch.Tensor | None = None,
        num_options: torch.Tensor | int | None = None,
    ) -> None:
        if option_features is not None:
            raise ValueError(
                "MultiLabelHead: option_features is not accepted — labels come from a fixed "
                f"linear head over `fused` producing {self.num_labels} logits."
            )
        if num_options is not None:
            raise ValueError(
                "MultiLabelHead: num_options is not applicable; label counts are fixed by "
                "num_labels (option_mask IS accepted to hide labels for variable schemas)."
            )
        super().validate_schema(fused, option_features, option_mask, num_options)

    def compute_logits(
        self,
        fused: torch.Tensor,
        option_features: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        return self.proj(fused)

    def build_output(
        self, logits: torch.Tensor, mask: torch.Tensor | None = None
    ) -> DecisionOutput:
        probs = torch.sigmoid(logits)
        if mask is not None:
            probs = probs.masked_fill(~mask, 0.0)
        return DecisionOutput(
            logits=logits,
            probabilities=probs,
            predicted_value=(probs >= 0.5),
        )


@register_decision_head("regression")
class RegressionHead(DecisionHead):
    """EXPERIMENTAL: scalar regression head.

    Contract: ``logits [B, 1]`` is the raw scalar prediction; there are no
    probabilities (``None``), no index (``None``) and ``predicted_value`` is
    the de-squeezed scalar ``[B]``. Options/masks are not applicable.


    The state pathway IS the scalar scorer here, for the same reason as
    :class:``jevx.decisions.noul.NoulHead``: a single regression output has no
    option rows, so no interaction residual exists and ``lambda`` is inert.
    Args:

        hidden_size: Input dim ``D``.
        state_pathway: ``"linear"`` (default) or ``"mlp"`` — see
            :class:``jevx.decisions.noul.NoulHead``.
        state_hidden_size: Hidden width ``H`` of the ``"mlp"`` pathway.
        state_activation: Elementwise activation of the ``"mlp"`` pathway.
        conditioning_scale_init: Accepted for a uniform recipe; inert.
        conditioning_enabled: Accepted for a uniform recipe; inert.
    """

    def __init__(
        self,
        hidden_size: int,
        *,
        state_pathway: str = "linear",
        state_hidden_size: int | None = None,
        state_activation: str = "gelu",
        conditioning_scale_init: float = 1.0,
        conditioning_enabled: bool = True,
    ) -> None:
        super().__init__(
            hidden_size,
            state_pathway=state_pathway,
            state_hidden_size=state_hidden_size,
            state_activation=state_activation,
            conditioning_scale_init=conditioning_scale_init,
            conditioning_enabled=conditioning_enabled,
            has_interaction=False,  # a scalar regression has no option rows
        )
        self.proj: nn.Module = (
            MLPStatePathway(hidden_size, 1, self.state_hidden_size, state_activation)
            if state_pathway == "mlp"
            else nn.Linear(hidden_size, 1)
        )

    def _state_module(self) -> nn.Module | None:
        """``proj`` is the state pathway of a regression head."""
        return self.proj

    def _expected_width(self, option_features: torch.Tensor | None) -> int | None:
        return 1

    def validate_schema(
        self,
        fused: torch.Tensor,
        option_features: torch.Tensor | None = None,
        option_mask: torch.Tensor | None = None,
        num_options: torch.Tensor | int | None = None,
    ) -> None:
        if option_features is not None or option_mask is not None or num_options is not None:
            raise ValueError(
                "RegressionHead: emits a single scalar per row — option_features / "
                "option_mask / num_options are not applicable."
            )
        super().validate_schema(fused, option_features, option_mask, num_options)

    def compute_logits(
        self,
        fused: torch.Tensor,
        option_features: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        return self.proj(fused)

    def build_output(
        self, logits: torch.Tensor, mask: torch.Tensor | None = None
    ) -> DecisionOutput:
        return DecisionOutput(
            logits=logits,
            predicted_value=logits.squeeze(-1),
        )
