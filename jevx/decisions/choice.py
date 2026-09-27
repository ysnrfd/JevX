# Copyright (c) 2026 ysnrfd
"""Choice decision head: softmax over mutually exclusive options."""

from __future__ import annotations

from typing import Any

import torch

from .generic import (
    DecisionHead,
    _build_interaction,
    _interact,
    register_decision_head,
)

__all__ = ["ChoiceHead"]


@register_decision_head("choice")
class ChoiceHead(DecisionHead):
    """Categorical choice over ``K`` mutually exclusive options.

    Mathematical contract::

        state_k = s^T W[k, :] + b[k]                     # "linear" pathway
        state_k = h_s^T W2[k, :] + b[k],                 # "mlp" pathway,
                  h_s = act(W1 s + b1)                  #   the PRIMARY score
        z_k     = state_k + lambda * r_k(h_s or s, e_k) # + the residual
        p       = softmax(z) over valid k                # rows sum to 1
        y       = argmax over valid k                    # in the valid set

    with ``lambda = interaction_scale`` a registered scalar buffer, ``state_k``
    the PRIMARY state score and ``r_k`` the option-interaction RESIDUAL:

    * ``"dot"``      — ``r_k = <s, e_k>`` (no residual parameters)
    * ``"bilinear"`` — ``r_k = s^T W_b e_k`` with ``W_b in R^{D x D}``
    * ``"mlp"``      — ``r_k = MLP([s ; e_k])``

    where the state vector the residual is scored against is the fused state
    ``s`` under the linear pathway and the hidden state ``h_s`` under the mlp
    one.

    Why the state score is the primary term (the math review that motivated
    the flag): with one fixed question and one fixed schema every option row
    ``e_k`` is identical on every batch row, so a BILINEAR residual collapses to
    ``z_k = s^T (W[k, :] + A e_k) + b[k]`` — still LINEAR in ``s``. A linear
    state path therefore supplies no nonlinearity at all and the head reduces to
    a linear classifier. ``state_pathway="mlp"`` makes the state score a genuine
    function of the state, and the option interaction is the additive residual
    it is scaled by.

    Why the additive term exists (EXP-H, benchmarks/run_head_quality.py): a
    softmax-over-options head that scores options ONLY through an interaction
    has no state-only path. For one fixed question and one fixed schema every
    option row ``e_k`` is identical on every batch row, so the head is
    a purely multiplicative reparameterization of a linear map whose per-class
    weights are reached only indirectly, through the schema encoder and the
    fusion option gate. On frozen features that indirection loses to a plain
    linear probe. With the additive term the head is a strict SUPERSET of a
    linear probe: zeroing the interaction residual leaves EXACTLY
    ``nn.Linear(D, K)`` applied to the fused state.

    Numerics: probabilities are computed with :func:`masked_softmax`, which
    upcasts fp16/bf16 to fp32, fills invalid positions with
    ``torch.finfo(dtype).min`` (never ``-inf`` — NaN gradients) and forces
    their probability to exactly 0.

    Args:
        hidden_size: Input dim ``D`` shared by ``fused`` and option
            embeddings (required for ``bilinear``/``mlp``).
        interaction: Option-interaction RESIDUAL mode, one of ``"dot"``,
            ``"bilinear"``, ``"mlp"`` (default
            ``"bilinear"``).
        num_options: Initial capacity of the state pathway.
            ``None`` uses ``DEFAULT_STATE_PATHWAY_OPTIONS``;
            the parameter is sliced to the observed ``K`` on every forward,
            so variable-``K`` batches never reallocate.
        state_pathway: Either the shipped legacy bool (``True`` = the linear
            probe, ``False`` = the pre-EXP-H interaction-only head, so the
            v1-vs-v2 ablation stays reproducible from the shipped class) or the
            pathway kind ``"linear"`` / ``"mlp"`` / ``"none"``. The selector
            is always readable as ``head.state_pathway_name``; the attribute
            ``head.state_pathway`` is the pathway MODULE (or ``None``), so the
            shipped ``state_pathway.weight`` state_dict keys keep working.
        state_hidden_size: Hidden width ``H`` of the ``"mlp"`` pathway
            (``None`` = ``hidden_size``).
        state_activation: Elementwise activation of the ``"mlp"`` pathway.
        conditioning_scale_init: Initial ``lambda``; the default 1.0 keeps the
            shipped ``z_k = state_k + r_k`` bit-for-bit.
        conditioning_enabled: When False the residual is exactly zero.

    Duplicate-option tie: two schema options whose rows are bit-identical
    receive the same score (see the block comment above
    :meth:`_tie_duplicate_options`). The tie is a property of the SCHEMA'S
    option rows, so a caller that has no option rows to report - the
    state-only path, where the ``[B, K, D]`` block is a shape placeholder and
    not a schema - passes ``duplicate_tie=False`` and the head scores the K
    classes from the state alone. That is the only sanctioned way to switch
    the tie off; every other call keeps the documented behaviour.
    """

    def __init__(
        self,
        hidden_size: int,
        interaction: str = "bilinear",
        *,
        num_options: int | None = None,
        state_pathway: bool | str = True,
        state_hidden_size: int | None = None,
        state_activation: str = "gelu",
        conditioning_scale_init: float = 1.0,
        conditioning_enabled: bool = True,
    ) -> None:
        # The shipped bool spelling (state_pathway=False) is the pre-EXP-H v1
        # head and stays valid; a string selects the pathway kind.
        kind = (
            ("linear" if state_pathway else "none")
            if isinstance(state_pathway, bool)
            else state_pathway
        )
        super().__init__(
            hidden_size,
            state_pathway=kind,
            state_hidden_size=state_hidden_size,
            state_activation=state_activation,
            conditioning_scale_init=conditioning_scale_init,
            conditioning_enabled=conditioning_enabled,
        )
        self.interaction = interaction
        self.bilinear_weight, self.mlp = _build_interaction(interaction, hidden_size)
        self.state_pathway = self._make_state_pathway(kind, num_options)

    def interaction_logits(
        self, fused: torch.Tensor, option_features: torch.Tensor
    ) -> torch.Tensor:
        """The option-interaction RESIDUAL alone: ``r_k = f(s, e_k)``, [B, K]."""
        return _interact(
            fused,
            option_features,
            self.interaction,
            bilinear_weight=self.bilinear_weight,
            mlp=self.mlp,
        )

    def validate_schema(
        self,
        fused: torch.Tensor,
        option_features: torch.Tensor | None = None,
        option_mask: torch.Tensor | None = None,
        num_options: torch.Tensor | int | None = None,
    ) -> None:
        if option_features is None:
            raise ValueError(
                "ChoiceHead: option_features [B, K, D] is required — a choice scores every "
                "option against the fused state; without options there is no K."
            )
        super().validate_schema(fused, option_features, option_mask, num_options)

    # -- duplicate-option tie ----------------------------------------------
    #
    # A schema may legitimately list the same option twice (the schema encoder
    # documents duplicates as accepted; D3.2 belongs to upstream validation).
    # Two identical options are the SAME decision, so they must receive the
    # SAME score: with a free positional state pathway they would otherwise get
    # different per-column weights and drift apart. Averaging the logits within
    # each group of identical option rows restores that invariant exactly.
    #
    # A group is applied to a row only when EVERY member is (a) bit-identical
    # to the others in that row and (b) a VALID option in that row, so padded /
    # masked columns - which may hold arbitrary values - can never be averaged
    # together and can never contaminate a valid column.
    #
    # SCOPE OF THE TIE (why it must not fire on the state-only path).
    # The tie is a statement about the SCHEMA'S OPTION ROWS, so it may only
    # fire when the rows handed to the head really are schema rows. The
    # state-only path carries NO option rows: JevXForDecision.forward
    # substitutes EXACT ZEROS for them purely to keep the [B, K, D] / mask /
    # schema-metadata shape contract, and zeros are 'identical' for EVERY
    # schema, so every K column would look duplicated. Tying there would
    # average the K columns of the pure state term state_k = s^T W[k, :] +
    # b[k] - a deliberately free PER-CLASS quantity - into one value and
    # collapse the K state-only logits to a single number. That degeneracy
    # cannot be trained out of. The tie makes every logit in the group the
    # SAME value m, so dL/dz_k = dL/dm / K is identical for all k and their
    # sum - the only quantity the per-class state pathway can respond to -
    # is zero (dL/dm = sum_k (p_k - y_k) = 1 - 1 = 0 on a uniform softmax).
    # The analytic gradient into EVERY state-pathway parameter is therefore
    # ZERO; only float32 round-off survives (measured 2.8e-07 for the weight
    # gradient on a [6, 8] fused batch, against 2.7e+00 for the same batch
    # with the tie off), far too small to move anything, and every row keeps
    # receiving the same round-off-sized update, so the K logits stay tied for
    # good. Suppressing the tie there - and ONLY there - keeps the documented
    # per-schema duplicate tie bit-for-bit everywhere else.

    def _tie_duplicate_options(
        self,
        logits: torch.Tensor,
        option_features: torch.Tensor,
        option_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Average logits within each group of identical VALID option rows.

        A group is applied to a row only when every member is (a) bit-identical
        to the others in that row and (b) a VALID option in that row, so padded /
        masked columns - which may hold arbitrary values - can never be averaged
        together nor contaminate a valid column. Groups are detected PER ROW (a
        variable-K batch mixes schemas, so row 0 is not representative) and each
        row keeps its own independent decision. When no row has duplicates - the
        common case - every cell of the result is the corresponding input cell,
        so the logits come out bit-for-bit unchanged.

        The grouping is computed ON DEVICE (bit-cast row signatures, one exact
        ``unique`` and per-row index algebra) and applied with a
        data-independent ``where``, so the common case costs no host transfer
        and no device synchronisation: only the SHAPE of the batch ever reaches
        the host. `tests/unit/test_tie_performance.py` pins that, and pins the
        logits bit-for-bit against the host-transfer reference it replaced.
        """
        rows = int(option_features.shape[0])
        k = int(option_features.shape[1])
        if k < 2:
            return logits
        device = logits.device
        # The reference keyed each option row on the raw bytes of its float32
        # view, so two rows are duplicates exactly when their BIT patterns agree
        # (-0.0 stays distinct from 0.0, as do distinct NaN payloads). The int32
        # view of the float32 upcast is that same bit pattern, and the fp16/bf16
        # upcast to float32 is lossless, so the partition below is the one the
        # byte comparison produced.
        keys = option_features.detach().to(dtype=torch.float32).contiguous()
        keys = keys.view(torch.int32)
        if option_mask is not None:
            if not isinstance(option_mask, torch.Tensor):
                option_mask = torch.as_tensor(option_mask)
            valid = option_mask.detach().to(device=device, dtype=torch.bool)
            valid = valid[:rows, :k].unsqueeze(-1)
            # A masked-out column must join no group. Dropping it instead would
            # make the group layout a host-side question, so it is given a tag
            # column no other column can hold (0 for every valid column, a
            # per-cell id >= 1 for every masked one) and so can never merge.
            tag = torch.arange(rows, device=device).unsqueeze(1) * k + torch.arange(
                k, device=device
            )
            tag = torch.where(
                valid,
                torch.zeros((), dtype=torch.int32, device=device),
                (tag + 1).to(torch.int32).unsqueeze(-1),
            )
            keys = torch.cat([keys, tag], dim=-1)
        # Exact partition of the rows: unique() compares the key columns, so only
        # bit-identical rows share a class id (no hashing, no collision). Ids are
        # batch-global, which is harmless - every use below is a comparison or a
        # sort WITHIN one row.
        _, inverse = torch.unique(
            keys.reshape(rows * k, -1), dim=0, return_inverse=True
        )
        group = inverse.view(rows, k)
        # size[r, j] = the number of columns of row r in j's class; start[r, j] =
        # how many columns sort before the class, i.e. where the class begins in
        # `order`. A K x K comparison is exact because K is the option count, not
        # the feature width.
        equal = group.unsqueeze(1) == group.unsqueeze(2)
        size = equal.sum(-1)
        start = (group.unsqueeze(2) > group.unsqueeze(1)).sum(-1)
        order = torch.argsort(group, dim=1, stable=True)
        # values[r, j, t] = the t-th (by column) member of j's class in row r. t
        # is clamped at K-1, which only ever lands outside a class shorter than K
        # and is therefore discarded by the `where` below, so one gather serves
        # every group size.
        positions = (start.unsqueeze(-1) + torch.arange(k, device=device)).clamp_(
            max=k - 1
        )
        columns = order.gather(1, positions.reshape(rows, k * k))
        values = logits.gather(1, columns).view(rows, k, k)
        # One selector per group SIZE, built in a single comparison.
        selectors = (torch.arange(2, k + 1, device=device).view(-1, 1, 1) == size).unbind(
            0
        )
        out = logits
        for n, selector in enumerate(selectors, start=2):
            # One [B, K, n] mean PER GROUP SIZE reproduces `logits[r, cols]
            # .mean()` bit-for-bit; zero-padding every class to one fixed width
            # would not (the padded lanes change the summation order), which is
            # why the sizes are iterated instead of padded.
            out = torch.where(selector, values[:, :, :n].mean(-1), out)
        return out

    def compute_logits(
        self,
        fused: torch.Tensor,
        option_features: torch.Tensor | None = None,
        *,
        option_mask: torch.Tensor | None = None,
        conditioning_enabled: bool | None = None,
        duplicate_tie: bool | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Raw scores ``z_k = state_k + lambda * r_k(vector, e_k)``, ``[B, K]``.

        Args:
            fused: ``[B, D]`` fused state.
            option_features: ``[B, K, D]`` option rows (required; the
                ``DecisionHead`` shape contract is validated in
                :meth:`validate_schema`).
            option_mask: ``[B, K]`` resolved mask, ``True`` = valid option.
            conditioning_enabled: Per-call override of the head's own flag;
                ``False`` (or ``lambda == 0``) makes the residual exactly zero.
            duplicate_tie: ``False`` skips the duplicate-option tie for this
                call, ``True`` forces it, ``None`` (the default) applies the
                documented rule. It arrives through
                ``DecisionHead.forward``'s ``**kwargs`` and is how the
                state-only path declares that its ``option_features`` are a
                shape placeholder rather than a schema: averaging the state
                term over "duplicated" zeros would collapse the K
                state-only logits to one value and zero the gradient into the
                state pathway.

        Returns:
            ``[B, K]`` raw logits; masking happens in ``build_output``.
        """
        if option_features is None:  # pragma: no cover - validate_schema rejects first
            raise ValueError("ChoiceHead: option_features is required.")
        width = int(option_features.shape[1])
        # state_k, sliced to this batch K so variable-K stays exact, and the
        # vector the residual is scored against: s under the linear pathway,
        # h_s under the mlp one.
        state_logits, vector = self._state_scores(fused, width)
        # z_k = state_k + lambda * r_k(vector, e_k)
        logits = self._combine(
            state_logits,
            self.interaction_logits(vector, option_features),
            conditioning_enabled,
        )
        if duplicate_tie is False:
            return logits
        return self._tie_duplicate_options(logits, option_features, option_mask)
