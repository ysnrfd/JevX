# Copyright (c) 2026 ysnrfd
"""Q-012 calibration masking regression tests.

Calibration used to run on the RAW, unmasked ``[g, K_max]`` logits, so
isotonic's top-label protocol (``probs.max`` over the LAST axis) could read
padded columns in a variable-K batch and silently no-op or misplace the
correction for that row. The fix calibrates per-question over VALID columns
only (``CalibrationEngine.apply_valid_columns``), re-padding afterwards.

Contract pinned here (variable-K {2, 5, 12}, isotonic AND temperature
fitted):

1. masked columns stuffed with ``±1e6`` → valid-column calibrated outputs
   are IDENTICAL (atol 1e-6) to the same call with clean masked columns;
2. the contrast: the unmasked path DOES change under stuffing (isotonic) —
   i.e. masking is what makes the identity hold;
3. calibrated probabilities still form a simplex;
4. noul (K = 1, no option mask) is unaffected by the masking path.
"""

from __future__ import annotations

import copy

import pytest
import torch

from jevx import JevXConfig, JevXForDecision
from jevx.calibration import CalibrationEngine

HIDDEN = 32
STATE = "hello world"
QUESTIONS = ["first question?", "second question?", "third question?"]
#: Per-question option counts: variable-K {2, 5, 12} in one call.
COUNTS = [2, 5, 12]
K_MAX = max(COUNTS)
OPTIONS = [[f"o{i}" for i in range(k)] for k in COUNTS]
METHODS = ["isotonic", "temperature"]


def _variable_k_mask(counts, k_max=None) -> torch.Tensor:
    """Bool ``[n, k_max]`` with ``counts[i]`` leading valid columns."""
    k_max = k_max or max(counts)
    return torch.arange(k_max).unsqueeze(0) < torch.tensor(counts).unsqueeze(1)


def _stuff_masked(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Replace masked columns with alternating ±1e6 (deterministic)."""
    k = x.shape[1]
    signs = torch.where(torch.arange(k) % 2 == 0, 1e6, -1e6).to(x.dtype)
    return torch.where(mask, x, signs.unsqueeze(0).expand_as(x))


def _build_wrapper(backbone, tokenizer=None, **config_overrides) -> JevXForDecision:
    """Wrap ``backbone`` (deep-copied) with a tiny decision config, eval mode."""
    overrides = dict(
        decision_hidden_size=HIDDEN,
        question_hidden_size=HIDDEN,
        schema_hidden_size=HIDDEN,
    )
    overrides.update(config_overrides)
    wrapper = JevXForDecision(
        JevXConfig(**overrides),
        backbone=copy.deepcopy(backbone),
        tokenizer=tokenizer,
    )
    wrapper.eval()
    return wrapper


def _fit(engine: CalibrationEngine, k_max: int = K_MAX) -> CalibrationEngine:
    """Fit on a synthetic calibration split of the same width."""
    torch.manual_seed(11)
    logits = torch.randn(64, k_max)
    labels = torch.randint(0, k_max, (64,))
    return engine.fit(logits, labels)


# ---------------------------------------------------------------------------
# 1-2. Engine level: sliced blocks are immune to stuffed masked columns
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", METHODS)
def test_apply_valid_columns_ignores_stuffed_masked_columns(method: str) -> None:
    """±1e6 in masked columns cannot move the calibrated VALID columns
    (atol 1e-6); masked columns keep their raw values untouched."""
    torch.manual_seed(0)
    counts = COUNTS
    mask = _variable_k_mask(counts)
    logits = torch.randn(len(counts), K_MAX)
    engine = _fit(CalibrationEngine(method))

    clean = engine.apply_valid_columns(logits, mask, "logits")
    stuffed = _stuff_masked(logits, mask)
    dirty = engine.apply_valid_columns(stuffed, mask, "logits")

    assert torch.allclose(clean[mask], dirty[mask], atol=1e-6, rtol=0.0), (
        f"{method}: stuffed masked columns changed the calibrated valid columns"
    )
    # re-pad contract: masked columns are passed through raw
    assert torch.equal(dirty[~mask], stuffed[~mask])

    # a row with no valid column is still rejected (input contract)
    with pytest.raises(ValueError, match=r"no valid column"):
        engine.apply_valid_columns(logits, torch.zeros_like(mask), "logits")


def test_unmasked_isotonic_path_would_differ_proving_masking_matters() -> None:
    """Contrast: applying calibration to the FULL row (the pre-fix path)
    reads the stuffed columns — so the identity above is caused by the
    masking, not by the calibrator ignoring garbage by accident.

    Only isotonic is discriminating: ``temperature`` is elementwise
    (z/T never mixes columns), so its unmasked valid columns are identical
    by construction (that case is pinned by the valid-column identity
    test above).
    """
    torch.manual_seed(0)
    mask = _variable_k_mask(COUNTS)
    logits = torch.randn(len(COUNTS), K_MAX)
    engine = _fit(CalibrationEngine("isotonic"))

    stuffed = _stuff_masked(logits, mask)
    un_clean = engine.apply(logits, "logits")
    un_dirty = engine.apply(stuffed, "logits")
    assert not torch.allclose(un_clean, un_dirty, atol=1e-6), (
        "the unmasked path must be sensitive to the stuffed columns "
        "(otherwise this test could not prove masking matters)"
    )


def test_noul_single_column_path_unaffected_at_engine_level() -> None:
    """K=1 rows with no mask (or an all-valid mask) take exactly the plain
    ``apply`` path — the masking machinery cannot change them."""
    engine = _fit(CalibrationEngine("isotonic"))
    x = torch.randn(4, 1)
    assert torch.equal(
        engine.apply_valid_columns(x, None, "logits"), engine.apply(x, "logits")
    )
    ones = torch.ones(4, 1, dtype=torch.bool)
    assert torch.equal(
        engine.apply_valid_columns(x, ones, "logits"), engine.apply(x, "logits")
    )


# ---------------------------------------------------------------------------
# 3-4. Model level: forward over a variable-K batch with fitted calibration
# ---------------------------------------------------------------------------


def _run_variable_k(wrapper: JevXForDecision):
    """One forward call over the {2, 5, 12} options batch."""
    with torch.no_grad():
        return wrapper(state=STATE, questions=QUESTIONS, options=OPTIONS)


@pytest.mark.parametrize("method", METHODS)
def test_variable_k_forward_ignores_stuffed_masked_logits(
    method: str, tiny_bert, tiny_tokenizer
) -> None:
    """Head logits for masked columns stuffed with ±1e6 do not move the
    calibrated probabilities of the VALID columns (atol 1e-6), masked
    probabilities stay exactly 0, and rows stay a simplex."""
    wrapper = _build_wrapper(
        tiny_bert, tokenizer=tiny_tokenizer, calibration_method=method
    )
    _fit(wrapper.calibration_engine)

    clean = _run_variable_k(wrapper)

    # Doctored head: keep valid logits bit-identical, stuff the masked ones.
    head = wrapper.heads["choice"]
    original_forward = head.forward

    def doctored(fused, option_features=None, *, option_mask=None, **kwargs):
        out = original_forward(
            fused, option_features, option_mask=option_mask, **kwargs
        )
        if option_mask is not None:
            out.logits = _stuff_masked(out.logits, option_mask)
        return out

    head.forward = doctored
    try:
        dirty = _run_variable_k(wrapper)
    finally:
        del head.forward

    mask = _variable_k_mask(COUNTS)
    for i, k in enumerate(COUNTS):
        probs_clean = clean.decisions[i].probabilities
        probs_dirty = dirty.decisions[i].probabilities
        assert torch.allclose(
            probs_clean[:, :k], probs_dirty[:, :k], atol=1e-6, rtol=0.0
        ), (
            f"{method}: question {i} valid-column probabilities differ "
            f"(max diff {float((probs_clean[:, :k] - probs_dirty[:, :k]).abs().max())})"
        )
        # masked columns are exactly 0 in both runs (re-mask contract)
        assert bool((probs_clean[:, k:] == 0).all())
        assert bool((probs_dirty[:, k:] == 0).all())
        # calibrated probabilities still form a simplex
        assert float(probs_clean.sum()) == pytest.approx(1.0, abs=1e-5)
        assert float(probs_dirty.sum()) == pytest.approx(1.0, abs=1e-5)
        # the predicted option is unchanged
        assert torch.equal(
            clean.decisions[i].predicted_index, dirty.decisions[i].predicted_index
        )
    assert bool(mask.any())  # the batch really was variable-K

    # Isotonic contrast on the real forward output: the PRE-FIX unmasked
    # path, replayed on the stuffed raw logits, lands elsewhere — masking is
    # what preserves the valid columns.
    if method == "isotonic":
        i, k = 0, COUNTS[0]
        raw = dirty.decisions[i].logits  # raw, masked columns stuffed
        row_mask = mask[i : i + 1]
        unmasked_cal = wrapper.calibration_engine.apply(raw, "logits")
        unmasked_probs = head.build_output(unmasked_cal, row_mask).probabilities
        assert not torch.allclose(
            unmasked_probs[:, :k],
            dirty.decisions[i].probabilities[:, :k],
            atol=1e-6,
        ), "replaying the unmasked path must differ — otherwise masking is moot"


@pytest.mark.parametrize("method", METHODS)
def test_noul_k1_probabilities_unaffected(
    method: str, tiny_bert, tiny_tokenizer
) -> None:
    """Noul decisions (K=1, option_mask=None) are calibrated exactly as
    before the masking fix: probabilities equal the plain-apply reference
    and stay strictly inside (0, 1)."""
    wrapper = _build_wrapper(
        tiny_bert, tokenizer=tiny_tokenizer, calibration_method=method
    )
    _fit(wrapper.calibration_engine)
    with torch.no_grad():
        out = wrapper.noul(STATE, ["is it so?"])
    decision = out.decisions[0]
    assert decision.probabilities.shape == (1, 1)

    reference = wrapper.heads["noul"].build_output(
        wrapper.calibration_engine.apply(decision.logits, "logits"), None
    ).probabilities
    assert torch.allclose(
        decision.probabilities, reference, atol=1e-7, rtol=0.0
    ), "K=1 calibration must be identical to the plain apply path"
    p = decision.probabilities
    assert float(p.min()) > 0.0 and float(p.max()) < 1.0
