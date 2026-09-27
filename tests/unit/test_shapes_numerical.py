# Copyright (c) 2026 ysnrfd
"""Unit tests for jevx.utils: validators with useful errors, stable
softmax/log-softmax, masked softmax and low-precision paths."""

from __future__ import annotations

import math

import pytest
import torch

from jevx.utils import (
    assert_batch_alignment,
    assert_same_device_dtype,
    assert_shape,
    check_finite,
    has_inf,
    has_nan,
    masked_softmax,
    one_hot,
    stable_log_softmax,
    stable_softmax,
)


# ---------------------------------------------------------------------------
# assert_shape
# ---------------------------------------------------------------------------


def test_assert_shape_passes_and_returns_tensor():
    t = torch.zeros(7, 4)
    assert assert_shape(t, (None, 4), "mytensor") is t
    assert assert_shape(t, None, "mytensor") is t  # None spec: tensor check only
    assert assert_shape(t, (7, 4), "mytensor") is t


def test_assert_shape_wrong_dim_size():
    t = torch.zeros(7, 4)
    with pytest.raises(ValueError, match="mytensor"):
        assert_shape(t, (None, 5), "mytensor")
    with pytest.raises(ValueError, match="expected shape"):
        assert_shape(t, (None, 5), "mytensor")


def test_assert_shape_wrong_rank():
    with pytest.raises(ValueError, match="rank"):
        assert_shape(torch.zeros(7, 4), (None, None, 4), "mytensor")


def test_assert_shape_not_a_tensor():
    with pytest.raises(TypeError, match="mytensor"):
        assert_shape([[1, 2]], (None, 2), "mytensor")


# ---------------------------------------------------------------------------
# assert_batch_alignment
# ---------------------------------------------------------------------------


def test_assert_batch_alignment_ok():
    b = assert_batch_alignment(
        state=torch.zeros(3, 8), question=torch.zeros(3, 5, 8), options=None
    )
    assert b == 3


def test_assert_batch_alignment_mismatch():
    with pytest.raises(ValueError, match="batch size mismatch"):
        assert_batch_alignment(state=torch.zeros(3, 8), question=torch.zeros(2, 8))
    with pytest.raises(ValueError, match="question"):
        assert_batch_alignment(state=torch.zeros(3, 8), question=torch.zeros(2, 8))


def test_assert_batch_alignment_scalar_and_empty():
    with pytest.raises(ValueError, match="0-dim"):
        assert_batch_alignment(state=torch.tensor(5.0))
    with pytest.raises(ValueError, match="no tensors"):
        assert_batch_alignment(state=None, other=None)


# ---------------------------------------------------------------------------
# check_finite / has_nan / has_inf
# ---------------------------------------------------------------------------


def test_check_finite_passes():
    t = torch.linspace(-1e6, 1e6, 10)
    assert check_finite(t, "h") is t


def test_check_finite_nan_counts_and_index():
    t = torch.zeros(2, 3)
    t[1, 2] = float("nan")
    with pytest.raises(ValueError, match="h_state"):
        check_finite(t, "h_state")
    with pytest.raises(ValueError, match="NaN"):
        check_finite(t, "h_state")
    with pytest.raises(ValueError, match=r"\(1, 2\)"):
        check_finite(t, "h_state")  # first offending index is reported


def test_check_finite_inf_counts():
    pos = torch.zeros(4)
    pos[0] = float("inf")
    with pytest.raises(ValueError, match="Inf"):
        check_finite(pos, "pos")
    neg = torch.zeros(4)
    neg[3] = float("-inf")
    with pytest.raises(ValueError, match="Inf"):
        check_finite(neg, "neg")
    assert has_inf(pos) and has_inf(neg) and not has_inf(torch.zeros(2))
    assert has_nan(torch.tensor([float("nan")]))
    assert not has_nan(torch.zeros(2))
    # integer tensors cannot be non-finite
    assert not has_nan(torch.arange(4)) and not has_inf(torch.arange(4))


def test_check_finite_rejects_non_float():
    with pytest.raises(TypeError, match="floating-point"):
        check_finite(torch.arange(4), "labels")


# ---------------------------------------------------------------------------
# assert_same_device_dtype
# ---------------------------------------------------------------------------


def test_assert_same_device_dtype_ok():
    a, b = torch.zeros(2), torch.ones(2)
    assert_same_device_dtype(a, b)
    assert_same_device_dtype(a, b, names=["fused", "question"])


def test_assert_same_device_dtype_mismatch():
    a = torch.zeros(2, dtype=torch.float32)
    b = torch.zeros(2, dtype=torch.float64)
    with pytest.raises(ValueError, match="dtype mismatch"):
        assert_same_device_dtype(a, b, names=["fused", "question"])
    with pytest.raises(ValueError, match="fused"):
        assert_same_device_dtype(a, b, names=["fused", "question"])


def test_assert_same_device_dtype_device_mismatch():
    cpu = torch.zeros(2)
    meta = torch.zeros(2, device="meta")
    with pytest.raises(ValueError, match="device mismatch"):
        assert_same_device_dtype(cpu, meta, names=["on_cpu", "on_meta"])


def test_assert_same_device_dtype_argument_errors():
    with pytest.raises(TypeError, match="at least two"):
        assert_same_device_dtype(torch.zeros(2))
    with pytest.raises(TypeError, match="torch.Tensor"):
        assert_same_device_dtype(torch.zeros(2), [1, 2])
    with pytest.raises(ValueError, match="names"):
        assert_same_device_dtype(torch.zeros(2), torch.zeros(2), names=["only-one"])


# ---------------------------------------------------------------------------
# stable softmax vs naive
# ---------------------------------------------------------------------------


def test_stable_softmax_vs_naive_on_large_logits():
    x = torch.tensor([1000.0, 0.0, -1000.0])
    naive = torch.exp(x) / torch.exp(x).sum()
    assert torch.isnan(naive).any(), "naive softmax must overflow to NaN here"
    stable = stable_softmax(x)
    assert torch.isfinite(stable).all()
    assert stable.sum().item() == pytest.approx(1.0, abs=1e-6)
    assert stable[0].item() == pytest.approx(1.0, abs=1e-6)
    assert stable[1].item() == pytest.approx(0.0, abs=1e-6)


def test_stable_softmax_extreme_symmetric():
    x = torch.tensor([[1e6, -1e6, 0.0]])
    p = stable_softmax(x)
    assert torch.isfinite(p).all()
    assert p.sum().item() == pytest.approx(1.0, abs=1e-6)
    # agrees with torch's internal softmax (also max-subtracted)
    assert torch.allclose(p, torch.nn.functional.softmax(x, dim=-1), atol=1e-6)


def test_stable_softmax_ignores_shift():
    x = torch.randn(5, 6)
    a = stable_softmax(x)
    b = stable_softmax(x + 1000.0)
    assert torch.allclose(a, b, atol=1e-3)


def test_stable_log_softmax_rows_sum_to_zero():
    x = torch.tensor([[1e6, 0.0, -5.0], [3.0, 3.0, 3.0]])
    log_p = stable_log_softmax(x)
    assert torch.isfinite(log_p).all()
    assert torch.allclose(
        torch.logsumexp(log_p, dim=-1), torch.zeros(2), atol=1e-5
    )
    assert torch.allclose(log_p[1], torch.full((3,), -math.log(3)), atol=1e-6)


# ---------------------------------------------------------------------------
# low-precision (fp16 / bf16) paths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_softmax_finite_and_downcast(dtype):
    x = torch.tensor([[5.0, -5.0, 0.0], [1.0, 1.0, 1.0]], dtype=dtype)
    p = stable_softmax(x)
    assert p.dtype == dtype, "output must return the input dtype"
    assert torch.isfinite(p).all()
    # computed in fp32: matches the fp32 reference before downcast
    ref = torch.nn.functional.softmax(x.float(), dim=-1)
    assert torch.allclose(p.float(), ref, atol=1e-2)
    assert torch.allclose(
        p.float().sum(-1), torch.ones(2, dtype=torch.float32), atol=1e-2
    )
    assert p[1, 0].float().item() == pytest.approx(1 / 3, abs=2e-2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_softmax_large_logits(dtype):
    # 1e4 is representable in the dtype but exp(1e4) overflows to +Inf, so
    # only the max-subtraction inside softmax (after the fp32 upcast) can
    # produce a valid result.
    x = torch.tensor([1e4, -1e4, 0.0], dtype=dtype)
    assert torch.isinf(torch.exp(x)).any(), "raw exp(1e4) must overflow"
    p = stable_softmax(x)
    assert p.dtype == dtype
    assert torch.isfinite(p).all()
    assert p.sum().item() == pytest.approx(1.0, abs=1e-2)
    assert p[0].item() == 1.0
    assert p[1].item() == 0.0


def test_low_precision_dtype_validation():
    with pytest.raises(TypeError, match="floating-point"):
        stable_softmax(torch.zeros(2, dtype=torch.int64))
    with pytest.raises(TypeError, match="torch.Tensor"):
        stable_softmax([1.0, 2.0])


# ---------------------------------------------------------------------------
# masked softmax
# ---------------------------------------------------------------------------


def test_masked_softmax_invalid_zero_valid_sum_to_one():
    logits = torch.tensor([[2.0, -1.0, 0.5], [0.0, 3.0, -2.0]])
    mask = torch.tensor([[True, False, True], [False, True, False]])
    p = masked_softmax(logits, mask)
    assert (p[~mask] == 0.0).all(), "invalid probs must be exactly 0"
    assert torch.allclose(p.sum(-1), torch.ones(2), atol=1e-6)
    # the single valid option in row 1 carries all the mass
    assert p[1, 1].item() == pytest.approx(1.0, abs=1e-6)


def test_masked_softmax_none_mask_matches_stable():
    logits = torch.randn(3, 4)
    assert torch.equal(masked_softmax(logits, None), stable_softmax(logits))


def test_masked_softmax_extreme_logits_with_mask():
    # position 0 is invalid yet carries a huge logit: it must still get
    # exactly zero while the two valid extremes split the mass evenly
    logits = torch.tensor([[1e6, 1e6, 1e6]])
    mask = torch.tensor([[False, True, True]])
    p = masked_softmax(logits, mask)
    assert torch.isfinite(p).all()
    assert p[0, 0].item() == 0.0
    assert p[0, 1].item() == pytest.approx(0.5, abs=1e-6)
    assert p[0, 2].item() == pytest.approx(0.5, abs=1e-6)
    assert p.sum().item() == pytest.approx(1.0, abs=1e-6)


def test_masked_softmax_errors():
    logits = torch.randn(2, 3)
    with pytest.raises(ValueError, match="same shape"):
        masked_softmax(logits, torch.ones(2, 4, dtype=torch.bool))
    with pytest.raises(TypeError, match="torch.bool"):
        masked_softmax(logits, torch.ones(2, 3, dtype=torch.uint8))
    with pytest.raises(ValueError, match="at least one valid"):
        masked_softmax(
            logits, torch.tensor([[True, True, True], [False, False, False]])
        )
    with pytest.raises(TypeError, match="floating-point"):
        masked_softmax(torch.ones(2, 3, dtype=torch.int64), None)


def test_masked_softmax_is_per_row_not_cross_row():
    # row normalization must not leak across questions
    logits = torch.tensor([[100.0, 0.0], [0.0, 0.0]])
    mask = torch.tensor([[True, True], [True, True]])
    p = masked_softmax(logits, mask)
    assert p[1, 0].item() == pytest.approx(0.5, abs=1e-6)
    assert p[1, 1].item() == pytest.approx(0.5, abs=1e-6)
    assert p[0, 0].item() == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# one_hot
# ---------------------------------------------------------------------------


def test_one_hot_values_and_validation():
    oh = one_hot(torch.tensor([0, 2]), 3)
    assert oh.shape == (2, 3)
    assert oh.dtype == torch.float32
    expected = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert torch.equal(oh, expected)
    with pytest.raises(ValueError, match="must lie in"):
        one_hot(torch.tensor([0, 3]), 3)
    with pytest.raises(ValueError, match="num_classes"):
        one_hot(torch.tensor([0]), 0)


# ---------------------------------------------------------------------------
# metric edge cases
# ---------------------------------------------------------------------------


def test_nll_clips_zero_probability():
    from jevx.utils.numerical import negative_log_likelihood

    probs = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    labels = torch.tensor([1])  # impossible under this distribution
    value = negative_log_likelihood(probs, labels)
    assert math.isfinite(value)
    assert value == pytest.approx(-math.log(1e-12), rel=1e-6)


def test_nll_binary_hand_value():
    from jevx.utils.numerical import negative_log_likelihood

    probs = torch.tensor([[0.25], [0.75]])
    labels = torch.tensor([1, 1])
    hand = -(math.log(0.25) + math.log(0.75)) / 2
    assert negative_log_likelihood(probs, labels) == pytest.approx(hand, abs=1e-9)


def test_n_bins_validation():
    from jevx.utils.numerical import expected_calibration_error

    probs = torch.full((4, 2), 0.5)
    labels = torch.zeros(4, dtype=torch.long)
    with pytest.raises(ValueError, match="n_bins"):
        expected_calibration_error(probs, labels, n_bins=0)


def test_metrics_reject_mismatched_labels():
    from jevx.utils.numerical import expected_calibration_error

    probs = torch.full((4, 2), 0.5)
    labels = torch.zeros(3, dtype=torch.long)
    with pytest.raises(ValueError, match="one label per sample"):
        expected_calibration_error(probs, labels)
