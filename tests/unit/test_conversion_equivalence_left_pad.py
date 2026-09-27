# Copyright (c) 2026 ysnrfd
"""Left/right/unpadded agreement of the independent pooling REFERENCE.

`jevx.conversion.equivalence._manual_pool` is the reference side of
`check_representation_equivalence`: it exists so the adapter's pooled
`H_S` is compared against something computed independently of it. That
is only worth anything if the reference measures the same quantity, so
it must be MASK-driven (math spec Definition 4, invariant P4.3) and may
not assume a padding side. The bug this file pins: the reference used
to index by POSITION (`seq[:, 0]` for `cls`, `count - 1` for
`last_valid`), which is a right-padding assumption — on left-padded
input it measures a padded position and the checker compares the
adapter against a wrong baseline instead of failing loudly.

The six-cell grid (2 strategies x 3 paddings) is the core assertion:
`cls` and `last_valid` over unpadded, right-padded and LEFT-PADDED
input, all with the same token content, must agree BITWISE between
the adapter path and the manual reference. Under the positional
reference exactly the two left-pad cells disagree, so this file fails
on a regression to position indexing.

Everything is in-memory (`tiny_bert` from `tests/conftest.py`,
deep-copied per test): no downloads, no network.
"""

from __future__ import annotations

import copy

import pytest
import torch

from jevx import convert
from jevx.conversion.equivalence import _manual_pool, check_representation_equivalence
from jevx.representations.pooling import pool

#: The three paddings of ONE logical 4-token content, all ids inside the
#: fixture vocab: row 0 is full, row 1 is padded to length 4.
#: `left_pad` puts the padding BEFORE the content — the layout a
#: right-padding assumption gets wrong.
PADDINGS: dict = {
    "unpadded": (
        torch.tensor([[2, 3, 4, 5], [6, 7, 8, 9]]),
        torch.tensor([[1, 1, 1, 1], [1, 1, 1, 1]]),
    ),
    "right_pad": (
        torch.tensor([[2, 3, 4, 5], [6, 7, 8, 9]]),
        torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]]),
    ),
    "left_pad": (
        torch.tensor([[1, 1, 2, 3], [1, 1, 1, 6]]),
        torch.tensor([[0, 0, 1, 1], [0, 0, 0, 1]]),
    ),
}

#: The six cells: every strategy that used to index by position, over
#: every padding side.
CELLS = tuple(
    (strategy, name)
    for strategy in ("cls", "last_valid")
    for name in PADDINGS
)


def _inputs(name: str) -> dict:
    """Fixed pre-tokenized batch for one padding layout."""
    input_ids, attention_mask = PADDINGS[name]
    return {
        "input_ids": input_ids.clone(),
        "attention_mask": attention_mask.clone(),
    }



# ---------------------------------------------------------------------------
# the six cells: adapter path vs the independent reference
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("strategy", "padding"), CELLS)
def test_manual_reference_matches_adapter_on_every_padding(
    tiny_bert, strategy, padding
) -> None:
    """Adapter pooling == the manual reference, bitwise, in all six cells."""
    wrapper, _ = convert(copy.deepcopy(tiny_bert), pooling_method=strategy)
    result = check_representation_equivalence(wrapper, _inputs(padding))
    # the manual reference is what was actually compared (not a native field)
    assert result["strategy"] == strategy, result
    assert result["reference"].startswith("independent fp32"), result
    assert "bitwise" in result and result["bitwise"] is True, (
        strategy,
        padding,
        result,
        "the independent reference disagrees with the adapter",
    )
    assert result["max_abs"] == 0.0, result
    assert result["passed"] is True, result


# ---------------------------------------------------------------------------
# the reference picks the MASK's first / last valid position
# ---------------------------------------------------------------------------


#: ONE logical row (content [1.0, 2.0]) written twice, in the two padding
#: layouts. Every position carries a distinct value and the padded ones are
#: 0.0, so a position-indexed reference is visibly wrong instead of
#: accidentally right.
LEFT_SEQ = torch.tensor([[[0.0], [0.0], [1.0], [2.0], [0.0]]])
RIGHT_SEQ = torch.tensor([[[1.0], [2.0], [0.0], [0.0], [0.0]]])
LEFT_MASK = torch.tensor([[False, False, True, True, False]])
RIGHT_MASK = torch.tensor([[True, True, False, False, False]])

#: The value a mask-driven reference must return, per strategy.
EXPECTED = {"cls": 1.0, "last_valid": 2.0, "masked_mean": 1.5}


@pytest.mark.parametrize("strategy", sorted(EXPECTED))
def test_manual_pool_is_independent_of_the_padding_side(strategy) -> None:
    """The same content pools identically left-padded or right-padded."""
    left = _manual_pool(LEFT_SEQ, LEFT_MASK, strategy)
    right = _manual_pool(RIGHT_SEQ, RIGHT_MASK, strategy)
    assert torch.equal(left, right), (left, right)
    # the mask's valid positions, never a padded one
    assert torch.equal(left, torch.tensor([[EXPECTED[strategy]]])), left
    # the positional answers the reference used to give, for the record:
    assert not torch.equal(left, LEFT_SEQ[:, 0])
    if strategy == "last_valid":
        assert not torch.equal(left, LEFT_SEQ[:, int(LEFT_MASK.sum()) - 1])


@pytest.mark.parametrize("strategy", ["cls", "last_valid"])
def test_manual_pool_agrees_with_the_shipped_operator_on_left_padding(
    strategy,
) -> None:
    """The reference is independent but must measure the SAME quantity."""
    reference = _manual_pool(LEFT_SEQ, LEFT_MASK, strategy)
    shipped = pool(LEFT_SEQ, LEFT_MASK, strategy)
    assert torch.equal(reference, shipped), (reference, shipped)


def test_manual_pool_keeps_its_other_strategies() -> None:
    """masked_mean / mean are untouched by the mask-driven rewrite."""
    manual = _manual_pool(LEFT_SEQ, LEFT_MASK, "masked_mean")
    assert torch.equal(manual, torch.tensor([[1.5]])), manual
    assert torch.equal(manual, _manual_pool(RIGHT_SEQ, RIGHT_MASK, "masked_mean"))
    assert torch.equal(
        _manual_pool(LEFT_SEQ, LEFT_MASK, "mean"), LEFT_SEQ.mean(dim=1)
    )
    assert torch.equal(
        _manual_pool(LEFT_SEQ, LEFT_MASK, "encoder_pool"),
        _manual_pool(LEFT_SEQ, LEFT_MASK, "masked_mean"),
    )
    # no mask at all: every position is valid -> position 0 / the last one
    assert torch.equal(_manual_pool(LEFT_SEQ, None, "cls"), LEFT_SEQ[:, 0])
    assert torch.equal(_manual_pool(LEFT_SEQ, None, "last_valid"), LEFT_SEQ[:, -1])
    with pytest.raises(ValueError, match="unsupported strategy"):
        _manual_pool(LEFT_SEQ, LEFT_MASK, "not_a_strategy")
