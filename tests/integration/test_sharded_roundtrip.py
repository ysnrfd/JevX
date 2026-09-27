# Copyright (c) 2026 ysnrfd
"""Q-017: REAL sharded-checkpoint round-trip.

Saves a JevX model with a tiny ``max_shard_size`` so ``save_pretrained``
actually shards the weights (multiple ``model-XXXXX-of-000NN.safetensors``
files + ``model.safetensors.index.json``), reloads it with ``from_pretrained``
and asserts the decisions are identical (atol 1e-6).

The model is an in-memory tiny BERT from ``tests/conftest.py`` (no network);
the save directory is a pytest ``tmp_path``. Measured on this host: the tiny
wrapper's state dict shards into **2+ weight files** at
``max_shard_size="0.001GB"`` (1 MB — the model is smaller than 2 MB total but
larger than 1 MB, so a 2-shard split is forced) and into ~34 files at
``max_shard_size=1024`` bytes; decisions round-trip with max probability
difference 0.0 in both cases.

Taxonomy note (docs/TESTING.md §1 "Not covered and why"): this file closes the
previously-absent **sharded checkpoint** category (quality-audit Q-017) — it is
the only suite file exercising the shard index/reload path.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import pytest
import torch

from jevx import JevXConfig, JevXForDecision

HIDDEN = 32  # matches the tiny_bert fixture's hidden_size
STATE = "hello world"
QUESTIONS = ["question option?", "hello cat?"]
OPTIONS = [["a", "b", "c"], ["foo", "bar", "baz"]]
#: Documented round-trip tolerance for decisions after sharded reload.
ATOL = 1e-6


def _build_wrapper(backbone, tokenizer=None) -> JevXForDecision:
    wrapper = JevXForDecision(
        JevXConfig(
            decision_hidden_size=HIDDEN,
            question_hidden_size=HIDDEN,
            schema_hidden_size=HIDDEN,
        ),
        backbone=copy.deepcopy(backbone),
        tokenizer=tokenizer,
    )
    wrapper.eval()
    return wrapper


def _weight_files(directory: Path) -> list[Path]:
    return sorted(
        p for p in directory.iterdir()
        if p.suffix in {".safetensors", ".bin"} and p.name.startswith("model")
    )


@pytest.mark.parametrize(
    "max_shard_size, minimum_shards",
    [("0.001GB", 2), (1024, 2)],
    ids=["str-1MB", "int-1024B"],
)
def test_sharded_save_load_roundtrip_is_identity(
    tiny_bert, tiny_tokenizer, tmp_path, max_shard_size, minimum_shards
) -> None:
    """``save_pretrained(max_shard_size=...)`` writes multiple weight files;
    ``from_pretrained`` reassembles them with identical decisions (atol 1e-6)."""
    wrapper = _build_wrapper(tiny_bert, tokenizer=tiny_tokenizer)
    with torch.no_grad():
        before = wrapper(state=STATE, questions=QUESTIONS, options=OPTIONS)

    save_dir = tmp_path / f"sharded_{str(max_shard_size).replace('.', '_')}"
    wrapper.save_pretrained(str(save_dir), max_shard_size=max_shard_size)

    # -- multiple weight files + shard index actually written --------------
    weight_files = _weight_files(save_dir)
    assert len(weight_files) >= minimum_shards, (
        f"expected >= {minimum_shards} weight shards at max_shard_size="
        f"{max_shard_size!r}, got {[p.name for p in weight_files]}"
    )
    index_path = save_dir / "model.safetensors.index.json"
    assert index_path.is_file(), (
        "sharded save must write model.safetensors.index.json; directory has "
        f"{sorted(p.name for p in save_dir.iterdir())}"
    )
    index = json.loads(index_path.read_text(encoding="utf-8"))
    assert "weight_map" in index and index["weight_map"], "empty shard weight_map"
    # every weight the index names lives in a file that exists on disk
    for tensor_name, shard_file in index["weight_map"].items():
        assert (save_dir / shard_file).is_file(), (
            f"weight_map points {tensor_name!r} at missing shard {shard_file!r}"
        )

    # -- reload and compare decisions --------------------------------------
    reloaded = JevXForDecision.from_pretrained(str(save_dir))
    reloaded.eval()
    with torch.no_grad():
        after = reloaded(state=STATE, questions=QUESTIONS, options=OPTIONS)

    assert len(after.decisions) == len(before.decisions)
    for j, (got, want) in enumerate(zip(after.decisions, before.decisions)):
        assert torch.allclose(
            got.probabilities, want.probabilities, atol=ATOL, rtol=0.0
        ), (
            f"decision {j}: probabilities differ by more than {ATOL} after "
            f"sharded reload (max "
            f"{(got.probabilities - want.probabilities).abs().max():.3e})"
        )
        assert torch.equal(got.predicted_index, want.predicted_index), (
            f"decision {j}: predicted index changed "
            f"{want.predicted_index.tolist()} -> {got.predicted_index.tolist()}"
        )

    # config survived the sharded round-trip too
    assert reloaded.config.decision_hidden_size == wrapper.config.decision_hidden_size
    assert reloaded.config.to_dict().get("model_type") == "jevx"


def test_unsharded_save_stays_single_file(tiny_bert, tiny_tokenizer, tmp_path) -> None:
    """Control: the default ``max_shard_size`` writes one weight file — the
    sharding above is caused by the argument, not by the save path."""
    wrapper = _build_wrapper(tiny_bert, tokenizer=tiny_tokenizer)
    save_dir = tmp_path / "unsharded"
    wrapper.save_pretrained(str(save_dir))
    weight_files = _weight_files(save_dir)
    assert len(weight_files) == 1, [p.name for p in weight_files]
    assert not (save_dir / "model.safetensors.index.json").exists()
