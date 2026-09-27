# Copyright (c) 2026 ysnrfd
"""§94 golden reference tests — re-derive captured outputs and compare.

Six FIXED models (``hf-internal-testing/tiny-random-bert``,
``hf-internal-testing/tiny-random-t5``, ``sshleifer/tiny-gpt2``,
``hf-internal-testing/tiny-random-RobertaModel``,
``hf-internal-testing/tiny-random-BartForConditionalGeneration``,
``hf-internal-testing/tiny-random-ViTModel``) with FIXED inputs (text families:
one sentence, two questions, two schemas; the vision family: the checked-in
``input_image_224.png`` fixture through the model's own image processor, same
questions/schemas), ``torch.manual_seed(0)``, ``model.eval()``. The fixtures
under ``tests/golden/fixtures/`` were captured by RUNNING this module's
:func:`derive` (generator: ``vendor/reports/_gen_golden_fixtures.py``) — never
hand-written.

Captured per model and asserted here:

* ``probs`` — every choice probability rounded to 7 decimals;
* ``predicted_indices`` — exact;
* ``state_representation`` — checksum (sum / mean / std, 7 decimals) of the
  fused ``H`` output;
* ``config`` — the pinned JevX config fields (exact equality);
* ``adapter`` / ``representation_field`` / ``jevx_conversion`` — routing
  identity (exact equality).

Documented tolerance: **1e-6** absolute for probabilities and checksums
(fixtures are rounded to 7 decimals; same-machine CPU fp32 re-derivation is
deterministic, so observed deltas are ~0). Any drift beyond tolerance means
representation extraction, pooling, normalization or serialization changed —
fail loudly and regenerate fixtures only with intent.

**Regeneration log (intent recorded).** 2026-09-25 — all six fixtures were
regenerated after the option-pathway closure (spec §5.f, `Fusion.option_normalize=True`
defaulting to a unit-DIRECTION option row before `W_o`). The drift was proven to come
from that one intentional change alone: re-deriving with `option_normalize=False`
reproduces the pre-change fixtures **bitwise** (max |Δp| = 0.000e+00 on all six).
Only `probs` and `predicted_indices` moved; `state_representation` checksums are
unchanged (the §5.f map touches the option branch only), and 5 of 6 predicted indices
are unchanged (t5's second decision flips 2 → 0 on a near-tie). Tolerance and every
assertion are
untouched — only the captured records were rewritten, via
`vendor/reports/_gen_golden_fixtures.py`.

**Second regeneration-log entry (2026-09-24, WAVE-2 follow-up / DECISION-3).**
The default `JevXConfig.conditioning_mode` changed from `"full"` to `"off"`,
and `"off"` now also neutralizes the question path in fusion, so a
default-constructed wrapper no longer reproduces these fixtures. The fixtures
were **NOT** regenerated. `GOLDEN_SPECS` now passes `conditioning_mode="full"`
explicitly, which is the pipeline the captured values were produced by, and with
it **every checked-in fixture JSON is unchanged and every assertion still passes
at the documented 1e-6 tolerance** (6/6). Regenerating under the new default was
rejected because the shipped linear state pathway is ZERO-initialized and the
option residual is off on a state-only call, so a fresh off-mode head emits one
identical logit per class: the recorded `probs` would become uniform
(`0.5/0.5/0.0` on the K=2 decision) and the reference would stop detecting
anything. The `"off"` path is pinned by
`tests/unit/test_conditioning_off_matches_state_only.py` instead, including a
bitwise equality against the WAVE-2 `state_only` reference arm. Tolerance and
every assertion are untouched; only the mode the spec requests became explicit.

Why these six of the 18 validated checkpoints: they pin the adapter routes
cheapest to re-derive offline — encoder + pooler (bert, roberta),
encoder–decoder (t5, bart), decoder (gpt2) and vision + pooler (vit). The
remaining 12 stay matrix-gated (live reference equality every run) — rationale
in docs/TESTING.md §1.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:  # allows `python tests/golden/test_golden_references.py`
    sys.path.insert(0, str(ROOT))

from jevx import JevX  # noqa: E402

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"

#: Documented comparison tolerance (§94).
TOLERANCE = 1e-6

SEED = 0

GOLDEN_STATE = (
    "The maintenance crew inspected the bridge deck and confirmed that the "
    "support damage had been repaired before the storm arrived."
)
GOLDEN_QUESTIONS = ["Is this content safe to deploy?", "Which disposition fits this item?"]
GOLDEN_OPTIONS = [["safe", "unsafe"], ["approve", "review", "reject"]]

#: key -> (model_id, extra JevX kwargs, fixture file name, state kind)
#: state kind: "text" (GOLDEN_STATE string) or "vision" (the checked-in
#: ``input_image_224.png`` fixture through the model's image processor).
#: key -> (model_id, extra JevX kwargs, fixture file name, state kind).
#:
#: The extra kwargs are EXPLICIT about the conditioning mode since DECISION-3
#: (WAVE-2 follow-up): JevXConfig() now defaults to conditioning_mode="off", a
#: state-only call. The captured fixtures were produced by the pre-DECISION-3
#: default, which was "full" - the conditioned per-call path - and the job of
#: this reference is to pin THE PIPELINE, so the pipeline it pins is now named
#: explicitly. The captured VALUES are unchanged and are NOT regenerated: with
#: conditioning_mode="full" the derived record reproduces every checked-in
#: fixture within the documented tolerance (verified: 6/6 pass, files unchanged).
#:
#: Regenerating under the new default was REJECTED, deliberately: with the
#: option residual off and the shipped ZERO-initialized linear state pathway, a
#: fresh state-only head emits the same logit for every class, so the recorded
#: probabilities would be uniform (0.5/0.5/0.0) and the fixture would stop
#: detecting anything. Pinning the mode keeps the assertion exactly as strong
#: as it was. The "off" path outputs are pinned instead by
#: tests/unit/test_conditioning_off_matches_state_only.py.
#:
#: state kind: "text" (GOLDEN_STATE string) or "vision" (the checked-in
#: input_image_224.png fixture through the model image processor).
GOLDEN_SPECS: Dict[str, Any] = {
    "tiny-random-bert": (
        "hf-internal-testing/tiny-random-bert",
        {"conditioning_mode": "full"},
        "golden_tiny_random_bert.json",
        "text",
    ),
    "tiny-random-t5": (
        "hf-internal-testing/tiny-random-t5",
        {"conditioning_mode": "full"},
        "golden_tiny_random_t5.json",
        "text",
    ),
    "tiny-gpt2": (
        "sshleifer/tiny-gpt2",
        {"conditioning_mode": "full"},
        "golden_tiny_gpt2.json",
        "text",
    ),
    "tiny-random-roberta": (
        "hf-internal-testing/tiny-random-RobertaModel",
        {"conditioning_mode": "full"},
        "golden_tiny_random_roberta.json",
        "text",
    ),
    "tiny-random-bart": (
        "hf-internal-testing/tiny-random-BartForConditionalGeneration",
        {"conditioning_mode": "full"},
        "golden_tiny_random_bart.json",
        "text",
    ),
    "tiny-random-vit": (
        "hf-internal-testing/tiny-random-ViTModel",
        {"conditioning_mode": "full"},
        "golden_tiny_random_vit.json",
        "vision",
    ),
}

#: Config fields captured verbatim (exact-match comparison).
CONFIG_FIELDS = [
    "model_type",
    "base_model_name_or_path",
    "base_model_type",
    "representation_strategy",
    "pooling_method",
    "fusion_type",
    "decision_hidden_size",
    "question_hidden_size",
    "schema_hidden_size",
    "decision_primitives",
    "calibration_method",
    "confidence_method",
    "freeze_base_model",
    "version",
]

_ENV_MARKERS = (
    "401", "403", "404", "connection", "timed out", "timeout", "max retries",
    "couldn't connect", "could not fetch", "offline", "not found", "gated",
    "permission", "proxy", "name resolution",
    "temporary failure in name resolution",
)


def _is_environment_error(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(marker in text for marker in _ENV_MARKERS)


def seed_everything(seed: int = SEED) -> None:
    """Fixed RNG state so the wrapper's randomly-initialized JevX-side layers
    (question encoder, schema encoder, projections, heads) are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _build_state(state_kind: str) -> Any:
    """Fixed input for one state kind (text string / vision image dict)."""
    if state_kind == "text":
        return GOLDEN_STATE
    if state_kind == "vision":
        from PIL import Image

        image_path = FIXTURES_DIR / "input_image_224.png"
        return {"images": Image.open(image_path).convert("RGB")}
    raise ValueError(f"unknown golden state kind {state_kind!r}")


def derive(key: str) -> Dict[str, Any]:
    """Re-derive the golden record for one model by RUNNING the pipeline."""
    model_id, kwargs, _fixture_name, state_kind = GOLDEN_SPECS[key]
    seed_everything()
    wrapper = JevX.from_pretrained(model_id, **kwargs)
    wrapper.eval()
    with torch.no_grad():
        out = wrapper.choice(
            state=_build_state(state_kind),
            questions=GOLDEN_QUESTIONS,
            options=GOLDEN_OPTIONS,
        )
    probs = [
        [round(float(v), 7) for v in dec.probabilities[0].tolist()]
        for dec in out.decisions
    ]
    indices = [int(dec.predicted_index[0]) for dec in out.decisions]
    sr = out.state_representation.detach().float()
    stats = {
        "sum": round(float(sr.sum()), 7),
        "mean": round(float(sr.mean()), 7),
        "std": round(float(sr.std(unbiased=False)), 7),
    }
    meta = out.capability_metadata or {}
    # JSON-safe state record: the vision state is the checked-in fixture path
    state_record: Any = (
        GOLDEN_STATE
        if state_kind == "text"
        else "tests/golden/fixtures/input_image_224.png"
    )
    return {
        "model_id": model_id,
        "seed": SEED,
        "decision_type": "choice",
        "state": state_record,
        "state_kind": state_kind,
        "questions": GOLDEN_QUESTIONS,
        "options": GOLDEN_OPTIONS,
        "tolerance": TOLERANCE,
        "adapter": meta.get("adapter"),
        "representation_field": meta.get("representation_field"),
        "jevx_conversion": wrapper.capability_report.jevx_conversion,
        "probs": probs,
        "predicted_indices": indices,
        "state_representation": stats,
        "config": {name: getattr(wrapper.config, name) for name in CONFIG_FIELDS},
    }


def generate_fixtures() -> Path:
    """(Re)write every golden fixture JSON — run via
    ``vendor/reports/_gen_golden_fixtures.py``."""
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    for key, (_model_id, _kwargs, fixture_name, _kind) in GOLDEN_SPECS.items():
        record = derive(key)
        path = FIXTURES_DIR / fixture_name
        path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
        print(f"wrote {path} ({key})", flush=True)
    return FIXTURES_DIR


def _load_fixture(fixture_name: str) -> Dict[str, Any]:
    path = FIXTURES_DIR / fixture_name
    if not path.is_file():
        pytest.fail(
            f"missing golden fixture {path} — capture it by running "
            "python vendor/reports/_gen_golden_fixtures.py (requires network)"
        )
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("key", sorted(GOLDEN_SPECS))
def test_golden_reference(key: str) -> None:
    """Re-derive the pipeline outputs and compare against the fixture."""
    _model_id, _kwargs, fixture_name, _kind = GOLDEN_SPECS[key]
    fixture = _load_fixture(fixture_name)
    try:
        derived = derive(key)
    except Exception as exc:  # noqa: BLE001 - classified below
        if _is_environment_error(exc):
            pytest.skip(
                f"{key}: environment-blocked ({type(exc).__name__}: {exc}) — golden "
                "re-derivation needs the checkpoint download"
            )
        raise

    # -- routing identity (exact) ------------------------------------------
    assert derived["adapter"] == fixture["adapter"], (
        f"{key}: adapter changed {fixture['adapter']!r} -> {derived['adapter']!r}"
    )
    assert derived["representation_field"] == fixture["representation_field"], (
        f"{key}: representation_field changed "
        f"{fixture['representation_field']!r} -> {derived['representation_field']!r}"
    )
    assert derived["jevx_conversion"] == fixture["jevx_conversion"]

    # -- config fields (exact) ---------------------------------------------
    for name in CONFIG_FIELDS:
        assert derived["config"][name] == fixture["config"][name], (
            f"{key}: config.{name} changed {fixture['config'][name]!r} -> "
            f"{derived['config'][name]!r}"
        )

    # -- probabilities (documented tolerance 1e-6) -------------------------
    assert len(derived["probs"]) == len(fixture["probs"])
    for j, (got_row, want_row) in enumerate(zip(derived["probs"], fixture["probs"])):
        assert len(got_row) == len(want_row), f"{key}: decision {j} K changed"
        for k, (got, want) in enumerate(zip(got_row, want_row)):
            assert abs(got - want) <= TOLERANCE, (
                f"{key}: probs[{j}][{k}] drifted by {abs(got - want):.3e} "
                f"(> {TOLERANCE:g}): fixture={want} derived={got}"
            )

    # -- predicted indices (exact) -----------------------------------------
    assert derived["predicted_indices"] == fixture["predicted_indices"], (
        f"{key}: predicted indices changed {fixture['predicted_indices']} -> "
        f"{derived['predicted_indices']}"
    )

    # -- state_representation checksum (tolerance 1e-6) ---------------------
    for stat in ("sum", "mean", "std"):
        got = derived["state_representation"][stat]
        want = fixture["state_representation"][stat]
        assert abs(got - want) <= TOLERANCE, (
            f"{key}: state_representation.{stat} drifted by {abs(got - want):.3e} "
            f"(> {TOLERANCE:g}): fixture={want} derived={got}"
        )


if __name__ == "__main__":  # pragma: no cover - fixture capture entry point
    generate_fixtures()
