# Copyright (c) 2026 ysnrfd
"""End-to-end JevX runs against REAL hub checkpoints (downloads required).

Per checkpoint family (encoder / encoder-decoder / decoder-only):
wrap -> choice/score/noul invariants -> save/reload equality -> base_forward
equivalence vs a freshly loaded raw model -> shared-state single forward ->
compact results table line (printed; visible with ``-s``).

Primary ids fall back to verified alternates from
``vendor/reports/_hf_verified_ids.json`` (GET 200 at audit time). If the
primary AND both verified alternates fail with environment/network errors,
that family is skipped with an explicit reason (never fails the suite for an
unavailable download). Implementation bugs (non-environment errors) re-raise
and DO fail the test.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import pytest
import torch
from transformers import AutoModel

from jevx import JevX, JevXForDecision

VERIFIED_IDS_FILE = (
    Path(__file__).resolve().parents[2] / "vendor" / "reports" / "_hf_verified_ids.json"
)

PRIMARY_ID = {
    "bert": "hf-internal-testing/tiny-random-bert",
    "t5": "hf-internal-testing/tiny-random-t5",
    "gpt2": "sshleifer/tiny-gpt2",
}
# Verified alternates (all present with GET 200 in _hf_verified_ids.json),
# tiny first; the larger same-family id is the second alternate.
ALTERNATE_IDS = {
    "bert": ["hf-internal-testing/tiny-random-BertModel", "google-bert/bert-base-uncased"],
    "t5": ["hf-internal-testing/tiny-random-T5Model", "google-t5/t5-small"],
    "gpt2": ["hf-internal-testing/tiny-random-GPT2Model", "gpt2"],
}

#: Substrings marking an environment/network problem (skip-eligible) rather
#: than an implementation bug (which must fail loudly).
_ENV_MARKERS = (
    "401",
    "403",
    "404",
    "connection",
    "timed out",
    "timeout",
    "max retries",
    "couldn't connect",
    "could not fetch",
    "offline",
    "not found",
    "gated",
    "permission",
    "proxy",
    "name resolution",
    "temporary failure in name resolution",
)


def _is_environment_error(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(marker in text for marker in _ENV_MARKERS)


def _candidate_ids(family: str) -> list[str]:
    alternates = list(ALTERNATE_IDS[family])
    if VERIFIED_IDS_FILE.is_file():  # keep only ids the audit verified
        import json

        verified = json.loads(VERIFIED_IDS_FILE.read_text(encoding="utf-8"))
        alternates = [a for a in alternates if a in verified] or alternates
    return [PRIMARY_ID[family], *alternates]


def _load_wrapper(family: str) -> Tuple[JevXForDecision, str]:
    attempts = []
    for model_id in _candidate_ids(family):
        try:
            return JevX.from_pretrained(model_id), model_id
        except Exception as exc:  # noqa: BLE001 - classified below
            attempts.append(f"{model_id}: {type(exc).__name__}: {exc}")
            if not _is_environment_error(exc):
                raise  # implementation bug — fail, do not mask as blocked
    pytest.skip(
        f"{family}: environment-blocked — the primary id and both verified "
        f"alternates from {VERIFIED_IDS_FILE.name} failed with network/HTTP "
        f"errors. Attempts: " + " | ".join(attempts)
    )
    raise AssertionError("unreachable")  # pytest.skip raises


class _ForwardCounter:
    """Counts outermost backbone forward calls via register_forward_hook.

    For encoder-decoder children the adapter calls ``model.encoder`` directly
    (audit Q6), so the encoder submodule is instrumented too; depth tracking
    ensures a nested encoder call inside a full forward counts once.
    """

    def __init__(self, modules) -> None:
        self.depth = 0
        self.calls = 0
        self._handles = []
        for module in modules:
            self._handles.append(module.register_forward_pre_hook(self._pre))
            self._handles.append(module.register_forward_hook(self._post))

    def _pre(self, module, args) -> None:
        if self.depth == 0:
            self.calls += 1
        self.depth += 1

    def _post(self, module, args, output) -> None:
        self.depth -= 1

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()


def _state_kwargs(wrapper: JevXForDecision) -> Dict[str, Any]:
    """Text state when a tokenizer resolved, else in-vocab input_ids."""
    if wrapper.tokenizer is not None:
        return {"state": "the cat sat on the bar"}
    vocab = int(wrapper.model.config.vocab_size)
    ids = torch.arange(1, 9, dtype=torch.long).unsqueeze(0) % max(vocab, 1)
    return {
        "state": None,
        "input_ids": ids,
        "attention_mask": torch.ones_like(ids),
    }


def _prepared_backbone_inputs(wrapper: JevXForDecision, state_kwargs: Dict[str, Any]):
    """Model kwargs for base_forward — through the adapter so encoder-decoder
    children get their mirrored decoder inputs (audit Q6)."""
    state = state_kwargs.get("state")
    if state is None:
        state = {
            key: state_kwargs[key]
            for key in ("input_ids", "attention_mask", "token_type_ids")
            if key in state_kwargs
        }
    return wrapper.adapter.prepare_inputs(
        state, tokenizer=wrapper.tokenizer, model=wrapper.model
    )


def _max_abs_error(a: Any, b: Any) -> float:
    """Max abs difference over all tensor leaves of two parallel outputs."""
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor) and a.shape == b.shape, (
            f"output structure differs: {type(a)} {getattr(a, 'shape', None)} vs "
            f"{type(b)} {getattr(b, 'shape', None)}"
        )
        if a.numel() == 0:
            return 0.0
        return float((a.float() - b.float()).abs().max())
    if isinstance(a, dict):
        assert isinstance(b, dict)
        keys_a = [k for k, v in a.items() if v is not None]
        keys_b = [k for k, v in b.items() if v is not None]
        assert keys_a == keys_b, f"output fields differ: {keys_a} vs {keys_b}"
        return max((_max_abs_error(a[k], b[k]) for k in keys_a), default=0.0)
    if isinstance(a, (tuple, list)):
        assert isinstance(b, (tuple, list)) and len(a) == len(b)
        return max((_max_abs_error(x, y) for x, y in zip(a, b)), default=0.0)
    return 0.0  # None / foreign objects (e.g. empty past_key_values caches)


@pytest.fixture(scope="module", params=["bert", "t5", "gpt2"])
def family_case(request):
    """(family, wrapper, used_model_id) — one instance per checkpoint family."""
    wrapper, used_id = _load_wrapper(request.param)
    wrapper.eval()
    return request.param, wrapper, used_id


def test_e2e_checkpoint(family_case, tmp_path) -> None:
    family, wrapper, used_id = family_case

    # -- 1. load facts ------------------------------------------------------
    n_params = sum(p.numel() for p in wrapper.parameters())
    conversion = wrapper.capability_report.jevx_conversion
    assert conversion == "supported", (
        f"{used_id}: unexpected jevx_conversion={conversion!r} "
        f"(reasons: {wrapper.capability_report.reasons})"
    )

    state_kwargs = _state_kwargs(wrapper)
    questions = ["is this safe?", "which option wins?"]
    options = [["yes", "no"], ["alpha", "beta", "gamma", "delta", "epsilon"]]

    # -- 2. choice invariants ----------------------------------------------
    with torch.no_grad():
        choice_out = wrapper.choice(
            questions=questions, options=options, **state_kwargs
        )
    assert len(choice_out.decisions) == 2
    for dec, keys in zip(choice_out.decisions, options):
        probs = dec.probabilities
        k = len(keys)
        assert (probs >= 0).all()
        assert bool((probs[0, k:] == 0).all())  # variable-K mask: exact zeros
        assert float(probs[0, :k].sum()) == pytest.approx(1.0, abs=1e-6)
        assert 0 <= int(dec.predicted_index[0]) < k
        assert dec.predicted_value in keys
        assert 0.0 <= float(dec.confidence[0]) <= 1.0

    # -- 3. score bounds + noul range --------------------------------------
    with torch.no_grad():
        score_out = wrapper.score(
            questions=["rate it"], options=[["0", "1", "2", "3"]], **state_kwargs
        )
        noul_out = wrapper.noul(questions=["is it?"], **state_kwargs)
    es = float(score_out.decisions[0].expected_score[0])
    assert -1e-6 <= es <= 3 + 1e-6
    p_yes = float(noul_out.decisions[0].probabilities[0, 0])
    assert 0.0 < p_yes < 1.0

    # -- 4. save -> reload: identical decisions ----------------------------
    save_dir = tmp_path / f"jevx-{family}"
    with torch.no_grad():
        before = wrapper.choice(questions=questions, options=options, **state_kwargs)
    wrapper.save_pretrained(save_dir)
    reloaded = JevXForDecision.from_pretrained(save_dir)
    reloaded.eval()
    with torch.no_grad():
        after = reloaded.choice(questions=questions, options=options, **state_kwargs)
    save_load_max_abs_error = max(
        float((a.probabilities - b.probabilities).abs().max())
        for a, b in zip(before.decisions, after.decisions)
    )
    assert save_load_max_abs_error <= 1e-6, (
        f"{used_id}: save/load changed decisions by {save_load_max_abs_error:.3e}"
    )

    # -- 5. base_forward vs a freshly loaded raw model ---------------------
    raw_model = AutoModel.from_pretrained(used_id)
    raw_model.eval()
    inputs = _prepared_backbone_inputs(wrapper, state_kwargs)
    with torch.no_grad():
        wrapped_base = wrapper.base_forward(**inputs)
        raw_base = raw_model(**inputs)
    base_max_abs_error = _max_abs_error(wrapped_base, raw_base)
    assert base_max_abs_error < 1e-5, (
        f"{used_id}: base_forward diverged from the raw checkpoint by "
        f"{base_max_abs_error:.3e}"
    )

    # -- 6. shared-state: exactly ONE backbone forward for m=4 -------------
    extra = []
    if wrapper.capability_report.encoder_decoder and hasattr(wrapper.model, "encoder"):
        extra.append(wrapper.model.encoder)
    counter = _ForwardCounter([wrapper.model, *extra])
    try:
        m = 4
        with torch.no_grad():
            wrapper.choice(
                questions=[f"question {i}?" for i in range(m)],
                options=[[f"q{i}-a", f"q{i}-b"] for i in range(m)],
                **state_kwargs,
            )
    finally:
        counter.close()
    assert counter.calls == 1, (
        f"{used_id}: state featurization ran the backbone {counter.calls} times "
        "for m=4 (contract: exactly once)"
    )

    # -- 7. results table line ---------------------------------------------
    print(
        f"E2E RESULT | family={family} | id={used_id} | params={n_params} | "
        f"conversion={conversion} | load=ok | forward=ok | "
        f"save_load_max_abs_error={save_load_max_abs_error:.3e} | "
        f"base_forward_max_abs_error={base_max_abs_error:.3e} | "
        f"backbone_forward_calls={counter.calls}"
    )
