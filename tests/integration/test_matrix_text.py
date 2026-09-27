# Copyright (c) 2026 ysnrfd
"""REAL pretrained validation matrix — TEXT families (Phases 20/46/70/92).

Every checkpoint here is HF-API verified (``vendor/reports/_hf_verified_ids.json``,
commit sha resolved in ``vendor/reports/_matrix_env_probe.json``) and executed
end-to-end through :func:`matrix_common.execute_matrix_case`: real sentences →
invariants → save/load equality → base_forward vs raw ``AutoModel`` → shared-
state single forward (m=4) → §71 extraction vs an independent manual reference.

Families covered (the bert/t5/gpt2 rows are ALSO executed by
``test_jevx_e2e_tiny.py`` — cited in ``reports/REAL_MODEL_VALIDATION.md`` — and
re-executed here so every row shares one harness):

* bert (encoder, native pooler), roberta, albert, distilbert, deberta-v2
* t5 + bart (encoder–decoder)
* gpt2 + qwen2 + llama (decoder-only)
* all-MiniLM-L6-v2 (semantic/embedding: REAL trained sentence-embedding
  checkpoint, wrapped with ``pooling_method="masked_mean"`` — the
  sentence-transformers pooling convention)

Network/HTTP failures skip with the exact attempts (never fail the suite on an
unavailable download); implementation bugs re-raise and fail.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import pytest

from matrix_common import (
    _ENV_MARKERS,  # noqa: F401 - re-exported marker tuple (documented)
    execute_matrix_case,
    is_environment_error,
    load_wrapper,
)

#: family -> (model_id, state kind, extra JevX.from_pretrained kwargs, notes)
TEXT_FAMILIES: Dict[str, Tuple[str, str, Dict[str, Any], str]] = {
    "bert": (
        "hf-internal-testing/tiny-random-bert",
        "text",
        {},
        "encoder; native pooler_output; also executed by test_jevx_e2e_tiny.py",
    ),
    "roberta": (
        "hf-internal-testing/tiny-random-RobertaModel",
        "text",
        {},
        "roberta-family row",
    ),
    "distilbert": (
        "hf-internal-testing/tiny-random-DistilBertModel",
        "text",
        {},
        "no pooler → CLS over bidirectional states (adapter default)",
    ),
    "albert": (
        "hf-internal-testing/tiny-random-AlbertModel",
        "text",
        {},
        "ALBERT row (cross-checked vs google/albert-base-v2 401 alternate)",
    ),
    "deberta-v2": (
        "hf-internal-testing/tiny-random-DebertaV2Model",
        "text",
        {},
        "DeBERTa tiny; base model has no pooler → CLS",
    ),
    "t5": (
        "hf-internal-testing/tiny-random-t5",
        "text",
        {},
        "encoder-decoder; encoder_state extraction; also test_jevx_e2e_tiny.py",
    ),
    "bart": (
        "hf-internal-testing/tiny-random-BartForConditionalGeneration",
        "text",
        {},
        "encoder-decoder second family",
    ),
    "gpt2": (
        "sshleifer/tiny-gpt2",
        "text",
        {},
        "decoder-only; last_valid pooling; also test_jevx_e2e_tiny.py",
    ),
    "qwen2": (
        "yujiepan/qwen2-tiny-random",
        "text",
        {},
        "modern decoder LLM family (verified alternate; hf-internal-testing id 401s)",
    ),
    "llama": (
        "hf-internal-testing/tiny-random-LlamaForCausalLM",
        "text",
        {},
        "decoder-only second family",
    ),
    "semantic-minilm": (
        "sentence-transformers/all-MiniLM-L6-v2",
        "text",
        {"pooling_method": "masked_mean"},
        "FUNCTIONALLY validated (real trained sentence-embedding checkpoint, "
        "sentence-transformers mean-pooling convention pinned); NOT quality-"
        "validated — quality metrics owned by the benchmark agent",
    ),
}


@pytest.fixture(scope="module", params=sorted(TEXT_FAMILIES))
def text_case(request) -> Tuple[str, Any, str, float]:
    """(family, wrapper, used_model_id, load_seconds) — one load per family."""
    family = request.param
    model_id, _kind, kwargs, _notes = TEXT_FAMILIES[family]
    wrapper, load_seconds = load_wrapper(model_id, **kwargs)
    return family, wrapper, model_id, load_seconds


def test_matrix_text_family(text_case, tmp_path) -> None:
    family, wrapper, model_id, load_seconds = text_case
    _kind, kwargs_note = TEXT_FAMILIES[family][1], TEXT_FAMILIES[family][3]
    execute_matrix_case(
        family,
        model_id,
        "text",
        wrapper,
        load_seconds,
        tmp_path,
        test_file="tests/integration/test_matrix_text.py",
        notes=kwargs_note,
    )
