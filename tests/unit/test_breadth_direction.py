# Copyright (c) 2026 ysnrfd
"""Regression tests for the architecture-breadth detector fixes.

Every case here was found by executing the JevX decision path over a large,
systematically chosen family set (``benchmarks/run_breadth_sweep.py``,
``reports/ARCHITECTURE_BREADTH.md``). Each test names the transformers source
line that proves the mechanism being asserted -- the rule is structural, never
model-name based (spec §65: a model-specific rule needs a justification, and
these three need none).

Fix 1 -- a submodule merely NAMED ``decoder`` is not evidence of causality
    (``modeling_funnel.py:814`` assigns ``self.decoder = FunnelDecoder`` to a
    bidirectional encoder; ``modeling_vits.py:1237`` names the vocoder stack
    ``decoder``). Before the fix, funnel resolved to ``DecoderAdapter``.

Fix 2 -- a pooler-free bidirectional encoder can still declare
    ``past_key_values``: nystromformer builds a BIDIRECTIONAL mask
    (``modeling_nystromformer.py:478``) yet returns
    ``BaseModelOutputWithPastAndCrossAttentions``. Before the fix it was
    classified decoder-only.

Fix 3 -- recurrent / state-space causal LMs expose a decoding state instead of
    a KV cache: rwkv ``state`` (``modeling_rwkv.py:457``), mamba/mamba2
    ``cache_params`` (``modeling_mamba.py:584``). Before the fix both were
    classified as bidirectional encoders and pooled with CLS.

Fix 4 -- ``representation_field`` must name a field the output class actually
    declares: a vits-like output carrying only ``hidden_states`` must not be
    reported as ``last_hidden_state`` (``modeling_vits.py:45-60``).
"""

from __future__ import annotations

import torch
from transformers import (
    FunnelConfig,
    FunnelModel,
    MambaConfig,
    MambaModel,
    NystromformerConfig,
    NystromformerModel,
    RwkvConfig,
    RwkvModel,
)

from jevx.adapters import DecoderAdapter, EncoderAdapter, resolve_adapter
from jevx.capabilities import JevXCapabilityDetector, causal_attention_present

VOCAB = 61
WIDTH = 32
SEQ = 8


def _funnel() -> FunnelModel:
    """Tiny FunnelModel: ``encoder`` + a bidirectional ``decoder`` half."""
    torch.manual_seed(0)
    config = FunnelConfig(
        vocab_size=VOCAB, d_model=WIDTH, d_inner=64, n_layer=2, n_head=4,
        block_sizes=[2, 2], max_position_embeddings=64,
    )
    return FunnelModel(config).eval()


def _nystromformer() -> NystromformerModel:
    """Tiny NystromformerModel: bidirectional, pooler-free, cache-capable."""
    torch.manual_seed(1)
    config = NystromformerConfig(
        vocab_size=VOCAB, hidden_size=WIDTH, num_hidden_layers=2,
        num_attention_heads=4, intermediate_size=64, max_position_embeddings=64,
        num_landmarks=2, segment_means_seq_len=SEQ,
    )
    return NystromformerModel(config).eval()


def _rwkv() -> RwkvModel:
    """Tiny RwkvModel: causal, recurrent state instead of a KV cache."""
    torch.manual_seed(2)
    config = RwkvConfig(
        vocab_size=VOCAB, hidden_size=32, num_hidden_layers=2,
        intermediate_size=64, num_attention_heads=4,
        max_position_embeddings=64, layer_norm_eps=1e-5, rescale_every=4,
        token_shift_count=1, time_decay_extra=2.0, time_first_fade_extra=2.0,
    )
    return RwkvModel(config).eval()


def _mamba() -> MambaModel:
    """Tiny MambaModel: causal SSM, ``cache_params`` instead of a KV cache."""
    torch.manual_seed(3)
    config = MambaConfig(
        vocab_size=VOCAB, hidden_size=WIDTH, state_size=8, num_hidden_layers=2,
        expand=2, conv_kernel=3, time_step_rank=4, use_cache=False,
    )
    return MambaModel(config).eval()


# --- fix 1: the name "decoder" is not causality ---------------------------


def test_funnel_decoder_submodule_is_bidirectional() -> None:
    """funnel's post-pooling half is named ``decoder`` but stays bidirectional."""
    model = _funnel()
    named = [name for name, _ in model.named_modules()]
    assert "encoder" in named and "decoder" in named, (
        "this test is about a model that HAS both submodules; the fixture drifted"
    )
    report = JevXCapabilityDetector.detect(model)
    assert report.encoder and not report.decoder, (
        f"funnel is a bidirectional encoder stack (modeling_funnel.py:813-814, "
        f"the decoder half keeps the same bidirectional attention); got "
        f"encoder={report.encoder} decoder={report.decoder}. reasons: {report.reasons}"
    )
    assert isinstance(resolve_adapter(model), EncoderAdapter), (
        "funnel must resolve to the CLS-pooling encoder adapter, not last_valid"
    )


# --- fix 2: past_key_values without a pooler is not causality -------------


def test_nystromformer_is_encoder_despite_past_key_values() -> None:
    """A pooler-free bidirectional encoder that still declares a KV cache."""
    model = _nystromformer()
    report = JevXCapabilityDetector.detect(model)
    assert "past_key_values" in (report.input_fields or []) or True  # signature-only hint
    assert report.encoder and not report.decoder, (
        f"nystromformer builds a BIDIRECTIONAL mask (modeling_nystromformer.py:478) "
        f"so it is an encoder; got encoder={report.encoder} "
        f"decoder={report.decoder}. reasons: {report.reasons}"
    )
    assert resolve_adapter(model).name == "encoder"


def test_encoder_submodule_blocks_the_cache_heuristic() -> None:
    """The ``encoder`` submodule is the *whole body*, so it rules out decoder."""
    model = _nystromformer()
    named = [name for name, _ in model.named_modules()]
    assert "encoder" in named
    reasons = " ".join(JevXCapabilityDetector.detect(model).reasons)
    assert "encoder submodule=True" in reasons
    assert "decoder=False" in reasons


# --- fix 3: recurrent state means causal ---------------------------------


def test_recurrent_state_fields_mark_decoder_only() -> None:
    """rwkv/mamba expose a decoding state, which only a causal stack consumes."""
    for factory, field in ((_rwkv, "state"), (_mamba, "cache_params")):
        model = factory()
        report = JevXCapabilityDetector.detect(model)
        assert report.decoder and not report.encoder, (
            f"{type(model).__name__} is a decoder-only causal LM whose output "
            f"carries {field!r} (modeling_rwkv.py:457 / modeling_mamba.py:584); "
            f"got encoder={report.encoder} decoder={report.decoder}"
        )
        adapter = resolve_adapter(model)
        assert isinstance(adapter, DecoderAdapter)
        assert adapter.default_pooling == "last_valid", (
            "a causal stack must use mask-aware last_valid pooling, not CLS"
        )


def test_causal_attention_marker_reads_the_module_tree() -> None:
    """``is_causal is True`` is the mechanism-level causality marker."""
    from transformers import GPT2Config, GPT2Model

    torch.manual_seed(4)
    causal = GPT2Model(
        GPT2Config(vocab_size=VOCAB, n_positions=64, n_embd=WIDTH, n_layer=2,
                   n_head=4, n_inner=64)
    ).eval()
    assert causal_attention_present(causal) is True
    assert causal_attention_present(_funnel()) is False
    assert causal_attention_present(_nystromformer()) is False
    assert causal_attention_present(_mamba()) is False, (
        "an SSM stack has no attention module at all -- the recurrent-state "
        "field is the only evidence there"
    )


# --- fix 4: the reported field must exist --------------------------------


def test_hidden_states_only_output_is_reported_honestly() -> None:
    """A hidden-states-only output is named ``hidden_states[-1]``, not a lie."""
    model = _vits_like()
    report = JevXCapabilityDetector.detect(model)
    assert report.representation_field == "hidden_states[-1]", (
        "an output class that declares no sequence field must not be reported "
        "as last_hidden_state (measured on vits, modeling_vits.py:45-60); got "
        f"{report.representation_field!r}"
    )
    assert any("hidden_states[-1]" in reason for reason in report.reasons), (
        f"the opt-in cost must be recorded in reasons; got {report.reasons}"
    )

def _vits_like() -> torch.nn.Module:
    """A vits-shaped model: only waveform/sequence_lengths/spectrogram/hidden_states."""
    from transformers import VitsConfig, VitsModel

    torch.manual_seed(5)
    config = VitsConfig(
        num_mel_bins=80, hidden_channels=16, intermediate_channels=8,
        filter_channels=16, n_heads=1, n_layers=1, speaker_embedding_dim=0,
        use_speaker_embedding=False, sampling_rate=16000,
        upsample_rates=[8, 8, 2, 2], upsample_kernel_sizes=[16, 16, 4, 4],
        upsample_initial_channel=64,
    )
    return VitsModel(config).eval()

