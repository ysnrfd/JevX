# Copyright (c) 2026 ysnrfd
"""Adapter for encoder-decoder stacks (t5/bart/whisper/...).

# justification: generic extraction *fails to pick the right state* here
# without a mode: ``Seq2SeqModelOutput.last_hidden_state`` is the DECODER
# state while ``encoder_last_hidden_state`` is the encoder's (audit Q6), and
# the v5 base stacks refuse a decoderless forward — ``T5Model.forward``
# raises ``"You must specify exactly one of input_ids or inputs_embeds"``
# unless ``decoder_input_ids`` is supplied (probe in this environment; the
# ``ForConditionalGeneration`` heads shift internally). Three modes:

* ``"encoder_state"`` (default) — call ``model.get_encoder()`` (the
  generic ``PreTrainedModel.get_encoder`` helper, audit §1.2) directly;
  audit Q6 verified ``model.encoder(...)`` is conventional and *equal* to
  the full model's ``encoder_last_hidden_state`` in ``eval()``.
* ``"decoder_state"`` — full forward; ``last_hidden_state`` is the decoder
  sequence; decoder-side mask preferred.
* ``"encoder_decoder_fusion"`` — full forward; pooled vector
  ``H = ½ (P(E_enc) + P(E_dec))`` (documented elementwise mean of the two
  pooled sides); the encoder sequence is kept as ``sequence_output`` and the
  full output stays in ``raw_output``.
"""

from __future__ import annotations

import inspect
from typing import Any, Dict, Optional

import torch

from jevx.capabilities import UnsupportedRepresentationError
from jevx.representations.pooling import pool

from .base import Representation, call_backbone, inject_no_cache, observed_output_fields
from .generic import GenericAdapter

__all__ = ["EncoderDecoderAdapter", "ENC_DECODER_MODES"]

ENC_DECODER_MODES: tuple[str, ...] = (
    "encoder_state",
    "decoder_state",
    "encoder_decoder_fusion",
)


class EncoderDecoderAdapter(GenericAdapter):
    """Encoder-decoder models; ``extract`` mode selects which side to read."""

    name = "encoder_decoder"
    default_pooling = "encoder_pool"  # documented alias of masked_mean (P4.a)

    def can_handle(self, model: Any) -> bool:
        """``config.is_encoder_decoder`` (audit Q6 families: t5/bart/whisper)."""
        return bool(self.describe_capabilities(model).encoder_decoder)

    def prepare_inputs(
        self,
        state: Any,
        *,
        tokenizer: Any = None,
        processor: Any = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Prepare inputs and default ``decoder_input_ids`` for base stacks.

        When the forward declares ``decoder_input_ids`` and it is absent, the
        encoder ``input_ids`` are mirrored onto it (probe: ``T5Model`` needs
        *some* decoder input for a full forward; encoder-side extraction
        simply drops the extra key when calling ``model.encoder``).
        """
        inputs = super().prepare_inputs(state, tokenizer=tokenizer, processor=processor, **kwargs)
        model = kwargs.get("model")
        if model is not None and "decoder_input_ids" not in inputs:
            try:
                params = set(inspect.signature(model.forward).parameters)
            except (TypeError, ValueError):  # pragma: no cover - uninspectable
                params = set()
            if "decoder_input_ids" in params:
                if "input_ids" in inputs:
                    # Shared vocabulary (t5/bart): mirror the encoder tokens,
                    # the standard encoder-decoder prompt.
                    inputs["decoder_input_ids"] = inputs["input_ids"]
                else:
                    # Encoder-side modality WITHOUT a shared vocabulary
                    # (whisper: input_features is a mel spectrogram): prompt
                    # the decoder with the config's decoder_start_token_id —
                    # what the ForConditionalGeneration heads construct
                    # internally before calling the base stack.
                    config = getattr(model, "config", None)
                    start = getattr(config, "decoder_start_token_id", None)
                    vocab = getattr(config, "vocab_size", None)
                    encoder_input = next(
                        (
                            inputs[key]
                            for key in ("input_features", "pixel_values")
                            if isinstance(inputs.get(key), torch.Tensor)
                            and inputs[key].dim() >= 1
                        ),
                        None,
                    )
                    if (
                        encoder_input is not None
                        and isinstance(start, int)
                        and start >= 0
                        and (not isinstance(vocab, int) or start < vocab)
                    ):
                        batch = int(encoder_input.shape[0])
                        inputs["decoder_input_ids"] = torch.full(
                            (batch, 1), int(start), dtype=torch.long
                        )
        return inputs

    # -- extraction modes ----------------------------------------------------

    def extract_representation(
        self,
        model: Any,
        inputs: Dict[str, Any],
        *,
        strategy: Optional[str] = None,
        output_hidden_states: bool = False,
        mode: Optional[str] = None,
    ) -> Representation:
        """Extract per ``mode`` (default ``"encoder_state"``).

        Raises:
            ValueError: Unknown ``mode`` (message lists the valid modes).
        """
        mode = mode or "encoder_state"
        if mode not in ENC_DECODER_MODES:
            raise ValueError(
                f"unknown encoder-decoder mode {mode!r}; expected one of "
                f"{list(ENC_DECODER_MODES)}."
            )
        if mode == "encoder_state":
            return self._extract_encoder_state(model, inputs, strategy, output_hidden_states)
        if mode == "encoder_decoder_fusion":
            return self._extract_encoder_decoder_fusion(model, inputs, strategy, output_hidden_states)
        return super().extract_representation(
            model, inputs, strategy=strategy, output_hidden_states=output_hidden_states, mode=mode
        )

    def _extract_encoder_state(
        self,
        model: Any,
        inputs: Dict[str, Any],
        strategy: Optional[str],
        output_hidden_states: bool,
    ) -> Representation:
        """Direct encoder call: ``field='encoder_last_hidden_state'``."""
        report = self.describe_capabilities(model)
        encoder = None
        get_encoder = getattr(model, "get_encoder", None)
        if callable(get_encoder):
            try:
                encoder = get_encoder()
            except (AttributeError, ValueError):  # pragma: no cover - odd wrappers
                encoder = None
        if encoder is None:
            encoder = getattr(model, "encoder", None)
        if encoder is None:
            # Fall back to the full forward and read the named field.
            outputs = self._run_full(model, inputs, output_hidden_states)
            sequence = outputs.get("encoder_last_hidden_state") if isinstance(outputs, dict) else None
            if sequence is None:
                self._raise_unsupported(
                    model,
                    report,
                    observed_output_fields(outputs),
                    attempted=["model.get_encoder()", "encoder submodule", "encoder_last_hidden_state"],
                    failed=["encoder_last_hidden_state: absent from full-forward output"],
                    suggestion="model exposes no encoder route — check config.is_encoder_decoder.",
                )
        else:
            # Shared keyword contract (audit Q-014): the encoder side never
            # needs a KV cache — inject use_cache=False via the one helper
            # (it was a per-path pop, i.e. the same gate in a second spelling).
            encoder_kwargs = inject_no_cache(inputs, encoder)
            if output_hidden_states:
                encoder_kwargs["output_hidden_states"] = True
            try:
                outputs = encoder(**encoder_kwargs)
            except TypeError as exc:
                self._raise_unsupported(
                    model,
                    report,
                    observed_output_fields("<encoder call failed>"),
                    attempted=["model.get_encoder()(**inputs)"],
                    failed=[f"encoder forward: {exc}"],
                    suggestion=(
                        "encoder rejected the prepared inputs — inspect "
                        "encoder.forward's signature (decoder-only keys are filtered; "
                        "remaining keys must be encoder inputs)."
                    ),
                    cause=exc,
                )
            sequence = (
                outputs.get("last_hidden_state") if isinstance(outputs, dict) else getattr(
                    outputs, "last_hidden_state", None
                )
            )
            if sequence is None:
                self._raise_unsupported(
                    model,
                    report,
                    observed_output_fields(outputs),
                    attempted=["encoder last_hidden_state"],
                    failed=["encoder output has no last_hidden_state"],
                    suggestion="unexpected encoder output class — report this model.",
                )
        mask = inputs.get("attention_mask")
        if not isinstance(mask, torch.Tensor) or tuple(mask.shape) != tuple(sequence.shape[:2]):
            mask = None
        effective = strategy if strategy not in (None, "native") else self.default_pooling
        if effective == "native":
            effective = self.default_pooling
        pooled = pool(sequence, mask, effective)
        return Representation(
            sequence_output=sequence,
            pooled=pooled,
            attention_mask=mask,
            raw_output=outputs,
            field="encoder_last_hidden_state",
            layout="btd",
        )

    def _extract_encoder_decoder_fusion(
        self,
        model: Any,
        inputs: Dict[str, Any],
        strategy: Optional[str],
        output_hidden_states: bool,
    ) -> Representation:
        """``H = ½ (P(E_enc) + P(E_dec))`` with the encoder sequence kept."""
        report = self.describe_capabilities(model)
        outputs = self._run_full(model, inputs, output_hidden_states)
        enc_seq = outputs.get("encoder_last_hidden_state") if isinstance(outputs, dict) else None
        dec_seq = outputs.get("last_hidden_state") if isinstance(outputs, dict) else None
        if enc_seq is None or dec_seq is None:
            self._raise_unsupported(
                model,
                report,
                observed_output_fields(outputs),
                attempted=["encoder_last_hidden_state + last_hidden_state (fusion)"],
                failed=[f"missing side(s): enc={enc_seq is not None}, dec={dec_seq is not None}"],
                suggestion="encoder-decoder fusion needs both sides — check the output class.",
            )
        enc_mask = inputs.get("attention_mask")
        dec_mask = inputs.get("decoder_attention_mask", enc_mask)
        if not isinstance(enc_mask, torch.Tensor) or tuple(enc_mask.shape) != tuple(
            enc_seq.shape[:2]
        ):
            enc_mask = None
        if not isinstance(dec_mask, torch.Tensor) or tuple(dec_mask.shape) != tuple(
            dec_seq.shape[:2]
        ):
            dec_mask = None
        effective = strategy if strategy not in (None, "native") else self.default_pooling
        if effective == "native":
            effective = self.default_pooling
        pooled = 0.5 * (pool(enc_seq, enc_mask, effective) + pool(dec_seq, dec_mask, effective))
        return Representation(
            sequence_output=enc_seq,
            pooled=pooled,
            attention_mask=enc_mask,
            raw_output=outputs,
            field="encoder_decoder_fusion",
            layout="btd",
        )

    @staticmethod
    def _run_full(
        model: Any,
        inputs: Dict[str, Any],
        output_hidden_states: bool,
    ) -> Any:
        """Full forward with the shared keyword rules (use_cache/hidden states).

        The use_cache gate is :func:`inject_no_cache` — audit Q-014 removed
        the old DECLARED-PARAMS-only check (``if "use_cache" in params``),
        which left kwargs-only v5 forwards allocating DynamicCaches in the
        ``decoder_state`` / ``encoder_decoder_fusion`` modes; the call runs
        through :func:`call_backbone` for JevX-context error wrapping (Q-013).
        """
        fwd = inject_no_cache(inputs, model)
        if output_hidden_states:
            fwd["output_hidden_states"] = True
        return call_backbone(model, fwd)
