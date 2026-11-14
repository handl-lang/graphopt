"""PackedSequence custom_op wrapper for Kokoro — tts_walls_fixed.md option 1.

``torch.export`` dies inside ``pack_padded_sequence``'s fake-tensor
propagation (``ValueError: batch_sizes should always be on CPU``)
before any FX graph exists, so the three ``pack → lstm → pad`` sites
in ``kokoro/modules.py`` (``TextEncoder``, ``ProsodyPredictor``,
``DurationEncoder`` — all ``batch_first``, all discarding the hidden
state) are wrapped here as ``torch.library.custom_op`` leaves.  The
``PackedSequence`` pytree never crosses the IR boundary; each region
lifts as one ``opaque`` passthrough leaf and replays verbatim — the
same contract ``stft``/``istft`` already live under (laws cannot
reach inside).

Two deliberate, documented semantics notes:

* ``pad_packed_sequence`` is called with ``total_length = T`` (the
  packed input's time dim) instead of the default ``max(lengths)``.
  ``max(lengths)`` is *data-dependent* — an output shape torch.export
  cannot determine statically, which is half of why the raw call
  walls.  The extra tail positions are zeros, and every downstream
  consumer re-masks the tail to zero anyway (``masked_fill_`` /
  zero-``x_pad`` in all three sites), so the delivered values are
  identical; only the static shape is fixed.
* Each op closes over its ``nn.LSTM`` module (weights stay in the
  model's ``state_dict`` / ``param_values`` harvest; the delivered
  module replays the same module through the opaque leaf).  Op names
  carry a per-install counter so re-installing on a freshly loaded
  model never binds stale weights.

Usage::

    import tts_kokoro_packed
    tts_kokoro_packed.install(model)   # after probe ``load()``
"""

from __future__ import annotations

import itertools
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

_NS = "tts_packed"
_counter = itertools.count()
_INSTALLED_ATTR = "_tts_packed_installed"


def _register_lstm_op(name: str, lstm: nn.LSTM):
    """Register ``tts_packed::<name>(x, lengths) -> padded output``."""
    qual = f"{_NS}::{name}"
    if hasattr(getattr(torch.ops, _NS, None), name):
        return getattr(torch.ops.tts_packed, name)

    @torch.library.custom_op(qual, mutates_args=())
    def _op(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        lstm.flatten_parameters()
        packed = nn.utils.rnn.pack_padded_sequence(
            x,
            lengths.reshape(-1).to(device="cpu", dtype=torch.long),
            batch_first=True,
            enforce_sorted=False,
        )
        out, _ = lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(
            out, batch_first=True, total_length=x.shape[1]
        )
        return out

    @_op.register_fake
    def _fake(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        h = lstm.hidden_size * (2 if lstm.bidirectional else 1)
        return x.new_empty(x.shape[0], x.shape[1], h)

    return _op


def _op_call(name: str):
    """Return the ``torch.ops.tts_packed.<name>`` packet for tracing."""
    return getattr(torch.ops.tts_packed, name)


# ---------------------------------------------------------------------------
#  Patched forwards — verbatim copies of kokoro/modules.py with ONLY the
#  pack→lstm→pad triple replaced by the custom-op call.
# ---------------------------------------------------------------------------


def _text_encoder_forward(self, x, input_lengths, m):
    """``TextEncoder.forward`` (modules.py:50-69), triple → custom op."""
    x = self.embedding(x)  # [B, T, emb]
    x = x.transpose(1, 2)  # [B, emb, T]
    m = m.unsqueeze(1)
    x.masked_fill_(m, 0.0)
    for c in self.cnn:
        x = c(x)
        x.masked_fill_(m, 0.0)
    x = x.transpose(1, 2)  # [B, T, chn]
    lengths = (
        input_lengths
        if input_lengths.device == torch.device("cpu")
        else input_lengths.to("cpu")
    )
    x = _op_call(self._tts_packed_op)(x, lengths)
    x = x.transpose(-1, -2)
    x_pad = torch.zeros(
        [x.shape[0], x.shape[1], m.shape[-1]], device=x.device
    )
    x_pad[:, :, : x.shape[-1]] = x
    x = x_pad
    x.masked_fill_(m, 0.0)
    return x


def _prosody_predictor_forward(self, texts, style, text_lengths, alignment, m):
    """``ProsodyPredictor.forward`` (modules.py:109-122), triple → op."""
    d = self.text_encoder(texts, style, text_lengths, m)
    m = m.unsqueeze(1)
    lengths = (
        text_lengths
        if text_lengths.device == torch.device("cpu")
        else text_lengths.to("cpu")
    )
    x = _op_call(self._tts_packed_op)(d, lengths)
    x_pad = torch.zeros(
        [x.shape[0], m.shape[-1], x.shape[-1]], device=x.device
    )
    x_pad[:, : x.shape[1], :] = x
    x = x_pad
    duration = self.duration_proj(F.dropout(x, 0.5, training=False))
    en = d.transpose(-1, -2) @ alignment
    return duration.squeeze(-1), en


def _make_duration_encoder_forward(op_names):
    """``DurationEncoder.forward`` (modules.py:148-176), triples → ops."""

    def forward(self, x, style, text_lengths, m):
        masks = m
        x = x.permute(2, 0, 1)
        s = style.expand(x.shape[0], x.shape[1], -1)
        x = torch.cat([x, s], axis=-1)
        x.masked_fill_(masks.unsqueeze(-1).transpose(0, 1), 0.0)
        x = x.transpose(0, 1)
        x = x.transpose(-1, -2)
        j = 0
        for block in self.lstms:
            # Original dispatches on ``isinstance(block, AdaLayerNorm)``;
            # the class lives in kokoro.modules — match by MRO name so
            # this helper needs no kokoro import.
            if any(c.__name__ == "AdaLayerNorm" for c in type(block).__mro__):
                x = block(x.transpose(-1, -2), style).transpose(-1, -2)
                x = torch.cat([x, s.permute(1, 2, 0)], axis=1)
                x.masked_fill_(
                    masks.unsqueeze(-1).transpose(-1, -2), 0.0
                )
            else:
                lengths = (
                    text_lengths
                    if text_lengths.device == torch.device("cpu")
                    else text_lengths.to("cpu")
                )
                x = x.transpose(-1, -2)
                x = _op_call(op_names[j])(x, lengths)
                j += 1
                x = F.dropout(x, p=self.dropout, training=False)
                x = x.transpose(-1, -2)
                x_pad = torch.zeros(
                    [x.shape[0], x.shape[1], m.shape[-1]],
                    device=x.device,
                )
                x_pad[:, :, : x.shape[-1]] = x
                x = x_pad
        return x.transpose(-1, -2)

    return forward


def install(model) -> dict[str, int]:
    """Patch the three pack→lstm→pad sites on a loaded KModel.

    Returns ``{op_name: lstm_param_count}`` for logging.  Idempotent
    per model instance; safe to call before ``load``'s dtype move?  No —
    call AFTER ``.to(device/dtype)`` so op shapes match.
    """
    if getattr(model, _INSTALLED_ATTR, False):
        return {}
    report: dict[str, int] = {}

    # --- site 1: TextEncoder.lstm (modules.py:60-63) ---
    enc = model.text_encoder
    name = f"lstm_{next(_counter)}"
    _register_lstm_op(name, enc.lstm)
    enc._tts_packed_op = name
    enc.forward = types.MethodType(_text_encoder_forward, enc)
    report[name] = sum(p.numel() for p in enc.lstm.parameters())

    # --- site 2: ProsodyPredictor.lstm (modules.py:113-116) ---
    pred = model.predictor
    name = f"lstm_{next(_counter)}"
    _register_lstm_op(name, pred.lstm)
    pred._tts_packed_op = name
    pred.forward = types.MethodType(_prosody_predictor_forward, pred)
    report[name] = sum(p.numel() for p in pred.lstm.parameters())

    # --- site 3: DurationEncoder.lstms LSTM blocks (modules.py:164-169) ---
    dur = pred.text_encoder
    op_names = []
    for block in dur.lstms:
        if isinstance(block, nn.LSTM):
            name = f"lstm_{next(_counter)}"
            _register_lstm_op(name, block)
            op_names.append(name)
            report[name] = sum(p.numel() for p in block.parameters())
    dur.forward = types.MethodType(
        _make_duration_encoder_forward(op_names), dur
    )

    setattr(model, _INSTALLED_ATTR, True)
    return report
