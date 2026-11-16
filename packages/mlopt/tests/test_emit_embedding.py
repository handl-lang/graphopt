"""``embedding`` emit coverage — the decoder-family gap.

``arch:whisper-tiny-decoder``/``bert-minilm-l6`` (and every
decoder-family arch spec) lift ``embedding(weight, ids)``; the host
``tensor.*`` surface has no ``tensor.embedding``, so the builder
spells the row gather as ``tensor.index_select(weight, 0, ids)`` —
correct for the rank-1 (flattened) index operand every shipped term
carries.  These tests pin the spelling and the honest declines.
"""

from __future__ import annotations

import pytest
from mlopt_core.ir import Op, Param, TensorType, Var

from mlopt.emit import UnsupportedOpError, emit

T = TensorType


def _w() -> Param:
    return Param("tok_embeddings", T((1000, 384)))


def _ids() -> Var:
    return Var("decoder_input_ids", T((64,)))


def test_emit_embedding_is_index_select_row_gather() -> None:
    term = Op.make("embedding", _w(), _ids())
    assert emit(term) == "tensor.index_select(tok_embeddings, 0, decoder_input_ids)"


def test_emit_embedding_inside_reshape_restores_batch() -> None:
    # The shipped spelling: flattened ids in, parent reshape back to
    # (batch, seq, dim) — index_select's [N, D] output feeds it.
    # Typed Vars let emit resolve the -1 wildcards statically: the
    # flat ids (64,) reshape to [64], index_select gathers 64 rows of
    # the (1000, 384) table → numel 64*384, and the parent restore
    # resolves -1 to 24576 / (1*32) = 768.
    flat = Op.make("reshape", _ids(), shape=(-1,))
    emb = Op.make("embedding", _w(), flat)
    out = Op.make("reshape", emb, shape=(1, 32, -1))
    assert emit(out) == (
        "tensor.reshape("
        "tensor.index_select(tok_embeddings, 0, "
        "tensor.reshape(decoder_input_ids, [64])), "
        "[1, 32, 768])"
    )


def test_emit_embedding_missing_indices_declines() -> None:
    with pytest.raises(UnsupportedOpError, match="embedding"):
        emit(Op.make("embedding", _w()))


def test_emit_embedding_padding_idx_declines() -> None:
    # padding_idx's zero-fill is only forward-equal while the table
    # row is already zeros — the gather cannot spell it, so decline.
    with pytest.raises(UnsupportedOpError, match="padding_idx"):
        emit(Op.make("embedding", _w(), _ids(), padding_idx=0))
    with pytest.raises(UnsupportedOpError, match="padding_idx"):
        emit(Op.make("embedding", _w(), _ids(), 0))
