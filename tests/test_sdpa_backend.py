"""Tests for the ``sdpa`` binding's decode-shape backend policy.

``torch_bridge._sdpa_math_backend`` pins ``SDPBackend.MATH`` for the
measured pathological regime — fp32/fp64, decode-like short query
length over a long key axis — where the default pick (mem-efficient
fmha under fp32+mask) runs ~2.4-4.7x slower
(``docs/reports/decode_step_fusion.md`` §4).  Prefill-shaped and
reduced-precision calls keep PyTorch's default backend choice.
"""

import torch
import torch.nn.functional as F
from mlopt_torch.torch_bridge import (
    _CORE_TORCH_BINDINGS,
    _sdpa_math_backend,
    _sdpa_torch,
)
from torch.nn.attention import SDPBackend

torch.manual_seed(0)


def _qk(shape_q, shape_k, dtype=torch.float32):
    q = torch.randn(*shape_q, dtype=dtype)
    k = torch.randn(*shape_k, dtype=dtype)
    v = torch.randn(*shape_k, dtype=dtype)
    return q, k, v


class TestBackendPick:
    def test_decode_fp32_pins_math(self):
        q, k, _ = _qk((1, 8, 1, 64), (1, 8, 1500, 64))
        assert _sdpa_math_backend(q, k) is SDPBackend.MATH

    def test_q_len_boundary(self):
        k = torch.randn(1, 8, 1500, 64)
        for qlen in (1, 8):
            q = torch.randn(1, 8, qlen, 64)
            assert _sdpa_math_backend(q, k) is SDPBackend.MATH
        q = torch.randn(1, 8, 9, 64)
        assert _sdpa_math_backend(q, k) is None

    def test_kv_len_boundary(self):
        q = torch.randn(1, 8, 1, 64)
        for kv in (256, 1500):
            k = torch.randn(1, 8, kv, 64)
            assert _sdpa_math_backend(q, k) is SDPBackend.MATH
        k = torch.randn(1, 8, 255, 64)
        assert _sdpa_math_backend(q, k) is None

    def test_reduced_precision_keeps_default(self):
        for dtype in (torch.float16, torch.bfloat16):
            q, k, _ = _qk((1, 8, 1, 64), (1, 8, 1500, 64), dtype=dtype)
            assert _sdpa_math_backend(q, k) is None

    def test_fp64_pins_math(self):
        q, k, _ = _qk(
            (1, 8, 1, 64), (1, 8, 1500, 64), dtype=torch.float64
        )
        assert _sdpa_math_backend(q, k) is SDPBackend.MATH

    def test_prefill_keeps_default(self):
        # Long-query (prefill) shapes keep the default pick — mem-
        # efficient wins there and math would materialise L*S scores.
        q, k, _ = _qk((1, 8, 1500, 64), (1, 8, 1500, 64))
        assert _sdpa_math_backend(q, k) is None


class TestSdpaTorch:
    def test_decode_shape_matches_reference(self):
        q, k, v = _qk((1, 8, 1, 64), (1, 8, 1500, 64))
        mask = torch.ones(1, 1500, dtype=torch.bool)
        out = _sdpa_torch(q, k, v, mask)
        ref = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        assert torch.allclose(out, ref, atol=1e-5)

    def test_decode_shape_unmasked(self):
        q, k, v = _qk((1, 8, 1, 64), (1, 8, 300, 64))
        out = _sdpa_torch(q, k, v)
        ref = F.scaled_dot_product_attention(q, k, v)
        assert torch.allclose(out, ref, atol=1e-5)

    def test_prefill_shape_matches_reference(self):
        q, k, v = _qk((1, 8, 64, 64), (1, 8, 1500, 64))
        mask = torch.ones(64, 1500, dtype=torch.bool)
        out = _sdpa_torch(q, k, v, attn_mask=mask)
        ref = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        assert torch.allclose(out, ref, atol=1e-5)

    def test_argN_attr_remap(self):
        # arg3→attn_mask, arg4→dropout_p, arg5→is_causal,
        # arg6→scale, arg7→enable_gqa; None values are filtered.
        q, k, v = _qk((1, 2, 1, 64), (1, 1, 300, 64))
        mask = torch.ones(1, 300, dtype=torch.bool)
        out = _sdpa_torch(
            q,
            k,
            v,
            mask,
            arg4=0.0,
            arg5=False,
            arg6=0.5,
            arg7=True,
            scale=None,  # filtered by the `vv is not None` guard
        )
        ref = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=0.0,
            is_causal=False,
            scale=0.5,
            enable_gqa=True,
        )
        assert torch.allclose(out, ref, atol=1e-5)

    def test_canonical_kwargs(self):
        q, k, v = _qk((1, 8, 9, 64), (1, 8, 41, 64))
        mask = torch.ones(9, 41, dtype=torch.bool)
        out = _sdpa_torch(q, k, v, attn_mask=mask, scale=0.25)
        ref = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, scale=0.25
        )
        assert torch.allclose(out, ref, atol=1e-5)

    def test_binding_table_entry(self):
        assert _CORE_TORCH_BINDINGS["sdpa"] is _sdpa_torch
