"""Held-out model zoo — architectures the intake corpus never saw.

``project/retros/real-corpus-yield.md`` measured ``shippable = 0``
on the real half of the intake corpus, and the remaining signal was
corpus-circular: every shipped rule's firing site is a purpose-built
spelling.  The question that retro could not answer is the honest
one — *does anything fire on architectures nobody targeted?*  This
module is the holdout set for that question: a registry of
``nn.Module`` workloads from architecture families the census has
never contained, kept deliberately OUT of
:func:`mlopt_discovery.intake.candidates`
(``tests/test_discovery_zoo.py`` pins the disjointness).  Nothing
here was written to spell a known law's LHS or to fill a guard's
truth region — the selection criterion was *family coverage*:

* attention variants the corpus lacks — additive (Bahdanau),
  cosine-normalized (SwinV2-style), talking-heads mixing, and the
  softmax-free gated attention unit (GAU);
* conditioning spellings — FiLM and DiT's AdaLN (the real
  ``mul(unsqueeze, ·)`` sites, not the purpose-built corner);
* token mixers — MLP-Mixer, a Fourier neural operator block, GCN;
* routing — soft-MoE slot dispatch, expert-choice top-k, capsule
  routing;
* recurrence/scan — chunked RetNet retention, a delta-rule
  unrolled matrix-state recurrence (DeltaNet — its chunked form
  is not in the law library), a deep-equilibrium unroll, WaveNet
  gated dilated conv;
* SSM/linear-attention — the unrolled Mamba selective scan
  (diagonal decay + selective rank-1 write), GLA gated linear
  attention, an RWKV-ish token-shift/wkv block, and a Hyena
  generated-kernel long conv;
* adapters/heads — LoRA, mixture-of-softmaxes, a VICReg-style
  covariance head, a KAN basis layer, Highway gates, RealNVP;
* deployment spellings — a top-k router MoE with the load-balance
  aux folded in, a sigmoid-routed shared+routed bank, STE
  fake-quant and per-channel dequant matmuls, the KV-cache
  decode step;
* modern block details — GEGLU (the gated-MLP sibling the intake
  lacks), SkipInit scalar-gated residuals, the soft-capped logit
  head, and the interleaved-pair (GPT-J) RoPE spelling;
* the second expansion — patterned attention (sliding-window band,
  dilated selection, block-sparse top-k), the depthwise-conv
  stack (ConvNeXt, squeeze-excite, inverted-bottleneck MBConv),
  fused-projection gates (GRU cell, conv GLU), manual norm
  spellings (GroupNorm, InstanceNorm, per-head RMSNorm) plus the
  norm-free dynamic tanh, positional-table ops (ALiBi slope grid,
  clamped relative-position gather, learned abs-emb slice-add),
  and the grouped-quant / paged-KV deployment shapes — picked
  because their exported graphs carry the pointwise-over-view and
  same-view op-tuples the ``zoo-real`` mint reads.
* the third expansion — post-2024 structures the rewrite catalogs
  (TASO/Inductor/XLA fusion banks) have not caught: the gated
  delta rule and the chunked SSD scan (the Mamba-2 dual form),
  decoupled-rope MLA and partial rotary, sigmoid and
  differential attention, the Griffin RG-LRU recurrence, FiLM'd
  conv conditioning, a banked SwiGLU MoE under a shared expert,
  attention- and final-logit soft-capping, the squared-relu FFN,
  Taylor-feature linear attention, ternary-weight dequant, and
  sink+window masking — the substrate where originals could
  live precisely because no prior-art bank covers them.

The registry reuses :class:`mlopt_discovery.intake.Workload` so
:func:`mlopt_discovery.intake.ingest` consumes it unchanged — the
holdout is a *provenance* property (the zoo is never registered in
``candidates()`` and never written to the cache census (``mlopt_discovery.intake_dir()``)
side-file), not a different ingestion path.  ``kind="zoo"`` keeps
the provenance visible in every report table and ``fire_cases`` name.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mlopt_discovery.intake import Workload

__all__ = ["zoo", "zoo_variants"]


# ---------------------------------------------------------------------------
#  Attention variants the corpus lacks
# ---------------------------------------------------------------------------


class _GatedAttentionUnit(nn.Module):
    """GAU (FLASH) — softmax-free gated attention on squared scores.

    Hua et al. 2022: two cheap projections, a quadratic
    self-similarity score (``relu(u uᵀ)²``), and an output gate —
    no softmax anywhere.
    """

    def __init__(self, d: int) -> None:
        """Build the u/v projections and the output gate."""
        super().__init__()
        self.u = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.gate = nn.Linear(d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``sigmoid(gate(x)) ⊙ (relu(u uᵀ)² / s) v``."""
        u = torch.relu(self.u(x))
        a = (u @ u.transpose(-1, -2)) / (x.shape[-2] ** 0.5)
        return torch.sigmoid(self.gate(x)) * ((a * a) @ self.v(x))


class _AdditiveAttention(nn.Module):
    """Bahdanau additive attention — ``wᵀ tanh(Wq q + Wk k)`` scores.

    The original seq2seq attention: scores come from a learned MLP on
    the *sum* of projected query and key, broadcast over the (tq, tk)
    grid — the ``add(unsqueeze, unsqueeze)`` spelling, not a matmul.
    """

    def __init__(self, d: int) -> None:
        """Build the query/key/value projections and score vector."""
        super().__init__()
        self.qp = nn.Linear(d, d)
        self.kp = nn.Linear(d, d)
        self.vp = nn.Linear(d, d)
        self.score = nn.Linear(d, 1)
        self.out = nn.Linear(d, d)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """Score by the additive MLP, softmax over keys, mix values."""
        e = self.score(
            torch.tanh(
                self.qp(q).unsqueeze(2) + self.kp(k).unsqueeze(1)
            )
        ).squeeze(-1)
        a = torch.softmax(e, dim=-1)
        mixed = (a.unsqueeze(-1) * self.vp(v).unsqueeze(1)).sum(2)
        return self.out(mixed)


class _TalkingHeadsAttention(nn.Module):
    """Talking-heads attention — linear head-mixing around softmax.

    Shazeer 2020: per-head-pair weights mix the attention logits
    before the softmax and the probabilities after it, spelled as
    ``permute`` + ``matmul`` over the head axis.
    """

    def __init__(self, d: int, heads: int) -> None:
        """Build the projections and the (heads x heads) mixers."""
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.pre = nn.Parameter(torch.randn(heads, heads) / heads)
        self.post = nn.Parameter(torch.randn(heads, heads) / heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Attend with head-mixed logits and head-mixed probabilities."""
        b, t, d = x.shape
        h = self.heads
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.reshape(b, t, h, d // h).transpose(1, 2)
        k = k.reshape(b, t, h, d // h).transpose(1, 2)
        v = v.reshape(b, t, h, d // h).transpose(1, 2)
        logits = q @ k.transpose(-1, -2) / (d // h) ** 0.5
        logits = (logits.permute(0, 2, 3, 1) @ self.pre).permute(
            0, 3, 1, 2
        )
        p = torch.softmax(logits, dim=-1)
        p = (p.permute(0, 2, 3, 1) @ self.post).permute(0, 3, 1, 2)
        return self.out((p @ v).transpose(1, 2).reshape(b, t, d))


class _CosineAttention(nn.Module):
    """Cosine attention — normalized q/k, learned logit scale.

    SwinV2-style: ``softmax(τ · q̂ k̂ᵀ)`` where q̂/k̂ are L2-normalized
    per head — the ``linalg_vector_norm`` spelling of attention.
    """

    def __init__(self, d: int, heads: int) -> None:
        """Build the projections and the clamped logit scale."""
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.logit_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Attend on cosine similarities scaled by a learned τ."""
        b, t, d = x.shape
        h = self.heads
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.reshape(b, t, h, d // h).transpose(1, 2)
        k = k.reshape(b, t, h, d // h).transpose(1, 2)
        v = v.reshape(b, t, h, d // h).transpose(1, 2)
        logits = (
            F.normalize(q, dim=-1)
            @ F.normalize(k, dim=-1).transpose(-1, -2)
        ) * self.logit_scale.clamp(max=4.6)
        o = torch.softmax(logits, dim=-1) @ v
        return self.out(o.transpose(1, 2).reshape(b, t, d))


# ---------------------------------------------------------------------------
#  Conditioning spellings — the real mul(unsqueeze, ·) sites
# ---------------------------------------------------------------------------


class _FiLMHead(nn.Module):
    """FiLM conditioning — ``gamma(c) ⊙ x + beta(c)`` feature-wise modulation.

    Perez et al. 2018: a conditioning input produces per-channel gain
    and bias, broadcast over the token axis — the canonical real
    ``mul(unsqueeze, ·)`` site (the purpose-built corpus only spelled
    the broadcast-trivial corner of this family).
    """

    def __init__(self, d: int, cdim: int) -> None:
        """Build the conditioning MLP and the feature norm."""
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.to_gb = nn.Linear(cdim, 2 * d)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Modulate normalized features by broadcast gain+shift."""
        gamma, beta = self.to_gb(c).unsqueeze(1).chunk(2, dim=-1)
        return self.norm(x) * gamma + beta


class _AdaLNBlock(nn.Module):
    """DiT adaptive layer-norm block — conditioning at block scale.

    Peebles & Xie 2022: ``LN(x) ⊙ (1 + scale(c)) + shift(c)`` feeds
    the block map and the residual is gated by ``gate(c)`` — three
    conditioning broadcasts in one real block.
    """

    def __init__(self, d: int, cdim: int) -> None:
        """Build the norm, the block map and the conditioning table."""
        super().__init__()
        self.norm = nn.LayerNorm(d, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d)
        )
        self.cond = nn.Linear(cdim, 3 * d)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Condition the block: scale+shift the norm, gate the skip."""
        scale, shift, gate = self.cond(c).unsqueeze(1).chunk(3, dim=-1)
        h = self.mlp(self.norm(x) * (1.0 + scale) + shift)
        return x + gate * h


class _LoRAAdapter(nn.Module):
    """LoRA adapter — frozen dense map plus a low-rank residual branch.

    Hu et al. 2021: ``W x + alpha·(B (A x))`` — the genuine production
    spelling of the shared-input factorization the purpose-built
    ``SharedFactor*`` workloads mimicked elementwise; here the shared
    factor is a matmul chain, what real adapters actually look like.
    """

    def __init__(self, d: int, rank: int) -> None:
        """Build the dense map and the (down, up) low-rank pair."""
        super().__init__()
        self.base = nn.Linear(d, d)
        self.down = nn.Linear(d, rank, bias=False)
        self.up = nn.Linear(rank, d, bias=False)
        self.scale = 0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the dense map plus the scaled low-rank residual."""
        return self.base(x) + self.scale * self.up(self.down(x))


# ---------------------------------------------------------------------------
#  Token mixers / spectral / graphs
# ---------------------------------------------------------------------------


class _MLPMixerBlock(nn.Module):
    """MLP-Mixer block — token-mixing via transpose-sandwiched MLP.

    Tolstikhin 2021: mix along the token axis by transposing the
    feature map into a per-token linear — the transpose residuality
    family the corpus's Conformer block only touched by accident.
    """

    def __init__(self, d: int, t: int) -> None:
        """Build both norms and the token/channel mixing MLPs."""
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.token_mix = nn.Linear(t, t)
        self.channel_mix = nn.Sequential(
            nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mix tokens (transposed linear) then channels (MLP)."""
        y = self.token_mix(self.norm1(x).transpose(1, 2))
        x = x + y.transpose(1, 2)
        return x + self.channel_mix(self.norm2(x))


class _FNOBlock(nn.Module):
    """Fourier neural operator layer — spectral filter plus bypass.

    Li et al. 2021: ``irfft2(scale · rfft2(x)) + conv1x1(x)`` with the
    top modes truncated.  The FFT ops made this workload the zoo's
    binding-gap probe: it traced but could not be delivered until the
    bridge bound ``fft_rfft2``/``fft_irfft2``/``real``.
    """

    def __init__(self, ch: int, modes: int) -> None:
        """Build the spectral scale and the bypass projection."""
        super().__init__()
        self.modes = modes
        self.scale = nn.Parameter(torch.randn(ch, modes) / modes)
        self.bypass = nn.Conv2d(ch, ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Filter the lowest modes in frequency space, add bypass."""
        f = torch.fft.rfft2(x)
        lo = f[..., : self.modes] * self.scale[:, None, :]
        hi = torch.zeros_like(f[..., self.modes :])
        spec = torch.cat([lo, hi], dim=-1)
        return (
            self.bypass(x)
            + torch.fft.irfft2(spec, s=(x.shape[-2], x.shape[-1])).real
        )


class _GCNLayer(nn.Module):
    """Two-hop graph convolution — symmetric-normalized propagation.

    Kipf & Welling 2017: ``D^{-1/2} A D^{-1/2}`` built in-graph from
    the adjacency (two ``unsqueeze`` broadcast sites), then two
    propagation matmuls around the feature maps.
    """

    def __init__(self, d: int) -> None:
        """Build the two hop projections."""
        super().__init__()
        self.w1 = nn.Linear(d, d)
        self.w2 = nn.Linear(d, d)

    def forward(
        self, x: torch.Tensor, adj: torch.Tensor
    ) -> torch.Tensor:
        """Propagate twice through the normalized adjacency."""
        d_inv = torch.rsqrt(adj.sum(-1).clamp_min(1e-6))
        norm = d_inv.unsqueeze(-1) * adj * d_inv.unsqueeze(-2)
        return torch.relu(norm @ self.w2(torch.relu(norm @ self.w1(x))))


class _AffineCoupling(nn.Module):
    """RealNVP affine coupling — split, scale-and-shift, recombine.

    Dinh et al. 2017: one half flows through unchanged and drives a
    ``(s, t)`` net that affinely transforms the other half —
    ``chunk`` + ``exp`` + ``cat`` spelling of a normalizing flow.
    """

    def __init__(self, d: int) -> None:
        """Build the conditioner producing per-feature (scale, shift)."""
        super().__init__()
        self.cond = nn.Sequential(
            nn.Linear(d // 2, d), nn.ReLU(), nn.Linear(d, d)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Split channels, affine the second half, reconcatenate."""
        x1, x2 = x.chunk(2, dim=-1)
        s, t = self.cond(x1).chunk(2, dim=-1)
        y2 = x2 * torch.exp(s.clamp(-3.0, 3.0)) + t
        return torch.cat([x1, y2], dim=-1)


# ---------------------------------------------------------------------------
#  Routing / gating families
# ---------------------------------------------------------------------------


class _SoftSlotMoE(nn.Module):
    """Soft MoE — continuous slot dispatch instead of top-k routing.

    Puigcerver et al. 2023: every token is a convex combination over
    slots and each slot is a convex combination of tokens — dispatch
    and combine are softmaxes, not index selects.
    """

    def __init__(self, d: int, slots: int) -> None:
        """Build the slot logits and the per-slot MLP weights."""
        super().__init__()
        self.slot = nn.Linear(d, slots)
        self.w1 = nn.Parameter(torch.randn(slots, d, 2 * d) / d)
        self.w2 = nn.Parameter(torch.randn(slots, 2 * d, d) / d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Combine tokens into slots, transform, dispatch back."""
        phi = self.slot(x)
        slots = torch.softmax(phi, dim=1).transpose(1, 2) @ x
        h = torch.einsum("bsd,sde->bse", slots, self.w1)
        h = torch.einsum("bse,sef->bsf", F.gelu(h), self.w2)
        return torch.softmax(phi, dim=-1) @ h


class _ExpertChoiceMoE(nn.Module):
    """Expert-choice routing — experts pick tokens, not vice versa.

    Zhou et al. 2022: per-expert top-k over the *token* axis, gather
    the chosen rows, apply each expert, scatter the results back —
    the gather/scatter_add spelling of MoE dispatch.
    """

    def __init__(self, d: int, n_exp: int, k: int) -> None:
        """Build the token scorer and the per-expert projections."""
        super().__init__()
        self.k = k
        self.score = nn.Linear(d, n_exp)
        self.w = nn.Parameter(torch.randn(n_exp, d, d) / d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Each expert selects its top-k tokens and writes them back."""
        b, t, d = x.shape
        e = self.w.shape[0]
        w = torch.softmax(self.score(x), dim=-1).transpose(1, 2)
        gate, idx = torch.topk(w, self.k, dim=-1)
        sel = torch.gather(
            x.unsqueeze(1).expand(b, e, t, d),
            2,
            idx.unsqueeze(-1).expand(b, e, self.k, d),
        )
        y = torch.einsum("bekd,edf->bekf", sel, self.w)
        out = torch.zeros_like(x)
        flat_i = idx.reshape(b, e * self.k, 1).expand(b, e * self.k, d)
        flat_y = (y * gate.unsqueeze(-1)).reshape(b, e * self.k, d)
        return out.scatter_add(1, flat_i, flat_y)


class _CapsuleRouting(nn.Module):
    """Dynamic routing between capsule layers (Sabour et al. 2017).

    Two static routing iterations: agreement votes from a per-capsule
    transform, softmax coupling coefficients, squash nonlinearity.
    """

    def __init__(self, din: int, dout: int, n_out: int) -> None:
        """Build the vote transform ``(din, n_out, dout)``."""
        super().__init__()
        self.n_out = n_out
        self.w = nn.Parameter(torch.randn(din, n_out, dout) / din)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Route lower capsules to output capsules, then squash."""
        votes = torch.einsum("bti,ioj->btoj", x, self.w)
        logits = torch.zeros(
            x.shape[0],
            x.shape[1],
            self.n_out,
            dtype=x.dtype,
            device=x.device,
        )
        out = votes.sum(1)
        for _ in range(2):
            c = torch.softmax(logits, dim=-1)
            s = (c.unsqueeze(-1) * votes).sum(1)
            n2 = s.square().sum(-1, keepdim=True)
            out = s * n2 / ((1.0 + n2) * n2.sqrt().clamp_min(1e-6))
            logits = logits + (votes * out.unsqueeze(1)).sum(-1)
        return out


# ---------------------------------------------------------------------------
#  Recurrence / scan / conv families
# ---------------------------------------------------------------------------


class _ChunkedRetention(nn.Module):
    """RetNet inner-chunk retention — decay-masked quadratic attention.

    Sun et al. 2023: ``(QKᵀ ⊙ D) V`` where the decay matrix
    ``D[c,s] = gamma^{c-s}`` is built in-graph by ``tril`` over a power
    of a position difference — the same in-graph-bias family as
    ALiBi, but multiplicative rather than additive.
    """

    def __init__(self, d: int) -> None:
        """Build the q/k/v projections and record the decay rate."""
        super().__init__()
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.gamma = 0.9

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply decay-masked attention within the chunk."""
        t = x.shape[-2]
        pos = torch.arange(t, dtype=x.dtype, device=x.device)
        decay = torch.tril(
            torch.pow(
                torch.tensor(
                    self.gamma, dtype=x.dtype, device=x.device
                ),
                (pos.unsqueeze(1) - pos.unsqueeze(0)).clamp_min(0),
            )
        )
        s = self.q(x) @ self.k(x).transpose(-1, -2) / (t**0.5)
        return (s * decay) @ self.v(x)


class _DeltaRuleBlock(nn.Module):
    """Delta-rule linear attention — an unrolled matrix-state recurrence.

    Schlag et al. 2021 (DeltaNet): the state is a ``(dk, dv)``
    matrix updated by ``S <- S·(I - beta k k^T) + beta k v^T``, the
    "write then partially erase" rule.  Unrolled T steps and spelled
    elementwise — ``matmul`` on per-step ``select`` slices plus
    ``sub``/``add``/``mul`` — it is a real program whose efficient
    form (the chunked delta-rule scan) is NOT in the law library:
    the aff/affd scan monoids lift ``h' = a·h + b`` maps, not a
    data-dependent ``(I - beta k k^T)`` contraction on a matrix
    state.
    """

    def __init__(self, dk: int, dv: int, steps: int) -> None:
        """Build the k/v/β projections and record the unroll count."""
        super().__init__()
        self.steps = steps
        self.dk = dk
        self.k = nn.Linear(dk, dk)
        self.v = nn.Linear(dk, dv)
        self.beta = nn.Linear(dk, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Unroll the delta rule; return the final read-out rows."""
        b = x.shape[0]
        kk = self.k(x)  # (B, T, dk)
        vv = self.v(x)  # (B, T, dv)
        bb = torch.sigmoid(self.beta(x))  # (B, T, 1)
        s = torch.zeros(
            b, self.dk, vv.shape[-1], dtype=x.dtype, device=x.device
        )
        for t in range(self.steps):
            k_t = kk[:, t]  # (B, dk)
            v_t = vv[:, t]  # (B, dv)
            beta_t = bb[:, t].unsqueeze(-1)  # (B, 1, 1)
            # residual read: v_t - k_t^T S  ->  (B, dv)
            read = v_t - (k_t.unsqueeze(1) @ s).squeeze(1)
            # rank-1 write: S += beta . k_t (x) read
            s = s + beta_t * k_t.unsqueeze(-1) * read.unsqueeze(-2)
        return s


class _DeepEquilibrium(nn.Module):
    """Deep-equilibrium block — an unrolled shared-weight fixed point.

    Bai et al. 2019: ``z ← tanh(Wz·z + Wx·x)`` iterated to a fixed
    point; unrolled three times it is a real program that reuses ONE
    parameter set across three stacked blocks — the spelling the
    param-sharing machinery exists for.
    """

    def __init__(self, d: int, iters: int) -> None:
        """Build the injection and the two fixed-point maps."""
        super().__init__()
        self.iters = iters
        self.inj = nn.Linear(d, d)
        self.fz = nn.Linear(d, d)
        self.fx = nn.Linear(d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Iterate the fixed-point map a statically unrolled count."""
        z = torch.tanh(self.inj(x))
        for _ in range(self.iters):
            z = torch.tanh(self.fz(z) + self.fx(x))
        return z


class _WaveNetGate(nn.Module):
    """WaveNet gated dilated conv block — ``tanh(f) ⊙ sigmoid(g)`` + res.

    van den Oord et al. 2016: a dilated causal conv pair whose filter
    and gate branches multiply, then a 1x1 residual projection.
    """

    def __init__(self, ch: int) -> None:
        """Build the dilated filter/gate convs and the residual mix."""
        super().__init__()
        self.filter = nn.Conv1d(ch, ch, 3, padding=2, dilation=2)
        self.gate = nn.Conv1d(ch, ch, 3, padding=2, dilation=2)
        self.res = nn.Conv1d(ch, ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the gated activation, cropped back to causal width."""
        f = torch.tanh(self.filter(x))[..., : x.shape[-1]]
        g = torch.sigmoid(self.gate(x))[..., : x.shape[-1]]
        return x + self.res(f * g)


class _TimeCondConv(nn.Module):
    """Diffusion residual block — sinusoidal time embedding + conv.

    The DDPM UNet piece: a sinusoidal timestep embedding projected
    into the conv block's channels and added as a broadcast bias —
    the 4-D ``unsqueeze`` conditioning site.
    """

    def __init__(self, ch: int, temb: int) -> None:
        """Build the time projection and the conv/norm stage."""
        super().__init__()
        self.temb = temb
        self.tproj = nn.Linear(2 * temb, ch)
        self.norm = nn.GroupNorm(4, ch)
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Add the sinusoidal time bias into the conv block."""
        freqs = torch.exp(
            torch.arange(self.temb, dtype=x.dtype, device=x.device)
            * (
                -torch.log(
                    torch.tensor(
                        10000.0, dtype=x.dtype, device=x.device
                    )
                )
                / self.temb
            )
        )
        ang = t.unsqueeze(-1) * freqs.unsqueeze(0)
        emb = self.tproj(torch.cat([ang.sin(), ang.cos()], dim=-1))
        y = self.conv(F.silu(self.norm(x)))
        return x + y + emb[:, :, None, None]


# ---------------------------------------------------------------------------
#  Heads / objectives
# ---------------------------------------------------------------------------


class _MoSHead(nn.Module):
    """Mixture-of-softmaxes output head (Yang et al. 2018).

    ``Σ_k π_k(h) · softmax(W_k h)`` — a learned prior over several
    cheap softmaxes, the real alternative to a single full softmax.
    """

    def __init__(self, d: int, ncls: int, k: int) -> None:
        """Build the prior net and the stacked component logits."""
        super().__init__()
        self.prior = nn.Linear(d, k)
        self.w = nn.Parameter(torch.randn(k, d, ncls) / d)
        self.b = nn.Parameter(torch.zeros(k, ncls))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the prior-weighted mixture of component softmaxes."""
        pi = torch.softmax(self.prior(x), dim=-1)
        logits = torch.einsum("bd,kdn->bkn", x, self.w) + self.b
        probs = torch.softmax(logits, dim=-1)
        return (pi.unsqueeze(-1) * probs).sum(1)


class _CovarianceHead(nn.Module):
    """VICReg-style covariance statistic — centered Gram matrix.

    Bardes et al. 2022: ``C = Ẋᵀ Ẋ / (n-1)`` over the batch — a real
    self-supervised objective term spelling ``sub(mean)``,
    ``transpose`` and a contracted ``matmul`` with no labels anywhere.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the feature covariance of the batch."""
        c = x - x.mean(dim=0, keepdim=True)
        return c.transpose(0, 1) @ c / (x.shape[0] - 1)


class _SplineKAN(nn.Module):
    """KAN edge mixing over a fixed sinusoid/polynomial basis.

    Liu et al. 2024 (Kolmogorov-Arnold networks): each edge is a
    learned combination of basis functions of the input — spelled
    here as a ``stack([x, x², sin x])`` bank times an
    ``(in, out, basis)`` weight table.
    """

    def __init__(self, din: int, dout: int) -> None:
        """Build the per-edge basis weight table."""
        super().__init__()
        self.w = nn.Parameter(torch.randn(din, dout, 3) / din)
        self.b = nn.Parameter(torch.zeros(dout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the basis bank and mix it per output."""
        feats = torch.stack([x, x * x, torch.sin(x)], dim=-1)
        return torch.einsum("bif,iof->bo", feats, self.w) + self.b


class _HighwayGate(nn.Module):
    """Highway network block — learned transform/carry interpolation.

    Srivastava et al. 2015: ``t ⊙ h(x) + (1 - t) ⊙ x`` — the convex
    gate between a transformed and an identity path.
    """

    def __init__(self, d: int) -> None:
        """Build the transform and the carry-gate maps."""
        super().__init__()
        self.h = nn.Linear(d, d)
        self.t = nn.Linear(d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Blend the transformed features with the identity."""
        t = torch.sigmoid(self.t(x))
        return t * torch.relu(self.h(x)) + (1.0 - t) * x


# ---------------------------------------------------------------------------
#  SSM / linear-attention / long-conv — the under-catalogued scans
# ---------------------------------------------------------------------------


class _MambaSelectiveScan(nn.Module):
    """Mamba S6 block — diagonal-decay state, selective write/read.

    Gu & Dao 2023: each channel carries an ``n``-dimensional state
    updated ``h ← exp(-softplus(Δ_t)·A) ⊙ h + x_t ⊗ B_t`` and read as
    ``y_t = C_tᵀ h_t`` — a *data-dependent* decay, spelled here as an
    unrolled loop over ``select`` slices with an outer-product rank-1
    write.  The intake's ``SelectiveScan`` is the parallel
    cumprod/cumsum spelling of a *static* elementwise gate; this is
    the matrix-state selectivity the shipped scan monoids don't
    reach.
    """

    def __init__(self, d: int, n: int, steps: int) -> None:
        """Build the Δ/B/C projections and the decay table."""
        super().__init__()
        self.d = d
        self.n = n
        self.steps = steps
        self.dt = nn.Linear(d, d)
        self.b = nn.Linear(d, n)
        self.c = nn.Linear(d, n)
        self.a_log = nn.Parameter(torch.randn(d, n) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Unroll the selective scan; return every step's read-out."""
        b = x.shape[0]
        dt = F.softplus(self.dt(x))  # (B, T, d)
        dec = torch.exp(
            -dt.unsqueeze(-1) * torch.exp(self.a_log)
        )  # (B, T, d, n)
        bb = self.b(x)  # (B, T, n)
        cc = self.c(x)  # (B, T, n)
        h = torch.zeros(
            b, self.d, self.n, dtype=x.dtype, device=x.device
        )
        outs = []
        for t in range(self.steps):
            h = dec[:, t] * h + x[:, t].unsqueeze(-1) * bb[
                :, t
            ].unsqueeze(1)
            outs.append(torch.einsum("bdn,bn->bd", h, cc[:, t]))
        return torch.stack(outs, dim=1)


class _GatedLinearAttention(nn.Module):
    """GLA — gated linear attention, vector-gated matrix state.

    Yang et al. 2024: ``S ← diag(g_t) S + k_t v_tᵀ``,
    ``o_t = q_tᵀ S`` — the halfway point between the intake's cumsum
    ``LinearAttention`` and the zoo's ``DeltaRuleBlock``: the decay is
    a data-dependent per-channel gate rather than a rank-1 erase, and
    the unrolled ``mul`` + ``outer`` scan is a third distinct
    spelling of the same state-update family.
    """

    def __init__(self, d: int, steps: int) -> None:
        """Build the q/k/v projections and the gate map."""
        super().__init__()
        self.steps = steps
        self.d = d
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.g = nn.Linear(d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Unroll the gated state update; return every read-out."""
        b = x.shape[0]
        qq = self.q(x)
        kk = self.k(x)
        vv = self.v(x)
        gg = torch.sigmoid(self.g(x))
        s = torch.zeros(
            b, self.d, self.d, dtype=x.dtype, device=x.device
        )
        outs = []
        for t in range(self.steps):
            write = kk[:, t].unsqueeze(-1) * vv[:, t].unsqueeze(-2)
            s = gg[:, t].unsqueeze(-1) * s + write
            outs.append(torch.einsum("bdv,bd->bv", s, qq[:, t]))
        return torch.stack(outs, dim=1)


class _RWKVMixBlock(nn.Module):
    """RWKV-ish block — token-shift, wkv attention, channel mix.

    Peng et al. 2023 (RWKV-4): the time-mix interpolates each token
    with its shifted predecessor (``x·μ + shift(x)·(1-μ)`` — the
    ``cat`` + ``slice`` token-shift spelling), then applies a
    softmax-free attention whose weights are
    ``exp((i - t)·w + k_i + [i = t]·u)`` — an in-graph
    data-dependent decay bias.  The channel-mix is the squared-relu
    MLP under a receptance gate.
    """

    def __init__(self, d: int) -> None:
        """Build the shift mixes, the wkv maps and the channel MLP."""
        super().__init__()
        self.mu_t = nn.Parameter(torch.rand(d))
        self.mu_c = nn.Parameter(torch.rand(d))
        self.decay = nn.Parameter(torch.rand(d))
        self.first = nn.Parameter(torch.rand(d))
        self.r_t = nn.Linear(d, d)
        self.k_t = nn.Linear(d, d)
        self.v_t = nn.Linear(d, d)
        self.o_t = nn.Linear(d, d)
        self.r_c = nn.Linear(d, d)
        self.k_c = nn.Linear(d, 2 * d)
        self.v_c = nn.Linear(2 * d, d)

    @staticmethod
    def _shift(x: torch.Tensor) -> torch.Tensor:
        """Prepend a zero token and drop the last — the causal shift."""
        zero = torch.zeros_like(x[:, :1])
        return torch.cat([zero, x[:, :-1]], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the receptance-gated wkv mix then the channel mix."""
        t = x.shape[-2]
        sx = self._shift(x)
        mix = x * self.mu_t + sx * (1.0 - self.mu_t)
        k = self.k_t(mix)
        v = self.v_t(mix)
        pos = torch.arange(t, dtype=x.dtype, device=x.device)
        rel = pos[None, :] - pos[:, None]  # (T, T): i - t
        diag = (pos[:, None] == pos[None, :]).to(x.dtype)
        w = -F.softplus(self.decay)
        bias = rel.unsqueeze(-1) * w + diag.unsqueeze(-1) * self.first
        tri = torch.tril(
            torch.ones(t, t, dtype=x.dtype, device=x.device)
        )
        att = torch.exp(bias.unsqueeze(0) + k.unsqueeze(1))
        att = att * tri.unsqueeze(0).unsqueeze(-1)  # (B, T, T, d)
        num = (att * v.unsqueeze(1)).sum(2)
        den = att.sum(2).clamp_min(1e-6)
        x = x + torch.sigmoid(self.r_t(mix)) * self.o_t(num / den)
        sx2 = self._shift(x)
        mixc = x * self.mu_c + sx2 * (1.0 - self.mu_c)
        sq = torch.relu(self.k_c(mixc)) ** 2
        return x + torch.sigmoid(self.r_c(mixc)) * self.v_c(sq)


class _HyenaLongConv(nn.Module):
    """Hyena operator — positional-MLP kernel + data-gated long conv.

    Poli et al. 2023: the convolution kernel is *generated* by a
    positional MLP evaluated at the filter taps rather than stored —
    a parameter-free-shape site — then applied depthwise between two
    elementwise data gates: ``u(x) ⊙ (h * (v(x) ⊙ x))``.
    """

    def __init__(self, d: int, ksize: int) -> None:
        """Build the kernel generator and the gate/output maps."""
        super().__init__()
        self.d = d
        self.ksize = ksize
        self.gen = nn.Sequential(
            nn.Linear(1, d), nn.SiLU(), nn.Linear(d, d)
        )
        self.uv = nn.Linear(d, 2 * d)
        self.out = nn.Linear(d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generate the depthwise kernel, convolve, gate, project."""
        t = x.shape[-2]
        taps = (
            torch.arange(self.ksize, dtype=x.dtype, device=x.device)
            / self.ksize
        ).unsqueeze(-1)
        ker = (
            self.gen(taps)
            .transpose(0, 1)
            .reshape(self.d, 1, self.ksize)
        )
        u, v = self.uv(x).chunk(2, dim=-1)
        z = (v * x).transpose(1, 2)
        y = F.conv1d(z, ker, padding=self.ksize - 1, groups=self.d)[
            ..., :t
        ]
        return self.out(u * y.transpose(1, 2))


# ---------------------------------------------------------------------------
#  Routing — deployment-shape dispatch spellings
# ---------------------------------------------------------------------------


class _TopKRouterMoE(nn.Module):
    """Top-k router MoE with the load-balance aux term folded in.

    The decoder-LLM dispatch (Switch/Mixtral family): a softmax
    router, top-k selection over a stacked expert bank, weighted
    recombination — plus the ``aux = E·Σ_e f_e·P_e`` load-balance
    objective folded into the output so the whole graph is one
    tensor.  Neither intake MoE spells the aux head or the
    ``one_hot`` top-k mask.
    """

    def __init__(self, d: int, n_exp: int, k: int) -> None:
        """Build the router and the stacked expert weight bank."""
        super().__init__()
        self.k = k
        self.router = nn.Linear(d, n_exp)
        self.w = nn.Parameter(torch.randn(n_exp, d, d) / d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Route top-k, recombine, add the load-balance penalty."""
        p = torch.softmax(self.router(x), dim=-1)
        _topv, topi = torch.topk(p, self.k, dim=-1)
        mask = F.one_hot(topi, p.shape[-1]).to(x.dtype).amax(dim=-2)
        w = p * mask
        h = torch.einsum("btd,edf->betf", x, self.w)
        out = torch.einsum("bte,betf->btf", w, h)
        frac = mask.mean(dim=(0, 1))
        aux = p.shape[-1] * (frac * p.mean(dim=(0, 1))).sum()
        return out + 0.01 * aux


class _DeepSeekSigMoE(nn.Module):
    """DeepSeekMoE-style routing — sigmoid gate over a fine-grained bank.

    DeepSeek-AI 2024: routing is *sigmoid* top-k (not softmax) over
    many small experts spelled as a stacked ``(E, d, f)`` weight bank
    and a single ``einsum`` contraction, plus the always-on shared
    expert.  The intake's ``SharedExpertMoE`` spells the softmax-gate
    ``ModuleList`` form — this is the banked-sigmoid sibling.
    """

    def __init__(self, d: int, n_exp: int, k: int) -> None:
        """Build the sigmoid gate, shared expert and expert bank."""
        super().__init__()
        self.k = k
        self.gate = nn.Linear(d, n_exp)
        self.shared = nn.Linear(d, d)
        self.w = nn.Parameter(torch.randn(n_exp, d, d) / d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Route by sigmoid top-k, add the shared expert output."""
        s = torch.sigmoid(self.gate(x))
        _topv, topi = torch.topk(s, self.k, dim=-1)
        mask = F.one_hot(topi, s.shape[-1]).to(x.dtype).amax(dim=-2)
        w = s * mask
        h = torch.einsum("btd,edf->betf", x, self.w)
        return self.shared(x) + torch.einsum("bte,betf->btf", w, h)


# ---------------------------------------------------------------------------
#  Quantization / cache — deployment-precision spellings
# ---------------------------------------------------------------------------


class _STEQuantLinear(nn.Module):
    """Straight-through fake-quant linear — round/clamp + detach.

    The QAT spelling: weights and activations are quantized by
    ``scale·clamp(round(x/scale))`` and the straight-through
    estimator is spelled ``x + (q - x).detach()`` — the
    ``detach`` + ``round`` pair the intake's ``FakeQuant`` never
    wrote (it quantizes activations per-tensor, no STE path, no
    weight quantization).
    """

    def __init__(self, d: int) -> None:
        """Build the dense map and the two quantization scales."""
        super().__init__()
        self.w = nn.Linear(d, d)
        self.w_scale = nn.Parameter(torch.ones(d, 1))
        self.x_scale = nn.Parameter(torch.ones(1))

    @staticmethod
    def _ste(t: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Quantize-dequantize *t* with a straight-through gradient."""
        q = torch.round(t / scale).clamp(-7.0, 7.0) * scale
        return t + (q - t).detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the linear map through STE-quantized operands."""
        wq = self._ste(self.w.weight, self.w_scale)
        xq = self._ste(x, self.x_scale)
        return F.linear(xq, wq, self.w.bias)


class _PerChannelQuantMatmul(nn.Module):
    """Per-channel-scale dequant matmul — ``x @ (s ⊙ W_q)ᵀ`` head.

    The W8A16 inference spelling: the stored weight is already
    integer-valued (rounded at build) and each output channel carries
    its own scale — the broadcast-``mul`` + ``matmul`` dequant site.
    ``FakeQuant`` scales per-tensor and ``LoRAAdapter`` factors but
    never dequantizes; nothing else spells per-channel weight scales.
    """

    def __init__(self, d: int) -> None:
        """Build the rounded weight, its per-channel scale, the bias."""
        super().__init__()
        w = torch.randn(d, d)
        s = (w.abs().amax(dim=0, keepdim=True) / 7.0).clamp_min(1e-3)
        self.w_q = nn.Parameter(torch.round(w / s))
        self.scale = nn.Parameter(s)
        self.bias = nn.Parameter(torch.zeros(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Dequantize the weight per channel and apply the head."""
        return x @ (self.scale * self.w_q).transpose(0, 1) + self.bias


class _KVCacheAttention(nn.Module):
    """KV-cache decode step — append, window-truncate, attend.

    The incremental-decoding spelling: past keys/values arrive as
    inputs, the new token's k/v are ``cat``'ed on, and only the last
    ``window`` entries feed the attention — the ``cat`` +
    ``narrow`` cache-update pair plus the length-mismatched
    ``(1, T)`` score site a streaming decoder actually emits.
    """

    def __init__(self, d: int, window: int) -> None:
        """Build the q/kv/out projections and record the window."""
        super().__init__()
        self.q = nn.Linear(d, d)
        self.kv = nn.Linear(d, 2 * d)
        self.out = nn.Linear(d, d)
        self.window = window

    def forward(
        self,
        x: torch.Tensor,
        kc: torch.Tensor,
        vc: torch.Tensor,
    ) -> torch.Tensor:
        """Append the new k/v to the cache, attend over the window."""
        k_new, v_new = self.kv(x).chunk(2, dim=-1)
        k = torch.cat([kc, k_new], dim=1)[:, -self.window :]
        v = torch.cat([vc, v_new], dim=1)[:, -self.window :]
        s = self.q(x) @ k.transpose(-1, -2) / (x.shape[-1] ** 0.5)
        return self.out(torch.softmax(s, dim=-1) @ v)


# ---------------------------------------------------------------------------
#  Modern block details — the variants the intake banks miss
# ---------------------------------------------------------------------------


class _GEGLUBlock(nn.Module):
    """GEGLU feed-forward block — the third gated-MLP sibling.

    Shazeer 2020's GLU variants: the intake banks hold ReGLU
    (``relu``) and SwiGLU (``silu``) at MLP and block scale; GEGLU —
    ``GELU(Wx) ⊙ Vx`` — is the uncovered member, spelled here at
    block scale with the pre-norm residual.
    """

    def __init__(self, d: int) -> None:
        """Build the norm, the packed gate map and the projection."""
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.w = nn.Linear(d, 2 * d)
        self.v = nn.Linear(d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the gelu-gated feed-forward under a residual."""
        a, b = self.w(self.norm(x)).chunk(2, dim=-1)
        return x + self.v(F.gelu(a) * b)


class _SkipInitNet(nn.Module):
    """SkipInit/Fixup residual — learnable scalar gates, no norms.

    De & Smith 2020: batch-norm-free residual nets replace the norm
    with a per-branch learnable scalar multiplier initialized at
    zero — ``x + a·block(x)`` stacked twice, the scalar-gated
    residual family the norm-centric corpus never wrote.
    """

    def __init__(self, d: int) -> None:
        """Build both branch MLPs and their scalar residual gates."""
        super().__init__()
        self.a1 = nn.Parameter(torch.zeros(1))
        self.a2 = nn.Parameter(torch.zeros(1))
        self.m1 = nn.Sequential(
            nn.Linear(d, 2 * d), nn.ReLU(), nn.Linear(2 * d, d)
        )
        self.m2 = nn.Sequential(
            nn.Linear(d, 2 * d), nn.ReLU(), nn.Linear(2 * d, d)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply both scalar-gated residual branches."""
        x = x + self.a1 * self.m1(x)
        return x + self.a2 * self.m2(x)


class _SoftcapLogitHead(nn.Module):
    """Logit soft-cap head — ``cap·tanh(logits/cap)`` (Gemma-2).

    The bounded-logit spelling Gemma-2/Grok use on the classifier
    output: ``div`` + ``tanh`` + ``mul`` with the cap a graph
    literal — a saturated head the transform lists never catalogued.
    """

    def __init__(self, d: int, ncls: int) -> None:
        """Build the classifier head and record the cap."""
        super().__init__()
        self.head = nn.Linear(d, ncls)
        self.cap = 15.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the soft-capped logits."""
        logits = self.head(x)
        return self.cap * torch.tanh(logits / self.cap)


class _RotaryPair(nn.Module):
    """Interleaved-pair rotary — the GPT-J RoPE spelling on (q, k).

    The intake's RoPE workloads spell the rotate-half (GPT-NeoX)
    form ``cat(-x₂, x₁)``.  This is the pairwise-interleaved form —
    ``unflatten`` to (…, d/2, 2), two real multiplies against the
    (cos, sin) tables, ``flatten`` back — a different shape-juggling
    site for the same rotation, scored through ``q·kᵀ``.
    """

    def __init__(self, d: int) -> None:
        """Build the q and k projections."""
        super().__init__()
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)

    @staticmethod
    def _rope(z: torch.Tensor) -> torch.Tensor:
        """Rotate the interleaved pairs of *z* by their positions."""
        t, d = z.shape[-2], z.shape[-1]
        inv = torch.pow(
            torch.tensor(10000.0, dtype=z.dtype, device=z.device),
            -torch.arange(0, d, 2, dtype=z.dtype, device=z.device) / d,
        )
        ang = torch.outer(
            torch.arange(t, dtype=z.dtype, device=z.device), inv
        )
        cos, sin = ang.cos(), ang.sin()  # (T, d/2)
        z1, z2 = z.unflatten(-1, (d // 2, 2)).unbind(-1)
        rot = torch.stack(
            [z1 * cos - z2 * sin, z1 * sin + z2 * cos], dim=-1
        )
        return rot.flatten(-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate q and k pairwise; return their score matrix."""
        q = self._rope(self.q(x))
        k = self._rope(self.k(x))
        return q @ k.transpose(-1, -2)


# ---------------------------------------------------------------------------
#  Sparse / patterned attention — masks and gathers as in-graph tensors
# ---------------------------------------------------------------------------


class _SlidingWindowAttention(nn.Module):
    """Sliding-window attention — a band mask built in-graph.

    The longformer/Mistral local window: full quadratic scores
    masked to a causal band of width ``w``.  The band itself is a
    graph term (``tril - tril``), applied by a broadcast ``mul``
    over an ``unsqueeze`` — the pointwise-over-view spelling the
    global-mask attention workloads never wrote.
    """

    def __init__(self, d: int, window: int) -> None:
        """Build the q/k/v/out projections and record the window."""
        super().__init__()
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.out = nn.Linear(d, d)
        self.window = window

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Attend inside the causal band; zero-scored keys fall off."""
        t, d = x.shape[-2], x.shape[-1]
        s = (self.q(x) @ self.k(x).transpose(-1, -2)) * (d**-0.5)
        ones = torch.ones(t, t, dtype=x.dtype, device=x.device)
        band = torch.tril(ones) - torch.tril(ones, -1 - self.window)
        s = s * band.unsqueeze(0) + (band.unsqueeze(0) - 1.0) * 1e4
        return self.out(torch.softmax(s, dim=-1) @ self.v(x))


class _DilatedBlockAttention(nn.Module):
    """Dilated attention — strided key/value selection with a gate.

    LongNet-style dilated attention: every ``r``-th key/value row
    feeds the score — the ``slice``-with-stride spelling of a sparse
    pattern — while a learned per-position mix gate blends the
    dilated read against the local window's ``slice``.
    """

    def __init__(self, d: int, dil: int, window: int) -> None:
        """Build the projections, the mix gate and the window."""
        super().__init__()
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.out = nn.Linear(d, d)
        self.mix = nn.Linear(d, 1)
        self.dil = dil
        self.window = window

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Read dilated keys, blend the local tail, project."""
        d = x.shape[-1]
        w = self.window
        kd = self.k(x)[:, :: self.dil]
        vd = self.v(x)[:, :: self.dil]
        s = (self.q(x) @ kd.transpose(-1, -2)) * (d**-0.5)
        far = torch.softmax(s, dim=-1) @ vd
        g = torch.sigmoid(self.mix(x))
        head = far[:, :-w] * g[:, :-w]
        tail = far[:, -w:] * g[:, -w:] + x[:, -w:] * (1.0 - g[:, -w:])
        return self.out(torch.cat([head, tail], dim=1))


class _BlockSparseAttention(nn.Module):
    """Block-sparse attention — per-query top-k over key blocks.

    The block-sparse dispatch spelling: keys are reshaped into a
    ``(blocks, block_size)`` grid, per-block scores are reduced to
    one number per block, top-k selects the live blocks, and the
    ``one_hot`` mask is broadcast back over the score tensor —
    ``mul(·, unsqueeze)`` over a gathered mask.
    """

    def __init__(
        self, d: int, blocks: int, bsize: int, ktop: int
    ) -> None:
        """Build the projections; record the block grid and top-k."""
        super().__init__()
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.out = nn.Linear(d, d)
        self.blocks = blocks
        self.bsize = bsize
        self.ktop = ktop

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Score key blocks, keep each query's top-k, attend."""
        b, _t, d = x.shape
        nb, bs = self.blocks, self.bsize
        kb = self.k(x).reshape(b, nb, bs, d)
        vb = self.v(x).reshape(b, nb, bs, d)
        s = torch.einsum("btd,bnsd->btns", self.q(x), kb) * (d**-0.5)
        blk = s.amax(-1)
        _topv, idx = torch.topk(blk, self.ktop, dim=-1)
        keep = F.one_hot(idx, nb).to(x.dtype).amax(-2).unsqueeze(-1)
        att = torch.softmax(s * keep + (keep - 1.0) * 1e4, dim=-1)
        o = torch.einsum("btns,bnsd->btd", att, vb)
        return self.out(o)


# ---------------------------------------------------------------------------
#  Depthwise-conv blocks — ConvNeXt, squeeze-excite, MBConv
# ---------------------------------------------------------------------------


class _ConvNeXtBlock(nn.Module):
    """ConvNeXt block — depthwise conv, channels-last LN, layer scale.

    Liu et al. 2022: a 7x7 depthwise conv, then the whole block runs
    channels-last (``permute`` in, ``permute`` out) with a per-channel
    learnable scale on the residual branch — the
    ``add(·, transpose)`` + ``mul(·, reshape)`` pair.
    """

    def __init__(self, ch: int) -> None:
        """Build the depthwise conv, LN, pointwise pair and scale."""
        super().__init__()
        self.dw = nn.Conv2d(ch, ch, 7, padding=3, groups=ch)
        self.norm = nn.LayerNorm(ch)
        self.pw1 = nn.Linear(ch, 2 * ch)
        self.pw2 = nn.Linear(2 * ch, ch)
        self.gamma = nn.Parameter(0.1 * torch.ones(ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the channels-last block; add the scaled residual."""
        y = self.dw(x).permute(0, 2, 3, 1)
        y = self.pw2(F.gelu(self.pw1(self.norm(y))))
        y = y * self.gamma.reshape(1, 1, 1, -1)
        return x + y.permute(0, 3, 1, 2)


class _SqueezeExcite(nn.Module):
    """Squeeze-excitation block — pooled gate broadcast back over space.

    Hu et al. 2018: global-average-pool to ``(b, c)``, a two-layer
    bottleneck, sigmoid, then ``reshape(b, c, 1, 1)`` broadcast over
    the feature map — the ``mul(·, reshape)`` channel-gate site.
    """

    def __init__(self, ch: int, r: int) -> None:
        """Build the squeeze/excite bottleneck pair."""
        super().__init__()
        self.fc1 = nn.Linear(ch, ch // r)
        self.fc2 = nn.Linear(ch // r, ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pool, gate and scale the feature map channel-wise."""
        b, c = x.shape[0], x.shape[1]
        s = torch.sigmoid(self.fc2(F.relu(self.fc1(x.mean((2, 3))))))
        return x * s.reshape(b, c, 1, 1)


class _MBConvBlock(nn.Module):
    """Inverted-bottleneck MBConv — expand, depthwise, SE, project.

    Sandler et al. 2018 (MobileNetV2) with the EfficientNet SE
    insert: a pointwise expansion, a depthwise spatial conv, the
    pooled channel gate, and a projection back to the residual —
    ``mul(conv, reshape)`` inside a depthwise sandwich.
    """

    def __init__(self, ch: int, ex: int) -> None:
        """Build the expand/depthwise/project convs and the SE pair."""
        super().__init__()
        self.expand = nn.Conv2d(ch, ex, 1)
        self.dw = nn.Conv2d(ex, ex, 3, padding=1, groups=ex)
        self.se1 = nn.Linear(ex, ex // 2)
        self.se2 = nn.Linear(ex // 2, ex)
        self.project = nn.Conv2d(ex, ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Expand, depthwise, gate by the pooled excite, project."""
        b = x.shape[0]
        h = F.silu(self.dw(F.silu(self.expand(x))))
        s = torch.sigmoid(self.se2(F.relu(self.se1(h.mean((2, 3))))))
        h = h * s.reshape(b, -1, 1, 1)
        return x + self.project(h)


# ---------------------------------------------------------------------------
#  Gated residuals — fused projections and conv gates
# ---------------------------------------------------------------------------


class _GRUFusedGate(nn.Module):
    """GRU update gate — a fused-projection cell step.

    Cho et al. 2014 spelled the deployment way: both input and
    hidden projections run as single fused linears and ``chunk``ed
    into (z, r, n) — so the gate arithmetic is ``add(chunk, chunk)``
    twice plus ``mul(sigmoid, chunk)``, the same-view two-operand
    tuple the scan workloads produce only via ``select``.
    """

    def __init__(self, d: int) -> None:
        """Build the fused input and hidden projections."""
        super().__init__()
        self.w = nn.Linear(d, 3 * d)
        self.u = nn.Linear(d, 3 * d)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """One GRU step: fused projections chunked into the gates."""
        xz, xr, xn = self.w(x).chunk(3, dim=-1)
        hz, hr, hn = self.u(h).chunk(3, dim=-1)
        z = torch.sigmoid(xz + hz)
        r = torch.sigmoid(xr + hr)
        n = torch.tanh(xn + r * hn)
        return (1.0 - z) * n + z * h


class _GatedConvUnit(nn.Module):
    """Gated linear unit over conv features — ``a ⊙ sigmoid(b)``.

    Dauphin et al. 2017 (the conv-seq2seq GLU): one conv produces
    both halves, ``chunk`` splits channels, and the sigmoid half
    gates the linear half — ``mul(chunk, sigmoid)`` under a residual.
    """

    def __init__(self, ch: int) -> None:
        """Build the doubled conv and the residual projection."""
        super().__init__()
        self.conv = nn.Conv1d(ch, 2 * ch, 3, padding=1)
        self.proj = nn.Conv1d(ch, ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Split the conv output; gate the first half, project."""
        a, b = self.conv(x).chunk(2, dim=1)
        return x + self.proj(a * torch.sigmoid(b))


# ---------------------------------------------------------------------------
#  Normalization variants — manual spellings and the norm-free tanh
# ---------------------------------------------------------------------------


class _ManualGroupNorm(nn.Module):
    """GroupNorm spelled by hand — group-reshape, center, scale.

    The ``nn.GroupNorm`` op as a program: reshape the channel axis
    into ``(groups, c·h·w/g)``, subtract the per-group mean, scale by
    the per-group rsqrt-variance, reshape back, apply the affine —
    ``sub(reshape, ·)``, ``mul(sub, rsqrt)`` and the
    ``mul(reshape, reshape)`` affine-broadcast triple.
    """

    def __init__(self, ch: int, groups: int) -> None:
        """Build the affine pair; record the group count."""
        super().__init__()
        self.groups = groups
        self.w = nn.Parameter(torch.ones(ch))
        self.b = nn.Parameter(torch.zeros(ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize within channel groups, then apply the affine."""
        b, c, hh, ww = x.shape
        xg = x.reshape(b, self.groups, c * hh * ww // self.groups)
        m = xg.mean(-1, keepdim=True)
        c0 = xg - m
        v = (c0 * c0).mean(-1, keepdim=True)
        y = (c0 * torch.rsqrt(v + 1e-5)).reshape(b, c, hh, ww)
        return y * self.w.reshape(1, c, 1, 1) + self.b.reshape(
            1, c, 1, 1
        )


class _InstanceNormBroadcast(nn.Module):
    """InstanceNorm spelled by hand — unsqueeze-broadcast statistics.

    The per-sample per-channel spelling: ``mean((2,3))`` drops the
    spatial axes so centering is ``x - m.unsqueeze(-1).unsqueeze(-1)``
    — two stacked ``unsqueeze`` broadcasts feeding a ``sub``, then
    the same double-``unsqueeze`` on the variance and the affine.
    """

    def __init__(self, ch: int) -> None:
        """Build the affine pair and record epsilon."""
        super().__init__()
        self.w = nn.Parameter(torch.ones(ch))
        self.b = nn.Parameter(torch.zeros(ch))
        self.eps = 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Center and scale each sample-channel map, then affine."""
        c = x.shape[1]
        m = x.mean(dim=(2, 3))
        c0 = x - m.unsqueeze(-1).unsqueeze(-1)
        v = (c0 * c0).mean(dim=(2, 3))
        y = c0 * torch.rsqrt(v.unsqueeze(-1).unsqueeze(-1) + self.eps)
        return y * self.w.reshape(1, c, 1, 1) + self.b.reshape(
            1, c, 1, 1
        )


class _RMSNormPerHead(nn.Module):
    """Per-head RMSNorm — reshape into heads, normalize each.

    The multi-head norm variant (QK-norm family): features are
    reshaped to ``(b, t, h, d/h)`` and the root-mean-square is taken
    per head — ``pow(reshape, ·)``, ``mul(reshape, rsqrt)`` and a
    ``mul(·, reshape)`` per-head gain, then ``reshape`` back.
    """

    def __init__(self, d: int, heads: int) -> None:
        """Record the head count and build the per-head gain."""
        super().__init__()
        self.heads = heads
        self.w = nn.Parameter(torch.ones(heads, d // heads))
        self.eps = 1e-6

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize each head's features by its RMS, regroup."""
        b, t, d = x.shape
        h = self.heads
        xh = x.reshape(b, t, h, d // h)
        ms = xh.pow(2).mean(-1, keepdim=True)
        xn = xh * torch.rsqrt(ms + self.eps)
        xn = xn * self.w.reshape(1, 1, h, d // h)
        return xn.reshape(b, t, d)


class _DynamicTanhBlock(nn.Module):
    """Dynamic tanh — the norm-free ``alpha·tanh`` activation block.

    Zhu et al. 2025 (Transformers without normalization): the
    elementwise ``tanh(alpha·x) · gamma + beta`` replaces the norm
    entirely — ``mul(tanh, reshape)`` and ``add(mul, reshape)``
    where the norm-family workloads wrote ``div``/``sub`` chains.
    """

    def __init__(self, d: int) -> None:
        """Build the alpha/gamma/beta tables and the residual MLP."""
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.gamma = nn.Parameter(torch.ones(d))
        self.beta = nn.Parameter(torch.zeros(d))
        self.mlp = nn.Sequential(
            nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the dynamic tanh in place of a norm, then the MLP."""
        y = torch.tanh(self.alpha * x) * self.gamma.reshape(
            1, 1, -1
        ) + self.beta.reshape(1, 1, -1)
        return x + self.mlp(y)


# ---------------------------------------------------------------------------
#  Positional / table ops — bias construction as graph terms
# ---------------------------------------------------------------------------


class _ALiBiAttention(nn.Module):
    """ALiBi attention — per-head slope times in-graph distance.

    Press et al. 2022: the bias is ``-slope_h · |i - j|`` built from
    a broadcast position difference and a per-head slope table —
    ``sub(unsqueeze, unsqueeze)`` for the distance matrix and
    ``mul(reshape, unsqueeze)`` for the slope grid, then
    ``add(·, unsqueeze)`` onto the scores.
    """

    def __init__(self, d: int, heads: int) -> None:
        """Build the projections and the per-head slope table."""
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.slopes = nn.Parameter(torch.rand(heads))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Attend with the slope-scaled distance bias added."""
        b, t, d = x.shape
        h = self.heads
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.reshape(b, t, h, d // h).transpose(1, 2)
        k = k.reshape(b, t, h, d // h).transpose(1, 2)
        v = v.reshape(b, t, h, d // h).transpose(1, 2)
        pos = torch.arange(t, dtype=x.dtype, device=x.device)
        rel = pos.unsqueeze(0) - pos.unsqueeze(1)
        bias = self.slopes.reshape(h, 1, 1) * rel.unsqueeze(0)
        s = q @ k.transpose(-1, -2) * ((d // h) ** -0.5)
        s = s + bias.unsqueeze(0)
        o = torch.softmax(s, dim=-1) @ v
        return self.out(o.transpose(1, 2).reshape(b, t, d))


class _RelPosTableAttention(nn.Module):
    """Relative-position table attention — clamped index gather.

    The T5/Shaw relative-position spelling: the in-graph distance
    matrix is clamped to ``[-m, m]``, shifted to table indices, and
    the learned scalar table is gathered — ``sub(unsqueeze,
    unsqueeze)`` feeding an ``index`` op, then ``add(·, unsqueeze)``
    onto the logits.
    """

    def __init__(self, d: int, span: int) -> None:
        """Build the projections and the relative-position table."""
        super().__init__()
        self.span = span
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.out = nn.Linear(d, d)
        self.table = nn.Parameter(torch.randn(2 * span + 1) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Gather the clamped relative bias and add it to scores."""
        t, d = x.shape[-2], x.shape[-1]
        pos = torch.arange(t, dtype=x.dtype, device=x.device)
        rel = pos.unsqueeze(1) - pos.unsqueeze(0)
        idx = (rel.clamp(-self.span, self.span) + self.span).long()
        bias = self.table[idx]
        s = self.q(x) @ self.k(x).transpose(-1, -2) * (d**-0.5)
        s = s + bias.unsqueeze(0)
        return self.out(torch.softmax(s, dim=-1) @ self.v(x))


class _LearnedPosSlice(nn.Module):
    """Learned absolute embedding — ``x + pe[:, :t]`` slice-add.

    The ViT/GPT positional spelling at its minimal real form: a
    stored ``(max_t, d)`` table sliced to the incoming length and
    added — the ``add(·, slice)`` tuple, then normed and projected
    like every real embedding block.
    """

    def __init__(self, d: int, max_t: int) -> None:
        """Build the position table, the norm and the projection."""
        super().__init__()
        self.pe = nn.Parameter(torch.randn(max_t, d) * 0.02)
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add the sliced positional table, norm, project."""
        return self.out(self.norm(x + self.pe[: x.shape[-2]]))


# ---------------------------------------------------------------------------
#  Second quant/kv shapes — grouped scales and paged cache
# ---------------------------------------------------------------------------


class _GroupedQuantLinear(nn.Module):
    """Grouped-quant dequant linear — per-group scales over input.

    The GPTQ/AWP deployment spelling: the stored weight is
    integer-valued per ``(out, group)`` tile and each group carries a
    scale broadcast along the group — ``mul(param, unsqueeze)`` then
    ``reshape`` back to the dense weight before the matmul.
    ``PerChannelQuantMatmul`` scales whole output rows; this is the
    finer input-group grid real W4 kernels use.
    """

    def __init__(self, d: int, groups: int) -> None:
        """Build the quantized weight tiles and their group scales."""
        super().__init__()
        self.d = d
        self.groups = groups
        w = torch.randn(d, d).reshape(d, groups, d // groups)
        s = (w.abs().amax(-1) / 7.0).clamp_min(1e-3)
        self.w_q = nn.Parameter(torch.round(w / s.unsqueeze(-1)))
        self.scale = nn.Parameter(s)
        self.bias = nn.Parameter(torch.zeros(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Dequantize per group and apply the linear map."""
        w = (self.w_q * self.scale.unsqueeze(-1)).reshape(
            self.d, self.d
        )
        return x @ w.transpose(0, 1) + self.bias


class _PagedKVAttention(nn.Module):
    """Paged-KV decode step — page-flatten, append, slice, scale.

    The vLLM paged-attention spelling at probe scale: keys/values
    live in fixed-size pages flattened to a flat cache, the new
    token's k/v are concatenated on, the cache is truncated to
    capacity by ``slice``, and a per-slot scale table weights the
    logits — ``mul(matmul, reshape)`` over the paged layout.
    """

    def __init__(self, d: int, cap: int) -> None:
        """Build the projections and the per-slot logit scales."""
        super().__init__()
        self.q = nn.Linear(d, d)
        self.kv = nn.Linear(d, 2 * d)
        self.out = nn.Linear(d, d)
        self.cap = cap
        self.slot_scale = nn.Parameter(torch.ones(cap))

    def forward(
        self,
        x: torch.Tensor,
        kp: torch.Tensor,
        vp: torch.Tensor,
    ) -> torch.Tensor:
        """Flatten the pages, append the new k/v, attend the window."""
        b = x.shape[0]
        k_new, v_new = self.kv(x).chunk(2, dim=-1)
        k = torch.cat([kp.reshape(b, -1, x.shape[-1]), k_new], dim=1)[
            :, : self.cap
        ]
        v = torch.cat([vp.reshape(b, -1, x.shape[-1]), v_new], dim=1)[
            :, : self.cap
        ]
        s = (
            self.q(x)
            @ k.transpose(-1, -2)
            * (x.shape[-1] ** -0.5)
            * self.slot_scale.reshape(1, 1, -1)
        )
        return self.out(torch.softmax(s, dim=-1) @ v)


# ---------------------------------------------------------------------------
#  Third expansion — the 2024-25 structures the rewrite catalogs miss
# ---------------------------------------------------------------------------


class _GatedDeltaRule(nn.Module):
    """Gated delta rule — the DeltaNet recurrence under a forget gate.

    Yang et al. 2024 (Gated DeltaNet): ``S ← alpha_t·(I - β_t k kᵀ)·S +
    β_t k vᵀ``, spelled elementwise as
    ``S ← alpha·S + β·k·(v - alpha·kᵀS)ᵀ`` — ``_DeltaRuleBlock``'s rank-1
    erase/write with a per-step scalar decay ``alpha_t`` multiplying the
    whole previous state.  The gate makes the update a
    ``mul(param-broadcast, sub)`` + ``mul(param-broadcast, outer)``
    pattern the shipped scan monoids (affine ``a·h + b``, diagonal
    ``affd``) cannot lift: the coefficient is a data-dependent
    *matrix*, not a scalar map.
    """

    def __init__(self, dk: int, dv: int, steps: int) -> None:
        """Build the k/v/β/alpha projections and record the unroll count."""
        super().__init__()
        self.steps = steps
        self.dk = dk
        self.k = nn.Linear(dk, dk)
        self.v = nn.Linear(dk, dv)
        self.beta = nn.Linear(dk, 1)
        self.dec = nn.Linear(dk, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Unroll the gated delta rule; return the final state."""
        b = x.shape[0]
        kk = self.k(x)  # (B, T, dk)
        vv = self.v(x)  # (B, T, dv)
        bb = torch.sigmoid(self.beta(x))  # (B, T, 1)
        aa = torch.exp(-F.softplus(self.dec(x)))  # (B, T, 1)
        s = torch.zeros(
            b, self.dk, vv.shape[-1], dtype=x.dtype, device=x.device
        )
        for t in range(self.steps):
            k_t = kk[:, t]  # (B, dk)
            v_t = vv[:, t]  # (B, dv)
            a_row = aa[:, t]  # (B, 1)
            a_cell = a_row.unsqueeze(-1)  # (B, 1, 1)
            b_cell = bb[:, t].unsqueeze(-1)  # (B, 1, 1)
            # gated residual read: v_t - alpha·kᵀS  →  (B, dv)
            read = v_t - a_row * (k_t.unsqueeze(1) @ s).squeeze(1)
            # decayed state + rank-1 write: S ← alphaS + β·k⊗read
            s = a_cell * s + b_cell * k_t.unsqueeze(
                -1
            ) * read.unsqueeze(-2)
        return s


class _ChunkedSSD(nn.Module):
    """Mamba-2 chunked SSD — intra-chunk masked attention + state pass.

    Dao & Gu 2024 (the SSD layer): the scalar-decay scan
    ``S_t = a_t·S + k_t ⊗ v_t`` in its chunked dual form — inside a
    chunk it is quadratic attention masked by the in-graph segment
    decay ``exp(cs_i - cs_j)`` (``cs`` the cumsum of the log-decays),
    across chunks the state ``S`` (an ``(n, p)`` matrix) carries
    ``Σ_j exp(cs_end - cs_j)·k_j ⊗ v_j`` and reads as
    ``exp(cs_i - cs_prev)·(q_i·S)``.  The ``cumsum`` → ``sub`` →
    ``exp`` → ``tril`` decay-mask chain plus the three einsum sites
    are the real chunked-scan spelling — not the unrolled
    ``select`` loop of ``_MambaSelectiveScan``.
    """

    def __init__(self, d: int, heads: int, n: int, chunk: int) -> None:
        """Build the q/k/v/decay projections and the output map."""
        super().__init__()
        self.h = heads
        self.n = n
        self.chunk = chunk
        self.q = nn.Linear(d, heads * n)
        self.k = nn.Linear(d, heads * n)
        self.v = nn.Linear(d, heads * (d // heads))
        self.dec = nn.Linear(d, heads)
        self.out = nn.Linear(d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Scan chunk-by-chunk: masked intra-chunk + decayed state."""
        b, t, d = x.shape
        h, n, c = self.h, self.n, self.chunk
        q = self.q(x).reshape(b, t, h, n).transpose(1, 2)  # (b,h,t,n)
        k = self.k(x).reshape(b, t, h, n).transpose(1, 2)
        v = self.v(x).reshape(b, t, h, d // h).transpose(1, 2)
        a = -F.softplus(self.dec(x))  # (b,t,h) — log-decay per step
        cs = a.cumsum(1).transpose(1, 2)  # (b,h,t) — prefix sums
        csp = torch.cat(
            [torch.zeros(b, h, 1, dtype=x.dtype, device=x.device), cs],
            dim=2,
        )
        tri = torch.tril(
            torch.ones(c, c, dtype=x.dtype, device=x.device)
        )
        s_state = torch.zeros(
            b, h, n, d // h, dtype=x.dtype, device=x.device
        )
        outs = []
        for c0 in range(0, t, c):
            cq = q[:, :, c0 : c0 + c]  # (b,h,c,n)
            ck = k[:, :, c0 : c0 + c]
            cv = v[:, :, c0 : c0 + c]  # (b,h,c,p)
            seg = cs[:, :, c0 : c0 + c]  # (b,h,c)
            cprev = csp[:, :, c0]  # (b,h) — cumsum at chunk start
            cend = cs[:, :, c0 + c - 1]  # (b,h) — cumsum at chunk end
            # intra-chunk decay mask: exp(cs_i - cs_j), i ≥ j
            dmat = seg.unsqueeze(-1) - seg.unsqueeze(-2)
            dec_intra = torch.exp(dmat) * tri
            scores = (cq @ ck.transpose(-1, -2)) / (n**0.5)
            y = (scores * dec_intra) @ cv  # (b,h,c,p)
            # inter-chunk read: decay from chunk start times q·S
            dec_q = torch.exp(seg - cprev.unsqueeze(-1))  # (b,h,c)
            y = y + (cq @ s_state) * dec_q.unsqueeze(-1)
            outs.append(y)
            # state update: decay old S to the chunk end, add the
            # decay-weighted rank-1 writes of this chunk's keys
            dec_k = torch.exp(cend.unsqueeze(-1) - seg)  # (b,h,c)
            dec_s = torch.exp(cend - cprev)  # (b,h)
            s_state = dec_s.unsqueeze(-1).unsqueeze(
                -1
            ) * s_state + torch.einsum(
                "bhcn,bhcp->bhnp", ck * dec_k.unsqueeze(-1), cv
            )
        y = torch.cat(outs, dim=2)  # (b,h,t,p)
        return self.out(y.transpose(1, 2).reshape(b, t, d))


class _DecoupledMLA(nn.Module):
    """MLA with a decoupled rope key — the DeepSeek/Kimi spelling.

    DeepSeek-V2/V3 (and Kimi k2): queries and the kv stream both pass
    through a low-rank latent (``W_dq``/``W_dkv`` down, ``W_uq``/
    ``W_uk``/``W_uv`` up), and the rotary component lives on a
    *decoupled* per-head pair — a shared ``k_r`` broadcast across
    heads — concatenated after the compressed dims.  The intake's
    ``MultiLatentAttention`` compresses then up-projects through
    ``sdpa`` with no rotary anywhere; this is the production
    spelling: ``cat`` of compressed + roped slices, broadcast
    ``expand`` of the shared rope key, manual softmax over the
    widened head dim.
    """

    def __init__(self, d: int, heads: int, rank: int, rpe: int) -> None:
        """Build the latent down/up maps and the decoupled rope key."""
        super().__init__()
        self.heads = heads
        self.rpe = rpe
        dn = d // heads - rpe
        self.dq = nn.Linear(d, rank)
        self.uq = nn.Linear(rank, heads * dn)
        self.qr = nn.Linear(rank, heads * rpe)
        self.dkv = nn.Linear(d, rank)
        self.uk = nn.Linear(rank, heads * dn)
        self.kr = nn.Linear(d, rpe)
        self.uv = nn.Linear(rank, d)
        self.out = nn.Linear(d, d)

    @staticmethod
    def _rope_tables(
        t: int, r: int, dtype: torch.dtype, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the ``(t, r/2)`` cosine/sine tables for rotate-half."""
        inv = torch.pow(
            torch.tensor(10000.0, dtype=dtype, device=device),
            -torch.arange(0, r, 2, dtype=dtype, device=device) / r,
        )
        ang = torch.outer(
            torch.arange(t, dtype=dtype, device=device), inv
        )
        return ang.cos(), ang.sin()

    @staticmethod
    def _rot(
        z: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        """Rotate-half *z* ``(b, t, h, r)`` by the ``(t, r/2)`` tables."""
        z1, z2 = z.chunk(2, dim=-1)
        c = cos.unsqueeze(1)  # (t, 1, r/2)
        s = sin.unsqueeze(1)
        return torch.cat([z1 * c - z2 * s, z1 * s + z2 * c], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Attend with compressed q/kv plus the decoupled rope dims."""
        b, t, d = x.shape
        h, r = self.heads, self.rpe
        dn = d // h - r
        cos, sin = self._rope_tables(t, r, x.dtype, x.device)
        qc = self.dq(x)
        qn = self.uq(qc).reshape(b, t, h, dn)
        qp = self._rot(self.qr(qc).reshape(b, t, h, r), cos, sin)
        q = torch.cat([qn, qp], dim=-1).transpose(1, 2)  # (b,h,t,dh)
        c = self.dkv(x)
        kn = self.uk(c).reshape(b, t, h, dn)
        kr = self._rot(self.kr(x).reshape(b, t, 1, r), cos, sin)
        k = torch.cat([kn, kr.expand(b, t, h, r)], dim=-1).transpose(
            1, 2
        )
        v = self.uv(c).reshape(b, t, h, d // h).transpose(1, 2)
        s = q @ k.transpose(-1, -2) / ((d // h) ** 0.5)
        o = torch.softmax(s, dim=-1) @ v
        return self.out(o.transpose(1, 2).reshape(b, t, d))


class _PartialRotary(nn.Module):
    """Partial rotary — rope on a head-dim slice, pass-through tail.

    The GPT-NeoX / Qwen ``rotary_pct < 1`` spelling: only the first
    ``rdim`` channels of q and k rotate (rotate-half, not the
    interleaved pairs of ``_RotaryPair``); the remainder passes
    through untouched and the halves re-join by ``cat`` — the
    ``slice`` + ``cat`` boundary the rope-family banks never wrote.
    """

    def __init__(self, d: int, rdim: int) -> None:
        """Build the q/k/v/out projections; record the rotary width."""
        super().__init__()
        self.rdim = rdim
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.out = nn.Linear(d, d)

    @staticmethod
    def _rot(z: torch.Tensor) -> torch.Tensor:
        """Rotate-half the last ``r`` dims of *z* ``(b, t, r)``."""
        t, r = z.shape[-2], z.shape[-1]
        inv = torch.pow(
            torch.tensor(10000.0, dtype=z.dtype, device=z.device),
            -torch.arange(0, r, 2, dtype=z.dtype, device=z.device) / r,
        )
        ang = torch.outer(
            torch.arange(t, dtype=z.dtype, device=z.device), inv
        )
        z1, z2 = z.chunk(2, dim=-1)
        cos, sin = ang.cos(), ang.sin()  # (t, r/2)
        return torch.cat(
            [z1 * cos - z2 * sin, z1 * sin + z2 * cos], dim=-1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate the rotary slice of q and k; attend; project."""
        r = self.rdim
        q = torch.cat(
            [self._rot(self.q(x)[..., :r]), self.q(x)[..., r:]], dim=-1
        )
        k = torch.cat(
            [self._rot(self.k(x)[..., :r]), self.k(x)[..., r:]], dim=-1
        )
        s = q @ k.transpose(-1, -2) / (x.shape[-1] ** 0.5)
        return self.out(torch.softmax(s, dim=-1) @ self.v(x))


class _SigmoidAttention(nn.Module):
    """Sigmoid attention — ``sig(QKᵀ/√d + b)·V``, softmax-free scoring.

    Ramapuram et al. 2024 (Sigmoid Self-Attention; the SigLIP-style
    nonlinearity applied to scores): the attention matrix is a
    per-pair sigmoid with a learned bias, NOT a normalized convex
    combination — the denominator softmax spends is dropped.  The
    ``sigmoid(matmul)`` → ``matmul`` spine is a scoring rule the
    softmax-centric fusion banks have no rule for.
    """

    def __init__(self, d: int, heads: int) -> None:
        """Build the packed qkv/out maps and the score bias."""
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.bias = nn.Parameter(torch.tensor(-2.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Score by sigmoid (unnormalized), mix values, project."""
        b, t, d = x.shape
        h = self.heads
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.reshape(b, t, h, d // h).transpose(1, 2)
        k = k.reshape(b, t, h, d // h).transpose(1, 2)
        v = v.reshape(b, t, h, d // h).transpose(1, 2)
        s = q @ k.transpose(-1, -2) / ((d // h) ** 0.5) + self.bias
        o = torch.sigmoid(s) @ v
        return self.out(o.transpose(1, 2).reshape(b, t, d))


class _RealGatedLRU(nn.Module):
    """RG-LRU — the Hawk/Griffin real-gated linear recurrence.

    De et al. 2024 (Griffin/Hawk): ``h_t = a_t ⊙ h_{t-1} +
    √(1 - a_t²) ⊙ (i_t ⊙ x_t)`` where
    ``a_t = exp(-c·softplus(Λ)·r_t)`` with ``r_t``/``i_t`` sigmoid
    recurrence and input gates — a *vector* data-dependent decay
    whose magnitude is restored by the ``sqrt(1 - a²)`` normalization
    term.  Unrolled over ``select`` slices it is the third distinct
    elementwise-recurrence spelling in the zoo (Mamba's rank-1 write,
    GLA's matrix gate, this one's magnitude-corrected gate) — and
    none of them is a carrier lift the banks own.
    """

    def __init__(self, d: int) -> None:
        """Build the input/gate/output maps and the log-rate table."""
        super().__init__()
        self.inp = nn.Linear(d, d)
        self.r = nn.Linear(d, d)
        self.i = nn.Linear(d, d)
        self.out = nn.Linear(d, d)
        self.lam = nn.Parameter(torch.randn(d) * 0.1)
        self.c = 8.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Unroll the gated recurrence; add the projected read-out."""
        b, t, d = x.shape
        u = self.inp(x)
        r = torch.sigmoid(self.r(x))
        i = torch.sigmoid(self.i(x))
        lam = F.softplus(self.lam)  # (d,) positive decay rate
        a = torch.exp(-self.c * lam * r)  # (b,t,d) in (0, 1]
        h = torch.zeros(b, d, dtype=x.dtype, device=x.device)
        outs = []
        for tt in range(t):
            h = a[:, tt] * h + torch.sqrt(1.0 - a[:, tt].square()) * (
                i[:, tt] * u[:, tt]
            )
            outs.append(h)
        return x + self.out(torch.stack(outs, dim=1))


class _AdaGNConv(nn.Module):
    """AdaGN conv block — FiLM scale+shift on a group-normed map.

    The DiT/SiT UNet conditioning at conv scale: a conditioning
    vector produces per-channel ``(scale, shift, gate)``, the
    affine-free GroupNorm output is modulated ``y·(1+s)+b``, and the
    conv branch re-enters through the gate — three
    ``mul/add(·, unsqueeze(unsqueeze))`` broadcast sites on a 4-D
    map, where ``_TimeCondConv`` only adds a bias.
    """

    def __init__(self, ch: int, cdim: int, groups: int) -> None:
        """Build the affine-free norm, the conv and the cond table."""
        super().__init__()
        self.norm = nn.GroupNorm(groups, ch, affine=False)
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)
        self.cond = nn.Linear(cdim, 3 * ch)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Scale+shift the norm by the conditioning, gate the conv."""
        scale, shift, gate = self.cond(c)[:, :, None, None].chunk(
            3, dim=1
        )
        h = self.norm(x) * (1.0 + scale) + shift
        return x + gate * self.conv(F.silu(h))


class _SharedSwiGLUMoE(nn.Module):
    """Banked-SwiGLU MoE — softmax top-k over two-layer experts.

    The production decoder-MoE shape (Mixtral/DeepSeek block MLPs):
    each routed expert is a *two-layer* SwiGLU banked as
    ``(E, d, 2f)`` / ``(E, f, d)`` weight tensors (the ``silu(a)·b``
    gate inside a 4-D ``einsum`` site), the router is softmax top-k
    with the ``one_hot`` mask, and an always-on shared expert adds
    in parallel.  ``TopKRouterMoE`` is single-layer with no shared
    arm; ``DeepSeekSigMoE`` is sigmoid-routed and single-layer.
    """

    def __init__(self, d: int, n_exp: int, k: int, f: int) -> None:
        """Build the router, the banked expert pair and the shared arm."""
        super().__init__()
        self.k = k
        self.router = nn.Linear(d, n_exp)
        self.w1 = nn.Parameter(torch.randn(n_exp, d, 2 * f) / (d**0.5))
        self.w2 = nn.Parameter(torch.randn(n_exp, f, d) / (f**0.5))
        self.shared = nn.Sequential(
            nn.Linear(d, 2 * d), nn.SiLU(), nn.Linear(2 * d, d)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Route top-k through the SwiGLU bank; add the shared arm."""
        p = torch.softmax(self.router(x), dim=-1)
        _topv, topi = torch.topk(p, self.k, dim=-1)
        mask = F.one_hot(topi, p.shape[-1]).to(x.dtype).amax(dim=-2)
        w = p * mask
        h = torch.einsum("btd,edg->bteg", x, self.w1)  # g = 2f
        a, b2 = h.chunk(2, dim=-1)
        act = F.silu(a) * b2  # (b,t,e,f)
        y = torch.einsum("btef,efg->bteg", act, self.w2)
        return self.shared(x) + torch.einsum("bte,bteg->btg", w, y)


class _AttnLogitSoftcap(nn.Module):
    """Attention logit soft-capping — Gemma-2's ``cap·tanh(s/cap)``.

    The bounded-logit trick applied *inside* the attention scores
    (Gemma-2 uses a 50.0 cap on logits): ``tanh`` saturates the
    ``(t, t)`` score grid before the softmax — a
    ``tanh(div(matmul))`` → ``softmax`` chain the fusion banks and
    the zoo's own head-level ``SoftcapLogitHead`` don't spell.
    """

    def __init__(self, d: int, cap: float) -> None:
        """Build the q/k/v/out projections and record the cap."""
        super().__init__()
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.out = nn.Linear(d, d)
        self.cap = cap

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Softcap the logits, softmax, mix values, project."""
        d = x.shape[-1]
        s = self.q(x) @ self.k(x).transpose(-1, -2) * (d**-0.5)
        s = self.cap * torch.tanh(s / self.cap)
        return self.out(torch.softmax(s, dim=-1) @ self.v(x))


class _NormedSoftcapHead(nn.Module):
    """Final-norm + softcapped logits — the Gemma-2 output head.

    The classifier's real tail: a manual RMS norm (``pow``/``mean``/
    ``rsqrt``/``mul``) feeds the head and the logits are soft-capped
    at 30 — the norm → linear → cap chain as one term, not the bare
    ``cap·tanh`` of ``SoftcapLogitHead``.
    """

    def __init__(self, d: int, ncls: int) -> None:
        """Build the gain, the head, record cap and eps."""
        super().__init__()
        self.g = nn.Parameter(torch.ones(d))
        self.head = nn.Linear(d, ncls)
        self.cap = 30.0
        self.eps = 1e-6

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """RMS-norm, project, softcap the logits."""
        ms = x.pow(2).mean(-1, keepdim=True)
        xn = x * torch.rsqrt(ms + self.eps) * self.g
        return self.cap * torch.tanh(self.head(xn) / self.cap)


class _SquaredReluFFN(nn.Module):
    """Squared-ReLU FFN — the Primer ``relu(Wx)²`` block.

    So et al. 2021 (Primer): the activation is ``relu(·)`` squared —
    ``pow(relu(linear))`` — inside the usual pre-norm residual MLP.
    A two-op chain no fusion bank owns (ReGLU gates a *product*,
    this is a pointwise power of a rectifier).
    """

    def __init__(self, d: int) -> None:
        """Build the norm and the up/down maps."""
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.up = nn.Linear(d, 2 * d)
        self.down = nn.Linear(2 * d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the squared-relu MLP under a residual."""
        h = torch.relu(self.up(self.norm(x))).square()
        return x + self.down(h)


class _DiffAttention(nn.Module):
    """Differential attention — two softmaxes, subtracted.

    Ye et al. 2024 (DIFF Transformer): each head's q/k split into
    two sub-groups and the attention is
    ``softmax(s₁) - λ·softmax(s₂)`` with
    ``λ = e^{λ₁}e^{λ₂} - e^{λ₃}e^{λ₄}`` a per-head learned constant —
    the ``sub(softmax, mul(param, softmax))`` noise-cancellation
    spelling.  Every softmax rule in the banks targets ONE softmax;
    a difference of two is uncatalogued.
    """

    def __init__(self, d: int, heads: int) -> None:
        """Build the split q/k, shared v, out map and the λ params."""
        super().__init__()
        self.heads = heads
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d // 2)
        self.out = nn.Linear(d // 2, d)
        self.l1 = nn.Parameter(torch.randn(heads) * 0.1)
        self.l2 = nn.Parameter(torch.randn(heads) * 0.1)
        self.l3 = nn.Parameter(torch.randn(heads) * 0.1)
        self.l4 = nn.Parameter(torch.randn(heads) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Attend by the difference of two softmaxes per head."""
        b, t, d = x.shape
        h = self.heads
        dh = d // (2 * h)
        q = self.q(x).reshape(b, t, 2, h, dh)
        k = self.k(x).reshape(b, t, 2, h, dh)
        q1, q2 = q.unbind(2)
        k1, k2 = k.unbind(2)
        v = self.v(x).reshape(b, t, h, dh).transpose(1, 2)
        s1 = q1.transpose(1, 2) @ k1.transpose(1, 2).transpose(-1, -2)
        s2 = q2.transpose(1, 2) @ k2.transpose(1, 2).transpose(-1, -2)
        s1 = s1 / (dh**0.5)
        s2 = s2 / (dh**0.5)
        lam = (
            torch.exp(self.l1) * torch.exp(self.l2)
            - torch.exp(self.l3) * torch.exp(self.l4)
        ).reshape(1, h, 1, 1)
        att = torch.softmax(s1, dim=-1) - lam * torch.softmax(
            s2, dim=-1
        )
        o = att @ v
        return self.out(o.transpose(1, 2).reshape(b, t, h * dh))


class _TaylorLinearAttention(nn.Module):
    """Taylor-feature linear attention — the "Based" kernel trick.

    Arora et al. 2024: the softmax kernel is approximated by a
    feature map ``φ(x) = cat(1, x, x²/2)`` and causality is kept by
    prefix-summing the outer products — ``cumsum`` over an
    ``einsum``-built ``(t, n, p)`` accumulator, normalized by the
    running feature mass.  Neither the intake's cumsum
    ``LinearAttention`` (elementwise feature map only) nor the zoo's
    matrix-state scans spell the ``(φq)·(Σ φk⊗v) / (φq)·(Σ φk)``
    quotient.
    """

    def __init__(self, d: int) -> None:
        """Build the q/k/v projections and the output map."""
        super().__init__()
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.out = nn.Linear(d, d)

    @staticmethod
    def _feat(z: torch.Tensor) -> torch.Tensor:
        """Return ``φ(z) = cat(1, z, z²/2)`` — the 2nd-order map."""
        one = torch.ones(
            *z.shape[:-1], 1, dtype=z.dtype, device=z.device
        )
        return torch.cat([one, z, z * z / 2.0], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Causal prefix-sum attention under the Taylor feature map."""
        fq = self._feat(self.q(x))  # (b,t,2d+1)
        fk = self._feat(self.k(x))
        v = self.v(x)  # (b,t,d)
        kv = torch.einsum("btn,btp->btnp", fk, v)
        acc = kv.cumsum(1)  # (b,t,n,p) running key-value state
        num = torch.einsum("btn,btnp->btp", fq, acc)
        den = torch.einsum("btn,btn->bt", fq, fk.cumsum(1))
        return self.out(num / den.unsqueeze(-1).clamp_min(1e-3))


class _TernaryWeightLinear(nn.Module):
    """Ternary-weight linear — the BitNet b1.58 dequant spelling.

    Ma et al. 2024: weights are quantized to ``{-1, 0, +1}`` by
    ``round(clamp(W/gamma))`` with ``gamma = mean|W|`` a per-tensor absmean
    scale, then the matmul runs on the dequantized ``gamma·W_q`` —
    ``round`` + ``clamp`` + the absmean-``div`` without the STE
    detach path ``STEQuantLinear`` spells (here the quantized weight
    IS the forward weight, no gradient fiction needed).
    """

    def __init__(self, d: int) -> None:
        """Build the dense map whose weight is quantized in-graph."""
        super().__init__()
        self.lin = nn.Linear(d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize the weight ternary-wise and apply the map."""
        g = self.lin.weight.abs().mean()
        wq = torch.round((self.lin.weight / g).clamp(-1.0, 1.0))
        return F.linear(x, wq * g, self.lin.bias)


class _SinkWindowAttention(nn.Module):
    """Attention-sink + sliding window — the StreamingLLM mask.

    Xiao et al. 2023: keep the band of the last ``window`` keys AND
    the first ``sinks`` columns (the attention sinks) — the keep
    mask is the union ``band + sink`` built in-graph as
    ``clamp_max(tril - tril + lt(arange))``, then the usual
    ``mul + (keep-1)·1e4`` masking.  ``SlidingWindowAttention``
    spells the band alone; the sink prefix is a different mask
    grammar (a column stripe, not a diagonal).
    """

    def __init__(self, d: int, window: int, sinks: int) -> None:
        """Build the projections; record window and sink counts."""
        super().__init__()
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.out = nn.Linear(d, d)
        self.window = window
        self.sinks = sinks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Attend inside band + sink-columns."""
        t, d = x.shape[-2], x.shape[-1]
        s = (self.q(x) @ self.k(x).transpose(-1, -2)) * (d**-0.5)
        ones = torch.ones(t, t, dtype=x.dtype, device=x.device)
        band = torch.tril(ones) - torch.tril(ones, -1 - self.window)
        pos = torch.arange(t, dtype=x.dtype, device=x.device)
        sink = (pos.unsqueeze(0) < self.sinks).to(x.dtype)  # (1,t)
        keep = (band + sink).clamp_max(1.0)
        s = s * keep + (keep - 1.0) * 1e4
        return self.out(torch.softmax(s, dim=-1) @ self.v(x))


# ---------------------------------------------------------------------------
#  The registry — same Workload thunks as intake, disjoint names
# ---------------------------------------------------------------------------


def _r(*shape: int) -> torch.Tensor:
    """Return a fresh fp64 CPU example input (intake's convention)."""
    return torch.randn(tuple(shape), dtype=torch.float64)


def _rand(*shape: int) -> torch.Tensor:
    """Return a fresh fp64 uniform input (adjacency-like feeds)."""
    return torch.rand(tuple(shape), dtype=torch.float64)


def zoo(d: int = 16, t: int = 8, b: int = 2) -> list[Workload]:
    """Return the held-out registry — thunks like the intake's.

    Every workload builds a fresh module plus its example input under
    the caller's seed, exactly like ``intake.candidates`` — the same
    ``(model, feed)`` contract, ``kind="zoo"`` for the report tables,
    and ``purpose_built=False``: these are architectures chosen for
    family coverage, not spellings of a known law's region.  ``d``
    is the model width — the default keeps the honest small-scale
    probe; a bigger ``d`` checks the wins at sizes where pointwise
    tails carry real work.
    """
    return [
        # --- attention variants ----------------------------------
        Workload(
            "GatedAttentionUnit",
            lambda: (_GatedAttentionUnit(d), _r(b, t, d)),
            kind="zoo",
        ),
        Workload(
            "AdditiveAttention",
            lambda: (
                _AdditiveAttention(d),
                (_r(b, 4, d), _r(b, t, d), _r(b, t, d)),
            ),
            kind="zoo",
        ),
        Workload(
            "TalkingHeadsAttention",
            lambda: (_TalkingHeadsAttention(d, 4), _r(b, t, d)),
            kind="zoo",
        ),
        Workload(
            "CosineAttention",
            lambda: (_CosineAttention(d, 4), _r(b, t, d)),
            kind="zoo",
        ),
        # --- conditioning ----------------------------------------
        Workload(
            "FiLMHead",
            lambda: (_FiLMHead(d, 8), (_r(b, t, d), _r(b, 8))),
            kind="zoo",
        ),
        Workload(
            "AdaLNBlock",
            lambda: (_AdaLNBlock(d, 8), (_r(b, t, d), _r(b, 8))),
            kind="zoo",
        ),
        Workload(
            "LoRAAdapter",
            lambda: (_LoRAAdapter(d, 4), _r(4, d)),
            kind="zoo",
        ),
        # --- token mixers / spectral / graphs --------------------
        Workload(
            "MLPMixerBlock",
            lambda: (_MLPMixerBlock(d, t), _r(b, t, d)),
            kind="zoo",
        ),
        Workload(
            "FNOBlock",
            lambda: (_FNOBlock(8, 4), _r(1, 8, 8, 8)),
            kind="zoo",
        ),
        Workload(
            "GCNLayer",
            lambda: (_GCNLayer(d), (_r(b, 6, d), _rand(b, 6, 6))),
            kind="zoo",
        ),
        Workload(
            "AffineCoupling",
            lambda: (_AffineCoupling(d), _r(4, d)),
            kind="zoo",
        ),
        # --- routing / gating ------------------------------------
        Workload(
            "SoftSlotMoE",
            lambda: (_SoftSlotMoE(d, 4), _r(b, t, d)),
            kind="zoo",
        ),
        Workload(
            "ExpertChoiceMoE",
            lambda: (_ExpertChoiceMoE(d, 4, 3), _r(b, t, d)),
            kind="zoo",
        ),
        Workload(
            "CapsuleRouting",
            lambda: (_CapsuleRouting(d, 8, 4), _r(b, t, d)),
            kind="zoo",
        ),
        # --- recurrence / scan / conv ----------------------------
        Workload(
            "ChunkedRetention",
            lambda: (_ChunkedRetention(d), _r(b, t, d)),
            kind="zoo",
        ),
        Workload(
            "DeepEquilibrium",
            lambda: (_DeepEquilibrium(d, 3), _r(4, d)),
            kind="zoo",
        ),
        Workload(
            "DeltaRuleBlock",
            lambda: (_DeltaRuleBlock(d, d, 4), _r(b, 4, d)),
            kind="zoo",
        ),
        Workload(
            "WaveNetGate",
            lambda: (_WaveNetGate(8), _r(1, 8, 16)),
            kind="zoo",
        ),
        Workload(
            "TimeCondConv",
            lambda: (_TimeCondConv(8, 4), (_r(1, 8, 8, 8), _r(1))),
            kind="zoo",
        ),
        # --- heads / objectives ----------------------------------
        Workload(
            "MoSHead",
            lambda: (_MoSHead(d, 10, 3), _r(4, d)),
            kind="zoo",
        ),
        Workload(
            "CovarianceHead",
            lambda: (_CovarianceHead(), _r(8, d)),
            kind="zoo",
        ),
        Workload(
            "SplineKAN",
            lambda: (_SplineKAN(d, 8), _r(4, d)),
            kind="zoo",
        ),
        Workload(
            "HighwayGate",
            lambda: (_HighwayGate(d), _r(4, d)),
            kind="zoo",
        ),
        # --- SSM / linear-attention / long-conv --------------------
        Workload(
            "MambaSelectiveScan",
            lambda: (_MambaSelectiveScan(d, 4, t), _r(b, t, d)),
            kind="zoo",
        ),
        Workload(
            "GatedLinearAttention",
            lambda: (_GatedLinearAttention(d, t), _r(b, t, d)),
            kind="zoo",
        ),
        Workload(
            "RWKVMixBlock",
            lambda: (_RWKVMixBlock(d), _r(b, t, d)),
            kind="zoo",
        ),
        Workload(
            "HyenaLongConv",
            lambda: (_HyenaLongConv(d, 4), _r(b, t, d)),
            kind="zoo",
        ),
        # --- routing: deployment-shape dispatch --------------------
        Workload(
            "TopKRouterMoE",
            lambda: (_TopKRouterMoE(d, 4, 2), _r(b, t, d)),
            kind="zoo",
        ),
        Workload(
            "DeepSeekSigMoE",
            lambda: (_DeepSeekSigMoE(d, 4, 2), _r(b, t, d)),
            kind="zoo",
        ),
        # --- quantization / cache ----------------------------------
        Workload(
            "STEQuantLinear",
            lambda: (_STEQuantLinear(d), _r(4, d)),
            kind="zoo",
        ),
        Workload(
            "PerChannelQuantMatmul",
            lambda: (_PerChannelQuantMatmul(d), _r(4, d)),
            kind="zoo",
        ),
        Workload(
            "KVCacheAttention",
            lambda: (
                _KVCacheAttention(d, 4),
                (_r(b, 1, d), _r(b, 5, d), _r(b, 5, d)),
            ),
            kind="zoo",
        ),
        # --- modern block details ----------------------------------
        Workload(
            "GEGLUBlock",
            lambda: (_GEGLUBlock(d), _r(b, t, d)),
            kind="zoo",
        ),
        Workload(
            "SkipInitNet",
            lambda: (_SkipInitNet(d), _r(b, t, d)),
            kind="zoo",
        ),
        Workload(
            "SoftcapLogitHead",
            lambda: (_SoftcapLogitHead(d, 10), _r(4, d)),
            kind="zoo",
        ),
        Workload(
            "RotaryPair",
            lambda: (_RotaryPair(d), _r(b, t, d)),
            kind="zoo",
        ),
        # --- sparse / patterned attention --------------------------
        Workload(
            "SlidingWindowAttention",
            lambda: (_SlidingWindowAttention(d, 3), _r(b, t, d)),
            kind="zoo",
        ),
        Workload(
            "DilatedBlockAttention",
            lambda: (_DilatedBlockAttention(d, 2, 4), _r(b, t, d)),
            kind="zoo",
        ),
        Workload(
            "BlockSparseAttention",
            lambda: (
                _BlockSparseAttention(d, max(t // 2, 1), 2, 2),
                _r(b, t, d),
            ),
            kind="zoo",
        ),
        # --- depthwise-conv blocks -----------------------------------
        Workload(
            "ConvNeXtStage",
            lambda: (_ConvNeXtBlock(8), _r(1, 8, 8, 8)),
            kind="zoo",
        ),
        Workload(
            "SqueezeExcite",
            lambda: (_SqueezeExcite(8, 2), _r(1, 8, 8, 8)),
            kind="zoo",
        ),
        Workload(
            "MBConvSEBlock",
            lambda: (_MBConvBlock(8, 16), _r(1, 8, 8, 8)),
            kind="zoo",
        ),
        # --- gated residuals -----------------------------------------
        Workload(
            "GRUFusedGate",
            lambda: (_GRUFusedGate(d), (_r(4, d), _r(4, d))),
            kind="zoo",
        ),
        Workload(
            "GatedConvUnit",
            lambda: (_GatedConvUnit(8), _r(1, 8, 16)),
            kind="zoo",
        ),
        # --- normalization variants -----------------------------------
        Workload(
            "ManualGroupNorm",
            lambda: (_ManualGroupNorm(8, 4), _r(1, 8, 8, 8)),
            kind="zoo",
        ),
        Workload(
            "InstanceNormBroadcast",
            lambda: (_InstanceNormBroadcast(8), _r(1, 8, 8, 8)),
            kind="zoo",
        ),
        Workload(
            "RMSNormPerHead",
            lambda: (_RMSNormPerHead(d, 4), _r(b, t, d)),
            kind="zoo",
        ),
        Workload(
            "DynamicTanhBlock",
            lambda: (_DynamicTanhBlock(d), _r(b, t, d)),
            kind="zoo",
        ),
        # --- positional / table ops -----------------------------------
        Workload(
            "ALiBiSlopeAttention",
            lambda: (_ALiBiAttention(d, 4), _r(b, t, d)),
            kind="zoo",
        ),
        Workload(
            "RelPosTableAttention",
            lambda: (_RelPosTableAttention(d, 3), _r(b, t, d)),
            kind="zoo",
        ),
        Workload(
            "LearnedPosSlice",
            lambda: (_LearnedPosSlice(d, max(t, 16)), _r(b, t, d)),
            kind="zoo",
        ),
        # --- second quant/kv shapes -----------------------------------
        Workload(
            "GroupedQuantLinear",
            lambda: (_GroupedQuantLinear(d, 4), _r(4, d)),
            kind="zoo",
        ),
        Workload(
            "PagedKVAttention",
            lambda: (
                _PagedKVAttention(d, 4),
                (_r(b, 1, d), _r(b, 2, 3, d), _r(b, 2, 3, d)),
            ),
            kind="zoo",
        ),
        # --- third expansion: uncatalogued 2024-25 structures ---------
        # gated/matrix-state scans and chunked dual forms
        Workload(
            "GatedDeltaRule",
            lambda: (_GatedDeltaRule(d, d, 4), _r(b, 4, d)),
            kind="zoo",
        ),
        Workload(
            "ChunkedSSD",
            lambda: (_ChunkedSSD(d, 4, 4, 4), _r(b, t, d)),
            kind="zoo",
        ),
        Workload(
            "RealGatedLRU",
            lambda: (_RealGatedLRU(d), _r(b, t, d)),
            kind="zoo",
        ),
        # compressed-latent + partial-rope attention spellings
        Workload(
            "DecoupledMLA",
            lambda: (_DecoupledMLA(d, 4, 8, 2), _r(b, t, d)),
            kind="zoo",
        ),
        Workload(
            "PartialRotary",
            lambda: (_PartialRotary(d, d // 4), _r(b, t, d)),
            kind="zoo",
        ),
        Workload(
            "SigmoidAttention",
            lambda: (_SigmoidAttention(d, 4), _r(b, t, d)),
            kind="zoo",
        ),
        Workload(
            "DiffAttention",
            lambda: (_DiffAttention(d, 4), _r(b, t, d)),
            kind="zoo",
        ),
        Workload(
            "AttnLogitSoftcap",
            lambda: (_AttnLogitSoftcap(d, 50.0), _r(b, t, d)),
            kind="zoo",
        ),
        Workload(
            "SinkWindowAttention",
            lambda: (_SinkWindowAttention(d, 3, 1), _r(b, t, d)),
            kind="zoo",
        ),
        Workload(
            "TaylorLinearAttention",
            lambda: (_TaylorLinearAttention(d), _r(b, t, d)),
            kind="zoo",
        ),
        # conditioned conv / routing / head details
        Workload(
            "AdaGNConv",
            lambda: (_AdaGNConv(8, 4, 4), (_r(1, 8, 8, 8), _r(1, 4))),
            kind="zoo",
        ),
        Workload(
            "SharedSwiGLUMoE",
            lambda: (_SharedSwiGLUMoE(d, 4, 2, 8), _r(b, t, d)),
            kind="zoo",
        ),
        Workload(
            "NormedSoftcapHead",
            lambda: (_NormedSoftcapHead(d, 10), _r(4, d)),
            kind="zoo",
        ),
        Workload(
            "SquaredReluFFN",
            lambda: (_SquaredReluFFN(d), _r(b, t, d)),
            kind="zoo",
        ),
        Workload(
            "TernaryWeightLinear",
            lambda: (_TernaryWeightLinear(d), _r(4, d)),
            kind="zoo",
        ),
    ]


def zoo_variants(d: int = 16, t: int = 8, b: int = 2) -> list[Workload]:
    """Return a second draw of the registry — same families, re-parametrized.

    Corpus-expansion siblings, NOT members of :func:`zoo`: every
    name here carries a ``*V2`` suffix so no provenance or
    disjointness check can confuse the two draws, and the draw
    mirrors ``zoo()`` one-for-one — a registry entry added there
    needs its re-parametrized sibling here.  Each entry is one
    of the registry's architecture families at a shifted
    configuration — wider width, a different head/expert/slot/unroll
    count — so the exported terms exercise the same module code with
    new leaf shapes and repetition depths.  That is the diversity a
    downstream miner (e.g. ``bench/drivers/policy/neural_corpus.py``) can
    actually use: fresh sites in known families rather than a
    re-measurement of the same trees.
    """
    d2 = 2 * d
    return [
        # --- attention variants ----------------------------------
        Workload(
            "GatedAttentionUnitV2",
            lambda: (_GatedAttentionUnit(d2), _r(b, t, d2)),
            kind="zoo",
        ),
        Workload(
            "AdditiveAttentionV2",
            lambda: (
                _AdditiveAttention(d2),
                (_r(b, 5, d2), _r(b, t, d2), _r(b, t, d2)),
            ),
            kind="zoo",
        ),
        Workload(
            "TalkingHeadsAttentionV2",
            lambda: (_TalkingHeadsAttention(d, 2), _r(b, t, d)),
            kind="zoo",
        ),
        Workload(
            "CosineAttentionV2",
            lambda: (_CosineAttention(d, 8), _r(b, t, d)),
            kind="zoo",
        ),
        # --- conditioning ----------------------------------------
        Workload(
            "FiLMHeadV2",
            lambda: (_FiLMHead(d, 4), (_r(b, t, d), _r(b, 4))),
            kind="zoo",
        ),
        Workload(
            "AdaLNBlockV2",
            lambda: (_AdaLNBlock(d, 4), (_r(b, t, d), _r(b, 4))),
            kind="zoo",
        ),
        Workload(
            "LoRAAdapterV2",
            lambda: (_LoRAAdapter(d, 8), _r(4, d)),
            kind="zoo",
        ),
        # --- token mixers / spectral / graphs --------------------
        Workload(
            "MLPMixerBlockV2",
            lambda: (_MLPMixerBlock(d, 6), _r(b, 6, d)),
            kind="zoo",
        ),
        Workload(
            "FNOBlockV2",
            lambda: (_FNOBlock(4, 2), _r(1, 4, 8, 8)),
            kind="zoo",
        ),
        Workload(
            "GCNLayerV2",
            lambda: (_GCNLayer(d2), (_r(b, 6, d2), _rand(2, 6, 6))),
            kind="zoo",
        ),
        Workload(
            "AffineCouplingV2",
            lambda: (_AffineCoupling(d2), _r(4, d2)),
            kind="zoo",
        ),
        # --- routing / gating ------------------------------------
        Workload(
            "SoftSlotMoEV2",
            lambda: (_SoftSlotMoE(d, 6), _r(b, t, d)),
            kind="zoo",
        ),
        Workload(
            "ExpertChoiceMoEV2",
            lambda: (_ExpertChoiceMoE(d, 6, 2), _r(b, t, d)),
            kind="zoo",
        ),
        Workload(
            "CapsuleRoutingV2",
            lambda: (_CapsuleRouting(d, 6, 6), _r(b, t, d)),
            kind="zoo",
        ),
        # --- recurrence / scan / conv ----------------------------
        Workload(
            "ChunkedRetentionV2",
            lambda: (_ChunkedRetention(d2), _r(b, t, d2)),
            kind="zoo",
        ),
        Workload(
            "DeepEquilibriumV2",
            lambda: (_DeepEquilibrium(d, 5), _r(4, d)),
            kind="zoo",
        ),
        Workload(
            "DeltaRuleBlockV2",
            lambda: (_DeltaRuleBlock(d, d // 2, 6), _r(b, 6, d)),
            kind="zoo",
        ),
        Workload(
            "WaveNetGateV2",
            lambda: (_WaveNetGate(6), _r(1, 6, 16)),
            kind="zoo",
        ),
        Workload(
            "TimeCondConvV2",
            lambda: (_TimeCondConv(8, 6), (_r(1, 8, 8, 8), _r(1))),
            kind="zoo",
        ),
        # --- heads / objectives ----------------------------------
        Workload(
            "MoSHeadV2",
            lambda: (_MoSHead(d, 12, 4), _r(4, d)),
            kind="zoo",
        ),
        Workload(
            "CovarianceHeadV2",
            lambda: (_CovarianceHead(), _r(6, d)),
            kind="zoo",
        ),
        Workload(
            "SplineKANV2",
            lambda: (_SplineKAN(d, 6), _r(4, d)),
            kind="zoo",
        ),
        Workload(
            "HighwayGateV2",
            lambda: (_HighwayGate(d2), _r(4, d2)),
            kind="zoo",
        ),
        # --- SSM / linear-attention / long-conv --------------------
        Workload(
            "MambaSelectiveScanV2",
            lambda: (_MambaSelectiveScan(d, 3, 6), _r(b, 6, d)),
            kind="zoo",
        ),
        Workload(
            "GatedLinearAttentionV2",
            lambda: (_GatedLinearAttention(d, 6), _r(b, 6, d)),
            kind="zoo",
        ),
        Workload(
            "RWKVMixBlockV2",
            lambda: (_RWKVMixBlock(d2), _r(b, t, d2)),
            kind="zoo",
        ),
        Workload(
            "HyenaLongConvV2",
            lambda: (_HyenaLongConv(d, 5), _r(b, 10, d)),
            kind="zoo",
        ),
        # --- routing: deployment-shape dispatch --------------------
        Workload(
            "TopKRouterMoEV2",
            lambda: (_TopKRouterMoE(d, 6, 3), _r(b, t, d)),
            kind="zoo",
        ),
        Workload(
            "DeepSeekSigMoEV2",
            lambda: (_DeepSeekSigMoE(d, 6, 3), _r(b, t, d)),
            kind="zoo",
        ),
        # --- quantization / cache ----------------------------------
        Workload(
            "STEQuantLinearV2",
            lambda: (_STEQuantLinear(d2), _r(4, d2)),
            kind="zoo",
        ),
        Workload(
            "PerChannelQuantMatmulV2",
            lambda: (_PerChannelQuantMatmul(d2), _r(4, d2)),
            kind="zoo",
        ),
        Workload(
            "KVCacheAttentionV2",
            lambda: (
                _KVCacheAttention(d, 3),
                (_r(b, 1, d), _r(b, 4, d), _r(b, 4, d)),
            ),
            kind="zoo",
        ),
        # --- modern block details ----------------------------------
        Workload(
            "GEGLUBlockV2",
            lambda: (_GEGLUBlock(d2), _r(b, t, d2)),
            kind="zoo",
        ),
        Workload(
            "SkipInitNetV2",
            lambda: (_SkipInitNet(d2), _r(b, t, d2)),
            kind="zoo",
        ),
        Workload(
            "SoftcapLogitHeadV2",
            lambda: (_SoftcapLogitHead(d, 12), _r(4, d)),
            kind="zoo",
        ),
        Workload(
            "RotaryPairV2",
            lambda: (_RotaryPair(d2), _r(b, 6, d2)),
            kind="zoo",
        ),
        # --- sparse / patterned attention --------------------------
        Workload(
            "SlidingWindowAttentionV2",
            lambda: (_SlidingWindowAttention(d2, 2), _r(b, t, d2)),
            kind="zoo",
        ),
        Workload(
            "DilatedBlockAttentionV2",
            lambda: (_DilatedBlockAttention(d, 3, 3), _r(b, 9, d)),
            kind="zoo",
        ),
        Workload(
            "BlockSparseAttentionV2",
            lambda: (
                _BlockSparseAttention(d2, 2, 3, 1),
                _r(b, 6, d2),
            ),
            kind="zoo",
        ),
        # --- depthwise-conv blocks -----------------------------------
        Workload(
            "ConvNeXtStageV2",
            lambda: (_ConvNeXtBlock(6), _r(1, 6, 8, 8)),
            kind="zoo",
        ),
        Workload(
            "SqueezeExciteV2",
            lambda: (_SqueezeExcite(8, 4), _r(1, 8, 6, 6)),
            kind="zoo",
        ),
        Workload(
            "MBConvSEBlockV2",
            lambda: (_MBConvBlock(6, 12), _r(1, 6, 8, 8)),
            kind="zoo",
        ),
        # --- gated residuals -----------------------------------------
        Workload(
            "GRUFusedGateV2",
            lambda: (_GRUFusedGate(d2), (_r(4, d2), _r(4, d2))),
            kind="zoo",
        ),
        Workload(
            "GatedConvUnitV2",
            lambda: (_GatedConvUnit(6), _r(1, 6, 12)),
            kind="zoo",
        ),
        # --- normalization variants -----------------------------------
        Workload(
            "ManualGroupNormV2",
            lambda: (_ManualGroupNorm(8, 2), _r(1, 8, 6, 6)),
            kind="zoo",
        ),
        Workload(
            "InstanceNormBroadcastV2",
            lambda: (_InstanceNormBroadcast(6), _r(1, 6, 8, 8)),
            kind="zoo",
        ),
        Workload(
            "RMSNormPerHeadV2",
            lambda: (_RMSNormPerHead(d, 2), _r(b, t, d)),
            kind="zoo",
        ),
        Workload(
            "DynamicTanhBlockV2",
            lambda: (_DynamicTanhBlock(d2), _r(b, t, d2)),
            kind="zoo",
        ),
        # --- positional / table ops -----------------------------------
        Workload(
            "ALiBiSlopeAttentionV2",
            lambda: (_ALiBiAttention(d, 8), _r(b, t, d)),
            kind="zoo",
        ),
        Workload(
            "RelPosTableAttentionV2",
            lambda: (_RelPosTableAttention(d2, 5), _r(b, t, d2)),
            kind="zoo",
        ),
        Workload(
            "LearnedPosSliceV2",
            lambda: (_LearnedPosSlice(d, 12), _r(b, 6, d)),
            kind="zoo",
        ),
        # --- second quant/kv shapes -----------------------------------
        Workload(
            "GroupedQuantLinearV2",
            lambda: (_GroupedQuantLinear(d, 8), _r(4, d)),
            kind="zoo",
        ),
        Workload(
            "PagedKVAttentionV2",
            lambda: (
                _PagedKVAttention(d2, 6),
                (_r(b, 1, d2), _r(b, 3, 2, d2), _r(b, 3, 2, d2)),
            ),
            kind="zoo",
        ),
        # --- third expansion: uncatalogued 2024-25 structures ---------
        Workload(
            "GatedDeltaRuleV2",
            lambda: (_GatedDeltaRule(d, d // 2, 6), _r(b, 6, d)),
            kind="zoo",
        ),
        Workload(
            "ChunkedSSDV2",
            lambda: (_ChunkedSSD(d, 2, 8, 2), _r(b, t, d)),
            kind="zoo",
        ),
        Workload(
            "RealGatedLRUV2",
            lambda: (_RealGatedLRU(d2), _r(b, t, d2)),
            kind="zoo",
        ),
        Workload(
            "DecoupledMLAV2",
            lambda: (_DecoupledMLA(d2, 4, 8, 4), _r(b, t, d2)),
            kind="zoo",
        ),
        Workload(
            "PartialRotaryV2",
            lambda: (_PartialRotary(d2, d2 // 4), _r(b, t, d2)),
            kind="zoo",
        ),
        Workload(
            "SigmoidAttentionV2",
            lambda: (_SigmoidAttention(d, 2), _r(b, t, d)),
            kind="zoo",
        ),
        Workload(
            "DiffAttentionV2",
            lambda: (_DiffAttention(d, 2), _r(b, t, d)),
            kind="zoo",
        ),
        Workload(
            "AttnLogitSoftcapV2",
            lambda: (_AttnLogitSoftcap(d2, 30.0), _r(b, t, d2)),
            kind="zoo",
        ),
        Workload(
            "SinkWindowAttentionV2",
            lambda: (_SinkWindowAttention(d2, 2, 2), _r(b, t, d2)),
            kind="zoo",
        ),
        Workload(
            "TaylorLinearAttentionV2",
            lambda: (_TaylorLinearAttention(d2), _r(b, t, d2)),
            kind="zoo",
        ),
        Workload(
            "AdaGNConvV2",
            lambda: (_AdaGNConv(6, 8, 3), (_r(1, 6, 8, 8), _r(1, 8))),
            kind="zoo",
        ),
        Workload(
            "SharedSwiGLUMoEV2",
            lambda: (_SharedSwiGLUMoE(d, 6, 3, 4), _r(b, t, d)),
            kind="zoo",
        ),
        Workload(
            "NormedSoftcapHeadV2",
            lambda: (_NormedSoftcapHead(d, 12), _r(4, d)),
            kind="zoo",
        ),
        Workload(
            "SquaredReluFFNV2",
            lambda: (_SquaredReluFFN(d2), _r(b, t, d2)),
            kind="zoo",
        ),
        Workload(
            "TernaryWeightLinearV2",
            lambda: (_TernaryWeightLinear(d2), _r(4, d2)),
            kind="zoo",
        ),
    ]
