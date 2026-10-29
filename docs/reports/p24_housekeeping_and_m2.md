# P24 — Housekeeping done, the schedule lever is flat, and that settles M2

Three things were asked: land the loose ends, attempt M2's lever, and record
the outcome either way. The lever failed, so this is the recording.

## 1. Housekeeping

- **ADR-0165 landed.** `host/project` turns out to be the `project`-branch
  *worktree*, so the ADR was one commit away (`032639c2`), not stuck behind
  the `main` ignore rule.
- **The search driver ran for real.** `search_real.py` fed it generated
  kernels instead of stubs: `qgemv/down_proj` chose `split8`, `qgemv/gate_proj`
  chose `split4`, both at 52.3 µs (47.0 % of roofline), incumbent timed
  in-run. Routes: `p24_search_real.json`. One honest number it surfaced:
  **`gate_proj` is 0.87× `awq_gemm`**, i.e. we are 13 % *slower* on that
  operator alone (the fused `gate_up` in P22 was 0.60×).

## 2. M2's lever, measured

`pipeline(depth)` added to the IR (tenth op, ADR-0166), lowered to
`num_stages`, swept 48 cells — `depth ∈ {1,2,3,4}` × `split ∈ {2,4,8}` ×
`BLOCK_K ∈ {128,256}` on both geometries, every cell gated against
`awq_gemm` (`p24_m2_schedule.json`).

| | depth 1 | depth 2 | depth 3 | depth 4 |
|---|---|---|---|---|
| `down_proj` best | 52.5 | 52.3 | 52.5 | **52.1** |
| `down_proj` median | 89.9 | 90.0 | 89.9 | 89.3 |
| `gate_proj` best | 52.4 | 52.3 | **52.1** | 52.2 |
| `gate_proj` median | 108.1 | 108.1 | 108.0 | 108.3 |

**0.4–0.8 %, against run-to-run noise of roughly the same size.** Pipelining
is not the lever. The pre-registered gate ("beat 52.3 µs") was **not met**.

## 3. What that means, stated plainly

- The gap to Marlin (86–90 % of roofline) is **arithmetic**, not latency:
  the unpack (shift, mask, subtract, multiply) and the `tl.sum` over the k
  axis. Prefetching more of the same bytes faster cannot fix a kernel that
  is not waiting on them.
- The 2026-10-27 review predicted the nine-op IR "can only express naive
  schedules". That prediction was about latency hiding, and this is the
  test: it was cheap, it was run, and it says the schedule was never the
  bottleneck. Recorded in ADR-0166 so nobody runs it again.
- **So the IR is not a compiler.** It is a config-search substrate with a
  faithful lowering: terms reproduce the hand kernel exactly, choose
  `split_k` (149.9 → 52.3 µs, a real 2.9×), and prove their own equality.
  Anything stronger is not supported by evidence.

## The pitch consequence

Three claims survive with numbers behind them:

1. **Exactness is checkable, not asserted** — the gate is external, and it
   caught our own 0.21-cosine bug (P21/P22).
2. **Search beats hand-tuning on structure** — `split_k` moves 2.9× on the
   same kernel body, and the driver picks it without a human.
3. **The unattended regime is where the wins live** — Marlin at 86–90 % on
   its tuned shapes, Humming at 34–43 % of bandwidth on SM80 decode (P20).

The claim that must be dropped: that a small IR plus a schedule vocabulary
gets us to Marlin-class utilisation. It does not, and we measured why.
