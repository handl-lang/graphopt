> **WITHDRAWN — see [p13_retraction_splitk.md](p13_retraction_splitk.md).** The split-k path used here
> accumulates incorrectly (max abs error 3.8-5.6 vs reference); every speedup
> in this report is invalid. The regime analysis stands; the numbers do not.

# P10 — the machine cannot yet pick the kernel, because the referee can't see

Phase 3's gate was: *the machine rediscovers the hand-tuned kernel without
being told its structure.* It did not. Three measurement protocols gave
three different winners, and the spread between them (58 → 116 µs on the
same code) is larger than any difference the search was meant to find.

## What happened

| protocol | `down_proj` winner | `down_proj` µs |
|---|---|---|
| hand-picked (single candidate, long warm loop) | word_once bn64 bk128 split4 w2 | **58.2** |
| sequential (measure each candidate fully, then next) | per_feature bn32 bk256 split1 w4 | 81.6 |
| round-robin (one call each, interleaved) | word_once bn32 bk256 split1 w4 | 116.7 |
| blocked random (warm block per candidate, shuffled rounds) | word_once bn32 bk64 split1 w2 | 107.5 |

**None of them agreed, and none reproduced 58 µs.** `agrees_with_hand`
was false on every shape.

## Why — the regime, not the search

The weight matrix for `down_proj` is 2.18 MB of packed int4. The card's
L2 is about 2 MB. So the measured number depends entirely on whether the
weights happen to be resident:

- a long warm loop on one kernel keeps them hot → **58 µs**;
- interleaving 72 candidates evicts them between rounds → **107–117 µs**.

And the deployment truth is the **cold** number: 24 layers × 2 MB of
weights cycle through a 2 MB L2, so when layer 0's `down_proj` runs again
its weights are gone. **Our warm-cache benchmarks flatter every kernel,
ours and vLLM's alike.**

One shape was immune: `k_proj` is 57 KB, fits comfortably, and the search
found **15.8 µs (2.13× over `awq_gemm`)** — better than my hand pick.
Launch-bound shapes are regime-insensitive; bandwidth-bound ones are not.

## What this means for the vision

The machine can search configs. It cannot yet *choose* kernels, because
the thing it optimizes against is not the thing that will run. A search
that scores candidates in a warm cache will pick a winner that loses in
production — confidently, and with good-looking numbers.

That is the referee problem I flagged in the plan, showing up earlier and
harder than expected. It is not a search-algorithm problem; it is a
**measurement-definition** problem: the objective must be defined in the
deployment regime (weights evicted, layer re-use distance respected)
before any search over it means anything.

## Next, precisely

1. **Define the deploy-regime benchmark**: measure with a cache-evicting
   pass between candidates (or a realistic layer sequence), on both arms,
   so ours-vs-vLLM is compared under the same conditions.
2. **Re-run the routing table** under that regime — the winner may flip
   (a kernel that wins hot can lose cold).
3. Only then does a search have a meaningful objective, and the
   "machine picks the kernel" gate can be re-attempted.
