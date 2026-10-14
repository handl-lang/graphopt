/** The payoff — each idea mapped to the project's open work.
 *
 * Mirrors `project/plans/0025-minting-and-substrate.md` and the
 * retros that name the blockers.  This is *not* the plan restated: it
 * is the connection the course is for — **idea → blocker → next
 * measurement**, plus the level that teaches the idea behind it.  A
 * gap's next step is a measurement, not a build; a measurement that
 * closes a family is a result, not a failure.
 */

export interface Gap {
  id: string;
  /** The idea the course taught (matches a ladder one-liner). */
  idea: string;
  /** What stands between "mints and ships" and "pays at scale". */
  blocker: string;
  /** The next step — a measurement, not a build. */
  next: string;
  /** The level that teaches the idea behind this gap. */
  level: string;
  levelNo: number;
  levelTitle: string;
  source: string;
}

/** The project's open work, as data (plan 0025 + the retros). */
export const GAPS: readonly Gap[] = [
  {
    id: "enum-ordering",
    idea: "the vocabulary grows itself — but the paying descriptor is unreachable",
    blocker:
      "the paying `fold_emit` sits >200k deep in `propose_bodies`' round-robin; it verifies at 6.7e-16 when reached directly, so the gap is ordering, not expressiveness",
    next: "re-measure the SelectiveScan funnel after a canonical emitted-scan seed — does the minted atom's law set fire on a real corpus term?",
    level: "mint",
    levelNo: 12,
    levelTitle: "The vocabulary grows itself",
    source: "plan 0025 §A",
  },
  {
    id: "grammar-gaps",
    idea: "the handler grammar is extended by its refusals, not by taste",
    blocker:
      "`fold`/`fold_emit` empty-side const fillers always raise in `eval_body`; reductions/softmax have zero candidates; the `a≡1` side is only spellable as `div(X,X)`, which the a/b partition can never propose",
    next: "extend the grammar by the refusals and watch each refusal count drop — an extension that adds candidates without dropping one is churn, revert it",
    level: "mint",
    levelNo: 12,
    levelTitle: "The vocabulary grows itself",
    source: "plan 0025 §B",
  },
  {
    id: "gate-3",
    idea: "the combinator presentation is the uniform proposal space",
    blocker:
      "ADR 0008's gate 3 — whether saturation actually *reaches* `loop` forms — is unmeasured; the bridge round-trips, but reach is the open question",
    next: "saturate the combinator presentation on a recurrence-erased workload (`cumsum` / `cumprod` / `lstm.input`) and ask whether a `loop`-headed member appears and is extractable",
    level: "substrate",
    levelNo: 15,
    levelTitle: "Sharing is a morphism",
    source: "plan 0025 §C",
  },
  {
    id: "precision",
    idea: "approximation is allowed — with a certificate and an economics test",
    blocker:
      "BiRefNet's dedup was too thin (0.78 % of the qkv weight) — the certified-bounded member priced 1.32× the dense member, so pricing refused it correctly",
    next: "open the precision lane on a model where the dedup is thick (a wide-MLP checkpoint), with weight-space members rather than row-gather ones",
    level: "bounded",
    levelNo: 18,
    levelTitle: "Approximation with a certificate",
    source: "plan 0025 §D1 · birefnet-bounded-lane.md",
  },
  {
    id: "data-movement",
    idea: "exposure is necessary, not sufficient — some wins are data movement",
    blocker:
      "hoisting invariant structure is real (Swin masks are recomputed per layer, identical) but Inductor already owns this turf — the risk is building a loss",
    next: "measure *before* building: compete with Inductor on a named workload or record the loss and close the family with numbers",
    level: "exposure",
    levelNo: 14,
    levelTitle: "Where discovery can pay",
    source: "plan 0025 §D2",
  },
  {
    id: "weight-space",
    idea: "the ledger of negatives — a family is closed by its own measurement",
    blocker:
      "43 `BatchNorm2d` modules measured in real BiRefNet, and no conv+BN fold law exists in the rule set — they lower as `batch_norm` nodes, not fusion sites",
    next: "feed measured `(W, b, μ, σ, γ, β) → (W′, b′)` bindings to `dspec_synth` and let it attempt the fold spec; whatever it cannot express becomes a named grammar gap (derive-or-refuse, never hand-write)",
    level: "honest",
    levelNo: 17,
    levelTitle: "The ledger of negatives",
    source: "plan 0025 §D3",
  },
];

/** The payoff chapter, playable: pick a gap, read the connection. */
export class Gaps {
  sel = 0;

  get gaps(): readonly Gap[] {
    return GAPS;
  }

  get gap(): Gap {
    return GAPS[this.sel];
  }

  pick(i: number): void {
    this.sel = Math.max(0, Math.min(GAPS.length - 1, i));
  }

  /** The three-step connection the level is about. */
  get connection(): { idea: string; blocker: string; next: string }[] {
    const g = this.gap;
    return [
      { idea: "the idea", blocker: g.idea, next: `taught by level ${g.levelNo} — ${g.levelTitle}` },
      {
        idea: "the blocker",
        blocker: g.blocker,
        next: "what stands between mints-and-ships and pays-at-scale",
      },
      {
        idea: "the next measurement",
        blocker: g.next,
        next: "a measurement, not a build — a closed family is a result",
      },
    ];
  }
}
