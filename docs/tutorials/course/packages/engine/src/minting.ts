/** The minting funnel — ADR 0007's four layers over a measured need bank.
 *
 * Mirrors `bench/mint/atom_synth_loop.py`: a measured need → a candidate
 * signature → a verified handler → synthesized laws → certified
 * admission → a pay probe.  The bank's stage outcomes, law counts and
 * pay verdicts are the measured ones (`project/retros/atom-synth-loop.md`,
 * `project/adrs/0007-semantic-atom-minting.md`); the funnel tallies are
 * computed *live* from the bank.
 *
 * The honest headline is the pay column: the loop composes end to end
 * and mints, but truth and pay are measured separately and are disjoint
 * on the real corpus.
 */

/** The vocabulary gap: law-invisible ops with no signature builder. */
export const VOCAB_GAP = 199;

/** Certified-law fires on the real corpus, across every admitted need. */
export const PAY_NULL_FIRES = 0;

export type SigVerdict = "ok" | "refused";
export type HandlerVerdict = "minted" | "ambient" | "refused";
export type PayVerdict = "pays" | "pay-null" | "unblocked" | "refused";

export interface Need {
  op: string;
  flops: number;
  sites: number;
  sig: SigVerdict;
  handler: HandlerVerdict;
  laws: number;
  pay: PayVerdict;
  note: string;
}

export const NEEDS: readonly Need[] = [
  {
    op: "cumsum",
    flops: 5376,
    sites: 6,
    sig: "ok",
    handler: "ambient",
    laws: 5,
    pay: "pay-null",
    note: "The manual cumulative sum — the top measured need. A torch binding already exists, so the handler is ambient: the need is the law set, not the body. Five certified laws (cum.add_same, cum.sub_same, cum.neg, cum.scale_same, cum.cat) — and none fire, because the corpus uses cumsum non-linearly.",
  },
  {
    op: "cumprod",
    flops: 576,
    sites: 3,
    sig: "ok",
    handler: "ambient",
    laws: 3,
    pay: "pay-null",
    note: "Three certified laws, including the cross-atom cp.exp: cumprod(exp x) = exp(cumsum x) — an equality that relates two atoms rather than one.",
  },
  {
    op: "logcumsumexp",
    flops: 256,
    sites: 1,
    sig: "ok",
    handler: "ambient",
    laws: 1,
    pay: "pay-null",
    note: "One certified law (lc.shift_same) — the numerically-stable sibling, admitted for completeness and paying nowhere real.",
  },
  {
    op: "fold (selective scan)",
    flops: 0,
    sites: 1,
    sig: "ok",
    handler: "minted",
    laws: 5,
    pay: "pay-null",
    note: "scan_fold — the one full semantic-atom mint. The recurrence h_t = a_t·h_{t-1} + b_t verified at 1.78e-15, with five certified primitive laws. The paying sibling (fold_emit, which emits the sequence) sits >200k deep in the round-robin enumeration: the gap is ordering, not expressiveness.",
  },
  {
    op: "pow",
    flops: 0,
    sites: 0,
    sig: "ok",
    handler: "ambient",
    laws: 1,
    pay: "unblocked",
    note: "The vocab lane: the 2**rank refusal → a pow proposal → a certified unblock (4/5 → 5/5 agreement). A synthetic need, not a corpus one — the scan over all shipped rules found zero real blocked derives.",
  },
  {
    op: `${VOCAB_GAP} law-invisible ops`,
    flops: 0,
    sites: 0,
    sig: "refused",
    handler: "refused",
    laws: 0,
    pay: "refused",
    note: "The vocabulary gap: no signature builder exists for these ops, so the funnel refuses at layer 2. The refusal list is a deliverable, not a failure — it is the measured frontier of what the vocabulary cannot yet spell.",
  },
];

export interface Stage {
  key: string;
  label: string;
  count: number;
}

/** The minting funnel, playable: pick a need, read the four layers. */
export class Minting {
  sel = 0;

  get needs(): readonly Need[] {
    return NEEDS;
  }

  get need(): Need {
    return NEEDS[this.sel];
  }

  pick(i: number): void {
    this.sel = Math.max(0, Math.min(NEEDS.length - 1, i));
  }

  /** The funnel tallies, computed live over the bank. */
  get funnel(): Stage[] {
    const ok = (n: Need): boolean => n.sig === "ok";
    return [
      { key: "need", label: "measured needs", count: NEEDS.length },
      { key: "sig", label: "signature builds", count: NEEDS.filter(ok).length },
      {
        key: "handler",
        label: "handler verified",
        count: NEEDS.filter((n) => n.handler === "minted" || n.handler === "ambient").length,
      },
      { key: "laws", label: "certified law rows", count: NEEDS.reduce((p, n) => p + n.laws, 0) },
      { key: "admit", label: "admitted", count: NEEDS.filter((n) => n.pay !== "refused").length },
      {
        key: "pay",
        label: "paying on the corpus",
        count: NEEDS.filter((n) => n.pay === "pays").length,
      },
    ];
  }

  /** Total measured FLOPs of the needs the census ranked. */
  get totalFlops(): number {
    return NEEDS.reduce((p, n) => p + n.flops, 0);
  }

  get verdict(): string {
    const n = this.need;
    if (n.sig === "refused") return "refused at layer 2 — no signature builder";
    if (n.pay === "unblocked") return "admitted — the vocabulary unblocked, nothing pays";
    if (n.pay === "pay-null") return "admitted through all four layers — and pays nothing";
    return "refused";
  }

  get verdictTone(): string {
    const n = this.need;
    return n.sig === "refused" ? "text-bad" : n.pay === "unblocked" ? "text-gold" : "text-eva";
  }

  /** The stage a need dies at, or `null` when it clears the funnel. */
  get stoppedAt(): string | null {
    const n = this.need;
    if (n.sig === "refused") return "signature";
    if (n.handler === "refused") return "handler";
    if (n.laws === 0) return "laws";
    return null;
  }

  /** The four layers for the picked need, for the funnel view. */
  get layers(): { key: string; label: string; ok: boolean; detail: string }[] {
    const n = this.need;
    return [
      {
        key: "need",
        label: "measured need",
        ok: true,
        detail: n.flops ? `${n.flops} flops / ${n.sites} sites` : `${n.sites} site`,
      },
      {
        key: "signature",
        label: "signature",
        ok: n.sig === "ok",
        detail:
          n.sig === "ok"
            ? "arity and attrs read off real occurrences"
            : "no signature builder — refused",
      },
      {
        key: "handler",
        label: "handler",
        ok: n.handler !== "refused",
        detail:
          n.handler === "minted"
            ? "minted body, verified numerically"
            : n.handler === "ambient"
              ? "a binding already exists — the need is the law set"
              : "refused",
      },
      {
        key: "laws",
        label: "laws",
        ok: n.laws > 0,
        detail: `${n.laws} certified law ${n.laws === 1 ? "row" : "rows"}`,
      },
      { key: "admission", label: "admission", ok: n.pay !== "refused", detail: this.verdict },
    ];
  }
}
