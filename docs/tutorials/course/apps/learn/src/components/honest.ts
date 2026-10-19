export interface Refusal {
  text: string;
  number: string;
  opts: string[];
  ok: string;
  why: string;
  picked: string;
}

/** The ledger of negatives — methodology as a result, not a footnote.
 *
 * Each card is a *measured* refusal from the real system; pick which
 * claim its number kills.
 */
export class Honest {
  items: Refusal[] = [
    {
      text: "Claim under test: trained checkpoint weights hide exact structure mlopt can exploit. Every hypothesis class was probed — Toeplitz/displacement rank, H-matrix, Kronecker-sum, cross-layer duplicates, equivariance, low-degree polynomials.",
      number:
        "0 relational hits · full column rank · 0 aligned · minimal polynomial = full degree (288 distinct eigenvalues) · structural compression kills quality (ppl 5.03 → 214)",
      opts: ["weight compression", "graph-level rewriting", "the four dimensions"],
      ok: "weight compression",
      why: "The door is closed (ADR 0001): weights are information-dense at the bit level. mlopt's value is graph-level — the ADR narrowed the project's scope rather than extending it.",
      picked: "",
    },
    {
      text: "Claim under test: a machine-minted atom that verifies numerically will pay. The composed loop minted scan_fold (h_t = a_t·h_{t-1} + b_t, verified at 1.78e-15) with five certified laws, and admitted cumsum / cumprod / logcumsumexp law sets.",
      number: "0 certified-law fires on every real term",
      opts: ["truth implies pay", "the four dimensions", "the certificate is the referee"],
      ok: "truth implies pay",
      why: "Truth and pay are measured separately, and disjoint here: the minted atom computes the *total*, the corpus *emits the sequence*. ADR 0007's falsification criteria working as designed — the honest negative is the result.",
      picked: "",
    },
    {
      text: "Claim under test: the bounded-error lane buys headroom on a real pretrained model where the exact lane was FLOP-neutral. The real BiRefNet checkpoint (220 M params, Swin-v1-L + ASPP) was swept at every budget.",
      number: "0 bounds delivered at every budget · 0.80–0.86× vs torch.compile",
      opts: [
        "bounded error always pays",
        "the certificate is the referee",
        "the frontier keeps options",
      ],
      ok: "bounded error always pays",
      why: "The bounded offers exist (20–56 certified sites) and the bound gate stayed honest — but only 0.78% of the qkv weight deduped, so the gather/re-expand wrapper cost 1.32× the dense member. Launch-aware pricing correctly declined it.",
      picked: "",
    },
    {
      text: "Claim under test: naming structure expands reach. Layer-1 definitional atoms (anti-unify + mint) were run on delta×6, deq×6 and retnet T=64/K=4.",
      number:
        "compression is real (12 ops vs 26 unfolded) · cheapest extractable member and class count identical with and without atoms",
      opts: ["compression is reach", "compression is compression", "the e-graph is unsound"],
      ok: "compression is reach",
      why: "The minted atoms are real, provenance-carrying and unfoldable — but naming structure did not expand reach on these workloads. The machinery stays as compression tooling, not a reach mechanism: falsification #1 measured NULL.",
      picked: "",
    },
    {
      text: "Claim under test: a learned law-proposer beats the pipeline's plain enumeration. A learned proposer trained on the corpus was compared at equal oracle calls.",
      number: "~10× fewer true candidates per oracle call than enumeration",
      opts: [
        "a learned proposer beats enumeration where structure is enumerable",
        "learning never helps anywhere",
        "the oracle is wrong",
      ],
      ok: "a learned proposer beats enumeration where structure is enumerable",
      why: "The corpus's shapes, not the player's imagination, carry the discoveries. The learned player earns its place in *contraction search* (0.87–0.98× vs best-of-N greedy at n = 40) — not in law proposal.",
      picked: "",
    },
  ];

  answer(i: number, option: string): void {
    this.items[i].picked = option;
  }

  get score(): number {
    return this.items.filter((t) => t.picked === t.ok).length;
  }
}
