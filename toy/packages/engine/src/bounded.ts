/** The bounded lane — approximation is allowed, with a certificate
 * *and* an economics test.
 *
 * Mirrors `project/retros/birefnet-bounded-lane.md`: on the real
 * BiRefNet checkpoint the precision lane offered certified-bounded
 * members at every budget — **20 offers at 1e-2, 56 at 3e-2**, bounds
 * inside budget — and delivered **none**.  The bound gate refused
 * nothing (`fresh_rel = 0.0`, bitwise-clean); the *pricing* refused,
 * because only 0.78 % of the qkv weight deduped, so the
 * gather/re-expand/bias-add wrapper cost **1.32×** the dense member.
 *
 * The tallies (offered → certified → delivered) and the economics
 * rule — a bounded member must beat dense — are computed *live* over
 * the member bank.  The thick-dedup toggle is the next target named
 * in plan 0025 (a wide-MLP checkpoint), where the dedup is thick
 * enough for the member to win; it is illustrative, not measured.
 */

export interface Member {
  site: string;
  /** How many real occurrences this class covers. */
  sites: number;
  /** The certified bound (`max_abs`) the offer carries. */
  bound: number;
  /** The backend-priced FLOPs of the dense member. */
  denseFlops: number;
  /** The backend-priced FLOPs of the bounded (elide) member, measured. */
  memberFlops: number;
  /** The measured share of weight rows that dedup. */
  dedup: number;
  note: string;
}

/** The BiRefNet offer frontier, as measured. */
export const MEMBERS: readonly Member[] = [
  {
    site: "attn.qkv · stages 2–3",
    sites: 20,
    bound: 9.95e-3,
    denseFlops: 832773,
    memberFlops: 1099421,
    dedup: 0.0078,
    note: "near-duplicate rows are real but thin — 0.78 % of the weight dedups",
  },
  {
    site: "mlp.fc1 / mlp.fc2 · stage 3",
    sites: 36,
    bound: 2.97e-2,
    denseFlops: 832773,
    memberFlops: 1099421,
    dedup: 0.0078,
    note: "the same thin structure, one budget step out",
  },
];

/** The budgets the sweep opened (0 = the exact lane). */
export const BUDGETS: readonly number[] = [0, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2];

/** The thick dedup of the plan-0025 next target — illustrative. */
export const THICK_DEDUP = 0.6;

/** The gather/re-expand/bias-add wrapper a bounded member pays. */
const wrapper = (m: Member): number => m.memberFlops - m.denseFlops * (1 - m.dedup);

/** The bounded member's price at a given dedup share — live. */
export function memberFlopsAt(m: Member, dedup: number): number {
  return m.denseFlops * (1 - dedup) + wrapper(m);
}

/** The bounded lane, playable: the economics is computed live. */
export class Bounded {
  sel = 5;
  thick = false;

  get budgets(): readonly number[] {
    return BUDGETS;
  }

  get members(): readonly Member[] {
    return MEMBERS;
  }

  get budget(): number {
    return BUDGETS[this.sel];
  }

  get budgetLabel(): string {
    return this.budget === 0 ? "none (exact)" : this.budget.toExponential(0);
  }

  select(i: number): void {
    this.sel = Math.max(0, Math.min(BUDGETS.length - 1, i));
  }

  toggle(): void {
    this.thick = !this.thick;
  }

  get dedup(): number {
    return this.thick ? THICK_DEDUP : MEMBERS[0].dedup;
  }

  /** One row per member class: certified, priced, delivered. */
  get rows(): {
    member: Member;
    certified: boolean;
    memberFlops: number;
    ratio: number;
    delivered: boolean;
  }[] {
    return MEMBERS.map((member) => {
      const certified = this.budget > 0 && member.bound <= this.budget;
      const memberFlops = memberFlopsAt(member, this.dedup);
      return {
        member,
        certified,
        memberFlops,
        ratio: memberFlops / member.denseFlops,
        delivered: certified && memberFlops < member.denseFlops,
      };
    });
  }

  /** Offers whose certified bound is inside the budget (weighted by sites). */
  get offered(): number {
    return this.rows.reduce((p, r) => p + (r.certified ? r.member.sites : 0), 0);
  }

  /** The bound gate refuses nothing — every offer carries a certified bound. */
  get certified(): number {
    return this.offered;
  }

  /** Offers whose bounded member also beats dense (weighted by sites). */
  get delivered(): number {
    return this.rows.reduce((p, r) => p + (r.delivered ? r.member.sites : 0), 0);
  }

  /** The bound gate stayed bitwise-clean: nothing declined for the bound. */
  get gateClean(): boolean {
    return this.rows.every((r) => !r.certified || r.member.bound <= this.budget);
  }

  /** Certified members the *pricing* refused (the economics test). */
  get pricedRefused(): number {
    return this.certified - this.delivered;
  }

  /** The live verdict for the picked budget. */
  get verdict(): string {
    if (this.offered === 0) return "no offer — the exact lane is all there is at this budget";
    if (this.delivered === 0)
      return `the bound gate certified ${this.certified} offers and the pricing refused all of them`;
    return `${this.delivered} of ${this.certified} certified offers beat dense — delivered`;
  }
}
