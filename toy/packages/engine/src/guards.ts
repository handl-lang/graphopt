/** Guard synthesis as a live mini-sweep.
 *
 * Mirrors `project/retros/mul-unsq-nat.md`: the census minted
 * `mul(unsq u d, unsq v d) = unsq(mul u v, d)` and `auto_cond` proposed
 * the obvious guard — "the two output grids coincide".  That guard is
 * *unsound past its measured domain*: the adversarial sweep found
 * 212 bad accepts.  The exact guard checks grid coincidence **and**
 * both operands' flat-read maps — 0 bad accepts, 0 misses.
 *
 * This is the real sweep, in miniature: a shape domain, a naive guard,
 * an exact guard, and the verdicts computed live.
 */

/** `unsq(x, d)` inserts a size-1 axis at `d mod (rank+1)`. */
const normPos = (d: number, rank: number): number => {
  const m = rank + 1;
  return ((d % m) + m) % m;
};

export function unsq(shape: number[], d: number): number[] {
  const pos = normPos(d, shape.length);
  return [...shape.slice(0, pos), 1, ...shape.slice(pos)];
}

/** Right-aligned broadcasting, or `null` when the shapes disagree. */
export function bcast(a: number[], b: number[]): number[] | null {
  const r = Math.max(a.length, b.length);
  const out: number[] = [];
  for (let i = 0; i < r; i++) {
    const x = a[a.length - r + i] ?? 1;
    const y = b[b.length - r + i] ?? 1;
    if (x !== y && x !== 1 && y !== 1) return null;
    out.push(Math.max(x, y));
  }
  return out;
}

const strides = (s: number[]): number[] => {
  const st = Array.from({ length: s.length }, () => 1);
  for (let i = s.length - 2; i >= 0; i--) st[i] = st[i + 1] * s[i + 1];
  return st;
};

/** Flat index in `src` read at flat index `p` of the broadcast `out`. */
function broadcastRead(src: number[], out: number[], p: number): number {
  const ost = strides(out);
  const sst = strides(src);
  let idx = 0;
  let rem = p;
  for (let i = 0; i < out.length; i++) {
    const dim = Math.floor(rem / ost[i]);
    rem %= ost[i];
    const j = src.length - out.length + i;
    if (j < 0) continue; // leading broadcast dims read index 0
    idx += (src[j] === 1 ? 0 : dim) * sst[j];
  }
  return idx;
}

/** Flat index in `inner` for flat index `p` of `unsq(inner, pos)`. */
function unsqRead(inner: number[], pos: number, p: number): number {
  const g = unsq(inner, pos);
  const gst = strides(g);
  const ist = strides(inner);
  let idx = 0;
  let rem = p;
  for (let i = 0; i < g.length; i++) {
    const dim = Math.floor(rem / gst[i]);
    rem %= gst[i];
    if (i === pos) continue; // the inserted axis has extent 1
    idx += dim * ist[i < pos ? i : i - 1];
  }
  return idx;
}

const same = (a: number[], b: number[]): boolean =>
  a.length === b.length && a.every((x, i) => x === b[i]);

export interface Site {
  u: number[];
  v: number[];
  d: number;
  grid: number[];
  evaluable: boolean;
  gridEq: boolean;
  equal: boolean;
  reason: string;
}

/** Evaluate one site: is `mul(unsq u d, unsq v d)` equal to `unsq(mul u v, d)`? */
export function site(u: number[], v: number[], d: number): Site {
  const base = { u, v, d, grid: [] as number[], gridEq: false, equal: false };
  const lu = unsq(u, d);
  const lv = unsq(v, d);
  const grid = bcast(lu, lv);
  if (!grid) return { ...base, evaluable: false, reason: "lhs ill-typed" };
  const inner = bcast(u, v);
  if (!inner) return { ...base, grid, evaluable: false, reason: "rhs ill-typed" };
  const grid2 = unsq(inner, d);
  const gridEq = same(grid, grid2);
  let equal = gridEq;
  if (gridEq) {
    const n = grid.reduce((p, q) => p * q, 1);
    for (let p = 0; p < n && equal; p++) {
      const ru = broadcastRead(lu, grid, p);
      const rv = broadcastRead(lv, grid, p);
      const ri = unsqRead(inner, normPos(d, inner.length), p);
      const ru2 = broadcastRead(u, inner, ri);
      const rv2 = broadcastRead(v, inner, ri);
      if (ru !== ru2 || rv !== rv2) equal = false;
    }
  }
  return { u, v, d, grid, evaluable: true, gridEq, equal, reason: "" };
}

export interface Guard {
  key: "naive" | "exact";
  label: string;
  note: string;
  accept: (s: Site) => boolean;
}

export const GUARDS: readonly Guard[] = [
  {
    key: "naive",
    label: "grid-eq only",
    note: "The guard `auto_cond` proposed: accept when the two output grids coincide. It separates its measured domain perfectly — and is unsound past it.",
    accept: (s) => s.gridEq,
  },
  {
    key: "exact",
    label: "flat-map-unsq",
    note: "The synthesized guard: grid coincidence AND both operands' flat-read maps agree. One clause of pure data, serializable — the exact truth region.",
    accept: (s) => s.equal,
  },
];

export interface SweepRow {
  key: string;
  label: string;
  accepted: number;
  badAccept: number;
  missed: number;
}

/** The live mini-sweep: try to break the naive guard. */
export class Guards {
  sel = 0;
  guard: "naive" | "exact" = "naive";

  /** Shapes of rank `1..ranks` with extents drawn from `extents`. */
  private shapes(ranks = 3, extents = [1, 2, 3]): number[][] {
    const out: number[][] = [];
    const grow = (dims: number[]): void => {
      if (dims.length) out.push([...dims]);
      if (dims.length === ranks) return;
      for (const e of extents) grow([...dims, e]);
    };
    grow([]);
    return out;
  }

  /** Every site in the domain, evaluable or not. */
  get sites(): Site[] {
    const shapes = this.shapes();
    const out: Site[] = [];
    for (const u of shapes) {
      for (const v of shapes) {
        for (let d = -4; d <= 4; d++) out.push(site(u, v, d));
      }
    }
    return out;
  }

  get evaluable(): Site[] {
    return this.sites.filter((s) => s.evaluable);
  }

  /** Bad accepts: the guard says yes, the equality says no. */
  get badAccepts(): Site[] {
    const g = GUARDS.find((x) => x.key === this.guard)!;
    return this.evaluable.filter((s) => g.accept(s) && !s.equal);
  }

  /** Equal sites the guard rejects. */
  get missed(): Site[] {
    const g = GUARDS.find((x) => x.key === this.guard)!;
    return this.evaluable.filter((s) => !g.accept(s) && s.equal);
  }

  get sweep(): SweepRow[] {
    const ev = this.evaluable;
    return GUARDS.map((g) => ({
      key: g.key,
      label: g.label,
      accepted: ev.filter((s) => g.accept(s)).length,
      badAccept: ev.filter((s) => g.accept(s) && !s.equal).length,
      missed: ev.filter((s) => !g.accept(s) && s.equal).length,
    }));
  }

  get domain(): {
    sites: number;
    evaluable: number;
    equal: number;
    unequal: number;
    illTyped: number;
  } {
    const all = this.sites;
    const ev = all.filter((s) => s.evaluable);
    return {
      sites: all.length,
      evaluable: ev.length,
      equal: ev.filter((s) => s.equal).length,
      unequal: ev.filter((s) => !s.equal).length,
      illTyped: all.filter((s) => !s.evaluable).length,
    };
  }

  /** The witness to display: a bad accept when the naive guard is on. */
  get witness(): Site | undefined {
    const bad = this.badAccepts;
    if (bad.length) return bad[this.sel % bad.length];
    const g = GUARDS.find((x) => x.key === this.guard)!;
    const ok = this.evaluable.filter((s) => g.accept(s));
    return ok.length ? ok[this.sel % ok.length] : this.evaluable[0];
  }

  pick(i: number): void {
    this.sel = Math.max(0, i);
  }

  setGuard(g: "naive" | "exact"): void {
    this.guard = g;
    this.sel = 0;
  }

  /** The guard the sweep selects, as a human-readable clause. */
  get synthesized(): string {
    return "flat-map-unsq(u, d, v, unsq-out(v, d)) — grid coincidence plus both flat-read maps";
  }
}
