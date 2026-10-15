/** The combinator presentation — sharing is a morphism, recurrence is a `loop`.
 *
 * Mirrors `project/adrs/0008-combinator-presentation.md`: the op-graph
 * IR is the lowering surface; a parallel combinator presentation makes
 * the structure explicit — `seq` (composition), `par` (the monoidal
 * product `⊗`), `arr` (apply an op), `dup` (Δ, sharing), and `loop`
 * (a typed trace).  The bridge is a bijection: `dagToComb` fans shared
 * subterms out into explicit `dup` sites; `combToDag` inlines them back.
 *
 * The round trip is the checkable contract: `dag → comb → dag` is the
 * identity on the toy term language.
 */

import type { Term } from "./term.ts";
import { isLeaf, isOp, key } from "./term.ts";

export type Comb =
  | { k: "leaf"; name: string }
  | { k: "arr"; op: string; arity: number }
  | { k: "seq"; f: Comb; g: Comb }
  | { k: "par"; fs: Comb[] }
  | { k: "dup"; f: Comb }
  | { k: "loop"; step: Comb };

/** Occurrence count of every subterm key in a term. */
function census(t: Term, acc = new Map<string, number>()): Map<string, number> {
  const k = key(t);
  acc.set(k, (acc.get(k) ?? 0) + 1);
  if (isOp(t)) t.args.forEach((a) => census(a, acc));
  return acc;
}

/** `dag → combinator`: fan-out becomes explicit `dup`, `pack` becomes `⊗`. */
export function dagToComb(root: Term): Comb {
  const count = census(root);
  const go = (t: Term): Comb => {
    if (isLeaf(t)) return { k: "leaf", name: t.leaf };
    if (t.op === "loop" && t.args.length === 1) return { k: "loop", step: go(t.args[0]) };
    const args = t.args.map(go);
    const base: Comb =
      args.length === 1
        ? { k: "seq", f: args[0], g: { k: "arr", op: t.op, arity: 1 } }
        : { k: "seq", f: { k: "par", fs: args }, g: { k: "arr", op: t.op, arity: args.length } };
    // a subterm used more than once is a shared value: Δ made explicit
    return (count.get(key(t)) ?? 0) > 1 ? { k: "dup", f: base } : base;
  };
  return go(root);
}

/** `combinator → dag`: expand the structure morphisms back into the graph. */
export function combToDag(c: Comb): Term {
  switch (c.k) {
    case "leaf":
      return { leaf: c.name };
    case "dup":
      return combToDag(c.f);
    case "loop":
      return { op: "loop", args: [combToDag(c.step)] };
    case "par":
      return { op: "pack", args: c.fs.map(combToDag) };
    case "arr":
      return { leaf: c.op };
    case "seq": {
      if (c.g.k === "arr") {
        const args = c.f.k === "par" ? c.f.fs.map(combToDag) : [combToDag(c.f)];
        return { op: c.g.op, args };
      }
      return combToDag(c.f);
    }
  }
}

/** A structural key for a combinator term — the round-trip comparator. */
export function combKey(c: Comb): string {
  switch (c.k) {
    case "leaf":
      return c.name;
    case "arr":
      return `arr:${c.op}/${c.arity}`;
    case "dup":
      return `dup(${combKey(c.f)})`;
    case "loop":
      return `loop(${combKey(c.step)})`;
    case "par":
      return `par(${c.fs.map(combKey).join(",")})`;
    case "seq":
      return `seq(${combKey(c.f)},${combKey(c.g)})`;
  }
}

/** How many explicit Δ sites a combinator term carries. */
export function dupCount(c: Comb): number {
  switch (c.k) {
    case "leaf":
    case "arr":
      return 0;
    case "dup":
      return 1 + dupCount(c.f);
    case "loop":
      return dupCount(c.step);
    case "par":
      return c.fs.reduce((p, f) => p + dupCount(f), 0);
    case "seq":
      return dupCount(c.f) + dupCount(c.g);
  }
}

/** A readable spelling of a combinator term. */
export function renderComb(c: Comb): string {
  switch (c.k) {
    case "leaf":
      return c.name;
    case "arr":
      return `arr(${c.op})`;
    case "dup":
      return `dup(${renderComb(c.f)})`;
    case "loop":
      return `loop(${renderComb(c.step)})`;
    case "par":
      return `(${c.fs.map(renderComb).join(" ⊗ ")})`;
    case "seq":
      return `${renderComb(c.f)} ▸ ${renderComb(c.g)}`;
  }
}

/** A recurrence — `h_t = a_t·h_{t-1} + b_t` — as a term under `loop`. */
export const LOOP_TERM: Term = {
  op: "loop",
  args: [
    {
      op: "add",
      args: [{ op: "mul", args: [{ leaf: "a" }, { leaf: "h" }] }, { leaf: "b" }],
    },
  ],
};

/** The substrate, playable: translate a term and back, live. */
export class Substrate {
  sel = 0;
  programs: { id: string; term: Term }[] = [];

  constructor(programs: { id: string; term: Term }[]) {
    this.programs = programs;
  }

  get program(): { id: string; term: Term } {
    return this.programs[this.sel];
  }

  pick(i: number): void {
    this.sel = Math.max(0, Math.min(this.programs.length - 1, i));
  }

  get comb(): Comb {
    return dagToComb(this.program.term);
  }

  get back(): Term {
    return combToDag(this.comb);
  }

  get roundTrips(): boolean {
    return key(this.back) === key(this.program.term);
  }

  get dups(): number {
    return dupCount(this.comb);
  }

  /** The recurrence, spelled as `loop(step)`, and its round trip. */
  get loop(): { comb: Comb; back: Term; roundTrips: boolean } {
    const comb = dagToComb(LOOP_TERM);
    const back = combToDag(comb);
    return { comb, back, roundTrips: key(back) === key(LOOP_TERM) };
  }
}
