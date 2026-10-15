/** A backend is a port — swapping it changes nothing in core.
 *
 * Mirrors `catopt_core.ports` + ADR 0006: the whole-graph ends are
 * `Source` (`model → IR`) and `Sink` (`IR → runnable`, plus its
 * `supported_ops` set).  A second backend (ONNX) landed with **zero
 * core edits** — the only thing that changes is which forms the
 * extraction may select, because `supported_ops` re-bounds it.
 *
 * The checkable contract here is the counter: swapping the sink must
 * leave the core byte-identical and the core-mutation count at zero.
 */

import type { Term } from "./term.ts";
import { isOp, key } from "./term.ts";

export interface Source {
  name: string;
  toIR: (model: Term) => Term;
}

export interface Sink {
  name: string;
  /** The ops this backend can lower — the extraction's hard bound. */
  supported: string[];
  artifact: (t: Term) => string;
}

export const SOURCES: readonly Source[] = [
  { name: "torch.export", toIR: (m) => m },
  { name: "onnx.source", toIR: (m) => m },
];

export const SINKS: readonly Sink[] = [
  {
    name: "TorchSink",
    supported: ["matmul", "add", "mul", "scale2", "sq", "sqq"],
    artifact: (t) => `torch module over ${key(t)}`,
  },
  {
    name: "OnnxSink",
    supported: ["matmul", "add", "mul", "scale2", "sq"],
    artifact: (t) => `onnx graph over ${key(t)}`,
  },
];

/** Every op name a term mentions. */
export function opsOf(t: Term, acc = new Set<string>()): Set<string> {
  if (isOp(t)) {
    acc.add(t.op);
    t.args.forEach((a) => opsOf(a, acc));
  }
  return acc;
}

export interface CoreState {
  term: Term;
  rules: string[];
}

/** The core: the IR and the law set.  Nothing here knows a backend. */
export class Core {
  term: Term;
  rules: string[];
  /** Every change to core state — the counter the swap must not move. */
  mutations = 0;

  constructor(term: Term, rules: string[]) {
    this.term = term;
    this.rules = [...rules];
  }

  fingerprint(): string {
    return `${key(this.term)}|${this.rules.join(",")}`;
  }

  addRule(rule: string): void {
    this.rules = [...this.rules, rule];
    this.mutations += 1;
  }
}

/** The port pair, playable: swap the sink, watch core stay put. */
export class Ports {
  core: Core;
  sourceIndex = 0;
  sinkIndex = 0;
  swaps = 0;

  constructor(term: Term, rules: string[]) {
    this.core = new Core(term, rules);
  }

  get source(): Source {
    return SOURCES[this.sourceIndex];
  }

  get sink(): Sink {
    return SINKS[this.sinkIndex];
  }

  get coreMutations(): number {
    return this.core.mutations;
  }

  get coreFingerprint(): string {
    return this.core.fingerprint();
  }

  get ir(): Term {
    return this.source.toIR(this.core.term);
  }

  /** Swap the backend — a pure port change; core is untouched. */
  swap(name: string): void {
    const i = SINKS.findIndex((s) => s.name === name);
    if (i >= 0 && i !== this.sinkIndex) {
      this.sinkIndex = i;
      this.swaps += 1;
    }
  }

  /** Ops the current sink cannot lower — what re-bounds extraction. */
  get unsupported(): string[] {
    const sup = new Set(this.sink.supported);
    return [...opsOf(this.ir)].filter((op) => !sup.has(op));
  }

  get lowerable(): boolean {
    return this.unsupported.length === 0;
  }

  lower(): string {
    return this.lowerable
      ? this.sink.artifact(this.ir)
      : `refused: no binding for ${this.unsupported.join(", ")}`;
  }

  /** A core edit — the only thing that may move the counter. */
  editCore(rule: string): void {
    this.core.addRule(rule);
  }
}
