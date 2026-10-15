/** The gates — how the project stays honest.
 *
 * Plan 0028 §D6.  Every claim in the course is only as good as the
 * checks that keep the tree honest: the **coverage floor** (100 % on the
 * shipped six, a 99 % ratchet on discovery), the typecheck (**ty**), the
 * lint/format (**ruff**), the dead-code pass (**vulture**), the
 * hexagonal-boundary contract (**import-linter**), the security passes
 * (**bandit**, **semgrep**), the **radon** complexity ratchet, and the
 * **bench gate** against pinned baselines.  The rule that keeps them
 * honest: **no new suppressions** — a finding is fixed, never silenced.
 *
 * `AGENTS.md` is the source.  The mini-gate below runs over sample data:
 * break a gate and watch the run fail.
 */

export interface Gate {
  id: string;
  label: string;
  /** What the gate checks. */
  what: string;
  /** The floor it enforces. */
  floor: string;
}

/** The gates the project runs before finishing any change. */
export const GATES: readonly Gate[] = [
  {
    id: "coverage",
    label: "coverage",
    what: "statement + branch coverage",
    floor: "100% shipped · 99% discovery",
  },
  { id: "ty", label: "ty", what: "typecheck", floor: "0 errors" },
  { id: "ruff", label: "ruff", what: "lint + format", floor: "clean" },
  { id: "vulture", label: "vulture", what: "dead code", floor: "none" },
  {
    id: "import-linter",
    label: "import-linter",
    what: "hexagonal boundary",
    floor: "0 violations",
  },
  { id: "bandit", label: "bandit", what: "security SAST", floor: "0 findings" },
  { id: "semgrep", label: "semgrep", what: "dataflow analysis", floor: "0 findings" },
  { id: "radon", label: "radon", what: "complexity ratchet", floor: "≤ baseline" },
  { id: "bench", label: "bench", what: "pinned baselines", floor: "no regression" },
];

/** The rule that keeps the gates honest. */
export const NO_SUPPRESSIONS =
  "a new finding is fixed, never silenced — no `# noqa`, no `# ty: ignore`, no baseline entry";

/** The gates level, playable: break a gate and watch the run fail. */
export class GateLab {
  /** The gate currently broken, or `null` when all pass. */
  broken: string | null = null;

  get gates(): readonly Gate[] {
    return GATES;
  }

  /** Break a gate by id (an unknown id is ignored). */
  break(id: string): void {
    if (GATES.some((g) => g.id === id)) this.broken = id;
  }

  /** Repair the tree — every gate passes again. */
  repair(): void {
    this.broken = null;
  }

  /** The gate that is broken, or `null`. */
  get brokenGate(): Gate | null {
    return GATES.find((g) => g.id === this.broken) ?? null;
  }

  /** One row per gate: whether it passes under the current break. */
  get results(): { gate: Gate; pass: boolean }[] {
    return GATES.map((gate) => ({ gate, pass: gate.id !== this.broken }));
  }

  /** Whether every gate passes. */
  get passing(): boolean {
    return this.broken === null;
  }

  /** How many gates pass. */
  get passingCount(): number {
    return GATES.length - (this.broken === null ? 0 : 1);
  }

  /** The run's one-line verdict. */
  get report(): string {
    const g = this.brokenGate;
    if (!g) return `✓ all ${GATES.length} gates pass`;
    return `✕ ${g.label}: FAIL — ${g.what} is below "${g.floor}"`;
  }
}
