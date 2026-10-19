/** The cell store — one record per vocabulary item, merged from fragments.
 *
 * Plan 0028 §D4, implementing plan 0022.  The project spelled "a cell
 * as data" five ways; `mlopt_core.cells` normalises them to one
 * record — `Cell{role, name, body, kind, provenance}` where role ∈
 * `op` / `law` / `object` / `handle`.  `merge_fragments(*Fragment)`
 * assembles a `CellStore`: a duplicate `(role, name)` **raises** by
 * default (silent override is how vocabularies drift), while alpha-equal
 * 2-cells across fragments are a discovery fact, not an error.
 *
 * `Provenance.origin` ∈ `shipped` / `constructed` / `admitted` /
 * `minted` / `declared` / `imported`.  `mlopt_discovery.handl` is the
 * schema bridge: op cells → `AstShapeDef` rows, law/object cells →
 * `TransformDef` rows, handle cells → `!Native` records.
 */

export type CellRole = "op" | "law" | "object" | "handle";

export type Origin = "shipped" | "constructed" | "admitted" | "minted" | "declared" | "imported";

/** The provenance origins, in the order the level lists them. */
export const ORIGINS: readonly Origin[] = [
  "shipped",
  "constructed",
  "admitted",
  "minted",
  "declared",
  "imported",
];

/** One vocabulary item as one record. */
export interface Cell {
  role: CellRole;
  name: string;
  origin: Origin;
}

/** A named bundle of cells — the merge unit. */
export interface Fragment {
  id: string;
  cells: readonly Cell[];
}

/** The fragments a store assembles from — the shipped kernel and a mint. */
export const BASE_FRAGMENTS: readonly Fragment[] = [
  {
    id: "shipped-ops",
    cells: [
      { role: "op", name: "matmul", origin: "shipped" },
      { role: "op", name: "add", origin: "shipped" },
      { role: "op", name: "mul", origin: "shipped" },
    ],
  },
  {
    id: "shipped-laws",
    cells: [
      { role: "law", name: "assoc", origin: "shipped" },
      { role: "law", name: "comm", origin: "shipped" },
      { role: "law", name: "cse", origin: "shipped" },
    ],
  },
  {
    id: "constructed",
    cells: [{ role: "object", name: "aff", origin: "constructed" }],
  },
  {
    id: "minted",
    cells: [
      { role: "law", name: "mul_unsq_nat", origin: "minted" },
      { role: "handle", name: "silu_kernel", origin: "declared" },
    ],
  },
];

/** A fragment that collides with a shipped cell — the merge must refuse it. */
export const DUPLICATE_FRAGMENT: Fragment = {
  id: "rival-assoc",
  cells: [{ role: "law", name: "assoc", origin: "imported" }],
};

/** A `(role, name)` pair — the store's key. */
export function cellKey(c: Cell): string {
  return `${c.role}:${c.name}`;
}

/** The handl row kind a cell role serialises to. */
export function handlKind(role: CellRole): string {
  if (role === "op") return "AstShapeDef";
  if (role === "handle") return "!Native";
  return "TransformDef";
}

/** The cell store, playable: assemble fragments and watch a duplicate raise. */
export class CellStoreLab {
  fragments: Fragment[] = [...BASE_FRAGMENTS];
  /** The conflict the last merge reported, or `null`. */
  conflict: Cell | null = null;

  get cells(): Cell[] {
    return this.fragments.flatMap((f) => [...f.cells]);
  }

  /** How many distinct `(role, name)` keys the store holds. */
  get size(): number {
    return new Set(this.cells.map(cellKey)).size;
  }

  /** The origins present in the store. */
  get origins(): Origin[] {
    return ORIGINS.filter((o) => this.cells.some((c) => c.origin === o));
  }

  /** One cell per role — for the role→handl view. */
  get byRole(): { role: CellRole; count: number; handl: string }[] {
    const roles: CellRole[] = ["op", "law", "object", "handle"];
    return roles.map((role) => ({
      role,
      count: this.cells.filter((c) => c.role === role).length,
      handl: handlKind(role),
    }));
  }

  /** Merge a fragment.  A duplicate `(role, name)` is a conflict, not an override. */
  merge(frag: Fragment): void {
    const have = new Set(this.cells.map(cellKey));
    const clash = frag.cells.find((c) => have.has(cellKey(c)));
    if (clash) {
      this.conflict = clash;
      return;
    }
    this.conflict = null;
    this.fragments = [...this.fragments, frag];
  }

  /** Add the rival fragment — the demo that raises. */
  addDuplicate(): void {
    this.merge(DUPLICATE_FRAGMENT);
  }

  reset(): void {
    this.fragments = [...BASE_FRAGMENTS];
    this.conflict = null;
  }
}
