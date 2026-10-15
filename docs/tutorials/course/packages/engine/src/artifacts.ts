/** Delivery artifacts — what "runnable" means.
 *
 * Plan 0028 §D3: a delivered term has to ship as something that runs.
 * `catopt_torch.export.export_optimized` / `load_optimized` write a
 * `module` / `safetensors` / `state_dict` / `torchscript` artifact, and
 * a `.pt2` round-trips to a **standalone** program with no catopt at
 * inference.  Beyond a file, delivery is also a *runner* — the CUDA-graph
 * runner captures an executor into one replay (`catopt_cuda`), and the
 * opt-in `catopt-native` PyO3/Rust engine is a faster closure over the
 * same IR.
 *
 * This level is the vocabulary of *what each artifact needs to run
 * standalone*: a weights file needs the model class back, a `.pt2`
 * needs only torch, a CUDA graph needs the device and the captured
 * shapes.  `docs/api.md` is the source.
 */

/** What an artifact is — a file, a runner, or an engine. */
export type ArtifactKind =
  | "module"
  | "safetensors"
  | "state_dict"
  | "torchscript"
  | "pt2"
  | "cuda-graph"
  | "native";

export interface Artifact {
  id: ArtifactKind;
  label: string;
  /** The extension, or `—` for a runner/engine (not a file). */
  ext: string;
  /** What it needs to run standalone. */
  needs: readonly string[];
  /** Runs with no catopt at inference. */
  standalone: boolean;
  /** Runs with no torch at all. */
  torchFree: boolean;
  note: string;
  sourced: boolean;
}

/** The delivery artifacts — the shipped export formats plus the runners. */
export const ARTIFACTS: readonly Artifact[] = [
  {
    id: "module",
    label: "optimized module",
    ext: ".py",
    needs: ["the catopt runtime", "torch"],
    standalone: false,
    torchFree: false,
    note: 'the optimized `nn.Module`, run in-process — the default `fmt="module"`',
    sourced: true,
  },
  {
    id: "safetensors",
    label: "safetensors",
    ext: ".safetensors",
    needs: ["the model class", "torch"],
    standalone: false,
    torchFree: false,
    note: "weights only, no graph — needs the class back to reconstruct",
    sourced: true,
  },
  {
    id: "state_dict",
    label: "state_dict",
    ext: ".pt",
    needs: ["the model class", "torch"],
    standalone: false,
    torchFree: false,
    note: "the classic `torch.save(model.state_dict())` — class + weights",
    sourced: true,
  },
  {
    id: "torchscript",
    label: "torchscript",
    ext: ".pt",
    needs: ["torch (libtorch)"],
    standalone: true,
    torchFree: false,
    note: "the graph is baked — runs without the original class",
    sourced: true,
  },
  {
    id: "pt2",
    label: "`.pt2` exported program",
    ext: ".pt2",
    needs: ["torch"],
    standalone: true,
    torchFree: false,
    note: "round-trips standalone — no catopt at inference (docs/api.md)",
    sourced: true,
  },
  {
    id: "cuda-graph",
    label: "CUDA-graph replay",
    ext: "—",
    needs: ["a CUDA device", "the captured input shapes"],
    standalone: false,
    torchFree: false,
    note: "`CudaGraphRunner` captures the executor into one replay — compile-free",
    sourced: true,
  },
  {
    id: "native",
    label: "native engine",
    ext: "—",
    needs: ["the PyO3/Rust extension"],
    standalone: false,
    torchFree: false,
    note: "`catopt-native` — a faster closure over the same IR (~17× on match-bound closures)",
    sourced: true,
  },
];

/** The artifacts level, playable: pick one, read what it needs to run. */
export class Artifacts {
  sel = 0;

  get artifacts(): readonly Artifact[] {
    return ARTIFACTS;
  }

  get artifact(): Artifact {
    return ARTIFACTS[this.sel];
  }

  pick(i: number): void {
    this.sel = Math.max(0, Math.min(ARTIFACTS.length - 1, i));
  }

  /** What the selected artifact needs to run standalone. */
  get needs(): readonly string[] {
    return this.artifact.needs;
  }

  /** Whether the selected artifact runs with no catopt at inference. */
  get standalone(): boolean {
    return this.artifact.standalone;
  }

  /** How many of the artifacts run standalone. */
  get standaloneCount(): number {
    return ARTIFACTS.filter((a) => a.standalone).length;
  }

  /** The artifact that round-trips with no catopt at inference. */
  get portable(): Artifact {
    return ARTIFACTS.find((a) => a.id === "pt2")!;
  }
}
