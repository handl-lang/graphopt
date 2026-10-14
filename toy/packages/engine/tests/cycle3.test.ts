import { describe, expect, it } from "vite-plus/test";

import {
  CORPUS,
  Exposure,
  Guards,
  LOOP_TERM,
  Minting,
  PAY_NULL_FIRES,
  PROGRAMS,
  Ports,
  Substrate,
  VOCAB_GAP,
  bcast,
  combToDag,
  dagToComb,
  dupCount,
  key,
  renderComb,
  unsq,
} from "../src/index.ts";
import type { Term } from "../src/index.ts";

describe("minting funnel", () => {
  it("tallies the four layers live over the need bank", () => {
    const f = new Minting().funnel;
    const at = (k: string) => f.find((s) => s.key === k)?.count;
    expect(at("need")).toBe(6);
    expect(at("sig")).toBe(5);
    expect(at("handler")).toBe(5);
    expect(at("laws")).toBe(15);
    expect(at("admit")).toBe(5);
    expect(at("pay")).toBe(PAY_NULL_FIRES);
  });

  it("clamps the selection and reads the picked need", () => {
    const m = new Minting();
    m.pick(-4);
    expect(m.need.op).toBe("cumsum");
    m.pick(99);
    expect(m.need.sig).toBe("refused");
    expect(m.stoppedAt).toBe("signature");
  });

  it("measures truth and pay separately — every admitted need is pay-null", () => {
    const m = new Minting();
    m.pick(0);
    expect(m.verdict).toContain("pays nothing");
    expect(m.verdictTone).toBe("text-eva");
    expect(VOCAB_GAP).toBe(199);
  });

  it("marks the unblocked vocab need as admitted but not paying", () => {
    const m = new Minting();
    m.pick(4); // pow
    expect(m.verdict).toContain("nothing pays");
    expect(m.stoppedAt).toBeNull();
  });

  it("reads the four layers for the picked need", () => {
    const m = new Minting();
    m.pick(0);
    const layers = m.layers;
    expect(layers.map((l) => l.key)).toEqual(["need", "signature", "handler", "laws", "admission"]);
    expect(layers.every((l) => l.ok)).toBe(true);
    m.pick(5); // the vocabulary gap
    expect(m.layers.find((l) => l.key === "signature")!.ok).toBe(false);
    expect(m.layers.find((l) => l.key === "admission")!.ok).toBe(false);
  });
});

describe("guard synthesis sweep", () => {
  it("finds bad accepts for the naive grid guard, none for the exact guard", () => {
    const g = new Guards();
    const naive = g.sweep.find((r) => r.key === "naive")!;
    const exact = g.sweep.find((r) => r.key === "exact")!;
    expect(naive.badAccept).toBeGreaterThan(0);
    expect(naive.missed).toBe(0);
    expect(exact.badAccept).toBe(0);
    expect(exact.missed).toBe(0);
  });

  it("contains the classic mid-insert counterexample", () => {
    const g = new Guards();
    const witness = g.badAccepts.some(
      (s) => JSON.stringify(s.u) === "[2]" && JSON.stringify(s.v) === "[1,2,2]" && s.d === 1,
    );
    expect(witness).toBe(true);
  });

  it("partitions the domain into equal, unequal and ill-typed sites", () => {
    const d = new Guards().domain;
    expect(d.sites).toBe(d.evaluable + d.illTyped);
    expect(d.evaluable).toBe(d.equal + d.unequal);
    expect(d.equal).toBeGreaterThan(0);
    expect(d.unequal).toBeGreaterThan(0);
  });

  it("switching to the exact guard leaves no witness of a bad accept", () => {
    const g = new Guards();
    g.setGuard("exact");
    expect(g.badAccepts).toHaveLength(0);
    expect(g.witness?.equal).toBe(true);
    expect(g.synthesized).toContain("flat-map-unsq");
  });

  it("unsqueeze normalizes the dim modulo rank+1", () => {
    expect(unsq([2], 1)).toEqual([2, 1]);
    expect(unsq([2], -1)).toEqual([2, 1]);
    expect(unsq([1, 2, 2], 1)).toEqual([1, 1, 2, 2]);
    expect(bcast([2, 1], [1, 1, 2, 2])).toEqual([1, 1, 2, 2]);
    expect(bcast([2], [3])).toBeNull();
  });
});

describe("exposure screen", () => {
  it("computes a bimodal histogram live over the workload slice", () => {
    const e = new Exposure();
    const hist = e.histogram;
    expect(hist.reduce((p, b) => p + b.count, 0)).toBe(e.workloads.length);
    expect(hist[0].count).toBeGreaterThan(0);
    expect(hist.at(-1)!.count).toBeGreaterThan(0);
  });

  it("separates the fused-op mode from the spelled-math mode", () => {
    const e = new Exposure();
    expect(e.modes.zero).toBe(8);
    expect(e.modes.high).toBe(16);
    expect(e.maxBin).toBe(e.modes.high);
  });

  it("exposure is necessary, not sufficient — fully visible models that pay 0", () => {
    const e = new Exposure();
    expect(e.visibleButNull.length).toBe(4);
    expect(e.visibleButNull.map((w) => w.name)).toContain("DeepEquilibrium");
  });

  it("cites the measured corpus totals and yield", () => {
    expect(CORPUS.total).toBe(321);
    expect(CORPUS.zero).toBe(81);
    expect(CORPUS.high).toBe(140);
  });

  it("classifies the black-box ops as invisible", () => {
    const e = new Exposure();
    const conv = e.ops.find((o) => o.op === "conv2d")!;
    expect(conv.visible).toBe(false);
    expect(e.ops.find((o) => o.op === "matmul")!.visible).toBe(true);
  });
});

describe("combinator presentation", () => {
  it("round-trips every toy program dag → comb → dag", () => {
    for (const term of Object.values(PROGRAMS)) {
      expect(key(combToDag(dagToComb(term)))).toBe(key(term));
    }
  });

  it("makes sharing explicit as Δ sites", () => {
    const comb = dagToComb(PROGRAMS.dup);
    expect(dupCount(comb)).toBe(2);
    expect(renderComb(comb)).toContain("dup");
    expect(dupCount(dagToComb(PROGRAMS.chain))).toBe(0);
  });

  it("spells a recurrence as loop(step) and round-trips it", () => {
    const comb = dagToComb(LOOP_TERM);
    expect(comb.k).toBe("loop");
    expect(renderComb(comb).startsWith("loop(")).toBe(true);
    expect(key(combToDag(comb))).toBe(key(LOOP_TERM));
  });

  it("drives the level's interactive round trip", () => {
    const s = new Substrate(Object.entries(PROGRAMS).map(([id, term]) => ({ id, term })));
    expect(s.roundTrips).toBe(true);
    s.pick(1);
    expect(s.roundTrips).toBe(true);
    expect(s.loop.roundTrips).toBe(true);
  });

  it("combToDag expands a bare product to a pack op", () => {
    const t: Term = combToDag({
      k: "par",
      fs: [
        { k: "leaf", name: "a" },
        { k: "leaf", name: "b" },
      ],
    });
    expect(key(t)).toBe("pack(a,b)");
  });
});

describe("backend port", () => {
  it("swapping the sink leaves the core byte-identical and the counter at zero", () => {
    const p = new Ports(PROGRAMS.chain, ["assoc", "comm"]);
    const before = p.coreFingerprint;
    p.swap("OnnxSink");
    expect(p.sink.name).toBe("OnnxSink");
    expect(p.coreFingerprint).toBe(before);
    expect(p.coreMutations).toBe(0);
    expect(p.swaps).toBe(1);
  });

  it("re-bounds extraction through supported_ops", () => {
    const fused: Term = { op: "sqq", args: [{ leaf: "x" }, { leaf: "x" }] };
    const p = new Ports(fused, ["fuse"]);
    expect(p.lowerable).toBe(true);
    expect(p.lower()).toContain("torch");
    p.swap("OnnxSink");
    expect(p.lowerable).toBe(false);
    expect(p.unsupported).toEqual(["sqq"]);
    expect(p.lower()).toContain("refused");
    expect(p.coreMutations).toBe(0);
  });

  it("only a core edit moves the mutation counter", () => {
    const p = new Ports(PROGRAMS.dup, []);
    p.swap("OnnxSink");
    p.swap("TorchSink");
    expect(p.coreMutations).toBe(0);
    p.editCore("square");
    expect(p.coreMutations).toBe(1);
    expect(p.coreFingerprint).toContain("square");
  });
});
