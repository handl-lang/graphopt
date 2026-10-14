import { afterEach, describe, expect, it } from "vite-plus/test";

import { CONCEPTS, MODULES, tier } from "engine";

import { Coherence } from "../src/components/coherence.ts";
import { Compare } from "../src/components/compare.ts";
import { Corpus } from "../src/components/corpus.ts";
import { CostModelFamily } from "../src/components/models.ts";
import { Course, STORAGE_KEY } from "../src/components/course.ts";
import { Bounded } from "../src/components/bounded.ts";
import { CellStore } from "../src/components/cellstore.ts";
import { Deliver } from "../src/components/deliver.ts";
import { DeliveryArtifacts } from "../src/components/artifacts.ts";
import { Expose } from "../src/components/exposure.ts";
import { Game } from "../src/components/game.ts";
import { Gates } from "../src/components/gates.ts";
import { Honest } from "../src/components/honest.ts";
import { Improve } from "../src/components/improve.ts";
import { Mint } from "../src/components/mint.ts";
import { Ports } from "../src/components/ports.ts";
import { Pricing } from "../src/components/pricing.ts";
import { Reach } from "../src/components/reach.ts";
import { Substrate } from "../src/components/substrate.ts";
import { Tensions } from "../src/components/tensions.ts";
import { Truthpay } from "../src/components/truthpay.ts";

/** Install a Map-backed localStorage shim; returns the backing map. */
function withStorage(): Map<string, string> {
  const map = new Map<string, string>();
  (globalThis as unknown as { localStorage: unknown }).localStorage = {
    getItem: (k: string) => map.get(k) ?? null,
    setItem: (k: string, v: string) => {
      map.set(k, v);
    },
    removeItem: (k: string) => {
      map.delete(k);
    },
    clear: () => {
      map.clear();
    },
    key: () => null,
    length: 0,
  };
  return map;
}

/** Answer every part of the current level's checkpoint correctly. */
function pass(c: Course): void {
  c.level.checkpoint.parts.forEach((p, i) => c.answerAt(i, p.ok));
}

const CONCEPT_IDS = new Set(CONCEPTS.map((c) => c.id));
const MODULE_IDS = MODULES.map((m) => m.id);

/** The measured forward-reference count after step 2 — it may only go down. */
const FORWARD_REF_CEILING = 0;

afterEach(() => {
  delete (globalThis as unknown as { localStorage?: unknown }).localStorage;
});

describe("course gating", () => {
  it("cannot advance before passing the checkpoint", () => {
    const c = new Course();
    expect(c.i).toBe(0);
    expect(c.canAdvance).toBe(false);
    c.next();
    expect(c.i).toBe(0);
  });

  it("advances once every part of the checkpoint is picked", () => {
    const c = new Course();
    c.answer(c.level.checkpoint.ok); // the primary part alone is not enough
    expect(c.canAdvance).toBe(false);
    pass(c);
    expect(c.canAdvance).toBe(true);
    c.next();
    expect(c.i).toBe(1);
  });

  it("rejects a wrong answer but still explains", () => {
    const c = new Course();
    const wrong = c.level.checkpoint.opts.find((o) => o !== c.level.checkpoint.ok) ?? "";
    c.answer(wrong);
    expect(c.canAdvance).toBe(false);
    expect(c.answered(c.level.id)).toBe(true);
  });

  it("is complete only when every level is passed", () => {
    const c = new Course();
    for (let k = 0; k < c.n; k++) {
      c.goto(k);
      pass(c);
    }
    expect(c.complete).toBe(true);
    expect(c.done).toBe(c.n);
  });

  it("every checkpoint's correct option is one of its options", () => {
    for (const l of new Course().levels) {
      expect(l.checkpoint.opts).toContain(l.checkpoint.ok);
      for (const p of l.checkpoint.parts) expect(p.opts).toContain(p.ok);
    }
  });

  it("runs twenty-eight levels, 0 through 27", () => {
    const c = new Course();
    expect(c.n).toBe(28);
    expect(c.levels.at(-1)?.id).toBe("improve");
    expect(c.levels.map((l) => l.id).length).toBe(new Set(c.levels.map((l) => l.id)).size);
  });

  it("includes the six coverage levels (plan 0028 §D) in course order", () => {
    const c = new Course();
    const ids = c.levels.map((l) => l.id);
    for (const id of ["corpus", "models", "artifacts", "cellstore", "game", "gates"]) {
      expect(ids).toContain(id);
    }
    expect(ids.indexOf("corpus")).toBeLessThan(ids.indexOf("reach"));
    expect(ids.indexOf("models")).toBeLessThan(ids.indexOf("architecture"));
    expect(ids.indexOf("gates")).toBeLessThan(ids.indexOf("honest"));
  });

  it("gives every level a ladder one-liner and a recap in order", () => {
    const c = new Course();
    for (const l of c.levels) expect(l.idea.length).toBeGreaterThan(0);
    expect(c.recap).toHaveLength(28);
    expect(c.recap[0].idea).toBe("ops over data, one root");
    expect(c.recap.at(-1)?.idea).toBe("idea → blocker → next measurement");
    expect(c.recap.every((r) => !r.passed)).toBe(true);
  });

  it("reports a cleared percentage for the progress meter", () => {
    const c = new Course();
    expect(c.pct).toBe(0);
    c.goto(0);
    pass(c);
    expect(c.pct).toBe(Math.round(100 / 28));
    expect(c.recap[0].passed).toBe(true);
  });

  it("tracks checkpoint progress part by part", () => {
    const c = new Course();
    const total = c.level.checkpoint.parts.length;
    expect(c.checkpointProgress).toEqual({ correct: 0, total, done: false });
    c.answerAt(0, c.level.checkpoint.parts[0].ok);
    expect(c.checkpointProgress.correct).toBe(1);
    expect(c.checkpointProgress.done).toBe(false);
    pass(c);
    expect(c.checkpointProgress.done).toBe(true);
  });

  it("restart clears progress", () => {
    const c = new Course();
    pass(c);
    c.next();
    c.restart();
    expect(c.i).toBe(0);
    expect(c.done).toBe(0);
  });
});

describe("the per-level template (plan 0028 §B)", () => {
  it("gives every level the four template fields plus 'why' and 'module'", () => {
    for (const l of new Course().levels) {
      expect(l.why.length).toBeGreaterThan(0);
      expect(l.outcome.length).toBeGreaterThan(0);
      expect(l.terms.length).toBeGreaterThan(0);
      expect(l.repo.length).toBeGreaterThan(0);
      for (const r of l.repo) {
        expect(r.path.length).toBeGreaterThan(0);
        expect(r.note.length).toBeGreaterThan(0);
      }
      expect(MODULE_IDS).toContain(l.module);
    }
  });

  it("lists exactly the concepts the graph introduces at each level", () => {
    const c = new Course();
    for (const l of c.levels) {
      const terms = [...l.terms].sort();
      const graph = c
        .graphTerms(l.id)
        .map((x) => x.id)
        .sort();
      expect(terms).toEqual(graph);
    }
  });

  it("resolves the current level's new terms from the graph", () => {
    const c = new Course();
    expect(c.level.id).toBe("tree");
    expect(c.levelTerms.map((t) => t.id).sort()).toEqual(["program-tree", "work"]);
    for (const t of c.levelTerms) expect(t.def.length).toBeGreaterThan(0);
  });

  it("names a module for every level", () => {
    const c = new Course();
    for (const l of c.levels) {
      c.gotoLevel(l.id);
      expect(c.module?.id).toBe(l.module);
    }
  });
});

describe("multi-part checkpoints", () => {
  it("gives every level at least two parts, one from the interactive", () => {
    for (const l of new Course().levels) {
      expect(l.checkpoint.parts.length).toBeGreaterThanOrEqual(2);
      expect(l.checkpoint.parts.some((p) => p.interactive)).toBe(true);
    }
  });

  it("keeps parts[0] identical to the single-question fields", () => {
    for (const l of new Course().levels) {
      expect(l.checkpoint.parts[0].q).toBe(l.checkpoint.q);
      expect(l.checkpoint.parts[0].ok).toBe(l.checkpoint.ok);
      expect(l.checkpoint.parts[0].opts).toEqual(l.checkpoint.opts);
      expect(l.checkpoint.parts[0].why).toBe(l.checkpoint.why);
    }
  });

  it("tags every part with a concept from the graph", () => {
    for (const l of new Course().levels) {
      for (const p of l.checkpoint.parts) {
        expect(CONCEPT_IDS.has(p.concept)).toBe(true);
        expect(p.q.length).toBeGreaterThan(0);
        expect(p.why.length).toBeGreaterThan(0);
      }
    }
  });
});

describe("forward references (the ordering gate)", () => {
  it("has zero forward references — and the count can only go down", () => {
    const c = new Course();
    expect(c.forwardRefs.length).toBeLessThanOrEqual(FORWARD_REF_CEILING);
    expect(c.forwardRefs).toHaveLength(0);
  });
});

describe("the glossary", () => {
  it("lists every concept once, with a definition and a home level", () => {
    const c = new Course();
    expect(c.glossary).toHaveLength(CONCEPTS.length);
    expect(new Set(c.glossary.map((r) => r.id)).size).toBe(CONCEPTS.length);
    for (const r of c.glossary) {
      expect(r.label.length).toBeGreaterThan(0);
      expect(r.def.length).toBeGreaterThan(0);
      expect(r.title.length).toBeGreaterThan(0);
      expect(c.levels.some((l) => l.id === r.introduced)).toBe(true);
    }
  });

  it("jumps to a concept's home level", () => {
    const c = new Course();
    const row = c.glossary.find((r) => r.id === "inversion")!;
    expect(row.introduced).toBe("pricing");
    c.gotoLevel(row.introduced);
    expect(c.level.id).toBe("pricing");
  });
});

describe("prerequisites (the ladder is not flat)", () => {
  it("shows the current level's prerequisites as earlier levels", () => {
    const c = new Course();
    c.gotoLevel("pricing");
    expect(c.prerequisites.length).toBeGreaterThan(0);
    for (const p of c.prerequisites) {
      expect(p.index).toBeLessThan(c.i);
      expect(p.title.length).toBeGreaterThan(0);
    }
    expect(c.prerequisites.map((p) => p.id)).toContain("measure");
  });

  it("gives the entry level none", () => {
    const c = new Course();
    c.gotoLevel("tree");
    expect(c.prerequisites).toHaveLength(0);
  });
});

describe("gamification wiring (plan 0028 §C)", () => {
  it("awards the base only when the whole multi-part checkpoint passes", () => {
    const c = new Course();
    c.answer(c.level.checkpoint.ok); // part 0 only
    expect(c.levelXp.base).toBe(0);
    pass(c);
    expect(c.levelXp.base).toBeGreaterThan(0);
  });

  it("feeds the mastery meter from each part's concept tag", () => {
    const c = new Course();
    pass(c);
    const concepts = new Set(c.level.checkpoint.parts.map((p) => p.concept));
    for (const id of concepts) expect(tier(c.mastery[id])).not.toBe("weak");
  });

  it("counts a concept mastered after a part right in two places", () => {
    const c = new Course();
    const id = c.level.checkpoint.parts[0].concept;
    // answer the same concept's question at this level and again at another
    pass(c);
    c.gotoLevel("reach");
    pass(c);
    // at least the level's own concept is no longer weak
    expect(tier(c.mastery[id])).not.toBe("weak");
  });
});

describe("reach", () => {
  it("structural rewriting reaches strictly more than op-level", () => {
    const r = new Reach();
    const ops = r.count;
    r.setMode("structure");
    expect(r.count).toBeGreaterThan(ops);
    expect(r.bestCost).toBeLessThan(220);
  });
});

describe("compare", () => {
  it("shows two equal programs at different costs", () => {
    const c = new Compare();
    expect(c.sameFunction).toBe(true);
    expect(c.rightCost).toBeLessThan(c.leftCost);
  });

  it("states the dimensions the costs depend on", () => {
    const c = new Compare();
    expect(c.shapes).toContain("2×3");
    expect(c.rule).toContain("2·m·n·p");
  });

  it("its worked arithmetic matches the engine's cost", () => {
    // a·(b·c) = 2·(3·4·5) + 2·(2·3·5) = 120 + 60 = 180
    // (a·b)·c = 2·(2·3·4) + 2·(2·4·5) =  48 + 80 = 128
    const c = new Compare();
    expect(c.leftBreakdown).toBe("2·(3·4·5) + 2·(2·3·5) = 120 + 60");
    expect(c.rightBreakdown).toBe("2·(2·3·4) + 2·(2·4·5) = 48 + 80");
    expect(c.leftCost).toBe(180);
    expect(c.rightCost).toBe(128);
  });
});

describe("persistence", () => {
  it("writes progress on answer and on advance", () => {
    const map = withStorage();
    const c = new Course();
    pass(c);
    c.next();
    expect(map.has(STORAGE_KEY)).toBe(true);
    expect(c.i).toBe(1);
  });

  it("restores level and answers on init", () => {
    withStorage();
    const a = new Course();
    pass(a);
    a.next();

    const b = new Course();
    b.init();
    expect(b.i).toBe(1);
    expect(b.restored).toBe(true);
    expect(b.passed(b.levels[0].id)).toBe(true);
    expect(b.savedLabel).toBe("restored from this browser");
  });

  it("resetProgress wipes the stored key and the in-memory state", () => {
    const map = withStorage();
    const c = new Course();
    pass(c);
    c.next();
    c.resetProgress();
    expect(map.has(STORAGE_KEY)).toBe(false);
    expect(c.i).toBe(0);
    expect(c.done).toBe(0);
    expect(c.settings).toBe(false);
  });

  it("survives corrupt storage instead of crashing", () => {
    const map = withStorage();
    map.set(STORAGE_KEY, "{ not json");
    const c = new Course();
    c.init();
    expect(c.i).toBe(0);
    expect(map.has(STORAGE_KEY)).toBe(false);
  });

  it("clamps an out-of-range saved level", () => {
    const map = withStorage();
    map.set(STORAGE_KEY, JSON.stringify({ i: 999, answers: {} }));
    const c = new Course();
    c.init();
    expect(c.i).toBe(c.n - 1);
  });

  it("does nothing when storage is unavailable", () => {
    const c = new Course();
    pass(c);
    c.next();
    expect(c.i).toBe(1);
  });
});

describe("tensions", () => {
  it("has a valid correct option per item", () => {
    for (const t of new Tensions().items) {
      expect(t.opts).toContain(t.ok);
    }
  });
});

describe("coherence", () => {
  it("a reordering inside a fold's redex rejoins under the pair alone", () => {
    const c = new Coherence();
    c.pick(0);
    expect(c.current.def.a).toBe("comm");
    expect(c.current.def.b).toBe("cse");
    expect(c.current.verdict).toBe("pair-confluent");
  });

  it("the critical pair diverges without the fold and is mediated with it", () => {
    const c = new Coherence();
    c.pick(3); // expand × fuse
    expect(c.current.ra.length).toBeGreaterThan(0);
    expect(c.current.rb.length).toBeGreaterThan(0);
    c.withFold = false;
    expect(c.current.verdict).toBe("divergent");
    c.withFold = true;
    expect(c.current.verdict).toBe("library-mediated");
    expect(c.current.mediator).toBe("square");
  });

  it("an inverse pair with no shared redex reports no-cofire", () => {
    const c = new Coherence();
    c.pick(4); // expand × square on sqdup — square has no redex there
    expect(c.current.verdict).toBe("no-cofire");
  });
});

describe("mint", () => {
  it("moves the funnel through the four layers for a picked need", () => {
    const m = new Mint();
    expect(m.needs.length).toBe(6);
    m.pick(0);
    expect(m.layers.every((l) => l.ok)).toBe(true);
    expect(m.verdict).toContain("pays nothing");
    m.pick(5); // the vocabulary gap
    expect(m.layers.some((l) => !l.ok)).toBe(true);
  });

  it("exposes the live funnel tallies", () => {
    const m = new Mint();
    const laws = m.funnel.find((s) => s.key === "laws")!;
    const pay = m.funnel.find((s) => s.key === "pay")!;
    expect(laws.count).toBe(15);
    expect(pay.count).toBe(0);
  });
});

describe("truthpay", () => {
  it("breaks the naive guard and finds the exact one clean", () => {
    const t = new Truthpay();
    expect(t.guard).toBe("naive");
    expect(t.witnessCount).toBeGreaterThan(0);
    expect(t.witness?.bad).toBe(true);
    t.setGuard("exact");
    expect(t.witnessCount).toBe(0);
    expect(t.synthesized).toContain("flat-map-unsq");
  });

  it("steps the counterexample viewer", () => {
    const t = new Truthpay();
    expect(t.witnessIndex).toBe(0);
    t.next();
    expect(t.witnessIndex).toBeGreaterThanOrEqual(0);
    t.prev();
    expect(t.witnessIndex).toBe(0);
  });
});

describe("expose", () => {
  it("computes the live histogram and cites the corpus anchors", () => {
    const e = new Expose();
    expect(e.histogram.reduce((p, b) => p + b.count, 0)).toBe(e.workloads.length);
    expect(e.corpus.total).toBe(321);
    expect(e.modes.zero).toBe(8);
    expect(e.visibleButNull.length).toBe(4);
    e.setMode("ops");
    expect(e.mode).toBe("ops");
    expect(e.invisibleOps.some((o) => o.op === "conv2d")).toBe(true);
    expect(e.visibleOps.some((o) => o.op === "matmul")).toBe(true);
  });
});

describe("substrate", () => {
  it("round-trips every term and spells the loop", () => {
    const s = new Substrate();
    expect(s.roundTrips).toBe(true);
    s.pick(2); // dup — the shared program
    expect(s.dups).toBe(2);
    expect(s.comb).toContain("dup");
    expect(s.loopRoundTrips).toBe(true);
    expect(s.loopComb).toContain("loop(");
  });
});

describe("ports", () => {
  it("swaps the sink with the core counter at zero", () => {
    const p = new Ports();
    expect(p.sink).toBe("TorchSink");
    p.swap("OnnxSink");
    expect(p.sink).toBe("OnnxSink");
    expect(p.coreMutations).toBe(0);
    expect(p.unchanged).toBe(true);
    expect(p.unsupported).toContain("sqq");
    expect(p.lowerable).toBe(false);
  });

  it("only a core edit moves the counter", () => {
    const p = new Ports();
    p.edit();
    expect(p.coreMutations).toBe(1);
    expect(p.unchanged).toBe(false);
  });
});

describe("honest", () => {
  it("has a valid correct option per refusal", () => {
    for (const r of new Honest().items) {
      expect(r.opts).toContain(r.ok);
    }
  });

  it("scores answers", () => {
    const h = new Honest();
    expect(h.score).toBe(0);
    h.answer(0, h.items[0].ok);
    expect(h.score).toBe(1);
  });
});

describe("bounded", () => {
  it("certifies offers the pricing then refuses", () => {
    const b = new Bounded();
    b.select(5); // 3e-2
    expect(b.offered).toBe(56);
    expect(b.delivered).toBe(0);
    expect(b.pricedRefused).toBe(56);
    expect(b.gateClean).toBe(true);
    expect(b.verdict).toContain("pricing refused");
  });

  it("delivers once the dedup is thick", () => {
    const b = new Bounded();
    b.select(5);
    b.toggle();
    expect(b.thick).toBe(true);
    expect(b.delivered).toBe(56);
    expect(b.rows[0].ratio).toBeLessThan(1);
  });
});

describe("deliver", () => {
  it("picks the minted kernel on both terms, agreeing with the clock", () => {
    const d = new Deliver();
    expect(d.deliveryCase.id).toBe("retention");
    expect(d.modelPick.kind).toBe("minted");
    expect(d.winner.kind).toBe("minted");
    expect(d.agrees).toBe(true);
    expect(d.speedup).toBeGreaterThan(3);
    d.select(1);
    expect(d.deliveryCase.id).toBe("chainfuse");
    expect(d.rows.filter((r) => r.winner)).toHaveLength(1);
  });

  it("marks the illustrative delivery", () => {
    const d = new Deliver();
    expect(d.rows.some((r) => !r.delivery.sourced)).toBe(true);
  });
});

describe("pricing", () => {
  it("computes the inversion and flips it when calibrated", () => {
    const p = new Pricing();
    expect(p.inverted).toBe(true);
    expect(p.speedup).toBeGreaterThan(1.9);
    expect(p.speedup).toBeLessThan(2.7);
    expect(p.correct).toBe(false);
    p.toggle();
    expect(p.correct).toBe(true);
    expect(p.pick.name).toBe("generic");
  });

  it("selects a case and counts the model's hits", () => {
    const p = new Pricing();
    p.select(2);
    expect(p.routeCase.id).toBe("manual-softmax-small");
    expect(p.correctCount.right).toBe(1);
    expect(p.rows).toHaveLength(3);
  });
});

describe("improve", () => {
  it("reads the three-step connection for a picked gap", () => {
    const g = new Improve();
    expect(g.gaps.length).toBeGreaterThanOrEqual(6);
    expect(g.gap.level).toBe("mint");
    expect(g.connection).toHaveLength(3);
    g.pick(99);
    expect(g.gap.level).toBe("honest");
  });
});

describe("module reviews (plan 0028 §A/§C)", () => {
  it("gives every level a module, and every module a review", () => {
    const c = new Course();
    for (const l of c.levels) {
      const m = MODULES.find((x) => x.id === l.module);
      expect(m).toBeDefined();
      expect(m!.questions.length).toBeGreaterThanOrEqual(4);
    }
    expect(c.moduleStates).toHaveLength(6);
    expect(c.currentModule.id).toBe(c.level.module);
  });

  it("unlocks a module's review only when its levels are all passed", () => {
    const c = new Course();
    const m = MODULES[0];
    expect(c.moduleStates[0].unlocked).toBe(false);
    for (const id of m.levels) {
      c.gotoLevel(id);
      pass(c);
    }
    expect(c.moduleStates[0].levelsCleared).toBe(true);
    expect(c.moduleStates[0].unlocked).toBe(true);
    expect(c.moduleStates[0].cleared).toBe(false); // the review is still needed
  });

  it("passes the review, records it, and clears the module", () => {
    const c = new Course();
    const m = MODULES[0];
    for (const id of m.levels) {
      c.gotoLevel(id);
      pass(c);
    }
    for (const q of m.questions) c.answerReview(m.id, q.id, q.ok);
    const r = c.submitReview(m.id);
    expect(r.passed).toBe(true);
    expect(c.reviewed).toContain(m.id);
    expect(c.moduleStates[0].cleared).toBe(true);
    expect(c.reviewsPassed).toBe(1);
    expect(c.reviewPassed(m.id, m.questions[0].id)).toBe(true);
  });

  it("fails a review below threshold and sends the weak concepts back", () => {
    const c = new Course();
    const m = MODULES[0];
    for (const id of m.levels) {
      c.gotoLevel(id);
      pass(c);
    }
    m.questions.forEach((q, i) => c.answerReview(m.id, q.id, i < 3 ? "definitely wrong" : q.ok));
    const r = c.submitReview(m.id);
    expect(r.passed).toBe(false);
    expect(c.reviewed).not.toContain(m.id);
    expect(c.moduleStates[0].weak.length).toBeGreaterThan(0);
    expect(c.reviewWeak(m.id).length).toBeGreaterThan(0);
    expect(c.reviewWeak(m.id)[0].levels.length).toBeGreaterThan(0);
  });

  it("feeds the mastery meter from review answers (spaced retrieval)", () => {
    const c = new Course();
    const m = MODULES[0];
    const q = m.questions[0];
    c.answerReview(m.id, q.id, q.ok);
    expect(tier(c.mastery[q.concept])).not.toBe("weak");
  });

  it("persists review answers and the reviewed list", () => {
    const map = withStorage();
    const a = new Course();
    const m = MODULES[0];
    for (const q of m.questions) a.answerReview(m.id, q.id, q.ok);
    a.submitReview(m.id);
    expect(map.has(STORAGE_KEY)).toBe(true);

    const b = new Course();
    b.init();
    expect(b.reviewed).toContain(m.id);
    expect(b.reviewAnswers[m.id][m.questions[0].id]).toBe(m.questions[0].ok);
    expect(b.reviewsPassed).toBe(1);
  });

  it("resets a review and un-records the pass", () => {
    const c = new Course();
    const m = MODULES[0];
    for (const q of m.questions) c.answerReview(m.id, q.id, q.ok);
    c.submitReview(m.id);
    expect(c.reviewed).toContain(m.id);
    c.resetReview(m.id);
    expect(c.reviewed).not.toContain(m.id);
    expect(c.reviewAnswers[m.id]).toBeUndefined();
  });
});

describe("the completion state (plan 0028 §C)", () => {
  it("reports XP, rank, levels, mastery, badges and the gaps", () => {
    const c = new Course();
    for (let k = 0; k < c.n; k++) {
      c.goto(k);
      pass(c);
      c.engage();
    }
    const s = c.completion;
    expect(s.levelsCleared).toBe(28);
    expect(s.levelsTotal).toBe(28);
    expect(s.xp).toBeGreaterThan(0);
    expect(s.rank.name.length).toBeGreaterThan(0);
    expect(s.weak + s.familiar + s.mastered).toBe(s.conceptsTotal);
    expect(s.badgesEarned).toBeGreaterThan(0);
    expect(s.badgesTotal).toBeGreaterThanOrEqual(s.badgesEarned);
    expect(s.reviewsTotal).toBe(6);
    expect(s.gaps.length).toBeGreaterThanOrEqual(6);
    expect(s.gaps[0].blocker.length).toBeGreaterThan(0);
    expect(s.gaps[0].next.length).toBeGreaterThan(0);
  });

  it("is not fully complete until every review passes", () => {
    const c = new Course();
    for (let k = 0; k < c.n; k++) {
      c.goto(k);
      pass(c);
    }
    expect(c.complete).toBe(true);
    expect(c.fullyComplete).toBe(false);
    for (const m of MODULES) {
      for (const q of m.questions) c.answerReview(m.id, q.id, q.ok);
      c.submitReview(m.id);
    }
    expect(c.reviewsPassed).toBe(6);
    expect(c.fullyComplete).toBe(true);
  });
});

describe("corpus", () => {
  it("computes the funnel and cites the corpus totals", () => {
    const c = new Corpus();
    expect(c.classes.total).toBe(c.workloads.length);
    expect(c.funnel.reduce((s, f) => s + f.count, 0)).toBe(c.workloads.length);
    expect(c.classes.censusOnly).toBe(5);
    expect(c.corpus.total).toBe(166);
    expect(c.corpus.verified).toBe(114);
    expect(c.defects).toHaveLength(3);
    c.pick(99);
    expect(c.picked.status).toBe("rejected");
  });
});

describe("costmodels", () => {
  it("shows the inversion and its calibrated fix", () => {
    const m = new CostModelFamily();
    expect(m.models).toHaveLength(4);
    expect(m.inverted).toBe(true);
    expect(m.correct).toBe(false);
    expect(m.pick("roofline")).toContain("route-independent");
    m.toggle();
    expect(m.calibrated).toBe(true);
    expect(m.correct).toBe(true);
    expect(m.deliveredPick.kind).toBe("generic");
    expect(m.rows).toHaveLength(2);
    expect(m.rows.filter((r) => r.deliveredPick)).toHaveLength(1);
  });
});

describe("artifacts", () => {
  it("reads what each artifact needs to run", () => {
    const a = new DeliveryArtifacts();
    expect(a.standaloneCount).toBe(2);
    expect(a.portable.id).toBe("pt2");
    a.pick(5);
    expect(a.artifact.id).toBe("cuda-graph");
    expect(a.needs).toContain("a CUDA device");
    expect(a.standalone).toBe(false);
  });
});

describe("cellstore", () => {
  it("assembles a store and raises on a duplicate", () => {
    const c = new CellStore();
    expect(c.size).toBeGreaterThan(0);
    expect(c.conflict).toBeNull();
    c.addDuplicate();
    expect(c.conflict?.name).toBe("assoc");
    expect(c.byRole.find((r) => r.role === "handle")!.handl).toBe("!Native");
    c.reset();
    expect(c.conflict).toBeNull();
  });
});

describe("game", () => {
  it("steps the episode and compares the priors", () => {
    const g = new Game();
    expect(g.domains).toContain("open");
    expect(g.at.move).toBe("saturate");
    g.next();
    expect(g.at.move).toBe("declare");
    expect(g.best.id).toBe("learned");
    expect(g.priorComparison.learned.ap).toBeCloseTo(0.133, 6);
    expect(g.priorComparison.heuristic.ap).toBeCloseTo(0.045, 6);
    g.reset();
    expect(g.step).toBe(0);
  });
});

describe("gates", () => {
  it("breaks a gate and fails the run", () => {
    const g = new Gates();
    expect(g.passing).toBe(true);
    g.break("coverage");
    expect(g.passing).toBe(false);
    expect(g.report).toContain("FAIL");
    expect(g.rule).toContain("never silenced");
    g.repair();
    expect(g.passing).toBe(true);
  });
});
