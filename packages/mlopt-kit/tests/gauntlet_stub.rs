//! `handl-discovery`'s admission gauntlet driven over injected domain
//! legs — the proof that `GauntletLegs` is a usable injection surface.
//!
//! The legs here are deliberately *stubs* (an in-memory object store,
//! the shipped `GenericCodec`, a constant `SiteEval`, a one-line
//! measure) — the point is the *skeleton*: `run_gauntlet` drives the
//! real gate pipeline (reconstruct → full-data → measure → truth →
//! novelty → typed-pay → closure → cert) over whatever the domain
//! injects, and stops-on-failure honestly. The mlopt domain legs
//! (torch truth probes, the lemma store, the measure pipeline) plug
//! into the same trait unchanged.

use std::collections::BTreeMap;
use std::rc::Rc;

use handl_discovery::{
    Evidence, GauntletLegs, GauntletOptions, GenericCodec, GuardedRegion, Mint, ObjectCodec,
    ObjectRecord, ObjectStore, SiteEval, SiteOutcome, Term, run_gauntlet,
};
use handl_egraph::Rewrite;
use serde_json::json;

/// An in-memory declared-object store — the spec's sqlite `lemmas`
/// table as a `BTreeMap`.
struct MemStore {
    by_key: BTreeMap<String, ObjectRecord>,
    by_name: BTreeMap<String, ObjectRecord>,
}

impl MemStore {
    fn with(records: Vec<(String, ObjectRecord)>) -> MemStore {
        let mut by_key = BTreeMap::new();
        let mut by_name = BTreeMap::new();
        for (key, rec) in records {
            by_name.insert(rec.name.to_string(), rec.clone());
            by_key.insert(key, rec);
        }
        MemStore { by_key, by_name }
    }
}

impl ObjectStore for MemStore {
    fn record_by_name(&self, name: &str) -> Option<ObjectRecord> {
        self.by_name.get(name).cloned()
    }

    fn record_by_key(&self, key: &str) -> Option<ObjectRecord> {
        self.by_key.get(key).cloned()
    }
}

/// A bare `add(?a, 0) -> ?a` record — the `id_add` shape.
fn add_zero_record() -> ObjectRecord {
    ObjectRecord {
        name: Rc::from("stub_add_zero"),
        kind: Rc::from("law"),
        lhs: Term::opl(
            "add",
            vec![Term::var("a"), Term::leaf("0")],
        ),
        rhs: Term::var("a"),
        law: "a + 0 = a".to_string(),
        missing_hooks: Vec::new(),
        guard: None,
        transported: Vec::new(),
        data: json!({}),
    }
}

/// The stub legs: an oracle that always answers `Equal`, a measure leg
/// returning the configured evidence template.
struct StubLegs {
    store: MemStore,
    codec: GenericCodec,
    base: Vec<Rc<Mint>>,
    eval: SiteEvalOwned,
    /// What `measure` reports — `num_true` is the knob the tests turn.
    num_true: Option<bool>,
}

/// `SiteEval` is a `dyn Fn` alias; the legs hold it boxed.
type SiteEvalOwned = Box<dyn Fn(&Term, &Term) -> SiteOutcome>;

impl StubLegs {
    fn new(records: Vec<(String, ObjectRecord)>, num_true: Option<bool>) -> StubLegs {
        // One axiom in the base — `comm_add` as a bare mint — so the
        // premise-closure leg evaluates a non-empty axiom set.
        let base = vec![Rc::new(Mint::bare(Rewrite::new(
            "comm_add",
            Term::opl("add", vec![Term::var("a"), Term::var("b")]),
            Term::opl("add", vec![Term::var("b"), Term::var("a")]),
        )))];
        StubLegs {
            store: MemStore::with(records),
            codec: GenericCodec,
            base,
            eval: Box::new(|_lhs, _rhs| SiteOutcome::Equal),
            num_true,
        }
    }

    fn evidence(&self) -> Evidence {
        Evidence {
            num_true: self.num_true,
            relation: "new".to_string(),
            matches: 1,
            census_sites: 1,
            fires: 1,
            fires_typed: 1,
            paid: 1,
            closure_safe: true,
            closure_ratio: 1.0,
            ..Evidence::default()
        }
    }
}

impl GauntletLegs for StubLegs {
    fn store(&self) -> &dyn ObjectStore {
        &self.store
    }

    fn codec(&self) -> &dyn ObjectCodec {
        &self.codec
    }

    fn base_rules(&self) -> &[Rc<Mint>] {
        &self.base
    }

    fn measure(
        &mut self,
        _rule: &Rewrite,
        _record: &ObjectRecord,
        _premise_extras: &[Rc<Mint>],
    ) -> Result<Evidence, String> {
        Ok(self.evidence())
    }

    fn guarded_truth(
        &mut self,
        _rule: &Rewrite,
        _limit: usize,
    ) -> (GuardedRegion, GuardedRegion) {
        // Never reached: the stub record's rule is unguarded. The
        // honest empty regions anyway — a guarded record would make
        // the gate read them.
        (GuardedRegion::default(), GuardedRegion::default())
    }

    fn site_eval(&self) -> &SiteEval {
        &*self.eval
    }

    fn stored_certificate(&self, _record: &ObjectRecord) -> Result<Option<usize>, String> {
        Ok(None)
    }
}

#[test]
fn a_clean_object_clears_every_gate() {
    let mut legs = StubLegs::new(
        vec![("k:add_zero".to_string(), add_zero_record())],
        Some(true),
    );
    let report = run_gauntlet(&mut legs, "k:add_zero", &GauntletOptions::default());
    assert!(report.usable, "gauntlet refused: {}", report.reason());
    // The full pipeline ran, in order — not a stubbed-out admit.
    let names: Vec<&str> = report.stages.iter().map(|s| s.name.as_str()).collect();
    assert_eq!(
        names,
        [
            "reconstruct",
            "full-data",
            "measure",
            "truth",
            "novelty",
            "typed-pay",
            "closure",
            "cert"
        ]
    );
    assert!(report.stages.iter().all(|s| s.passed));
    assert_eq!(report.name, "stub_add_zero");
}

#[test]
fn a_measured_counterexample_refuses_at_truth() {
    // The oracle measured lhs ≠ rhs: `num_true = Some(false)` must
    // fail the truth gate even though every other leg is benign —
    // stop-on-failure, so later stages never run.
    let mut legs = StubLegs::new(
        vec![("k:add_zero".to_string(), add_zero_record())],
        Some(false),
    );
    let report = run_gauntlet(&mut legs, "k:add_zero", &GauntletOptions::default());
    assert!(!report.usable);
    let names: Vec<&str> = report.stages.iter().map(|s| s.name.as_str()).collect();
    assert_eq!(names, ["reconstruct", "full-data", "measure", "truth"]);
    assert!(report.stages.last().is_some_and(|s| !s.passed));
    assert!(report.reason().starts_with("truth:"));
}

#[test]
fn an_unknown_key_refuses_at_reconstruct() {
    let mut legs = StubLegs::new(vec![], Some(true));
    let report = run_gauntlet(&mut legs, "k:absent", &GauntletOptions::default());
    assert!(!report.usable);
    assert_eq!(report.stages.len(), 1);
    assert_eq!(report.stages[0].name, "reconstruct");
    assert!(!report.stages[0].passed);
}
