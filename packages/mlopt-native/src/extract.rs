//! Native extraction: the cheapest-term DP over the e-graph plus the
//! ``dag_cost`` true-DAG cost walk — a faithful port of
//! ``mlopt_core.egraph.extract._ExtractMixin.extract_best`` and
//! ``mlopt_core.cost.params.dag_cost``.
//!
//! The cost model itself stays Python: ``extract`` calls the injected
//! ``pricer`` once per candidate term — ``("leaf", key)`` /
//! ``("op", name, (child term ids), attrs)`` — and the adapter
//! materialises the interned ``Op``/leaf objects, runs ``cfn`` and the
//! same ``eo``/``rc`` additive-memo pre-seeding the reference
//! extractor does, and hands back ``(term_id, cost, flags)``.  Only
//! the *walk* is native: member ordering, cycle skipping, the
//! per-class ``(total, nops)`` tie-break, the bill-each-e-class-once
//! DAG accounting and the param-only discount — all bit-parity with
//! the Python code.
//!
//! Determinism note: the billing walk accumulates ``sub_cost`` over a
//! child's new classes in *ascending e-class id* order; Python iterates
//! the set difference in its own (deterministic but layout-dependent)
//! order.  Sums of the same floats can differ by an ulp either way —
//! the result is deterministic on both engines.

use crate::attr::AttrVal;
use crate::graph::{ClassId, ENode, NativeCore};
use crate::pyrepr::repr_attrs;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyString, PyTuple};
use std::collections::{BTreeSet, HashMap, HashSet};
use std::rc::Rc;

/// One extraction cache entry — mirrors the Python tuple
/// ``(total, term, used, param_only, nops, cfn(term))``; ``term`` is an
/// index into the Python pricer's term table.
pub struct Entry {
    pub total: f64,
    pub term_id: Option<usize>,
    pub used: Rc<BTreeSet<ClassId>>,
    pub param_only: bool,
    pub nops: u64,
    pub ct: f64,
}

/// A kept candidate under an armed fusion band
/// ``(total, term_id, used, local, ct, param_only, nops)``.
struct Cand {
    total: f64,
    term_id: usize,
    used: BTreeSet<ClassId>,
    local: f64,
    ct: f64,
    param_only: bool,
    nops: u64,
}

/// The per-call extraction environment.
pub struct ExtractCtx {
    pub cache: HashMap<ClassId, Rc<Entry>>,
    /// ``eid -> billable local`` — the param-only discount zeroes it.
    pub adj_of: HashMap<ClassId, f64>,
    pub in_progress: HashSet<ClassId>,
    pub bill_params: bool,
    pub foldable: HashSet<String>,
    pub fusion_epsilon: f64,
    pub fuser: Option<Py<PyAny>>,
    pub overrides: HashMap<ClassId, ENode>,
    pub bans: HashMap<ClassId, HashSet<ENode>>,
}

impl ExtractCtx {
    pub fn new(
        bill_params: bool,
        foldable: HashSet<String>,
        fusion_epsilon: f64,
        fuser: Option<Py<PyAny>>,
        overrides: HashMap<ClassId, ENode>,
        bans: HashMap<ClassId, HashSet<ENode>>,
    ) -> Self {
        ExtractCtx {
            cache: HashMap::new(),
            adj_of: HashMap::new(),
            in_progress: HashSet::new(),
            bill_params,
            foldable,
            fusion_epsilon,
            fuser,
            overrides,
            bans,
        }
    }
}

/// ``(eid, total, term_id, used, param_only, nops, cfn(term))`` per
/// cached class — what the Python adapter needs to reconstruct
/// ``extract_best``'s ``_cache_out`` view.
pub type CacheDump = Vec<(ClassId, f64, Option<usize>, Vec<ClassId>, bool, u64, f64)>;

impl NativeCore {
    /// ``pricer(tag, payload, child_ids, attrs) -> (term_id, cost,
    /// flags)`` — the adapter's per-candidate pricing call.  ``tag``
    /// is ``"leaf"`` (payload = leaf key, flags bit0 = is_var) or
    /// ``"op"`` (payload = op name, children = term ids).
    fn price_leaf(
        &mut self,
        py: Python<'_>,
        pricer: &Bound<'_, PyAny>,
        key: &AttrVal,
    ) -> PyResult<(usize, f64, i64)> {
        let key_obj = key.to_py(py)?;
        let args = (
            PyString::new(py, "leaf"),
            key_obj.bind(py),
            PyTuple::empty(py).into_any(),
            PyTuple::empty(py).into_any(),
        );
        let r = pricer.call1(args)?;
        let (t, c, f): (i64, f64, i64) = r.extract()?;
        Ok((t as usize, c, f))
    }

    fn price_op(
        &mut self,
        py: Python<'_>,
        pricer: &Bound<'_, PyAny>,
        node: &ENode,
        child_ids: &[usize],
    ) -> PyResult<(usize, f64)> {
        let kids = PyTuple::new(py, child_ids.iter().copied())?;
        let pairs: Vec<Py<PyAny>> = node
            .attrs
            .iter()
            .map(|(k, v)| -> PyResult<Py<PyAny>> {
                Ok(PyTuple::new(py, [py_str(py, k), v.to_py(py)?])?
                    .unbind()
                    .into())
            })
            .collect::<PyResult<_>>()?;
        let args = (
            PyString::new(py, "op"),
            PyString::new(py, &node.op),
            kids.unbind().into_any(),
            PyTuple::new(py, pairs)?.unbind().into_any(),
        );
        let r = pricer.call1(args)?;
        let (t, c, _f): (i64, f64, i64) = r.extract()?;
        Ok((t as usize, c))
    }

    /// The canonical member order — ``sorted(nodes, key=_member_key)``
    /// with ``_mk = (op, children, repr(attrs))``.
    fn member_order(nodes: &[ENode]) -> Vec<ENode> {
        if nodes.len() <= 1 {
            return nodes.to_vec();
        }
        let mut keyed: Vec<(String, Vec<ClassId>, String, &ENode)> = nodes
            .iter()
            .map(|n| (n.op.clone(), n.children.clone(), repr_attrs(&n.attrs), n))
            .collect();
        keyed.sort_by(|a, b| (&a.0, &a.1, &a.2).cmp(&(&b.0, &b.1, &b.2)));
        keyed.into_iter().map(|(_, _, _, n)| n.clone()).collect()
    }

    /// ``best(eid)`` — the per-class DP of ``extract_best``.
    fn xbest(
        &mut self,
        py: Python<'_>,
        pricer: &Bound<'_, PyAny>,
        ctx: &mut ExtractCtx,
        cid0: ClassId,
    ) -> PyResult<Rc<Entry>> {
        let cid = self.find(cid0);
        if let Some(e) = ctx.cache.get(&cid) {
            return Ok(e.clone());
        }
        if ctx.in_progress.contains(&cid) {
            // Cycle back to an ancestor — not extractable (and, as in
            // Python, NOT cached: the transient failure may resolve
            // once the ancestor's walk completes).
            return Ok(Rc::new(Entry {
                total: f64::INFINITY,
                term_id: None,
                used: Rc::new(BTreeSet::new()),
                param_only: false,
                nops: 0,
                ct: 0.0,
            }));
        }
        ctx.in_progress.insert(cid);
        let members: Vec<ENode> = match ctx.overrides.get(&cid) {
            Some(ov) => vec![ov.clone()],
            None => {
                let raw: Vec<ENode> = self.classes[&cid].nodes.iter().cloned().collect();
                Self::member_order(&raw)
            }
        };
        // Cloned: the recursive child walk below needs ``&mut ctx``.
        let banned = ctx.bans.get(&cid).cloned();

        let mut best_total: Option<f64> = None;
        let mut best_term: Option<usize> = None;
        let mut best_used: Rc<BTreeSet<ClassId>> = Rc::new(BTreeSet::from([cid]));
        let mut best_local = 0.0f64;
        let mut best_ct = 0.0f64;
        let mut best_param_only = false;
        let mut best_nops = 0u64;
        let mut cands: Vec<Cand> = Vec::new();

        for node in &members {
            if let Some(b) = &banned {
                if b.contains(node) {
                    continue;
                }
            }
            if node.op == "leaf" {
                let key = node
                    .attrs
                    .first()
                    .map(|p| p.1.clone())
                    .unwrap_or_else(|| AttrVal::Str("??".to_string()));
                let (tid, total, flags) = self.price_leaf(py, pricer, &key)?;
                let po = flags & 1 == 0; // not a Var
                if ctx.fusion_epsilon != 0.0 {
                    cands.push(Cand {
                        total,
                        term_id: tid,
                        used: BTreeSet::from([cid]),
                        local: total,
                        ct: total,
                        param_only: po,
                        nops: 0,
                    });
                }
                if best_total.is_none() || total < best_total.unwrap() {
                    best_total = Some(total);
                    best_term = Some(tid);
                    best_used = Rc::new(BTreeSet::from([cid]));
                    best_local = total;
                    best_ct = total;
                    best_param_only = po;
                    best_nops = 0;
                }
                continue;
            }

            let mut child_tids: Vec<usize> = Vec::with_capacity(node.children.len());
            let mut child_tc = 0.0f64;
            let mut child_nops = 0u64;
            let mut used: BTreeSet<ClassId> = BTreeSet::from([cid]);
            let mut sub_cost = 0.0f64;
            let mut param_only = true;
            let mut valid = true;
            for &ceid in &node.children {
                let entry = match ctx.cache.get(&ceid) {
                    Some(e) => e.clone(),
                    None => {
                        let canon = self.find(ceid);
                        if canon == cid {
                            valid = false; // direct self-reference
                            break;
                        }
                        self.xbest(py, pricer, ctx, canon)?
                    }
                };
                let Some(ctid) = entry.term_id else {
                    valid = false;
                    break;
                };
                child_tids.push(ctid);
                child_tc += entry.ct;
                child_nops += entry.nops;
                param_only = param_only && entry.param_only;
                // Bill each distinct e-class once — the classes the
                // child's subtree introduced that this candidate has
                // not seen yet.
                for &u in entry.used.iter() {
                    if !used.contains(&u) {
                        sub_cost += ctx.adj_of.get(&u).copied().unwrap_or(0.0);
                        used.insert(u);
                    }
                }
            }
            if !valid {
                continue;
            }
            let (tid, ct) = self.price_op(py, pricer, node, &child_tids)?;
            // This node's own op must be foldable too — a param-only
            // subtree over trace/inv still runs its solver call.
            param_only = param_only && ctx.foldable.contains(&node.op);
            let mut local = ct - child_tc;
            if local < 0.0 {
                local = 0.0;
            }
            if param_only && !ctx.bill_params {
                local = 0.0; // whole subtree folds at compile time
            }
            let total = local + sub_cost;
            let nops = child_nops + 1;
            if ctx.fusion_epsilon != 0.0 {
                cands.push(Cand {
                    total,
                    term_id: tid,
                    used: used.clone(),
                    local,
                    ct,
                    param_only,
                    nops,
                });
            }
            let bt = best_total;
            if bt.is_none() || total < bt.unwrap() || (total == bt.unwrap() && nops < best_nops) {
                best_total = Some(total);
                best_term = Some(tid);
                best_used = Rc::new(used);
                best_local = local;
                best_ct = ct;
                best_param_only = param_only;
                best_nops = nops;
            }
        }

        // Fusion-preferred near-tie: members priced within
        // ``fusion_epsilon`` (relative) of the class minimum compete
        // on the Python-side fusion key — ``(*fuser(term), nops)`` —
        // stable in scan order, exactly like the reference extractor.
        if ctx.fusion_epsilon != 0.0 && cands.len() > 1 {
            let floor = cands.iter().map(|c| c.total).fold(f64::INFINITY, f64::min);
            let band_top = floor + ctx.fusion_epsilon * floor.abs();
            let band: Vec<&Cand> = cands.iter().filter(|c| c.total <= band_top).collect();
            if band.len() > 1 {
                let fuser = ctx.fuser.as_ref().unwrap();
                let mut win: Option<((i64, i64, u64), &Cand)> = None;
                for c in band {
                    let fk: (i64, i64) = fuser.call1(py, (c.term_id,))?.extract(py)?;
                    let key = (fk.0, fk.1, c.nops);
                    if win.is_none_or(|(wk, _)| key < wk) {
                        win = Some((key, c));
                    }
                }
                if let Some((_, c)) = win {
                    best_total = Some(c.total);
                    best_term = Some(c.term_id);
                    best_used = Rc::new(c.used.clone());
                    best_local = c.local;
                    best_ct = c.ct;
                    best_param_only = c.param_only;
                    best_nops = c.nops;
                }
            }
        }

        ctx.in_progress.remove(&cid);
        let mut bt = best_total.unwrap_or(f64::INFINITY);
        if best_term.is_none() {
            bt = f64::INFINITY;
        }
        ctx.adj_of.insert(
            cid,
            if ctx.bill_params || !best_param_only {
                best_local
            } else {
                0.0
            },
        );
        let entry = Rc::new(Entry {
            total: if bt == 0.0 { 0.0 } else { bt },
            term_id: best_term,
            used: best_used,
            param_only: best_param_only,
            nops: best_nops,
            ct: best_ct,
        });
        ctx.cache.insert(cid, entry.clone());
        Ok(entry)
    }

    /// ``extract_best`` over the native graph — returns
    /// ``(total, term_id, nops, cache_dump)``; the Python adapter maps
    /// ``term_id`` to the materialised term and reconstructs
    /// ``_cache_out`` from the dump.
    #[allow(clippy::too_many_arguments)]
    pub fn extract(
        &mut self,
        py: Python<'_>,
        root: ClassId,
        pricer: &Bound<'_, PyAny>,
        fuser: Option<Py<PyAny>>,
        bill_params: bool,
        foldable: HashSet<String>,
        fusion_epsilon: f64,
        overrides: HashMap<ClassId, ENode>,
        bans: HashMap<ClassId, HashSet<ENode>>,
    ) -> PyResult<(f64, Option<usize>, u64, CacheDump)> {
        let mut ctx = ExtractCtx::new(
            bill_params,
            foldable,
            fusion_epsilon,
            fuser,
            overrides,
            bans,
        );
        let root_entry = self.xbest(py, pricer, &mut ctx, root)?;
        let mut dump: CacheDump = ctx
            .cache
            .iter()
            .map(|(&eid, e)| {
                (
                    eid,
                    e.total,
                    e.term_id,
                    e.used.iter().copied().collect(),
                    e.param_only,
                    e.nops,
                    e.ct,
                )
            })
            .collect();
        dump.sort_by_key(|e| e.0);
        Ok((root_entry.total, root_entry.term_id, root_entry.nops, dump))
    }
}

fn py_str(py: Python<'_>, s: &str) -> Py<PyAny> {
    PyString::new(py, s).unbind().into()
}

// ---------------------------------------------------------------------------
//  dag_cost — the true-DAG cost walk over a serialized term
// ---------------------------------------------------------------------------

/// One flat term node for ``dag_cost`` — ``("leaf", key, kind)`` /
/// ``("op", name, (child idxs), attrs)`` post-order, where ``kind``
/// carries the leaf-classification bits the Python walk reads via
/// ``isinstance`` (bit0 Var, bit1 Param, bit2 Const).
pub enum DNode {
    Leaf { kind: u8 },
    Op { op: String, children: Vec<usize> },
}

/// ``dag_cost`` over a flat serialized term — mirrors
/// ``mlopt_core.cost.params.dag_cost``: each distinct node bills
/// ``c(t) - sum(c(children))`` once, param-only foldable subtrees
/// (``_folds_to_param`` under ``source_tensors=None``) skip entirely
/// unless ``bill_params`` (``charges_param_only``), and the fold
/// decision + ``has_var_leaf`` are computed bottom-up here instead of
/// by memoized Python recursion.  ``pricer(i) -> float`` prices the
/// i-th flat node's *original* Python term — the same ``c(t)`` calls
/// the reference walk issues, over the same term objects.
pub fn dag_cost_flat(
    nodes: &[DNode],
    pricer: &Bound<'_, PyAny>,
    bill_params: bool,
    foldable: &HashSet<String>,
) -> PyResult<f64> {
    let n = nodes.len();
    // has_var[i]: subtree reads a Var leaf.
    let mut has_var = vec![false; n];
    // folds[i]: param-only AND foldable under the lowerer's rules.
    let mut folds = vec![false; n];
    for (i, nd) in nodes.iter().enumerate() {
        match nd {
            DNode::Leaf { kind } => {
                has_var[i] = kind & 1 != 0;
            }
            DNode::Op { op, children } => {
                has_var[i] = children.iter().any(|&c| has_var[c]);
                if has_var[i] {
                    folds[i] = false;
                    continue;
                }
                let to_param = |a: usize, allow_const: bool| -> bool {
                    match &nodes[a] {
                        DNode::Leaf { kind } => {
                            if kind & 2 != 0 {
                                true // Param — resolves (source_tensors=None)
                            } else if kind & 4 != 0 {
                                allow_const // Const
                            } else {
                                false
                            }
                        }
                        DNode::Op { .. } => folds[a],
                    }
                };
                folds[i] = if (op == "matmul" || op == "concat") && children.len() >= 2 {
                    children.iter().all(|&a| to_param(a, false))
                } else if foldable.contains(op) {
                    children.iter().all(|&a| to_param(a, true))
                } else {
                    false
                };
            }
        }
    }
    // Reachability: root is last (post-order); a skipped (param-only
    // foldable) node does not propagate to its children — but IS
    // still priced when its parent bills it (``sum(c(a))``), so every
    // reached node gets exactly one pricer call, same as the Python
    // walk's ``c(t)``/``c(a)`` invocations.
    let mut reached = vec![false; n];
    if n > 0 {
        reached[n - 1] = true;
    }
    for i in (0..n).rev() {
        if !reached[i] {
            continue;
        }
        if let DNode::Op { children, .. } = &nodes[i] {
            let skipped = !has_var[i] && !bill_params && folds[i];
            if !skipped {
                for &c in children {
                    reached[c] = true;
                }
            }
        }
    }
    let mut cost = vec![0.0f64; n];
    for i in 0..n {
        if reached[i] {
            cost[i] = pricer.call1((i,))?.extract::<f64>()?;
        }
    }
    let mut total = 0.0f64;
    for (i, nd) in nodes.iter().enumerate() {
        if !reached[i] {
            continue;
        }
        match nd {
            DNode::Leaf { .. } => total += cost[i],
            DNode::Op { children, .. } => {
                let skipped = !has_var[i] && !bill_params && folds[i];
                if skipped {
                    continue;
                }
                let mut local = cost[i];
                for &c in children {
                    local -= cost[c];
                }
                if local < 0.0 {
                    local = 0.0;
                }
                total += local;
            }
        }
    }
    Ok(total)
}
