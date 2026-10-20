//! Contraction-ordering primitives — the pure combinatorial core of
//! ``mlopt_torch.contraction_policy``'s board
//! (``pair_cost``/``merge_tensors``/``greedy``/``cost_of_order``).
//!
//! A network is ``(tensors, sizes)``: each tensor is a set of index
//! labels (crossing as any Python iterable of ints, normalised to a
//! sorted dedup'd vector here) and ``sizes`` maps each label to its
//! extent.  A pair contraction merges two tensors: the result's index
//! set is the symmetric difference (shared labels are contracted
//! away) and its scalar-multiply count is ``∏ sizes`` over the
//! *union* — the classic pairwise contraction cost.
//!
//! Semantics mirror the Python functions exactly: the merged tensor
//! is APPENDED to the tensor list (positions matter — a stored order
//! is a list of index pairs into the evolving list), and the
//! randomised greedy arm pre-draws its candidate indices in Python
//! (``picks``) so the ``random.Random`` stream stays on the Python
//! side with identical draw order.

use pyo3::exceptions::PyKeyError;
use pyo3::prelude::*;
use pyo3::types::PyAny;
use std::collections::HashMap;

/// Sorted-set helpers over ``Vec<i64>`` labels.
pub(crate) fn set_from_iter(obj: &Bound<'_, PyAny>) -> PyResult<Vec<i64>> {
    let mut v: Vec<i64> = Vec::new();
    for it in obj.try_iter()? {
        v.push(it?.extract::<i64>()?);
    }
    v.sort_unstable();
    v.dedup();
    Ok(v)
}

pub(crate) fn parse_net(tensors: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<i64>>> {
    let mut ts = Vec::new();
    for t in tensors.try_iter()? {
        ts.push(set_from_iter(&t?)?);
    }
    Ok(ts)
}

pub(crate) fn parse_sizes(sizes: &Bound<'_, PyAny>) -> PyResult<HashMap<i64, f64>> {
    let mut m = HashMap::new();
    for (k, v) in sizes.cast::<pyo3::types::PyDict>()?.iter() {
        m.insert(k.extract::<i64>()?, v.extract::<f64>()?);
    }
    Ok(m)
}

/// Symmetric difference of two sorted sets (the contracted result).
fn set_xor(a: &[i64], b: &[i64]) -> Vec<i64> {
    let (mut i, mut j) = (0usize, 0usize);
    let mut out = Vec::with_capacity(a.len() + b.len());
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => {
                out.push(a[i]);
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                out.push(b[j]);
                j += 1;
            }
            std::cmp::Ordering::Equal => {
                i += 1;
                j += 1;
            }
        }
    }
    out.extend_from_slice(&a[i..]);
    out.extend_from_slice(&b[j..]);
    out
}

/// ``pair_cost`` — ``∏ sizes`` over the union of the two index sets.
pub(crate) fn union_cost(a: &[i64], b: &[i64], sizes: &HashMap<i64, f64>) -> PyResult<f64> {
    let (mut i, mut j) = (0usize, 0usize);
    let mut c = 1.0f64;
    while i < a.len() || j < b.len() {
        let lab = if j >= b.len() || (i < a.len() && a[i] < b[j]) {
            i += 1;
            a[i - 1]
        } else if i >= a.len() || b[j] < a[i] {
            j += 1;
            b[j - 1]
        } else {
            i += 1;
            j += 1;
            a[i - 1]
        };
        c *= *sizes.get(&lab).ok_or_else(|| PyKeyError::new_err(lab))?;
    }
    Ok(c)
}

/// ``merge_tensors`` — drop positions ``a``/``b``, append the merge.
pub(crate) fn merge(ts: &[Vec<i64>], a: usize, b: usize) -> Vec<Vec<i64>> {
    let merged = set_xor(&ts[a], &ts[b]);
    let mut out: Vec<Vec<i64>> = ts
        .iter()
        .enumerate()
        .filter(|(k, _)| *k != a && *k != b)
        .map(|(_, t)| t.clone())
        .collect();
    out.push(merged);
    out
}

/// One greedy round's candidate list, ``(cost, a, b)`` sorted —
/// identical order to Python's ``sorted((pair_cost, a, b) ...)``.
fn candidates(ts: &[Vec<i64>], sizes: &HashMap<i64, f64>) -> PyResult<Vec<(f64, usize, usize)>> {
    let mut cands = Vec::with_capacity(ts.len() * (ts.len() - 1) / 2);
    for a in 0..ts.len() {
        for b in (a + 1)..ts.len() {
            cands.push((union_cost(&ts[a], &ts[b], sizes)?, a, b));
        }
    }
    cands.sort_by(|x, y| {
        (x.0, x.1, x.2)
            .partial_cmp(&(y.0, y.1, y.2))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    Ok(cands)
}

/// ``greedy`` — contract the cheapest pair per round; ``picks`` (one
/// index per round, pre-drawn by the Python wrapper under its own
/// ``random.Random``) selects within the sorted candidate list.
/// Returns ``(total_cost, order)``.
pub fn greedy_order(
    tensors: &Bound<'_, PyAny>,
    sizes: &Bound<'_, PyAny>,
    picks: Option<Vec<usize>>,
) -> PyResult<(Vec<(usize, usize)>, f64)> {
    let mut ts = parse_net(tensors)?;
    let size_map = parse_sizes(sizes)?;
    let mut total = 0.0f64;
    let mut order = Vec::new();
    let mut round = 0usize;
    while ts.len() > 1 {
        let cands = candidates(&ts, &size_map)?;
        let pick = match &picks {
            None => 0,
            Some(p) => *p.get(round).ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err("picks shorter than the contraction rounds")
            })?,
        };
        let Some(&(c, a, b)) = cands.get(pick) else {
            return Err(pyo3::exceptions::PyIndexError::new_err(
                "greedy pick out of range",
            ));
        };
        total += c;
        order.push((a, b));
        ts = merge(&ts, a, b);
        round += 1;
    }
    Ok((order, total))
}

/// ``cost_of_order`` — replay an index-pair order and sum the costs.
pub fn cost_of_order(
    tensors: &Bound<'_, PyAny>,
    sizes: &Bound<'_, PyAny>,
    order: &Bound<'_, PyAny>,
) -> PyResult<f64> {
    let mut ts = parse_net(tensors)?;
    let size_map = parse_sizes(sizes)?;
    let mut total = 0.0f64;
    for pair in order.try_iter()? {
        let p = pair?;
        let a: usize = p.get_item(0)?.extract()?;
        let b: usize = p.get_item(1)?.extract()?;
        total += union_cost(&ts[a], &ts[b], &size_map)?;
        ts = merge(&ts, a, b);
    }
    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn net(v: &[&[i64]]) -> Vec<Vec<i64>> {
        v.iter().map(|t| t.to_vec()).collect()
    }

    #[test]
    fn xor_drops_shared_labels() {
        assert_eq!(set_xor(&[1, 2, 3], &[2, 4]), vec![1, 3, 4]);
        assert_eq!(set_xor(&[1, 2], &[1, 2]), Vec::<i64>::new());
        assert_eq!(set_xor(&[], &[1, 2]), vec![1, 2]);
    }

    #[test]
    fn union_cost_products() {
        let mut sizes = HashMap::new();
        sizes.insert(1, 4.0);
        sizes.insert(2, 8.0);
        sizes.insert(3, 2.0);
        assert_eq!(
            union_cost(&[1, 2], &[2, 3], &sizes).unwrap(),
            4.0 * 8.0 * 2.0
        );
    }

    #[test]
    fn merge_appends_result() {
        let ts = net(&[&[1], &[2], &[3]]);
        let m = merge(&ts, 0, 1);
        assert_eq!(m, net(&[&[3], &[1, 2]]));
    }
}
