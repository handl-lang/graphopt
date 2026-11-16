"""SCRATCH spike — mlopt IR term <-> handl ``Term`` JSON feasibility.

Proof of concept for the "tensor programs as handl Terms, e-graph as
engine over them" convergence question.  NOT shipped machinery: this
driver measures round-trip fidelity and the serde boundary cost, and
optionally feeds the emitted JSONL to the scratch Rust probe
(``term_spike_rs``, built from ``/tmp/term_spike_rs`` against
``handl-lang/engine``'s ``hnl-engine`` crate).

Run from the graphopt repo root::

    .venv/bin/python bench/drivers/probes/term_ir_spike.py

Phases:

1. ``fidelity`` — lift a small torch module, encode every reachable
   subterm as a serde-shaped ``Term`` JSON dict, decode back, compare.
   Also attempts every op in ``mlopt_core.opmeta.REGISTRY`` with a
   best-effort attr payload.
2. ``emit`` — write the corpus as JSONL (one serde ``Term`` per line)
   for the Rust probe.
3. ``perf`` — 10k-node random program: time mlopt-side encode +
   ``term_to_data``/``json.dumps`` vs the Rust parse/canonicalize/hash
   numbers the probe reports (invoked here if the binary exists).
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from mlopt_core.ir import (
    Const,
    IR,
    Op,
    Param,
    TensorType,
    Var,
    attr_from_data,
    attr_to_data,
    term_to_data,
)
from mlopt_core.opmeta import REGISTRY

RUST_BIN = "/tmp/term_spike_rs/target/release/term_spike_rs"
OUT_DIR = Path("/tmp/term_ir_spike")

# serde defaults: Term::Op/Arr dim = Const(1); we always emit it so the
# JSON is a faithful Term (the field has a serde default but serializes).
_DIM1 = {"Const": {"value": 1}}
# ``Foreign`` is the host-function boundary: the name is a module-
# registry key — "tensor" matches host-tensor's ``tensor.*`` surface.
_TENSOR = {"Foreign": "tensor"}


def _shape_data(typ: TensorType) -> Any:
    return None if typ.shape is None else list(typ.shape)


def _leaf_record(kind: str, name: str, typ: TensorType) -> dict:
    return {
        "__mlopt_leaf__": kind,
        "name": name,
        "shape": _shape_data(typ),
    }


def enc(t: Any) -> dict:
    """mlopt term -> serde-shaped handl ``Term`` JSON (dict form)."""
    if isinstance(t, Op):
        lit = None
        if t.attrs:
            lit = {
                "__mlopt_attrs__": {
                    k: attr_to_data(v) for k, v in t.attrs.items()
                }
            }
        node = {
            "dim": _DIM1,
            "name": t.op,
            "args": [enc(a) for a in t.args],
            "source": _TENSOR,
        }
        if lit is not None:
            node["lit"] = lit
        return {"Op": node}
    if isinstance(t, Var):
        return {
            "Arr": {
                "dim": _DIM1,
                "value": _leaf_record("var", t.name, t.typ),
            }
        }
    if isinstance(t, Param):
        return {
            "Arr": {
                "dim": _DIM1,
                "value": _leaf_record("param", t.name, t.typ),
            }
        }
    if isinstance(t, Const):
        return {"Const": {"value": t.value}}
    raise TypeError(f"unencodable term: {t!r}")


def dec(j: Any) -> Any:
    """serde-shaped ``Term`` JSON -> mlopt term (inverse of :func:`enc`)."""
    if "Op" in j:
        node = j["Op"]
        attrs = {}
        lit = node.get("lit")
        if lit is not None:
            attrs = {
                k: attr_from_data(v)
                for k, v in lit["__mlopt_attrs__"].items()
            }
        return Op.make(
            node["name"],
            *[dec(a) for a in node["args"]],
            **attrs,
        )
    if "Arr" in j:
        rec = j["Arr"]["value"]
        shape = rec["shape"]
        typ = TensorType(tuple(shape) if shape is not None else None)  # type: ignore[arg-type]
        if rec["__mlopt_leaf__"] == "var":
            return Var(rec["name"], typ)
        return Param(rec["name"], typ)
    if "Const" in j:
        return Const(j["Const"]["value"])
    raise ValueError(f"bad term encoding: {list(j)[:3]}")


def termjson_to_eg(j: Any, eg: Any) -> int:
    """Drive the Python e-graph from serde ``Term`` JSON alone.

    The minimal adapter the spike proposes: post-order walk; ``Op``
    nodes become ``eg.add_enode(name, child_eids, attrs)`` (attrs out
    of the ``lit`` record), ``Arr``/``Const`` leaves become
    ``eg.add_leaf``.  No mlopt ``Op``/``ENode`` objects are built —
    node identity is the e-graph's own ``(op, children, attrs)``
    signature, which is exactly what a ``Term::Op`` carries.
    """
    from mlopt_core.egraph import EGraph  # noqa: F401 — type only

    def walk(node: Any) -> int:
        if "Op" in node:
            op = node["Op"]
            children = tuple(walk(a) for a in op["args"])
            attrs: dict[str, Any] = {}
            lit = op.get("lit")
            if lit is not None:
                attrs = {
                    k: attr_from_data(v)
                    for k, v in lit["__mlopt_attrs__"].items()
                }
            return eg.add_enode(op["name"], children, attrs)
        if "Arr" in node:
            return eg.add_leaf(json.dumps(node["Arr"]["value"], sort_keys=True))
        if "Const" in node:
            return eg.add_leaf(f"const:{node['Const']['value']!r}")
        raise ValueError(f"unsupported term node: {list(node)}")

    return walk(j)


def dag_postorder(root: Any) -> list[Any]:
    """Deduped post-order over the term DAG (Op nodes + leaves)."""
    seen: set[int] = set()
    out: list[Any] = []

    def walk(t: Any) -> None:
        if id(t) in seen:
            return
        seen.add(id(t))
        if isinstance(t, Op):
            for a in t.args:
                walk(a)
        out.append(t)

    walk(root)
    return out


# ---------------------------------------------------------------------------
#  Phase 1 — fidelity on a lifted program
# ---------------------------------------------------------------------------


def lift_tiny_transformer() -> IR:
    """Small real lift: conv + attention-ish + norms, two outputs (pack)."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from kit_torch.torch_bridge import export_to_ir

    class Block(nn.Module):
        def __init__(self, dim: int = 32) -> None:
            super().__init__()
            self.conv = nn.Conv2d(3, 8, 3, padding=1)
            self.lin1 = nn.Linear(dim, dim)
            self.ln = nn.LayerNorm(dim)
            self.lin2 = nn.Linear(dim, dim)
            self.q = nn.Linear(dim, dim)
            self.k = nn.Linear(dim, dim)

        def forward(self, x, t):
            c = F.relu(self.conv(x))  # (B,8,H,W)
            c = c.mean(dim=(2, 3))  # (B,8)
            c = c.unsqueeze(-1).expand(-1, -1, t.shape[-1])
            h = F.gelu(self.lin1(t) + c.transpose(1, 2).reshape(t.shape))
            h = self.ln(h)
            q = self.q(h)
            k = self.k(h)
            a = F.scaled_dot_product_attention(
                q.unsqueeze(1), k.unsqueeze(1), k.unsqueeze(1)
            )
            out = self.lin2(h + a.squeeze(1))
            return out, c.sum()

    torch.manual_seed(0)
    m = Block().eval()
    x = torch.randn(2, 3, 8, 8)
    t = torch.randn(2, 8, 32)
    ir, _values = export_to_ir(m, (x, t))
    return ir


def phase_fidelity() -> dict:
    report: dict[str, Any] = {}
    ir = lift_tiny_transformer()
    nodes = dag_postorder(ir.root)
    ops = [n for n in nodes if isinstance(n, Op)]
    report["lifted"] = {
        "nodes": len(nodes),
        "ops": len(ops),
        "op_names": sorted({o.op for o in ops}),
        "leaves": sorted(
            {type(n).__name__ for n in nodes if not isinstance(n, Op)}
        ),
        "attr_keys": sorted(
            {k for o in ops for k in o.attrs}
        ),
    }

    # whole-program round trip
    j = enc(ir.root)
    failures: list[str] = []
    try:
        back = dec(j)
        ok = back == ir.root
    except Exception as e:  # noqa: BLE001 — spike records, not raises
        ok = False
        failures.append(f"decode: {e!r}")
    report["roundtrip_root_equal"] = ok

    # per-node round trip (catches per-op/attr losses the root eq might
    # hide — actually root eq IS structural, so this is a second opinion
    # counting exactly which nodes fail)
    bad_nodes = []
    for n in nodes:
        try:
            if dec(enc(n)) != n:
                bad_nodes.append(("neq", repr(n)[:120]))
        except Exception as e:  # noqa: BLE001
            bad_nodes.append(("err", f"{e!r} :: {repr(n)[:100]}"))
    report["bad_nodes"] = bad_nodes[:20]
    report["bad_node_count"] = len(bad_nodes)

    # attr stress: every distinct attr *spelling* seen -> JSON-safe?
    attr_failures = []
    for o in ops:
        for k, v in o.attrs.items():
            try:
                attr_to_data(v)
            except TypeError as e:
                attr_failures.append((o.op, k, repr(v), str(e)))
    report["attr_encode_failures"] = attr_failures

    # --- Q2 demo: drive the e-graph from the Term JSON alone -----------
    from mlopt_core.egraph import EGraph

    eg_direct = EGraph()
    eg_direct.add_term(ir.root)
    n_direct = sum(len(c.nodes) for c in eg_direct._classes.values())

    eg_via_term = EGraph()
    root_eid = termjson_to_eg(j, eg_via_term)
    n_via = sum(len(c.nodes) for c in eg_via_term._classes.values())
    report["egraph_adapter"] = {
        "enodes_via_add_term": n_direct,
        "enodes_via_term_json": n_via,
        "root_eid": root_eid,
        "classes": len(eg_via_term._classes),
    }

    return report, ir


# ---------------------------------------------------------------------------
#  Phase 1b — op-vocabulary coverage: can every REGISTRY op mint + round-trip?
# ---------------------------------------------------------------------------

_ATTR_PROBES: dict[str, dict[str, Any]] = {
    # canonical attrs observed in real exports / ATTR_SCHEMA names —
    # best-effort probe values per op family.
    "concat": {"dim": 0},
    "stack": {"dim": 0},
    "unbind": {"dim": 0},
    "chunk": {"chunks": 2, "dim": 0},
    "split": {"sizes": [2, 2], "dim": 0},
    "getitem": {"index": 0},
    "transpose": {"dim0": 0, "dim1": 1},
    "permute": {"dims": [0, 2, 1]},
    "reshape": {"shape": [4]},
    "view": {"shape": [4]},
    "slice": {"dim": 0, "start": 0, "end": 2, "step": 1},
    "narrow": {"dim": 0, "start": 0, "length": 2},
    "select": {"dim": 0, "index": 0},
    "unsqueeze": {"dim": 0},
    "squeeze": {"dim": 0},
    "expand": {"sizes": [2, 4]},
    "repeat": {"repeats": [2, 1]},
    "conv2d": {
        "stride": [1, 1],
        "padding": [0, 0, 0, 0],
        "dilation": [1, 1],
        "groups": 1,
    },
    "linear": {"bias": True},
    "matmul": {},
    "sdpa": {"scale": None, "is_causal": False},
    "softmax": {"dim": -1},
    "log_softmax": {"dim": -1},
    "layer_norm": {"dim": -1, "normalized_shape": [4], "eps": 1e-5, "cudnn_enabled": True},
    "batch_norm": {
        "eps": 1e-5,
        "momentum": 0.1,
        "cudnn_enabled": True,
        "running_stats": False,
    },
    "group_norm": {"num_groups": 1, "eps": 1e-5},
    "max_pool2d": {"kernel_size": [2, 2], "stride": [2, 2]},
    "adaptive_avg_pool2d": {"output_size": [2, 2]},
    "upsample_bilinear2d": {"scale_factor": [2.0, 2.0]},
    "pad": {"pad": [0, 0], "mode": "constant", "value": 0.0},
    "clamp": {"min": 0.0, "max": 1.0},
    "clamp_min": {"min": 0.0},
    "clamp_max": {"max": 1.0},
    "hardtanh": {"min": -1.0, "max": 1.0},
    "leaky_relu": {"negative_slope": 0.01},
    "prelu": {"weight": 0.25},
    "dropout": {"p": 0.5, "train": False},
    "embedding": {},
    "one_hot": {"num_classes": 4},
    "topk": {"k": 2, "dim": -1, "largest": True, "sorted": True},
    "sort": {"dim": -1, "descending": False},
    "argmax": {"dim": -1, "keepdim": False},
    "argmin": {"dim": -1, "keepdim": False},
    "kthvalue": {"k": 1, "dim": -1},
    "var_mean": {"dim": -1, "unbiased": True, "keepdim": False},
    "std_mean": {"dim": -1, "unbiased": True, "keepdim": False},
    "lstm.input": {},
    "einsum": {"equation": "ij,jk->ik"},
    "tril": {"diagonal": 0},
    "triu": {"diagonal": 0},
    "roll": {"shifts": [1], "dims": [0]},
    "flip": {"dims": [0]},
    "cumsum": {"dim": 0},
    "cumprod": {"dim": 0},
    "arange": {"start": 0, "end": 4, "step": 1},
    "full": {"fill_value": 0.0},
    "full_like": {"fill_value": 0.0},
    "new_zeros": {"size": [2, 4]},
    "eye": {"n": 4},
    "to": {"dtype": "float32"},
    "type_as": {},
    "pow": {"exponent": 2.0},
    "addcdiv": {"value": 1.0},
    "addcmul": {"value": 1.0},
    "lerp": {"weight": 0.5},
    "nan_to_num": {"nan": 0.0, "posinf": 0.0, "neginf": 0.0},
    "where": {},
    "masked_fill": {"value": 0.0},
    "index_select": {"dim": 0},
    "index_add": {"dim": 0, "alpha": 1.0},
    "scatter": {"dim": 0, "reduce": "add"},
    "scatter_add": {"dim": 0},
    "scatter_reduce": {"dim": 0, "reduce": "sum"},
    "gather": {"dim": 0},
    "take_along_dim": {"dim": 0},
    "index_put": {"accumulate": False},
    "nonzero": {},
    "isclose": {"rtol": 1e-5, "atol": 1e-8},
    "heaviside": {"value": 0.5},
    "searchsorted": {"right": False},
    "flatten": {"start_dim": 0, "end_dim": -1},
    "unflatten": {"dim": 0, "sizes": [2, 2]},
    "movedim": {"source": [0], "destination": [1]},
    "pixel_shuffle": {"upscale_factor": 2},
    "pixel_unshuffle": {"downscale_factor": 2},
    "glu": {"dim": -1},
    "om": {},
}


def phase_vocab() -> dict:
    """Try minting every REGISTRY op once; round-trip what mints."""
    leaf = Var("x", TensorType((4,)))
    leaf2 = Var("y", TensorType((4,)))
    ok, mint_fail, rt_fail, attr_fail = [], [], [], []
    for name, meta in sorted(REGISTRY.items()):
        arity = meta.arity if meta.arity is not None else 1
        args = tuple(leaf if i % 2 == 0 else leaf2 for i in range(arity))
        attrs = dict(_ATTR_PROBES.get(name, {}))
        try:
            t = Op.make(name, *args, **attrs)
        except Exception as e:  # noqa: BLE001
            mint_fail.append((name, str(e)[:100]))
            continue
        try:
            enc(t)  # may raise on unserializable attr
        except Exception as e:  # noqa: BLE001
            attr_fail.append((name, str(e)[:100]))
            continue
        try:
            back = dec(enc(t))
            if back != t:
                rt_fail.append((name, "neq"))
            else:
                ok.append(name)
        except Exception as e:  # noqa: BLE001
            rt_fail.append((name, f"{type(e).__name__}: {e}"[:100]))
    return {
        "minted_and_roundtripped": len(ok),
        "mint_failures": mint_fail,
        "attr_encode_failures": attr_fail,
        "roundtrip_failures": rt_fail,
        "roundtripped_ops": ok,
    }


# ---------------------------------------------------------------------------
#  Phase 2/3 — emit JSONL + perf
# ---------------------------------------------------------------------------


def gen_program(n_ops: int, seed: int = 7) -> Any:
    """Random n-op *tree* (no sharing — Terms have no DAG refs)."""
    rng = random.Random(seed)
    pool = [
        ("add", {}), ("mul", {}), ("sub", {}), ("div", {"rounding_mode": None}),
        ("matmul", {}), ("relu", {}), ("sigmoid", {}), ("tanh", {}),
        ("gelu", {}), ("sum", {"dim": [0], "keepdim": True}),
        ("mean", {"dim": [1], "keepdim": False}),
        ("reshape", {"shape": [2, 8]}), ("transpose", {"dim0": 0, "dim1": 1}),
        ("concat", {"dim": 0}), ("stack", {"dim": 0}),
        ("softmax", {"dim": -1}), ("layer_norm", {"normalized_shape": [8], "eps": 1e-5}),
        ("linear", {}), ("sdpa", {"is_causal": True}),
        ("conv2d", {"stride": [1, 1], "padding": [1, 1, 1, 1], "dilation": [1, 1], "groups": 1}),
        ("clamp", {"min": -1.0, "max": 1.0}), ("pow", {"exponent": 2.0}),
    ]
    leaves = [
        Var(f"in{i}", TensorType((2, 8))) for i in range(8)
    ] + [
        Param(f"w{i}", TensorType((8, 8))) for i in range(8)
    ]
    counter = {"i": 0}

    def build(depth: int) -> Any:
        if depth <= 0 or counter["i"] >= n_ops:
            return rng.choice(leaves)
        counter["i"] += 1
        op, attrs = rng.choice(pool)
        if op in ("add", "mul", "sub", "div", "concat", "stack", "matmul", "sdpa", "clamp"):
            nargs = 2 if op != "concat" and op != "stack" else rng.choice([2, 3])
        elif op == "linear":
            nargs = 2
        else:
            nargs = 1
        # concat/stack want list-ish? mlopt concat takes variadic tensor args
        args = [build(depth - 1) for _ in range(nargs)]
        try:
            return Op.make(op, *args, **attrs)
        except Exception:  # noqa: BLE001 — fall back to a leaf on schema miss
            return rng.choice(leaves)

    # grow a forest of trees until the op budget is spent, then pack
    # them under one root (pack is the IR's own multi-output pairing).
    roots = []
    while counter["i"] < n_ops:
        roots.append(build(7))
    if len(roots) == 1:
        return roots[0]
    return Op.make("pack", *roots)


def phase_perf(n_ops: int = 10_000) -> dict:
    root = gen_program(n_ops)
    nodes = dag_postorder(root)
    n_actual = sum(1 for n in nodes if isinstance(n, Op))

    # --- mlopt-side costs -------------------------------------------------
    t0 = time.perf_counter()
    td = term_to_data(root)
    py_data_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    blob = json.dumps(td, sort_keys=True)
    py_json_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    h = hash(blob)
    py_hash_s = time.perf_counter() - t0

    # --- boundary encode (mlopt term -> serde Term JSON) ------------------
    t0 = time.perf_counter()
    tj = enc(root)
    enc_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    tj_blob = json.dumps(tj)
    tj_json_s = time.perf_counter() - t0

    # --- Rust side ---------------------------------------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    jsonl = OUT_DIR / "terms_10k.jsonl"
    jsonl.write_text(tj_blob + "\n")
    rust = {}
    if Path(RUST_BIN).exists():
        t0 = time.perf_counter()
        out = subprocess.run(
            [RUST_BIN, str(jsonl)], capture_output=True, text=True, check=True
        )
        rust_wall_s = time.perf_counter() - t0
        rust = json.loads(out.stdout.strip().splitlines()[-1])
        rust["wall_s_incl_python"] = rust_wall_s
    else:
        rust = {"error": f"{RUST_BIN} not built"}

    # --- e-graph insertion baseline ---------------------------------------
    from mlopt_core.egraph import EGraph

    eg = EGraph()
    t0 = time.perf_counter()
    eg.add_term(root)
    eg_add_s = time.perf_counter() - t0

    return {
        "ops_in_program": n_actual,
        "dag_nodes": len(nodes),
        "term_to_data_s": py_data_s,
        "term_to_data_json_s": py_json_s,
        "term_to_data_hash_s": py_hash_s,
        "mlopt_to_termjson_s": enc_s,
        "termjson_dumps_s": tj_json_s,
        "termjson_bytes": len(tj_blob),
        "rust": rust,
        "egraph_add_term_s": eg_add_s,
    }


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    result: dict[str, Any] = {}
    if which in ("all", "fidelity"):
        rep, ir = phase_fidelity()
        result["fidelity"] = rep
        # keep the lifted root for emit
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "lifted_root.json").write_text(
            json.dumps(enc(ir.root))
        )
        lifted_nodes = dag_postorder(ir.root)
        with open(OUT_DIR / "lifted_terms.jsonl", "w") as f:
            for n in lifted_nodes:
                f.write(json.dumps(enc(n)) + "\n")
        print("== fidelity ==")
        print(json.dumps(rep, indent=2, default=str)[:4000])
    if which in ("all", "vocab"):
        v = phase_vocab()
        result["vocab"] = {
            k: v[k]
            for k in (
                "minted_and_roundtripped",
                "roundtrip_failures",
                "attr_encode_failures",
            )
        }
        result["vocab"]["mint_failures"] = v["mint_failures"]
        result["vocab"]["roundtripped_ops"] = v["roundtripped_ops"]
        print("== vocab ==")
        print(
            f"roundtripped {v['minted_and_roundtripped']}/{len(REGISTRY)}"
        )
        print("mint failures:", v["mint_failures"][:15])
        print("rt failures:", v["roundtrip_failures"][:15])
        print("attr failures:", v["attr_encode_failures"][:15])
    if which in ("all", "perf"):
        p = phase_perf()
        result["perf"] = p
        print("== perf ==")
        print(json.dumps(p, indent=2))
    (OUT_DIR / "result.json").write_text(
        json.dumps(result, indent=2, default=str)
    )
    print(f"wrote {OUT_DIR}/result.json")


if __name__ == "__main__":
    main()
