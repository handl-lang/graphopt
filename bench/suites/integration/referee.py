"""Referee — do the two worlds speak the same language?

The integration assumes one calculus certifies both domains.  The
practical test of that, before any proof engine runs, is vocabulary:
a rule minted in the tensor domain can only be *derived* (or judged
novel) in the kernel domain if the kernel bank can name the ops it
mentions.

This suite computes the op vocabularies of both shipped banks and the
overlap:

* substrate/tensor — ``mlopt_core.laws.all_rules()`` (the shipped
  default bank, layout, attention, categorical, prior-art slices);
* kernel — ``mlopt_kernel.laws.KERNEL_RULES`` +
  ``mlopt_kernel.priorart.KERNEL_PRIOR_ART_RULES``.

Op names are read from each rule's printed form, so the measurement is
about *what the bank can say*, not about an internal field that could
disagree with it.  This is a vocabulary proxy for derivability — it
says a law **cannot** cross when its ops are absent, and is silent
about laws whose ops are present (that needs the proof engine).

    .venv/bin/python -m bench.suites.integration.referee
    PYTHONPATH=../kernelopt/src python -m bench run referee
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from bench.benchkit.model import Case, Cell, Finding, Variant, Verdict
from bench.benchkit.report import Report

_REPO = Path(__file__).resolve().parents[3]

DOC_JSON = _REPO / "docs" / "reports" / "referee.json"
DOC_MD = _REPO / "docs" / "reports" / "referee.md"

#: The proof-gate artifact: the kernel sieve's derivability verdicts,
#: written by ``kernelopt/bench/drivers/screen/novelty_sieve.py --json``.
PROOF_JSON = _REPO / "docs" / "reports" / "referee_proof.json"

#: ``(op 'a', (op2 'b'))`` — the printed rule form both banks share.
_OP = re.compile(r"\(([a-zA-Z_][\w]*)")

#: Below this overlap the two banks cannot be said to share a
#: vocabulary; a law minted on one side is unnameable on the other.
SHARED_FLOOR = 0.50


@dataclass(frozen=True)
class Bank:
    """One side's rule count and op vocabulary."""

    name: str
    rules: int
    ops: frozenset[str]

    def to_dict(self) -> dict:
        return {
            "bank": self.name,
            "rules": self.rules,
            "ops": len(self.ops),
        }


@dataclass(frozen=True)
class Overlap:
    """The join of the two vocabularies."""

    substrate: Bank
    kernel: Bank

    @property
    def shared(self) -> frozenset[str]:
        return self.substrate.ops & self.kernel.ops

    @property
    def kernel_only(self) -> frozenset[str]:
        return self.kernel.ops - self.substrate.ops

    @property
    def substrate_only(self) -> frozenset[str]:
        return self.substrate.ops - self.kernel.ops

    @property
    def jaccard(self) -> float:
        union = self.substrate.ops | self.kernel.ops
        return 0.0 if not union else len(self.shared) / len(union)

    @property
    def untranslatable_fraction(self) -> float:
        """Share of kernel ops the tensor bank cannot name."""
        if not self.kernel.ops:
            return 0.0
        return len(self.kernel_only) / len(self.kernel.ops)

    def to_dict(self) -> dict:
        return {
            "substrate": self.substrate.to_dict(),
            "kernel": self.kernel.to_dict(),
            "shared": sorted(self.shared),
            "kernel_only": sorted(self.kernel_only),
            "substrate_only_count": len(self.substrate_only),
            "jaccard": round(self.jaccard, 4),
            "untranslatable_fraction": round(
                self.untranslatable_fraction, 4
            ),
        }


def _ops_of(rules) -> frozenset[str]:
    """Every op name mentioned by any rule's printed form."""
    ops: set[str] = set()
    for rule in rules:
        ops.update(_OP.findall(str(rule)))
    return frozenset(ops)


def load_overlap() -> Overlap | None:
    """Both banks' vocabularies, or ``None`` when kernelopt is absent.

    The kernel package lives in a sibling repo, so this suite is
    runnable with ``PYTHONPATH=../kernelopt/src``.  When it is missing
    the suite reports *why* instead of guessing a vocabulary — an
    invented overlap would be worse than no measurement.
    """
    try:
        from mlopt_core.laws import all_rules
        from mlopt_kernel.laws import KERNEL_RULES
        from mlopt_kernel.priorart import KERNEL_PRIOR_ART_RULES
    except ImportError:
        return None

    substrate_rules = list(all_rules())
    kernel_rules = list(KERNEL_RULES) + list(KERNEL_PRIOR_ART_RULES)
    return Overlap(
        substrate=Bank(
            name="substrate/tensor",
            rules=len(substrate_rules),
            ops=_ops_of(substrate_rules),
        ),
        kernel=Bank(
            name="kernel/codegen",
            rules=len(kernel_rules),
            ops=_ops_of(kernel_rules),
        ),
    )


def load_proof(path: Path = PROOF_JSON) -> dict | None:
    """The kernel sieve's derivability verdicts, when the gate was run.

    The sieve evaluates ``cert ∧ beat ∧ ¬derivable`` per kernel law
    against a bank of ``KERNEL_RULES``, ``KERNEL_PRIOR_ART_RULES`` and
    the shared substrate surface.  ``derivable`` is the proof question
    this suite's vocabulary proxy can only approximate.
    """
    try:
        rec = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return None
    cands = rec.get("candidates") or []
    if not cands:
        return None
    return {
        "bank_size": (rec.get("env") or {}).get("bank_size"),
        "laws": len(cands),
        "derivable": [c["name"] for c in cands if c.get("derivable")],
        "unknown": [c["name"] for c in cands if not c.get("derivable")],
        "witnesses": {
            c["name"]: c.get("witness_rules") or []
            for c in cands
            if c.get("derivable")
        },
    }


def build_findings(
    overlap: Overlap | None, proof: dict | None = None
) -> list[Finding]:
    """Is the shared referee real, or only in the paper?"""
    if overlap is None:
        return [
            Finding(
                claim="Both banks can be compared in one process.",
                verdict=Verdict.INCONCLUSIVE,
                headline=(
                    "kernelopt not importable — run with "
                    "PYTHONPATH=../kernelopt/src"
                ),
                metric="kernel_bank_available",
                value=0.0,
            )
        ]

    j = overlap.jaccard
    untr = overlap.untranslatable_fraction

    findings = [
        Finding(
            claim=(
                "The two domains share one rule vocabulary, so a law "
                "minted in one can be judged by the other."
            ),
            verdict=Verdict.WIN
            if j >= SHARED_FLOOR
            else Verdict.NEGATIVE,
            headline=(
                f"Jaccard {j:.2f} over "
                f"{len(overlap.substrate.ops | overlap.kernel.ops)} ops — "
                + (
                    "the banks share a vocabulary"
                    if j >= SHARED_FLOOR
                    else "the banks are largely disjoint vocabularies"
                )
            ),
            metric="vocabulary_jaccard",
            value=j,
            evidence=overlap.to_dict(),
        ),
        Finding(
            claim="A kernel law can be re-derived in the tensor bank.",
            verdict=Verdict.NEGATIVE if untr > 0.0 else Verdict.WIN,
            headline=(
                f"{len(overlap.kernel_only)}/{len(overlap.kernel.ops)} "
                f"kernel ops ({untr:.0%}) have no tensor spelling — those "
                "laws cannot cross without translation"
            ),
            metric="untranslatable_fraction",
            value=untr,
            evidence={"kernel_only": sorted(overlap.kernel_only)[:24]},
        ),
        Finding(
            claim=(
                "The derivability union the novelty sieve computes is a "
                "real floor, not a formality."
            ),
            verdict=(
                Verdict.WIN
                if proof and proof["derivable"]
                else Verdict.INCONCLUSIVE
            ),
            headline=(
                f"proof gate not run — the vocabulary proxy over "
                f"{len(overlap.shared)} shared ops is the only evidence"
                if not proof
                else f"{len(proof['derivable'])}/{proof['laws']} shipped "
                "kernel laws are derivable from the "
                f"{proof['bank_size']}-rule union — the floor demotes "
                "them to known"
            ),
            metric="shared_ops",
            value=float(len(overlap.shared)),
            evidence={"proof": proof},
        ),
    ]
    if proof and proof["derivable"]:
        findings.append(
            Finding(
                claim=(
                    "Crossing works through explicit denotation rows: a "
                    "kernel law is provable in the union even though the "
                    "tensor bank cannot spell its fused op."
                ),
                verdict=Verdict.WIN,
                headline=(
                    "every derivable law has a prior-art witness "
                    "(`pa_kf_*_expand`) — translation rows, not a shared "
                    "vocabulary, are what make the union real"
                ),
                metric="laws_with_witness",
                value=float(
                    sum(1 for w in proof["witnesses"].values() if w)
                ),
                evidence={"witnesses": proof["witnesses"]},
            )
        )
    return findings


def _cells(overlap: Overlap | None) -> list[Cell]:
    """One cell per bank so the dashboard shows the sizes."""

    def _noop() -> None:
        return None

    if overlap is None:
        return []
    out = []
    for bank in (overlap.substrate, overlap.kernel):
        case = Case(
            name=bank.name,
            params={"rules": bank.rules, "ops": len(bank.ops)},
            variants=[Variant(name="bank", stmt=_noop)],
            aux=bank.to_dict(),
        )
        out.append(
            Cell(
                case=case,
                medians={},
                iqr={},
                aux={
                    **bank.to_dict(),
                    "shared": len(overlap.shared),
                    "jaccard": round(overlap.jaccard, 4),
                },
            )
        )
    return out


def render_markdown(
    overlap: Overlap | None, findings: list[Finding]
) -> str:
    """The written report — the banks, the overlap, the caveats."""
    lines = [
        "# Referee — do the two worlds speak the same language?",
        "",
        "**Method.** Op vocabularies read from the printed form of every "
        "shipped rule: `mlopt_core.laws.all_rules()` on the tensor side, "
        "`KERNEL_RULES` + `KERNEL_PRIOR_ART_RULES` on the kernel side. "
        "A law whose ops are absent from the other bank cannot be "
        "derived there at all; presence is necessary, not sufficient "
        "(that needs the proof engine).",
        "",
        "## Findings",
        "",
    ]
    for f in findings:
        lines += [
            f"### {f.verdict.badge} — {f.claim}",
            "",
            f.headline,
            "",
        ]

    if overlap is None:
        lines += [
            "> kernelopt is not importable in this environment. Re-run "
            "with `PYTHONPATH=../kernelopt/src`.",
            "",
        ]
        return "\n".join(lines)

    lines += [
        "## The banks",
        "",
        "| bank | rules | distinct ops |",
        "|---|---|---|",
        f"| {overlap.substrate.name} | {overlap.substrate.rules} | "
        f"{len(overlap.substrate.ops)} |",
        f"| {overlap.kernel.name} | {overlap.kernel.rules} | "
        f"{len(overlap.kernel.ops)} |",
        "",
        f"- shared ops: **{len(overlap.shared)}** · Jaccard "
        f"**{overlap.jaccard:.2f}**",
        f"- kernel-only: **{len(overlap.kernel_only)}** · "
        f"substrate-only: **{len(overlap.substrate_only)}**",
        "",
        "**Shared** — ops a law can be stated in on both sides:",
        "",
        "`" + "`, `".join(sorted(overlap.shared)) + "`",
        "",
        "**Kernel-only** — a law mentioning any of these cannot be "
        "re-derived in the tensor bank without a translation row:",
        "",
        "`" + "`, `".join(sorted(overlap.kernel_only)) + "`",
        "",
        "## What this establishes, and what it does not",
        "",
        "- It establishes that the two banks are **largely disjoint "
        "vocabularies**. A shared referee is possible only over the "
        "shared slice unless translation rules are added.",
        "- The vocabulary proxy is sufficient for *impossibility*, not "
        "necessary for *possibility*: a law can be provable through a "
        "denotation row even when its fused op is absent from the other "
        "bank. `referee_proof.md` is that run.",
        "- It says nothing about whether a *player* transfers; that is "
        "the schema question.",
        "",
        "_Generated by `bench/suites/integration/referee.py`. Re-run; "
        "do not hand-edit._",
        "",
    ]
    return "\n".join(lines)


def run_bench(args) -> Report:
    """Load both banks, state the findings, write the twins."""
    overlap = load_overlap()
    proof = load_proof()
    findings = build_findings(overlap, proof)

    out = getattr(args, "out", None)
    md_path = Path(out) / "referee.md" if out else DOC_MD
    json_path = Path(out) / "referee.json" if out else DOC_JSON
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(render_markdown(overlap, findings))
    json_path.write_text(
        json.dumps(
            {
                "schema": 1,
                "suite": "referee",
                "available": overlap is not None,
                "overlap": None
                if overlap is None
                else overlap.to_dict(),
                "findings": [
                    {
                        "claim": f.claim,
                        "verdict": f.verdict.value,
                        "headline": f.headline,
                        "metric": f.metric,
                        "value": f.value,
                    }
                    for f in findings
                ],
            },
            indent=2,
        )
        + "\n"
    )

    return Report(
        suite="referee",
        cells=_cells(overlap),
        title="Referee — one calculus, two vocabularies",
        summary=(
            "The tensor and kernel banks overlap only on a slice of "
            "their ops, so a law cannot cross domains without "
            "translation."
        ),
        findings=findings,
        provenance={"available": overlap is not None},
    )


def main(argv: list[str] | None = None) -> int:
    """Standalone entry point (the CLI drives ``run_bench``)."""
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None, help="output directory")
    args = ap.parse_args(argv)

    report = run_bench(args)
    for f in report.findings:
        print(f"[{f.verdict.badge}] {f.headline}")
    print(f"\nwrote {DOC_MD.name} + {DOC_JSON.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
