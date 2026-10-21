# paper/

**Draft — proposed, not accepted.**  No human revision or external
review yet.

- `paper.qmd` — the quarto source
- `references.bib` — bibliography
- `paper.pdf` — the rendered draft

Rebuild: `quarto render paper.qmd --to pdf` (quarto + a LaTeX engine;
TinyTeX works).

## Convention

A repo that publishes a paper keeps it in a top-level `paper/`
directory: `paper.qmd` is the source of truth, `references.bib` and
any assets live beside it, and the committed `paper.pdf` is the
rendered output.  CI rebuilds the pdf from the qmd —
`handl-lang/docs/ci/paper-build.sh`, same command — and pushes the
result back to main when it changes, so a qmd edit never leaves a
stale pdf.

## This paper

Every quantitative claim cites the probe, test, or retro that produced
it — the evidence ledger is [`../bench/RESULTS.md`](../bench/RESULTS.md)
and the extended working draft (full win/loss tables, evidence table)
is [`../docs/paper_draft.md`](../docs/paper_draft.md).
