# paper/

**Draft — proposed, not accepted.**  No human revision or external
review yet.

- `paper.qmd` — the quarto source
- `references.bib` — bibliography
- `paper.pdf` — the rendered draft

Rebuild: `quarto render paper.qmd --to pdf` (quarto + a LaTeX engine;
TinyTeX works).

Every quantitative claim cites the probe, test, or retro that produced
it — the evidence ledger is [`../bench/RESULTS.md`](../bench/RESULTS.md)
and the extended working draft (full win/loss tables, evidence table)
is [`../docs/paper_draft.md`](../docs/paper_draft.md).
