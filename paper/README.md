# Adaptive Green AI — arXiv preprint (scaffold)

Working draft generated from `SOLUTION.md` and the patent inventive-core
(`docs/archimate_patent_core.puml`, mechanisms M1–M6).

## ⚠️ Before you post to arXiv (read this first)

This paper describes an **HPE patent disclosure**. Posting to arXiv (or any
public server) is a **public disclosure** and can affect patent rights:

- **US:** 12-month grace period after first public disclosure to file.
- **EPO / most of the world:** *absolute novelty* — a public post before the
  filing date can bar the patent entirely.

**Do not upload until HPE IP/legal has cleared it and the provisional is filed.**
The clean order is usually: *file provisional → then publish.* This is not legal
advice; confirm with your HPE patent contact.

## Status: scaffold, not a finished paper

Real content (equations, mechanism descriptions, architecture) is filled in from
`SOLUTION.md`. Everything you must supply is marked in the source:

- `\TODO{...}` — required content (author list, evaluation numbers, setup,
  figures, intro prose). Renders in **red**.
- `\NOTE{...}` — guidance/decisions for you. Renders in **purple**.

Grep them:

```bash
grep -n 'TODO\|NOTE' paper/main.tex paper/references.bib
```

Delete the `\TODO`/`\NOTE` macro definitions in `main.tex` (and all uses) before
submission.

### The evaluation is the critical gap

Your internal three-arm benchmark is a **negative result** (always-full is
competitive on carbon and quality — the ladder has no genuinely cheaper capable
rung). `\Cref{sec:eval}` documents two honest framings; pick one before writing
numbers. Do not overclaim.

## Build

```bash
cd paper
latexmk -pdf main.tex          # preferred
# or:
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Figures

`\Cref{fig:arch}` is a placeholder. Render the architecture from PlantUML:

```bash
plantuml -tpdf ../docs/archimate_patent_core.puml   # → docs/*.pdf
# then place/point figs/architecture.pdf and uncomment \includegraphics
```

## Uploading to arXiv (after clearance)

arXiv wants **source**, not just a PDF. Upload:

- `main.tex`, `references.bib`
- the generated **`main.bbl`** (arXiv does not run BibTeX in the default flow —
  include the `.bbl` produced by your local build)
- any figure files under `figs/`

Suggested categories: primary **cs.DC**, cross-list **cs.LG**, **cs.AI**.
Scholar indexes arXiv automatically within days once it's live.
