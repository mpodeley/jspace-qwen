# LaTeX builds

**Submission (BlackboxNLP / OpenReview): `acl/main.pdf`** — anonymized ([review] mode,
line numbers, links redacted). Built by `acl/build.py` + tectonic.
**Public preprint (site): `acl/main_public.pdf`** — de-anonymized ([final] + author), built
with `acl/build.py --public`; copied to `docs/assets/paper.pdf`.
⚠️ Do not upload `main_public.pdf` to OpenReview.

# LaTeX build — arXiv preprint (single-column, legacy)

`main.tex` is the preprint; the body is generated from `../docs/paper.md` via pandoc and
lives in `body.tex`. Figures are the vector PDFs in `figs/` (produced by
`../scripts/plots.py`).

## Build

Compiles with **pdfLaTeX** (all Unicode mapped to LaTeX macros — no XeLaTeX needed):

```bash
# tectonic (single binary, fetches its own TeX) — what we tested with
tectonic -X compile main.tex
# or a standard TeX Live
pdflatex main.tex && pdflatex main.tex        # twice for refs/toc
# or upload this folder to Overleaf (compiler: pdfLaTeX)
```

## Regenerating the body from the Markdown source

The paper of record is `../docs/paper.md`. To re-sync after editing it:

```bash
pandoc -f gfm -t latex ../docs/paper.md -o body.tex   # (then re-apply the fixes below)
```

Post-conversion fixes baked in (see git history): strip the `\def\LTcaptype{none}`
longtable lines; map Unicode (μ ×  ≫ ⊕ ≠ → · ° ± §) to `\ensuremath{…}`/text macros;
force `width=\linewidth` on the figures; `\setcounter{secnumdepth}{-1}` so the manual
section numbers (4.1, …) are not double-numbered.

## Before submitting to arXiv
- Fill the real title of **Christ et al. 2025** (`[title TK]` in the references).
- Consider a proper `\begin{abstract}` and an author affiliation/email.
