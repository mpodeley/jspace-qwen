#!/usr/bin/env python
"""Build the anonymized ACL (BlackboxNLP) submission body from docs/paper.md.

Pipeline: pandoc gfm->latex, then post-process for two-column ACL:
  - strip the docs-only header (draft note + subtitle) and ALL identifying links
    (site, repo) -- double-blind;
  - extract the Abstract section into abstract.tex (main.tex wraps it);
  - demote manual section numbers ("1. Introduction" -> "Introduction"; ACL
    renumbers identically, so in-text \\S-references stay correct);
  - longtable -> table/tabular (longtable breaks in two-column mode);
  - figures -> figure* environments at \\textwidth with the markdown alt text as
    the caption (the italic caption paragraphs in the md are dropped);
  - the same unicode->macro map as paper/main.tex.

Usage:  .venv/bin/python paper/acl/build.py   (from the repo root or anywhere)
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ACL = ROOT / "paper" / "acl"
MD = ROOT / "docs" / "paper.md"

UNI = {
    # multi-char sequences first (dict order is application order)
    "10⁻⁸": r"\ensuremath{10^{-8}}",
    "²": r"\ensuremath{^{2}}", "Σ": r"\ensuremath{\Sigma}",
    "−": "-", "–": "--", "—": "---", "×": r"\ensuremath{\times}",
    "Δ": r"\ensuremath{\Delta}", "∈": r"\ensuremath{\in}",
    "≡": r"\ensuremath{\equiv}",
    "≈": r"\ensuremath{\approx}", "≠": r"\ensuremath{\neq}",
    "≫": r"\ensuremath{\gg}", "≪": r"\ensuremath{\ll}",
    "⊕": r"\ensuremath{\oplus}", "→": r"\ensuremath{\to}",
    "≤": r"\ensuremath{\leq}", "≥": r"\ensuremath{\geq}",
    "μ": r"\ensuremath{\mu}", "α": r"\ensuremath{\alpha}",
    "ℓ": r"\ensuremath{\ell}", "°": r"\textdegree{}",
    "·": r"\textperiodcentered{}", "±": r"\ensuremath{\pm}",
    "§": r"\S{}", "↔": r"\ensuremath{\leftrightarrow}", "○": r"$\circ$",
    "‖": r"\ensuremath{\,\Vert\,}",
    "“": "``", "”": "''", "‘": "`", "’": "'", "…": r"\ldots{}",
}


# prose citation -> natbib. \citet for narrative mentions, \citealp for mentions
# already inside prose parentheses. Patterns are whitespace-tolerant (pandoc wraps
# lines) and match the LaTeX-escaped text (& -> \&). Order: \citet forms first.
CITES = [
    (r"Geva et al\.\s*\(EMNLP\s+2023\)", r"\\citet{geva2023dissecting}"),
    (r"Wang\s+\\&\s+Xu\s*\(2025\)", r"\\citet{wang2025functional}"),
    (r"Wang\s+\\&\s+Xu\s+2025", r"\\citealp{wang2025functional}"),
    (r"Wang\s+et al\.\s*\(2024\)", r"\\citet{wang2024locating}"),
    (r"Wang\s+et al\.\s+2024", r"\\citealp{wang2024locating}"),
    (r"Christ et al\.\s*\(2025\)", r"\\citet{christ2025structure}"),
    (r"Christ et al\.\s*\(Oct\s+2025\)", r"\\citet{christ2025structure}"),
    (r"the J-space paper", r"\\citet{gurnee2026workspace}"),
    (r"Gurnee,\s+Sofroniew,\s+Lindsey et al\.\s+\(2026\)", r"\\citet{gurnee2026workspace}"),
    (r"Gurnee,\s+Sofroniew,\s+Lindsey et al\.\s+2026", r"\\citealp{gurnee2026workspace}"),
    (r"Hendel et al\.,?\s+(?:EMNLP\s+)?2023", r"\\citealp{hendel2023icl}"),
    (r"Todd et al\.,?\s+(?:ICLR\s+)?2024", r"\\citealp{todd2024function}"),
    (r"Liu et al\.,?\s+(?:ICML\s+)?2024", r"\\citealp{liu2024icv}"),
    (r"Turner et al\.\s+2023", r"\\citealp{turner2023actadd}"),
    (r"Hernandez et al\.,?\s+(?:ICLR\s+)?2024", r"\\citealp{hernandez2024lre}"),
    (r"Merullo et al\.\s+2024", r"\\citealp{merullo2024vector}"),
    (r"Chughtai et al\.\s+2024", r"\\citealp{chughtai2024summing}"),
    (r"Huang et al\.,?\s+(?:ACL\s+)?2024", r"\\citealp{huang2024ravel}"),
    (r"Geva et al\.\s+\(?EMNLP\s+2023\)?", r"\\citealp{geva2023dissecting}"),
    (r"Nikankin et al\.\s+2025", r"\\citealp{nikankin2025heuristics}"),
    (r"Nanda et al\.\s+2023", r"\\citealp{nanda2023grokking}"),
    (r"Zhong et al\.\s+2023", r"\\citealp{zhong2023clock}"),
    (r"Kantamneni \\& Tegmark\s+2025", r"\\citealp{kantamneni2025trig}"),
    (r"Gurnee \\& Tegmark\s+2024", r"\\citealp{gurnee2024space}"),
    (r"Marks \\& Tegmark\s+2023", r"\\citealp{marks2023geometry}"),
    (r"Hong et al\.\s+2024", r"\\citealp{hong2024implies}"),
    (r"Christ et al\.\s+2025", r"\\citealp{christ2025structure}"),
    (r"Christ et al\.", r"\\citeauthor{christ2025structure}"),
]


def md_source() -> str:
    lines = MD.read_text().splitlines()
    out, skip_para, after_image = [], False, 0
    for ln in lines:
        if ln.startswith("# "):          # H1 title -> handled by main.tex
            continue
        if ln.startswith("*Subtitle:") or ln.startswith("*Draft, "):
            skip_para = True             # docs-only preamble paragraphs
        if ln.startswith("[:material-"):
            continue                     # mkdocs button lines (site-only)
        # italic caption paragraph following an image: the alt text becomes the
        # LaTeX \caption, so the site-facing italic caption would be a duplicate
        if after_image and ln.startswith("*"):
            skip_para = True
        if ln.startswith("!["):
            after_image = 2
        elif ln.strip():
            after_image = max(0, after_image - 1)
        if skip_para:
            if not ln.strip():
                skip_para = False
            continue
        out.append(ln)
    return "\n".join(out)


def longtable_to_table(tex: str) -> str:
    """Convert every pandoc longtable to a floating table/tabular (booktabs kept).
    Tables whose widest row exceeds a column's worth of text go to table* (full
    text width) — a two-column layout cannot hold them in one column."""
    CAPTIONS = [  # wide floats need captions to be findable once they float away.
        # Patterns must be UNIQUE to one table's body (several tables share terms
        # like "operator variance" -- match on a row label only that table has).
        (r"logic \(compare\)", "Cross-domain factorization: relational operators "
         "factorize and generalize; arithmetic and logic do not, under our setup."),
        (r"add-N \(operator = addend\)", "Reconciliation with the add-N cut: the "
         "linear-parameter family sits between relations and true two-operand "
         "arithmetic."),
        (r"all-pairs swap contrast", "Cross-architecture replication: every "
         "signature reproduces on Gemma-2-9B (logit units are not comparable "
         "across architectures; compare flip rates and variance shares)."),
        (r"read position", "Two-way ANOVA of the workspace state: variance shares "
         "by read position (1.7B / 8B)."),
        (r"full donor", "Decomposing the donor activation (1.7B, 224 cells; clean "
         "greedy ceiling 53\\%): a state composed from the factorization's additive "
         "parts makes the model emit the target answer as well as the real donor "
         "does; the interaction term is not needed. \"Held-out cell\" rebuilds the "
         "components without ever seeing the target cell; \"wrong operand\" "
         "redirects the answer to the swapped operand."),
        (r"permuted labels \(per-operand\)", "The null battery (1.7B, $\\alpha=4$). "
         "Semantic nulls -- permuted relation labels and random directions inside "
         "the operator subspace -- abolish the effect; structural probes remain "
         "nonzero for mechanistically explained reasons (\\S4.1)."),
        (r"wrong \(t = 0\)", "Where the vector must land (1.7B, "
         "$\\alpha=4$): position $\\times$ layer scope, with the metrics flips "
         "hide. Ranks are medians over 224 cells; clean-baseline rank 570."),
        (r"reading-position control", "The paper's claims, each with its main "
         "evidence, the control that could kill it, and its scope."),
    ]

    # the claims table has long prose cells: no uniform scaling can make it both
    # fit and stay legible — give it wrapping p{} columns at \footnotesize instead.
    CLAIMS_COLSPEC = (r"@{}>{\raggedright\arraybackslash}p{0.185\textwidth}"
                      r">{\raggedright\arraybackslash}p{0.315\textwidth}"
                      r">{\raggedright\arraybackslash}p{0.27\textwidth}"
                      r">{\raggedright\arraybackslash}p{0.14\textwidth}@{}")

    def repl(m):
        colspec = m.group(1)
        body = m.group(2)
        # pandoc longtable: [toprule+header+midrule] \endhead [bottomrule]
        # \endlastfoot [data]. Reassemble as header + data + closing bottomrule,
        # dropping the foot content from the middle (else \bottomrule lands ABOVE
        # the data rows and the table renders with no closing rule).
        sec = re.search(r"(.*?)\\endhead(?:.*?)\\endlastfoot(.*)", body, re.S)
        if sec:
            body = sec.group(1).strip() + "\n" + sec.group(2).strip() + "\n\\bottomrule"
        body = re.sub(r"\\end(head|firsthead|foot|lastfoot)", "", body)
        body = body.replace(r"\noalign{}", "")
        plain = re.sub(r"\\[a-zA-Z]+\{?|[{}]", "", body)
        widest = max((len(r) for r in plain.splitlines() if "&" in r), default=0)
        if re.search(r"reading-position control", body):
            # the claims table: wrapping p{} columns, never scaled
            cap = next((c for pat, c in CAPTIONS if re.search(pat, body)), "")
            captex = f"\\caption{{{cap}}}\n" if cap else ""
            return ("\\begin{table*}[t]\n\\centering\\footnotesize\n"
                    "\\setlength{\\tabcolsep}{4pt}\n"
                    f"\\begin{{tabular}}{{{CLAIMS_COLSPEC}}}\n{body.strip()}\n"
                    f"\\end{{tabular}}\n{captex}\\end{{table*}}")
        if widest > 55:  # wide: span both columns, floats to a page top -> caption it
            cap = next((c for pat, c in CAPTIONS if re.search(pat, body)), "")
            captex = f"\\caption{{{cap}}}\n" if cap else ""
            return ("\\begin{table*}[t]\n\\centering\\small\n"
                    "\\setlength{\\tabcolsep}{4pt}\n"
                    "\\begin{adjustbox}{max width=\\textwidth}\n"
                    f"\\begin{{tabular}}{{{colspec}}}\n{body.strip()}\n"
                    f"\\end{{tabular}}\n\\end{{adjustbox}}\n{captex}\\end{{table*}}")
        # narrow: keep inline exactly where the prose introduces it; shrink-only
        # (a 5-column row can overflow the column, but must never be blown up)
        return ("\\begin{table}[H]\n\\centering\\small\n"
                "\\begin{adjustbox}{max width=\\columnwidth}\n"
                f"\\begin{{tabular}}{{{colspec}}}\n{body.strip()}\n"
                "\\end{tabular}\n\\end{adjustbox}\n\\end{table}")
    return re.sub(
        r"\\begin\{longtable\}\[\]\{@\{\}(\w+)@\{\}\}(.*?)\\end\{longtable\}",
        repl, tex, flags=re.S)


def _balanced(s: str, start: int) -> int:
    """Index of the '}' closing the '{' at s[start]."""
    depth = 0
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return i
    raise ValueError("unbalanced braces")


def figures(tex: str) -> str:
    """\\pandocbounded{\\includegraphics[...alt={CAP}]{fig}} -> figure* with caption.
    Brace-matching, not regex: pandoc escapes [ ] as {[} {]} inside the alt text."""
    key = r"\pandocbounded"
    out, i = [], 0
    while True:
        j = tex.find(key + "{", i)
        if j < 0:
            out.append(tex[i:])
            break
        out.append(tex[i:j])
        end = _balanced(tex, j + len(key))
        inner = tex[j:end + 1]
        cap = ""
        a = inner.find("alt={")
        if a >= 0:
            q = _balanced(inner, a + len("alt="))
            cap = inner[a + len("alt={"):q].strip()
        pm = re.search(r"\]\{([^{}]*)\}\}$", inner)
        path = pm.group(1) if pm else ""
        out.append("\\begin{figure*}[t]\n\\centering\n"
                   f"\\includegraphics[width=\\textwidth]{{{path}}}\n"
                   f"\\caption{{{cap}}}\n\\end{{figure*}}")
        i = end + 1
    return "".join(out)


def main() -> None:
    import sys
    public = "--public" in sys.argv  # de-anonymized preprint build (site PDF)
    md = md_source()
    tex = subprocess.run(
        ["pandoc", "-f", "gfm", "-t", "latex", "--shift-heading-level-by=-1"],
        input=md, capture_output=True, text=True, check=True).stdout
    for u, t in UNI.items():
        tex = tex.replace(u, t)

    # figure paths: figs/op_x.png -> op_x; copy the vector PDFs in (self-contained)
    figdir = ACL / "figs"
    figdir.mkdir(exist_ok=True)
    for stem in set(re.findall(r"figs/(op_[a-z_]+)\.png", tex)):
        # prefer the print-targeted variants (PAPER=1 scripts/plots.py): white
        # surface, serif type, drawn near final physical size
        src = ROOT / "docs" / "figs" / "paper" / f"{stem}.pdf"
        if not src.exists():
            src = ROOT / "docs" / "figs" / f"{stem}.pdf"
        if src.exists():
            (figdir / src.name).write_bytes(src.read_bytes())
    tex = re.sub(r"figs/(op_[a-z_]+)\.png", r"\1", tex)
    tex = figures(tex)
    tex = longtable_to_table(tex)
    tex = tex.replace(r"\def\LTcaptype{none} % do not increment counter", "")

    # drop italic caption paragraphs (they duplicate the figure captions)
    tex = re.sub(r"^\\emph\{(Injecting the operator|Every ordered swap|Where operand"
                 r"|Operation \\ensuremath\{\\neq\} realization)[^\n]*\}$",
                 "", tex, flags=re.M)

    # strip manual numbers from section headings; ACL renumbers identically
    tex = re.sub(r"(\\(?:sub)*section)\{\d+(?:\.\d+)*\.?\s*", r"\1{", tex)
    # kill pandoc's labels (avoid duplicate-label warnings after renaming)
    tex = re.sub(r"\\label\{[^}]*\}", "", tex)

    if not public:
        # anonymity: no identifying URLs anywhere
        for pat in (r"https?://mpodeley\.github\.io[^\s}]*",
                    r"https?://github\.com/mpodeley[^\s}]*"):
            tex = re.sub(pat, r"[link redacted for review]", tex)
    # internal docs links (reproduce.md etc.) -> plain text
    tex = re.sub(r"\\href\{[^}]*\}\{([^}]*)\}", r"\1", tex)

    # prose citations -> natbib, then drop the prose References section (bibtex now)
    for pat, rep in CITES:
        tex = re.sub(pat, rep, tex)
    tex = re.sub(r"\\section\{References\}.*\Z", "", tex, flags=re.S)

    # split out the abstract
    m = re.search(r"\\section\{Abstract\}\s*(.*?)(?=\\section\{)", tex, re.S)
    abstract = m.group(1).strip() if m else ""
    if m:
        tex = tex[:m.start()] + tex[m.end():]

    # appendices: split out and let main.tex place them after \bibliography
    appendix = ""
    ai = tex.find(r"\section{Appendix A:")
    if ai >= 0:
        appendix = re.sub(r"\\section\{Appendix [A-Z]: ", r"\\section{", tex[ai:])
        tex = tex[:ai]
        # the appendix is set \onecolumn (main.tex): floats become [H] blocks
        # pinned exactly where their prose introduces them — no more headers
        # divorced from their tables and half-empty float pages
        appendix = (appendix
                    .replace("\\begin{table*}[t]", "\\begin{table}[H]")
                    .replace("\\end{table*}", "\\end{table}")
                    .replace("\\begin{figure*}[t]", "\\begin{figure}[H]")
                    .replace("\\end{figure*}", "\\end{figure}")
                    .replace("\\begin{adjustbox}{max width=\\columnwidth}",
                             "\\begin{adjustbox}{max width=\\textwidth}")
                    .replace("\\includegraphics[width=\\textwidth]",
                             "\\includegraphics[width=0.95\\textwidth]"))

    suffix = "_public" if public else ""
    (ACL / f"abstract{suffix}.tex").write_text(abstract + "\n")
    (ACL / f"appendix{suffix}.tex").write_text(appendix + "\n")
    (ACL / f"body{suffix}.tex").write_text(tex)
    bad = sorted({c for c in tex + abstract if ord(c) > 127})
    print(f"wrote paper/acl/body{suffix}.tex ({len(tex)} chars) + abstract{suffix}.tex "
          f"({len(abstract)} chars); non-ascii remaining: {bad if bad else 'none'}")


if __name__ == "__main__":
    main()
