# Writing Guide — Extended Springer Chapter

Working conventions for writing and editing the extended book chapter
`author/Falconi-Synthetic-SQuAD-Methodology-EQA-Robbery.tex` (Springer **SNmult** book-chapter class).

## Target file
- **Chapter:** `author/Falconi-Synthetic-SQuAD-Methodology-EQA-Robbery.tex` — this is the file that is
  being written and will keep evolving as the chapter is built.

## AUTHORITATIVE FORMAT GUIDE
- **`author/author.tex`** is the authoritative reference for the intended Springer
  **SNmult** chapter formatting (author block, sections, abstract, tables, figures,
  environments, etc.). Resolve any format question against this file rather than guessing.
- `editor/author.tex` is the same template in its "editor" build variant (some lines
  commented out plus trivial wording differences); use it only as a secondary check.
- **Use the Springer blocks/environments shown in `author.tex` for special content:**
  use `definition` for formal definitions (e.g. metrics), `backgroundinformation` /
  `important` / `warning` / `tips` / `overview` / `legaltext` for their intended
  highlighted content, and `programcode` for prompts/code excerpts. Prefer these
  template-provided environments over inventing custom formatting.

## CLASS NOTE
- The target chapter uses the **SNmult** book-chapter class.
- The reference `.tex` sources in `author/tex-references/` use the **llncs** conference
  class (different commands, e.g. `\begin{abstract}` vs `\abstract*{}`, `\inst{}` vs
  `\at`, `\orcidID`, etc.).
- When adapting content into the chapter, decide the correct SNmult form **per case** and
  explicitly flag the decision made (do not blanket-convert blindly).

## KEY REFERENCE MATERIALS AND DATA
- **Previous/original paper:** `author/tex-references/ica9637.tex` — source for parts of
  the earlier work (e.g. author names, methodology pieces, F1 = 82.88).
- **Initial extended draft:** `author/tex-references/initial_draft_extended_legal_eqa.tex`
  — extended-chapter skeleton with placeholders (`[N]`, `[--]`) to fill with measured data.
- **Research journal / methodology & results:** `research.org` (repo root).
- **Dataset / QC outputs:** `out_qc_M2/` (includes ready-made LaTeX tables).
- **Audit (artificial manual labelling) report:** `Reports/audit_glm_v1/` (includes
  ready-made LaTeX tables under `outputs/`).
- **Ablation experiment results:** `out_experiments/` — aggregate views also under
  `paper/results_summary.csv` / `paper/results_summary.json`.

## REFERENCES / BIBTEX
- References are managed with **BibTeX** using `.bib` files (the author prefers this over
  a manual `thebibliography`).
- **Source of entries:** `author/references.bib`. Add or update entries there, never inline.
- **In the chapter** (`Falconi-Synthetic-SQuAD-Methodology-EQA-Robbery.tex`) the bibliography is produced
  at the end with:
  ```
  \bibliographystyle{spmpsci}
  \bibliography{references}
  ```
  and each reference is cited in the text with `\cite{key}`.
- **Citation format:** Springer prefers **numeric** citations (by number) over
  author/year. The provided Springer styles are:
  - `spmpsci.bst` — **numeric**, standard LaTeX, for mathematics/computer
    science/physical sciences (**use this one**).
  - `spbasic.bst` — author-year by default (needs `natbib[numbers]` to become
    numeric); not the right default for this chapter.
  - `spphys.bst` — numeric but physics/APS-specific.
  Use `spmpsci` so `\cite{key}` renders as a number, e.g. `[1]`.
- Do **not** use `\input{references}` (that old line pulled in the sample
  `references.tex` template, not real entries).
- `references.tex` is only a template/example; it is not part of the BibTeX workflow.
- Cite the correct bib key from `references.bib`; never invent keys or entries.
- **Academic searching for model references:** when the chapter describes a model or
  method, perform an academic web search to locate the supporting paper/technical
  report. Add the paper's BibTeX entry to `author/references.bib`, cite it in the
  text, and also cite the Hugging Face model card when the exact checkpoint matters.
- **Marking new bib entries:** when adding entries in `references.bib`, surround them
  with a comment banner such as
  `% NEW ADDITIONS -- MANUALLY CHECK ...` so the author can verify the
  bibliographic details before submission. Do not silently edit existing entries.

## FIGURES AND TABLES
- **Figures:** if a figure is to be used in the paper, it must be copied into
  `author/figures/` (create the directory if needed). Reference the figure in the
  chapter by its local path under `author/figures/`, e.g.
  `\includegraphics{figures/<filename>}`. Do not reference figures that live
  only in another repository folder.
- **Tables:** if a table is copied from a file in the repository (e.g. a
  ready-made LaTeX table under `out_qc_M2/` or
  `Reports/audit_glm_v1/outputs/`), add a comment above the table in the chapter
  noting the origin file path, e.g.
  `% Source: out_qc_M2/table_metrics_summary.tex`. This makes the provenance of
  every table traceable.
- When adapting tables, keep the measured values consistent with the source file
  and never invent values; flag placeholders that still need data.

## LLM INSTRUCTION COMMENTS IN THE CHAPTER
- The chapter may contain comments like `%% LLM: ...` or `% LLM: ...` that carry a
  writing instruction left by the author.
- When such a comment is found, **solve the requirement** (implement the requested
  change/write the content) and then add an answer marker immediately after the
  solved area: `%% LLM-SOLVED`.
- Do your best to satisfy the requirement; if it is ambiguous or would invent data,
  ask the author before solving.

## RESULT AGGREGATION PROVENANCE
- Experiment results are aggregated from per-run outputs (e.g. under
  `out_experiments/`) using `scripts/aggregate_results.py`.
- When reporting such results in the chapter, add a comment above the table/figure
  explaining how the results were aggregated and from which root, for example:
  ```
  % Results aggregated with scripts/aggregate_results.py from
  % out_experiments/option_b_abblation/ (per-experiment metrics_summary.json,
  % gold_audit_metrics.json, and metrics_by_question_type.csv).
  ```
- Keep the numbers exactly consistent with the aggregated output; never invent or
  round silently.

## WRITING STYLE AND TONE
- When asked to write, act as a **professional computer science researcher**.
- Write with an **academic tone** and proper grammar and punctuation.
- This behavior is expected whenever the author commands you to write (not only for
  full drafts; apply it to any prose you produce).
- The author may provide a previous work so you can **mimic their writing style**.
- For the author's writing style, see the open-access paper:
  https://www.astesj.com/v05/i02/p20/

## EXPECTED BEHAVIOUR OF THE WRITING ASSISTANT
- Be careful: ask before acting if there is any doubt, especially on format or data.
- At the initial stage, focused mostly on tasks like **move, copy, and adapt** (and
  solving questions), rather than drafting new prose.
- When citing or filling tables, keep evidence consistent with the referenced folders
  above and never invent measured values; flag placeholders that still need data.
