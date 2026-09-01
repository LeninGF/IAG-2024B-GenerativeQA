# Instructions for Restructuring the Extended Paper

**Audience:** an LLM/agent (or a human editor) that will produce the next draft of the manuscript.
**Do not use the word "chapter" anywhere in the manuscript or in any commentary about it.** Use "paper", "study", or "research" instead. Springer's own invitation and guideline site call the deliverable a "manuscript"/"chapter" only in its own boilerplate; in every author-facing part of the text (title, headers, prose, captions) use "paper"/"study"/"research."

This document does not rewrite the paper. It tells whoever does the rewrite exactly what to change, why, and where to find the evidence.

---

## 0. Sources this document is based on

- Current draft: `Falconi-Synthetic-SQuAD-Methodology-EQA-Robbery.pdf` (25 pages, all sections read in full: Abstract → References).
- Original published paper (now recovered — see §7 "Corrections"): `696374_1_En_16_Chapter_Author-Question-Answering-Robbery-Chrimes.pdf`, i.e. Falconi, L.G., Barona López, L.I., Hernandez-Alvarez, M., "Leveraging Transformer Models for Extractive Question Answering in Criminal Report Analysis," in H.R. Arabnia et al. (eds.), *Internet Computing, Internet of Things, Artificial Intelligence, and Applications*, Springer Nature Switzerland, Cham (2026). DOI: [10.1007/978-3-032-22190-2_16](https://doi.org/10.1007/978-3-032-22190-2_16).
- Research journal `research.org` (772 lines, read in full) — the primary source for the honest, unpublished methodological narrative (overfitting, leakage, threshold calibration, inter-generator agreement).
- Notebooks: `qc_squadv2_datasets.ipynb`, `audit_report.ipynb`, `dataset_xplore_and_upload_M2.ipynb`, `report_m2_results.ipynb`.
- Springer guideline site content (`LLMs AI Tools, NLP & Use Cases.pdf`, extracted from the project's guideline page): manuscript must contain **at least 50% new material**, be **15–25 pages**, cite the original paper, and use a **different title and abstract**.

Whoever executes this instruction set should re-open these same files rather than trust numbers restated here from memory; treat every number below as coming from the specific artifact cited next to it.

---

## 1. Mandatory terminology changes

| Current term | Problem | Replace with |
|---|---|---|
| "chapter" | Springer program-wide word; author-facing text should read as a standalone paper | "paper" / "study" / "research" |
| "semantic audit" / "audit" (Section 3.4, 4.2, 6, Conclusion) | "Audit" implies a formal compliance/financial-style inspection and slightly overstates the rigor of a single-reviewer, 200-row pass | "LLM-based semantic review" (first use), then "review" / "reviewer-labeled assessment" |
| "gold" / "gold set" / "gold_test" / "gold standard" (Sections 3.5, 4.3, Table names, HF dataset split) | The 200-row subset was labeled by **another LLM acting as a simulated reviewer**, not by a human annotator or an independently verified ground truth. Calling it "gold" implies a certainty the data does not have — and the QC notebook's own conclusions explicitly warn "auto-generated labels are not ground truth." | "LLM-reviewed subset" / "reviewer-labeled evaluation subset" (prose); if the HF dataset split name `gold_test` cannot be renamed without breaking the public dataset card, keep the technical split name in code/tables but never call it "gold" in prose — call it "the reviewer-labeled split (`gold_test` in the released dataset)" |
| reviewer id `computer-v1` | This internal identifier could be misread as naming a specific commercial AI product/assistant. Keep the paper's language product-neutral. | Rename in the paper (not necessarily in your internal files) to something generic, e.g. "Reviewer-LLM" or "R1" |
| "GLM-5.3 (to be confirmed)" placeholder, and any description implying "the model available through Perplexity" | Underspecified/unverifiable version number, and revealing the interface used to access the model is off-topic for a methods section and could read as an advertisement | See §2 below — cite the actual model family with a verifiable technical report |

**Action for the rewriting agent:** do a full-text search of the current draft for `audit`, `gold`, `chapter`, and `computer-v1`, and resolve every instance using the table above. Do not do a blind find-and-replace — some occurrences (e.g., "financial audit" is never used, but "Threats to Validity" section reads naturally with "review" substituted; check each sentence still parses).

---

## 2. How to cite the LLM reviewer model correctly

Do not write "the model provided by Perplexity" or any equivalent. Cite the model family with a verifiable technical report, exactly as you already do for Gemma and Qwen.

**What is confirmed:**
- The GLM family (Zhipu AI / Z.ai) publishes open technical reports on arXiv. The GLM-4.5 report is confirmed and citable: Zeng, A. et al. (Z.ai / Zhipu AI), *"GLM-4.5: Agentic, Reasoning, and Coding (ARC) Foundation Models,"* arXiv:2508.06471 (2025). [https://arxiv.org/abs/2508.06471](https://arxiv.org/abs/2508.06471). The GLM-4.6 model card on Hugging Face explicitly points back to this same report (no separate GLM-4.6 technical report has been published as of this writing) — [https://huggingface.co/zai-org/GLM-4.6](https://huggingface.co/zai-org/GLM-4.6).
- The broader lineage report is already reference [28] in your draft: Zeng, A. et al., *"ChatGLM: A Family of Large Language Models from GLM-130B to GLM-4 All Tools,"* arXiv:2406.12793 (2024).
- A GLM-5 technical report may exist as of early 2026 (a "GLM-5: from Vibe Coding to Agentic Engineering" preprint was referenced in a secondary technical-report tracker, arXiv:2602.15763) — **this was found only in a secondary/community-maintained source, not verified directly on arxiv.org**, so treat it as unconfirmed until checked at the primary source.

**Action for the rewriting agent / author:**
1. Determine the exact model identifier that was actually invoked as the reviewer (check your own request logs, the API/model string used when the 200-row sample was labeled, or the environment that ran `audit_report.ipynb`).
2. If it is a GLM-4.5 or GLM-4.6-class model: cite arXiv:2508.06471 (GLM-4.5 report, which GLM-4.6 explicitly reuses) plus the Hugging Face model card URL for the exact checkpoint. Do not call it "GLM-5.3."
3. If it is confirmed to be a GLM-5.x model: verify the GLM-5 report directly on arXiv (search "GLM-5" on arxiv.org) before citing arXiv:2602.15763, since it was not independently confirmed here.
4. Phrase the methods text as, e.g.: *"A second, larger open-weight LLM from the GLM family (Zhipu AI/Z.ai) [cite report] was used as an automated reviewer to label a stratified sample of question–answer pairs, functioning as a simulated human annotator."* Never mention the specific consumer product or API used to access it.
5. Remove the "(to be confirmed)" placeholder from the References section — either confirm and finalize, or keep an internal `TODO` outside the manuscript, not inside the reference list.

---

## 3. How to describe the framework (use this consistently in Abstract, Intro, Methods overview, Discussion, Conclusion)

Use one consistent formulation, adapted from the earlier title discussion but stripped of "audit"/"gold"/"chapter":

> We propose an end-to-end methodology that uses open LLMs to generate candidate SQuAD v2-style question–answer annotations from Spanish robbery reports; applies deterministic span, offset, and answerability controls; assesses residual semantic errors through an LLM-based reviewer pass on a stratified sample; and evaluates the resulting data through zero-shot and fine-tuned extractive QA models under a leakage-aware, context-disjoint evaluation protocol.

Key framing rules:
- The LLMs are **candidate annotation generators**, not the final QA system.
- The reports are real source narratives; the **synthetic artifact is the QA annotation layer** (question, answer, answerability decision, span) — never call the reports themselves synthetic.
- The scientific thesis to repeat throughout the paper: **a literal string match and a correct character offset are necessary but not sufficient conditions for reliable synthetic EQA supervision.** This is the sentence that ties QC (§3.3), the LLM-based review (§3.4), and the results (§4.2) together — make sure it appears near-verbatim in the Introduction, the start of the Methods section, and the Conclusion so a reader can trace the argument.

---

## 4. Contributions list (revise the Introduction's list to this)

> The main contributions of this study are:
>
> 1. **An LLM-based methodology for synthetic SQuAD v2 construction** from Spanish robbery narratives, producing answerable and unanswerable EQA instances with answer-span offsets and generator provenance.
> 2. **A reproducible quality-control protocol** (structural integrity → extractive validity → abstention consistency → span distribution/usefulness → inter-generator agreement) that filters raw LLM output into `strict`, `expanded`, and `merged` dataset variants before any human or LLM review is needed.
> 3. **An LLM-based reviewer protocol for semantic reliability assessment**, using a second, independent LLM as a simulated annotator over a stratified sample, with an explicit error typology (wrong semantic type, wrong event reference, partial location, incomplete object list, answered-when-unanswerable, incorrect impossible, false abstention, other) and Wilson-score confidence intervals — explicitly distinguishing structural validity from semantic correctness.
> 4. **A leakage-aware evaluation protocol** using context-disjoint train/validation/test partitions and a held-out, LLM-reviewed evaluation subset, correcting the row-level, test-set-leaking split used in the original proof-of-concept study.
> 5. **A four-model extractive QA ablation** (two Spanish BETO variants, a multilingual XLM-RoBERTa model, and a distilled BETO model) comparing zero-shot and fine-tuned performance, including training-dynamics diagnostics (overfitting behavior, early stopping, no-answer threshold calibration) that the original study did not report.
> 6. **A direct, controlled replication of the original study's result**, run under the new codebase, that reproduces its headline number and then shows, on the same trained model, why that number does not hold under the new, harder, leakage-free benchmark — turning the original result into a diagnosed baseline rather than a competitor to be beaten.

Contribution 6 is new relative to the earlier draft of this contributions list and should be added: it is exactly what Springer's "must differ from and build on the original paper" requirement wants to see made explicit, and it is your strongest, most defensible extension claim (see §8 "Corrections" for the underlying evidence).

---

## 5. Research questions

State these explicitly at the end of the Introduction and structure the Discussion around them:

- **RQ1.** Can open LLMs construct usable SQuAD v2-style EQA supervision from unlabeled Spanish robbery reports at scale, without per-example human authoring?
- **RQ2.** Are literal-span validity and correct character offsets sufficient evidence that a generated question–answer pair is semantically correct, or is a further, independent review step required to detect errors that string matching cannot see?
- **RQ3.** Does the constructed resource support effective extractive QA fine-tuning across encoder families and capacities, when evaluated under a leakage-free, context-disjoint protocol and against an independent, LLM-reviewed evaluation subset?
- **RQ4 (new — recommended addition).** Under a corrected, leakage-free evaluation protocol, does the original study's headline result still hold, and what specific factors (dataset composition, split methodology, metric implementation) explain any discrepancy?

RQ4 is what lets you honestly and constructively report the ~82.9 → ~0.73/~0.59 drop as a scientific finding about evaluation methodology rather than as your own model underperforming — see §8.

---

## 6. Abstract guidance

The abstract must differ from the original paper's abstract (Springer requirement) and must not use "chapter," "audit," or "gold." Suggested abstract (fill in the bracketed numbers only after re-confirming them against `report_m2_results.ipynb` output — do not carry over unverified numbers from any prior draft or chat without re-checking):

> This study extends prior work on transformer-based extractive question answering (EQA) for Spanish-language robbery reports by examining the reliability of large language model (LLM)-generated SQuAD v2-style supervision. We propose a methodology that uses two open LLMs to generate candidate question–answer pairs from robbery narratives, applies deterministic checks for schema conformance, literal span support, offset consistency, duplication, and answerability, and complements these checks with a semantic review performed by a third, independent LLM acting as a simulated annotator over a stratified sample. We evaluate the resulting dataset using a context-disjoint train/validation/test split and four pretrained extractive QA encoders under zero-shot and fine-tuned settings, and we replicate the original study's evaluation protocol on its own dataset to directly compare methodologies. The results show that literal span validity is a necessary but not sufficient condition for reliable synthetic EQA supervision, that fine-tuning under a realistic, abstention-inclusive benchmark yields lower but more defensible performance than the original proof-of-concept result, and that both dataset composition and evaluation design materially affect reported scores. We conclude with a reusable quality-control and review protocol for LLM-synthesized supervision in other under-resourced, sensitive-text domains.

---

## 7. Proposed paper structure

Follow this structure. Each subsection lists what goes in it, what can be reused from the original paper, and where the new material comes from.

### 1. Introduction
- Open with the same real-world motivation as the original paper: robbery statistics from Ecuador's Fiscalía General del Estado (67,705 reported incidents in 2024, ≈376 per 100,000 inhabitants — cf. original paper §1), the value of EQA for investigations, insurance claims, and public-safety planning. **This paragraph may be reused close to verbatim** — it is factual, well-cited, and is the kind of "shared context" Springer expects an extension to build from.
- State the research gap and this study's contribution explicitly: the original study demonstrated feasibility (an LLM-synthesized dataset can support EQA fine-tuning) but left open whether the synthesized supervision was *reliable*, whether the evaluation protocol was *leakage-free*, and whether the result *generalizes* across model families. This study addresses exactly those three gaps.
- Insert the research questions (§5 above).
- Insert the contributions list (§4 above).
- One paragraph explicitly stating how this study differs from and extends the original paper (title, dataset size, generator model(s), split methodology, number of models evaluated, evaluation depth) — this satisfies Springer's "must state what is new" requirement directly in the text, not just implicitly through content.

### 2. Theoretical Background and Related Work
Make this more didactic than the current draft — the current 2.1–2.5 reads like a related-work survey; add short, clearly-marked definition blocks before diving into literature.
- **2.1 Extractive vs. generative QA and unanswerability.** Define EQA formally (span prediction over context), define SQuAD v2-style unanswerability, and briefly define EM/F1 conceptually here (formal definitions with equations stay in Materials and Methods §3, not here).
- **2.2 Transformers for evidence-based / legal NLP.** Keep current related-work content (2.2), reorganized as literature, not mixed with definitions.
- **2.3 LLMs as synthetic data generators.** Keep current related-work content (2.3); add one paragraph explicitly defining "LLM-as-annotator" / "LLM-as-judge" as a concept with 1–2 supporting citations (e.g., existing LLM-as-judge literature), since this is now central to your methodology and deserves definitional grounding, not just a data-generation citation.
- **2.4 Research gap and paper positioning.** Keep current 2.5 content here, renumbered.
- **Move "Models Used in This Research" (current 2.4, with Tables 1–2) entirely to Materials and Methods**, since it describes *your* experimental setup, not background literature — group it with the fine-tuning subsection (§3.6 below).

### 3. Materials and Methods
1. **Robbery-narrative corpus.** Reuse and lightly update the original paper's corpus description (174,594 Spanish robbery reports, word-count filtering, mean 112 words/report — cf. original §3.1 and current draft §3.1). This is appropriate reuse since the underlying corpus is unchanged between studies.
2. **SQuAD v2-style candidate generation.** Describe the two generator models (Gemma-3-1B-IT, Qwen-2.5-3B-Instruct), the prompt structure (can closely follow the style of the original paper's prompt description, updated for the new schema and the two-generator design), and the stated **technical constraint that motivated using smaller, faster models across up to 8 NVIDIA A16 GPUs** rather than a single larger model — this is a legitimate, citable systems/engineering justification and should be stated plainly (compute budget and throughput, not just a preference).
3. **Automatic quality control (the core methodological contribution).** This is where the notebook `qc_squadv2_datasets.ipynb` methodology belongs, in detail:
   - Phase 1 — structural integrity (JSON parseability, required fields, valid labels, `(context_id, question)` uniqueness).
   - Phase 2 — extractive validity: **generated offsets are never trusted**; they are recomputed and validated against the context. Define the `qc_status` taxonomy (`clean_no_answer`, `empty_no_answer`, `free_text_no_answer`, `impossible_but_span_present`, `non_extractive_or_hallucinated`, etc.) and state explicitly that an answer failing literal-span verification is **not** silently converted to "impossible" — it is flagged as uncertain, preserving the distinction between "the model couldn't find an answer" and "the model hallucinated one."
   - Phase 3 — abstention consistency (sentinel/label coherence, answer-vs-abstention rates globally and by question type).
   - Phase 4 — span distribution/usefulness (answer length, duplicate answers, over-long-answer detection via a combined percentile + fixed-length rule).
   - Phase 5 — inter-generator agreement (Gemma vs. Qwen on identical `(context_id, question)` pairs): report the actual figures from the notebook/journal — answerability agreement, exact-answer agreement, no-answer agreement, and disagreement rate — and state plainly that low exact-answer agreement (documented as low in the QC notebook) is itself evidence that literal validity does not guarantee semantic correctness, motivating the review step.
   - Output: `strict_gemma`, `strict_qwen`, `expanded_gemma`, `expanded_qwen`, and `merged` dataset variants, with the merging rule (agreed spans kept; agreed no-answer kept; disagreements resolved to the unique unflagged valid span). State the final row counts and composition once confirmed against the notebook's exported summary table (do not restate from memory).
   - Then describe the **LLM-based semantic review** (§2 terminology): a second, independent LLM reviewer labels a stratified sample (~200–300 rows, chosen to target roughly ±5–7 percentage points at 95% confidence — this sizing rationale is already in the QC notebook and should be stated explicitly, with a Wilson score interval used because sample proportions are being estimated from a finite, stratified sample). Report per-generator review accuracy with confidence intervals and the dominant error categories.
4. **Dataset consolidation and splits.** Explain how `LeninGF/question-answering-robbery-m2` was constructed: which QC splits fed the merge, how the LLM-reviewed rows were excluded from train/validation/test to prevent using reviewed rows as both a design input and a held-out evaluation set, and why a **context-level** 80/10/10 split (not row-level) was chosen — because multiple questions share the same source report, and a row-level split would leak the same report's other questions across partitions. State the exact split sizes once you've re-confirmed them from the dataset card (`train`, `validation`, `test`, and the reviewed subset).
5. **Extractive QA models and fine-tuning ablation.** Move the "Models Used" table here (from §2.4 of the current draft). Describe the four encoders, hyperparameters, and **explicitly describe the early-stopping protocol and why it is necessary**: state that a preliminary plateau experiment (documented in the research journal) showed `train_loss` decreasing monotonically while `eval_loss` increased almost monotonically over 50 epochs — a textbook overfitting signature — and that dev-set `eval_f1` peaked mid-training and then declined, which is why the final ablation uses early stopping on `eval_f1` (patience 2–3) with a useful epoch range of roughly 10–15 rather than a large fixed epoch count. This is a genuinely reportable methodological finding, not just an implementation detail, and directly answers reviewer-style questions about training rigor.

### 4. Results
1. **Metrics for the synthetic dataset's quality.** Structural QC pass rates, inter-generator agreement figures, and the LLM-reviewer accuracy per generator with confidence intervals; the semantic error-type breakdown. (This can live here rather than in Methods, since these are *outcomes* of applying the protocol, not the protocol itself — see the open question below.)
2. **Metrics for QA performance.** Define EM, F1, HasAns/NoAns breakdown formally here (move the formal metric definitions from wherever they currently sit in Methods to here, since Methods should describe *what was measured* and Results should show *the numbers*). Explicitly describe and justify the **dev-tuned threshold protocol**: report plain F1 (threshold 0.0) as a conservative lower bound, and dev-tuned-threshold F1 (threshold frozen from the best dev epoch, then applied to test) as the headline number — and explicitly state that the test-set oracle threshold (`best_f1`) is a diagnostic upper bound only and must never be reported as the primary result. This is a concrete statistical-rigor improvement over the original paper and should be called out as such.
3. **Fine-tuning ablation results.** Report zero-shot vs. fine-tuned F1/EM for all four models, the HasAns/NoAns breakdown, and per-question-type results — explicitly naming `place` and `objects` as the weakest question types (as identified in the notebooks) and `date`/`value` as the easiest, since this localizes exactly where extractive QA fails on this benchmark and gives the Discussion something concrete to explain.
4. **Original-study replication.** Report the controlled replication: re-running the original study's exact protocol (legacy schema, row-level split, same hyperparameters) inside the new codebase, and report both (a) that this reproduces a comparable headline number to the original paper, confirming the replication is faithful, and (b) the same replicated model's score on the new, leakage-free, LLM-reviewed evaluation subset — which should be materially lower. Present the reasons for the drop as a short, explicit list (see §8 below) rather than a vague "task is harder" statement.

*Suggestion on where the "how many rows survived QC" figures go:* keep the raw QC funnel numbers (rows in → structural failures → extractive-validity failures → final split sizes) in Materials and Methods §3.3–3.4 as a description of the constructed artifact, and keep only the *evaluative* metrics (agreement rates, reviewer accuracy, downstream F1) in Results — this avoids duplicating tables between the two sections.

### 5. Discussion / Conclusion
- Organize the Discussion around the four research questions (§5), each answered with a short paragraph pointing to a specific result.
- Keep and lightly rephrase the existing "Threats to Validity and Responsible Use" section (construct/internal/external validity, responsible use) — this section is already well-structured and needs only the terminology fixes from §1.
- **Conclusion recommendations:**
  1. Restate the central thesis (§3) as the paper's take-home message, not just a summary of what was done.
  2. State plainly, as a finding rather than an apology, that the corrected, leakage-free evaluation protocol produces a lower but more trustworthy estimate of model performance than the original proof-of-concept study, and name the three concrete causes (see §8) — this is intellectually honest and is exactly the kind of methodological contribution book chapters/extensions are valued for.
  3. Explicitly state what a practitioner or future researcher should reuse: the QC pipeline, the review protocol and error typology, the context-disjoint split methodology, and the threshold-calibration protocol — position these as portable to other under-resourced or sensitive-text QA domains (medical records, customer complaints, other incident reports), not just this one.
  4. Name concrete, specific future work tied to your own weakest results: targeted data augmentation or prompt refinement for `place` and `objects` questions; scaling the LLM-reviewed sample size (the notebooks already compute that ~385 rows would be needed for ±5% at 95% confidence, versus the current ~200–300); and a human-in-the-loop verification pass to validate the LLM reviewer itself against human judgment on a small sub-sample, since the LLM reviewer's own reliability was never independently checked against a human.
  5. Do not claim the LLM-reviewed subset is a substitute for human-verified ground truth; state this limitation explicitly, consistent with the terminology fix in §1.

---

## 8. The 82.88 → lower-score story: what to say and why it is a strength, not a weakness

This is the single most valuable scientific narrative available in the research journal, and the current draft's Discussion does not yet make it explicit. State it directly, referencing your own replication experiment (§4.4):

The original study's headline F1 = 82.88 (EM = 55.74) is not wrong, but it was measured under conditions that make the task easier than a realistic deployment setting:
1. **All-answerable test set.** ~3,000 "impossible to find answer" rows were dropped before evaluation, removing all abstention difficulty from the test set. (Consistent with `eval_best_f1_thresh = 0.0` in the original paper's own results table — there was no no-answer case to calibrate a threshold against.)
2. **Test-set model selection.** The original training used the test set itself as the evaluation set for selecting the best checkpoint (`load_best_model_at_end` on the test split, with no separate dev split), which is a form of information leakage between model selection and reported performance.
3. **A metric implementation gap.** The original evaluation code computed start/end token predictions independently (no `start ≤ end` constraint) under `return_overflowing_tokens=True`, which is not a fully correct SQuAD v2 scorer, though this rarely changes scores on short robbery-report contexts.

Frame this precisely: "the original result is best understood as a proof-of-concept score on an easier, all-answerable, test-selected setup; under a corrected, leakage-free, abstention-inclusive protocol, the same modeling approach scores lower but the estimate is unbiased." This framing is more persuasive to reviewers than either (a) silently reporting a lower number with no explanation, or (b) implying the original work was flawed rather than an earlier iteration of the same research program.

---

## 9. Reuse budget (≤30% of the original paper)

Springer requires **at least 50% new material**; capping reuse at 30% of the original paper's text is comfortably inside that requirement, but confirm it quantitatively before submission rather than assuming: after the rewrite, run a rough overlap check (e.g., shared-sentence or shared-paragraph count) between the new manuscript and the original paper's ~11 published pages, and confirm reused text is under ~30% of the *original paper's* length (not 30% of the new, longer manuscript — those are different denominators and the guideline is ambiguous about which one is meant, so satisfying the stricter reading is safer).

Content safe to reuse near-verbatim (factual, uncontested, already well-cited):
- The robbery-statistics motivation paragraph (Introduction).
- The corpus description numbers (174,594 reports, word-count filtering criteria, mean/median word counts).
- Model-card-style descriptions of the shared models (BETO, mrm8488, XLM-R) where architecture facts are unchanged.

Content that must be substantially rewritten, not reused:
- Contributions list, abstract, title (Springer explicitly requires a different title and abstract).
- All methodology from candidate generation onward — the original used a single generator and no QC pipeline, so there is little to reuse here regardless of the 30% cap.
- Results, Discussion, and Conclusion — entirely new, since the empirical study is new.

---

## 10. Corrections found while reviewing the current draft (fix before submission)

1. **Reference [7]'s page range is wrong.** The current draft cites the original paper as "pp. 222–233." The original paper's own front matter states "CSCE 2025, CCIS 2934, pp. 1–12, 2026." Verify the correct final page range directly from the published SpringerLink record ([https://link.springer.com/chapter/10.1007/978-3-032-22190-2_16](https://link.springer.com/chapter/10.1007/978-3-032-22190-2_16)) before finalizing the reference — do not carry over either unverified number.
2. **Unedited LaTeX template boilerplate is present in the current draft's back matter** (Acknowledgements, Competing Interests, Ethics Approval, and Appendix sections). These currently contain the Springer template's own instructional placeholder text (e.g., "Please declare any competing interests... The following sentences can be regarded as examples," and a meaningless placeholder equation "a × b = c" in the Appendix) rather than actual content. This must be replaced with real acknowledgments/competing-interest statements or removed if not applicable, and the placeholder equation must be deleted — leaving it in would be a serious, easily-caught oversight at review.
3. **A production "Author Queries" proof page is embedded in the draft** (page 15 of the current PDF-derived text region, referencing "Chapter 16" author queries). This is publisher proof metadata, not manuscript content, and must not appear in the submitted manuscript.
4. **Several references carry "verify author list/pagination before final submission" notes already** (refs [11], [17], [18], [24]) — resolve these now rather than leaving them as open flags in a near-final draft.
5. **Page budget is tight.** The current draft is already at 25 pages, the stated maximum. Once the terminology fixes, moved subsections, and new RQ4/replication framing are added, re-check total length; trimming the boilerplate in point 2 above recovers some room.

---

## 11. Open items for the author to confirm

- Exact GLM checkpoint/version used as the reviewer model (§2) — needed to finalize the citation.
- Whether to rename the internal reviewer id `computer-v1` in the paper text (§1).
- Final row counts for the merged dataset and the train/validation/test/reviewed-subset split sizes — re-confirm against the notebooks' exported summary tables rather than reusing any number quoted in earlier chat discussion, since those were not independently re-verified in this pass.
- The correct page range for reference [7] (§10, item 1).
- Whether the Springer "≥50% new material" threshold should be checked against the original paper's page count or the new manuscript's page count (§9) — recommend using the stricter interpretation regardless.
