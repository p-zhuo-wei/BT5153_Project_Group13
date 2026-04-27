# BT5153 Group 14 Project Walkthrough

## What this project is trying to do

This project studies resume-job matching as a ranking problem. Given a job posting and a pool of resumes, the system should rank stronger candidates nearer the top so recruiters can review promising candidates earlier.

The project compares three approaches:

- **TF-IDF lexical matching** — a sparse vector space baseline
- **SBERT dense semantic matching** — sentence-transformer embeddings
- **LLM-assisted reranking** — a local Gemma model served through LM Studio

It also evaluates two additional requirements that matter in a hiring context:

- **Interpretability** — so a reviewer can see why a candidate was ranked highly
- **Fairness and robustness** — so the ranking can be stress-tested under controlled perturbations (gender swaps, name swaps, age-related edits)

## Data provenance

The project draws on two public Kaggle datasets:

1. **Resume Dataset** (gauravduttakiit, Kaggle) — a collection of resumes with category labels across 24 job families such as Accounting, Healthcare, Engineering, and Information Technology. Available at `kaggle.com/datasets/gauravduttakiit/resume-dataset`.

2. **Real or Fake Fake Job Posting Prediction** (Shivam Bansal, Kaggle) — a set of ~18,000 job postings tagged as fraudulent or legitimate. We retain only the non-fraudulent listings. Available at `kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction`.

These two datasets are independent — they were not collected as matched pairs. To our knowledge, no publicly available, human-annotated English-language resume-job matching benchmark exists (Yu et al., ConFit, ACM RecSys 2024). We therefore construct our own pairings using category structure and overlap heuristics, an approach consistent with published methodology in this domain.

## What the project ultimately contributes

The project does not end with a production-ready hiring model. The final result is more careful than that.

The strongest contribution is a comparative retrieval study under weak supervision:

- Build a reproducible benchmark from two public datasets
- Compare simple and advanced ranking methods under the same evaluation logic
- Show how quality, interpretability, fairness, and runtime trade off against one another
- Arrive at a conservative business recommendation grounded in the evidence

This is a good fit for BT5153 because the course values not only obtaining scores but also choosing appropriate methods, interpreting them properly, and arriving at useful real-world conclusions.

## How the current repo is organised

The project is now notebook-first.

```text
Group Project/
├── README.md
├── analysis/
│   └── group14_resume_job_matching_end_to_end.ipynb
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
├── results/
│   ├── figures/
│   ├── metrics/
│   └── tables/
├── archive/
│   ├── README.md
│   ├── legacy_docs/
│   ├── legacy_notebooks/
│   ├── legacy_results/
│   └── legacy_scripts/
├── report/
├── report13.pdf
├── presentation13.pptx
├── submission_checklist14.md
├── requirements.txt
└── ORCHESTRATOR_PROMPT.md
```

### Active review path

These are the files and folders that matter most now:

- `analysis/` - the single canonical notebook
- `data/` - raw inputs, processed datasets, and split files
- `results/` - the cleaned set of metrics, figures, and tables used in the final analysis
- `report/` - the LaTeX source for the report
- `report13.pdf` and `presentation13.pptx` - the current deliverables

### Archived material

The following are kept for provenance, not as the main interface:

- `archive/legacy_scripts/src/` - the earlier modular Python implementation
- `archive/legacy_notebooks/notebooks/` - the earlier multi-notebook layout
- `archive/legacy_results/` - older run-specific outputs and verbose traces
- `archive/legacy_docs/` - older code-first documentation

## The end-to-end execution logic

The easiest way to understand the project is to follow the notebook in order. The rest of this document explains that same sequence in plain language.

### Step 0 — Shared assumptions and configuration

The notebook begins by defining project paths, runtime switches, normalized job-family mappings, heuristic skill patterns, and model configuration values.

These settings matter because the whole project is driven by one core design choice: the benchmark is constructed, not given. That means the preprocessing logic is part of the modelling story, not just setup code.

### Step 1 — Build a weakly supervised benchmark

The raw data comes from two public Kaggle sources (see Data Provenance above):

- `data/raw/Resume/Resume.csv` — resumes with category labels
- `data/raw/fake_job_postings.csv` — non-fraudulent job postings

The notebook then:

1. Cleans and normalizes text (removing HTML tags, URLs, and formatting artifacts)
2. Masks obvious personal identifiers (email addresses, phone numbers, named entities)
3. Maps resumes and jobs into ten broad categories using a manually defined taxonomy
4. Extracts heuristic skill indicators from text using regex patterns
5. Constructs resume-job pairs using category structure and overlap heuristics
6. Assigns relevance labels `0`, `1`, and `2`
7. Creates stratified train, validation, and test splits

### What the labels mean

The labels are not recruiter judgments. They are heuristic proxy labels:

- `2` = strong match from the same normalized category, selected by highest lexical and skill overlap
- `1` = partial match from a related category in our taxonomy, with some skill alignment
- `0` = negative pair from a distant category with low semantic overlap

This is the central limitation of the project. It makes the benchmark reproducible and large enough for a course project, but it also means the final results should be interpreted as comparative evidence under weak supervision rather than as a direct estimate of hiring quality.

This category-based proxy approach is well-established in the person-job fit literature. The ConFit paper (Yu et al., ACM RecSys 2024) explicitly acknowledges that "there is no standardized public person-job fit dataset." The Brookings Institution's 2024 occupational matching study and the ResumeAtlas benchmark (ScienceDirect 2024) both use similar category-derived labels when human-annotated match judgments are unavailable.

## What the dataset looks like after preprocessing

Current saved summary:

- **1,309** processed resumes across 10 categories
- **2,119** processed jobs across 10 categories
- **10,472** resume-job pairs
- Label distribution: 3,927 negative (label 0), 2,618 partial (label 1), 3,927 positive (label 2)
- Split distribution: 7,328 train, 1,568 validation, 1,576 test pairs

Why this matters: the benchmark is large enough for a meaningful comparison, it covers multiple domains rather than only one narrow job family, and it remains a proxy dataset so absolute metric values are not the whole story.

## Step 2 — Evaluate rankings using a shared protocol

The notebook treats each job as a query and the resumes in the same split as the candidate pool.

It uses four ranking metrics plus significance testing:

- **Precision@10** — how many of the top 10 results are actually relevant?
- **MRR** — how early does the first relevant candidate appear?
- **NDCG@10** — are better matches ordered nearer the top, accounting for graded relevance?
- **MAP** — average precision across all recall points
- **Paired bootstrap significance tests** — to avoid over-interpreting small raw metric differences

### Business meaning of the metrics

This is worth making explicit for BT5153:

- `Precision@10` asks: if a recruiter only scans the first page of results, how useful is that shortlist?
- `MRR` asks: how quickly does the first relevant candidate appear?
- `NDCG@10` asks: are better matches actually ordered nearer the top?
- Runtime asks: is the quality gain worth the operational cost?

## Step 3 — TF-IDF baseline

The TF-IDF model is the lexical baseline.

What it does: builds sparse unigram/bigram features, represents jobs and resumes in the same vector space, and ranks candidates by cosine similarity.

Why it matters: it is cheap to run, easy to explain, and sets the minimum bar for any more complex model. If a more expensive model cannot beat it by a meaningful margin, the practical answer is to keep the simpler system.

Current full-test result:

- `P@10 = 0.0391`
- `MRR = 0.1409`
- `NDCG@10 = 0.1323`

## Step 4 — SBERT dense retrieval

The SBERT model tests whether sentence-level embeddings (`all-MiniLM-L6-v2`) recover better matches when wording changes but job fit is still similar.

What it does: encodes jobs and resumes into a shared dense embedding space, caches the embeddings for reuse, and ranks by cosine similarity against the same held-out test split used by TF-IDF.

Current full-test result:

- `P@10 = 0.0288`
- `MRR = 0.1103`
- `NDCG@10 = 0.0999`

Interpretation: on this benchmark, SBERT does not beat the lexical baseline. This is a useful negative finding, not a failed experiment. It suggests that under weak supervision and category-heavy matching, semantic generalization alone is not sufficient to outperform explicit keyword overlap.

## Step 5 — LLM reranking

The LLM is not used as a full first-stage ranker. Instead, the notebook:

1. Builds a short lexical shortlist for each job (top-3 by TF-IDF)
2. Sends only that shortlist to LM Studio running a local Gemma model
3. Asks the model to rerank the candidates with structured JSON output
4. Evaluates the reordered shortlist across the full held-out test split

The full-test run uses a compact prompt and concise JSON output so that the runtime remains tractable across all 823 test queries (~56 minutes on the hardware used).

Current full-test LLM result:

- `P@10 = 0.0368`
- `MRR = 0.1365`
- `NDCG@10 = 0.1254`

Interpretation: the LLM clearly outperforms SBERT on the full held-out test split, but TF-IDF still remains slightly stronger overall. The final ranking is TF-IDF > LLM > SBERT. This is why the final recommendation does not claim that the LLM should replace TF-IDF outright.

## Step 6 — Interpretability

Each model family gets a different explanation style appropriate to its architecture:

- **TF-IDF**: top matching terms and shared skills (highly transparent — you can see exactly which words drove the similarity score)
- **SBERT**: perturbation-style saliency — remove a skill phrase and observe how much the embedding similarity drops
- **LLM**: short rationale-style explanations when the model returns structured reasons, though the compact prompt configuration often produces terse fallback outputs

Current active interpretability export:

- `results/tables/interpretability_examples.csv`

Important current limitation:

- the active export now includes LLM rows
- however, the full-test LLM run uses a compact prompt configuration, so the LLM explanation strings are often terse fallback-style outputs rather than rich recruiter-facing rationales
- this should still be treated as a limitation, not as strong interpretability evidence

## Step 7 — Fairness and robustness

The fairness section is a perturbation-based stability check rather than a full fairness certification. The notebook changes demographic-style cues in resumes and measures whether rankings shift:

- **Gendered wording swaps** — replacing masculine pronouns/terms with feminine equivalents and vice versa
- **Name swaps** across four demographic groups (White American, African American, South Asian, East Asian)
- **Age-related wording reduction** — removing years of experience, graduation years, and age indicators

The current audit supports TF-IDF and SBERT. LLM fairness remains under-evaluated.

Current active fairness export:

- `results/tables/fairness_summary.csv`
- `results/tables/fairness_job_detail.csv`

Important limitation:

- the active notebook-first outputs do not support a strong LLM fairness summary claim
- older archived smoke-test artifacts exist, but they should not be treated as equivalent to the active TF-IDF/SBERT audit

## Step 8 — Final comparison and business recommendation

The project has one primary comparison view for all three models: `results/tables/model_comparison_full_test.csv`. All three approaches are now judged on the same held-out test split, which makes the comparison fair and the conclusions more defensible.

## What the results mean overall

The project does not support a simple “more complex is better” conclusion. The more careful reading is:

- TF-IDF is the strongest full-coverage model in this benchmark
- SBERT underperforms the lexical baseline — semantic generalization alone is not enough under category-heavy labels
- The LLM is promising as a reranker and validated on the full held-out test split, but it is still slower than TF-IDF and does not overtake it on the main metrics

This leads to the final business recommendation: use TF-IDF as the first-stage ranking engine, use the LLM only as a selective second-stage reranker for a small shortlist, and do not treat the current benchmark as production-ready evidence.

## How this aligns with BT5153

The project aligns well with the course in several ways:

- A real applied ML problem (recruitment screening) with clear business relevance
- A disciplined comparison of simple and advanced methods under a common evaluation protocol
- Held-out evaluation and paired bootstrap significance testing
- Discussion of model choice, trade-offs, and limitations rather than only score reporting
- Meaningful conclusions that follow from the evidence rather than from assumptions

## Where the project under-delivers

These are the main gaps that matter for evaluation:

1. **Proposal targets were not met.** The proposal aimed for Precision@10 >= 0.80 and MRR >= 0.70. The final benchmark does not reach those targets.

2. **Labels are weak proxies.** The relevance labels come from heuristic construction, not recruiter judgment. This is an acknowledged methodological constraint, not an oversight.

3. **The LLM full-test run uses a compact prompt.** This strengthens the ranking comparison by enabling full coverage, but reduces the richness of the explanation layer.

4. **Absolute scores are low.** The project is strongest as a comparative study, not as a high-performing matching engine.

## Current gaps in project delivery

### Academic and documentation gaps

- `report13.pdf` still needs a clearer reconciliation between the proposal’s success targets and the final contribution.
- The report currently lacks a GitHub/code link because this local repository does not yet have a remote attached.
- Runtime, interpretability, and fairness wording should always be checked against the active `results/` outputs rather than manually copied numbers.

### Submission gaps

- `video14.mp4` is still missing if the video requirement applies.
- `code14.zip` and `data14.zip` should be reviewed before submission because they were created before the latest notebook-readability cleanup and may no longer be the best representation of the current repo.

## Final takeaway

The project is strongest when framed honestly: not as a finished hiring product or proof that the most advanced model wins, but as a disciplined BT5153 project that compares realistic ML options, exposes trade-offs, and arrives at a practical recommendation under imperfect data.

The finding that a simple, interpretable baseline outperforms more complex alternatives under weak supervision is itself a meaningful result. It shows that the choice of method should depend on the quality of available labels and the operational constraints, not only on model sophistication.
