# 🪐 Kosmos: AI Scientist — Operational Runbook v2.0

**System Role:** Autonomous Data-Driven Research Engine  
**Standard Runtime:** Up to 12 hours (continuous operation)  
**Output:** Fully cited scientific reports, executable code repository, literature log, and figures

---

## 1. System Overview

**Kosmos** is an autonomous AI scientist that automates data-driven discovery. Unlike standard LLM agents that lose coherence after extended operation, Kosmos uses a **Structured World Model** to maintain consistent research direction across hundreds of agent actions.

### Core Innovation

The Structured World Model is a queryable database of entities, relationships, experimental results, and open questions that persists across the entire run. Information from Cycle 1 remains accessible at Cycle 20. This decouples memory from individual agents and prevents hallucination loops that plague conventional approaches.

### Operational Scale (Per Run)

| Metric | Typical Value |
|--------|---------------|
| Runtime | Up to 12 hours |
| Discovery Cycles | Up to 20 |
| Agent Rollouts | ~200 total (~166 data analysis, ~36 literature) |
| Code Executed | ~42,000 lines of Python |
| Papers Read | ~1,500 full-text papers |
| Output Reports | 3–4 discovery narratives |

---

## 2. Architecture: The World Model and Agents

Kosmos operates as a hub-and-spoke system where a central World Model coordinates two specialized agent types.

### 2.1 The Structured World Model (Hub)

The World Model functions as the shared brain. It stores:

- Current hypotheses under investigation
- Verified facts from data analysis
- Evidence from literature
- Negative results and refuted hypotheses
- Open questions and knowledge gaps

After each agent task completes, summaries of the output are integrated into the World Model. The system then queries this state to generate the next batch of tasks.

### 2.2 The Agents (Spokes)

| Agent Type | Role | Capabilities |
|------------|------|--------------|
| **Data Analysis Agent** | The Coder | Writes, executes, and debugs Python code in a sandboxed environment. Generates plots, statistical tests, and quantitative results. **Self-corrects on errors**: reads tracebacks, fixes code, and re-runs. |
| **Literature Search Agent** | The Reader | Queries external APIs (PubMed, Semantic Scholar, arXiv). Reads full-text papers and extracts evidence snippets to validate or challenge hypotheses. |

Agents operate independently and do not communicate directly with each other. All coordination flows through the World Model.

---

## 3. Pre-Run Configuration

*The Scientist-in-the-Loop Phase*

> **Critical**: Kosmos augments human scientists—it does not replace them. Input quality directly determines output quality. Preliminary runs in development showed significantly different results depending on data preprocessing choices.

### 3.1 Input Data Requirements

| Requirement | Specification |
|-------------|---------------|
| **Format** | Structured tabular data: CSV, TSV, Parquet, Excel, or scientific formats (e.g., `h5ad` for single-cell RNA-seq) |
| **Size Limit** | < 5 GB |
| **Structure** | Tables/matrices with clearly labeled columns. Kosmos struggles with unstructured inputs (raw FASTQ, raw images). |
| **Preprocessing** | Data should be normalized and quality-controlled before input. Include a data dictionary if variable names are non-obvious. |

**Data Hygiene Checklist:**

- [ ] Data is in tabular/matrix format
- [ ] Columns are clearly labeled
- [ ] Appropriate normalization applied (e.g., log-transform)
- [ ] Quality control completed
- [ ] Data dictionary provided (recommended)

### 3.2 Defining the Research Objective

Provide an **open-ended, high-level scientific goal** in natural language.

| Quality | Example | Problem |
|---------|---------|---------|
| ❌ Bad | "Analyze this data." | Too vague—leads to unfocused exploration |
| ❌ Bad | "Run a t-test on Column A vs B." | Too specific—use a script instead |
| ✅ Good | "Identify validated Type 2 diabetes protective mechanisms and prioritize them based on multi-omic evidence." | Clear objective with defined success criteria |
| ✅ Good | "Investigate how environmental parameters during spin coating affect perovskite solar-cell efficiency." | Domain-specific, exploratory, but bounded |
| ✅ Good | "Propose mechanisms contributing to tau accumulation and a temporal sequence of these events." | Requests both findings and methodology |

> **Warning**: Research directions are sensitive to prompt phrasing. Different phrasings of the same objective may yield different exploration paths.

### 3.3 External Dependencies

The Literature Search Agent requires API access to:
- PubMed
- Semantic Scholar  
- arXiv

Ensure network connectivity to these services before initiating a run.

---

## 4. The Autonomous Discovery Loop

Once initiated, Kosmos enters a recursive cycle that repeats up to 20 times or until the objective is satisfied.

### Phase 0: Initialization

1. **Ingestion**: System loads the dataset and parses the research objective
2. **Scoping**: World Model is initialized with broad initial questions (e.g., "What is the distribution of the target variable?", "Search literature for established markers in this domain")

### Phase A: Strategic Planning

1. **Query State**: System queries the World Model to assess current knowledge: *"What hypotheses exist? What evidence supports or refutes them? What gaps remain?"*
2. **Task Generation**: Generates a batch of **up to 10 parallel tasks** distributed across both agent types
   - Example: "Agent 1: Perform pathway enrichment on differentially expressed genes"
   - Example: "Agent 2: Search PubMed for prior work on [Gene X] in [Disease Y]"

### Phase B: Parallel Execution

Agents execute their assigned tasks independently.

**Data Analysis Agent Workflow:**
1. Writes Python script to address the task
2. Executes in sandboxed environment
3. If error occurs → reads traceback → fixes code → re-executes (iterative debugging)
4. Produces outputs: p-values, regression tables, plots, statistical models

**Literature Search Agent Workflow:**
1. Performs semantic search across APIs
2. Reads full-text papers
3. Extracts evidence snippets with citations
4. Returns findings to World Model

### Phase C: World Model Update

Agents return results. The World Model integrates findings using three categories:

| Update Type | Description | Example |
|-------------|-------------|---------|
| **Confirmation** | Data/literature supports existing hypothesis | "Pathway enrichment confirms nucleotide metabolism is affected" |
| **Conflict** | Data contradicts literature (or vice versa) | "Our data shows X, but literature reports Y" → flagged as potential novel finding |
| **Pruning** | Hypothesis refuted or dead end | "No significant correlation found; hypothesis marked as refuted" |

### Phase D: Hypothesis Refinement

Based on updated World Model state, the system formulates more specific questions for the next cycle.

*Example progression:*
- Cycle 3: "SOD2 shows significant association"
- Cycle 7: "Test if SOD2 effect is mediated by oxidative stress pathway"
- Cycle 12: "Search for SOD2 regulatory variants in 3' UTR"

### Phase E: Termination & Synthesis

**Trigger**: Cycle limit reached OR objective satisfied

1. World Model state is frozen
2. System synthesizes 3–4 discovery narratives into scientific reports
3. Citation mapping links every statement to its source

---

## 5. Output Artifacts

At run completion, users receive:

| Artifact | Description |
|----------|-------------|
| **Scientific Reports** | 3–4 discovery narratives in PDF/Markdown format. Each contains ~25 claims based on 8–9 agent trajectories. |
| **Code Repository** | All ~42,000 lines of executable Python code generated during the run (Jupyter notebooks) |
| **Literature Log** | Complete bibliography of all ~1,500 papers read |
| **Figures** | High-resolution plots generated by the Data Analysis Agent |

### Traceability System

Every statement in Kosmos reports includes one of two citation types:

1. **Code Citation**: Hyperlink to the exact Jupyter notebook and line of code that produced the claim
2. **Literature Citation**: Hyperlink to the primary source (DOI) supporting the claim

This enables independent verification of any finding.

---

## 6. Post-Run Validation

*Human Expert Review Protocol*

> **Kosmos is an augmentation tool, not a replacement for scientific judgment.** All findings require human validation.

### 6.1 Accuracy Profile by Statement Type

| Statement Type | Accuracy | Risk Level | Validation Approach |
|----------------|----------|------------|---------------------|
| **Data Analysis** | ~85% | Lower | Click code citation; verify logic, filtering, and statistical approach |
| **Literature Claims** | ~82% | Lower | Click DOI; confirm cited paper actually supports the statement |
| **Interpretation/Synthesis** | ~58% | **HIGH** | Requires domain expertise. Kosmos conflates statistical significance with scientific importance. |

> ⚠️ **Critical Warning**: Nearly half of interpretation claims contain errors. Kosmos tends to make excessively strong claims and may present statistically significant but biologically negligible effects as "major discoveries." Human judgment is essential.

### 6.2 Common Failure Modes

| Behavior | Description | Mitigation |
|----------|-------------|------------|
| **Over-interpretation** | Claims "major discovery" for small effect sizes | Check effect sizes, not just p-values |
| **Invented Metrics** | Creates novel scoring systems (e.g., "Mechanistic Ranking Score") that are mathematically sound but may lack scientific grounding | Verify the metric makes domain sense |
| **Pipeline Pivots** | If a tool fails (e.g., R package crashes), may silently switch methods | Check logs if methodology seems unexpected |
| **Rabbit Holes** | Fixates on statistically significant but scientifically irrelevant signals | Refine initial prompt constraints |

### 6.3 Stochasticity Warning

> **Kosmos is non-deterministic.** Multiple independent runs on the same data and objective may not converge on the same discoveries. If a finding is critical, run Kosmos multiple times and look for convergent results.

### 6.4 Current Interaction Limitations

- **No intermediate interaction**: Scientists cannot currently nudge Kosmos mid-run
- **No external data access**: Kosmos cannot autonomously fetch public datasets for orthogonal validation
- **No raw data processing**: Limited capability with unstructured data (images, raw sequencing files)

---

## 7. Validation Checklist

Before accepting any Kosmos discovery:

- [ ] Clicked code citations and verified computational logic
- [ ] Clicked literature citations and confirmed papers support claims
- [ ] Evaluated effect sizes (not just statistical significance)
- [ ] Assessed biological/scientific plausibility of interpretations
- [ ] Checked for methodology pivots in logs
- [ ] Considered running a replication run for critical findings
- [ ] Had domain expert review synthesis/interpretation claims

---

## 8. Quick Reference

### Expected Performance

| Metric | Value |
|--------|-------|
| Runtime | Up to 12 hours |
| Expert-equivalent time | ~6 months (per collaborator estimates) |
| Valuable findings | Scales linearly with cycle count (tested to 20 cycles) |
| Overall accuracy | 79.4% |
| Data analysis accuracy | 85.5% |
| Literature accuracy | 82.1% |
| Interpretation accuracy | 57.9% |

### Input Requirements Summary

- Structured tabular data < 5GB
- Clear column labels + data dictionary
- Pre-normalized and QC'd
- Open-ended natural language objective

### Output Summary

- 3–4 discovery narratives with full traceability
- ~42,000 lines of executable code
- ~1,500 paper bibliography
- High-resolution figures

---

## Appendix: Domains Validated

Kosmos has produced validated discoveries in:

- Metabolomics (neuroprotection mechanisms)
- Materials Science (perovskite solar cell fabrication)
- Connectomics (neuronal network distributions)
- Statistical Genetics (Mendelian randomization, GWAS)
- Proteomics (Alzheimer's disease temporal ordering)
- Transcriptomics (neuronal aging mechanisms)

The general-purpose architecture (world model + two domain-agnostic agents) enables operation in any data-rich field with structured datasets.

---

*Document version: 2.0*  
*Based on: Mitchener et al., "Kosmos: An AI Scientist for Autonomous Discovery" (arXiv:2511.02824v2)*