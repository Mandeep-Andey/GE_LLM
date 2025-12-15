# GE_LLM Research Enhancement Plan
## LLM-Powered Character Network Extraction for Literary Analysis

**Version:** 2.0 Draft  
**Timeline:** 3 Months (1-Month Prototype + 2-Month Refinement)  
**Author:** [Your Name]  
**Advisor:** [Professor's Name]

---

## 📋 Executive Summary

This document outlines the plan to transform GE_LLM from a prototype into a **publication-ready research tool** for extracting and analyzing character interaction networks from literary texts. The enhanced system introduces:

1. **Agentic Validation Pipeline** — Using Gemini as a "judge LLM" to verify extraction quality
2. **Crowdsourced Annotation Framework** — Survey app for building gold-standard datasets
3. **Rigorous Evaluation Metrics** — Precision, recall, F1, and inter-annotator agreement
4. **Reproducible Research Artifacts** — Versioned prompts, model outputs, and benchmarks

---

## 🎯 Research Objectives

### Primary Research Questions

1. **RQ1:** How accurately can LLMs extract character interactions from 19th-century literary prose?
2. **RQ2:** Does a two-stage agentic pipeline (Extractor → Judge) improve extraction quality?
3. **RQ3:** What types of interactions are most/least reliably extracted?
4. **RQ4:** How do network-derived character importance metrics align with literary scholarship?

### Publication Targets

| Venue | Type | Focus | Deadline (Check Latest) |
|-------|------|-------|------------------------|
| CHR (Computational Humanities Research) | Conference | DH + CS methods | ~Summer 2025 |
| Digital Scholarship in the Humanities | Journal | DH applications | Rolling |
| EMNLP | Conference | NLP methods | ~May 2025 |
| ACL | Conference | NLP methods | ~Feb 2025 |

---

## 🏗️ System Architecture (Enhanced)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           AGENTIC EXTRACTION PIPELINE v2.0                       │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   ┌────────────────┐                                                            │
│   │   Raw Text     │                                                            │
│   │ (Middlemarch)  │                                                            │
│   └───────┬────────┘                                                            │
│           │                                                                     │
│           ▼                                                                     │
│   ┌────────────────┐     ┌────────────────┐     ┌────────────────┐              │
│   │   Preprocessor │────▶│   EXTRACTOR    │────▶│   Candidate    │              │
│   │   (Chunking)   │     │   (qwen3:8b)   │     │   Interactions │              │
│   └────────────────┘     │   [Local LLM]  │     └───────┬────────┘              │
│                          └────────────────┘             │                       │
│                                                         ▼                       │
│                          ┌────────────────┐     ┌────────────────┐              │
│                          │     JUDGE      │◀────│   Batch for    │              │
│                          │ (gemini-1.5-pro)│     │   Validation   │              │
│                          │   [API LLM]    │     └────────────────┘              │
│                          └───────┬────────┘                                     │
│                                  │                                              │
│           ┌──────────────────────┼──────────────────────┐                       │
│           ▼                      ▼                      ▼                       │
│   ┌────────────────┐     ┌────────────────┐     ┌────────────────┐              │
│   │    ACCEPT      │     │  NEEDS REVIEW  │     │    REJECT      │              │
│   │  (score ≥ 0.7) │     │ (0.3 < s < 0.7)│     │  (score ≤ 0.3) │              │
│   └───────┬────────┘     └───────┬────────┘     └────────────────┘              │
│           │                      │                                              │
│           │              ┌───────▼────────┐                                     │
│           │              │  HUMAN REVIEW  │◀─── Crowdsourced                    │
│           │              │  (Survey App)  │     Annotations                     │
│           │              └───────┬────────┘                                     │
│           │                      │                                              │
│           └──────────┬───────────┘                                              │
│                      ▼                                                          │
│              ┌────────────────┐     ┌────────────────┐     ┌────────────────┐   │
│              │  FINAL GRAPH   │────▶│   ANALYSIS     │────▶│   REPORTS &    │   │
│              │  (NetworkX)    │     │   (Centrality) │     │   VISUALIZATIONS│   │
│              └────────────────┘     └────────────────┘     └────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📅 Timeline & Milestones

### Phase 1: Prototype (Weeks 1-4) — *Proving Viability*

| Week | Focus | Deliverables | Status |
|------|-------|--------------|--------|
| 1 | Gemini Judge Integration | `gemini_judge.py`, `run_judge_pipeline.py` | ✅ Done |
| 2 | Evaluation Framework | `evaluation_metrics.py`, sample evaluations | ✅ Done |
| 2 | Annotation Survey App | `annotation_survey_app.py`, deployed URL | ✅ Done |
| 3 | Initial Annotations | 2-3 chapters manually annotated | ⏳ Pending |
| 4 | Prototype Demo | Working pipeline, preliminary metrics | ⏳ Pending |

**Milestone 1 Deliverable:** Demo to professor showing:
- Full pipeline running on 1 book
- Judge acceptance/rejection rates
- Interactive visualization
- Survey app with sample annotations

### Phase 2: Annotation Campaign (Weeks 5-8)

| Week | Focus | Deliverables |
|------|-------|--------------|
| 5-6 | Survey Distribution | Share with DH community, collect responses |
| 7 | Quality Control | Remove low-quality annotations, compute agreement |
| 8 | Gold Standard | Finalized gold annotations for 5+ chapters |

**Milestone 2 Deliverable:** Gold-standard dataset with inter-annotator agreement > 0.6 Kappa

### Phase 3: Evaluation & Paper (Weeks 9-12)

| Week | Focus | Deliverables |
|------|-------|--------------|
| 9 | Comprehensive Evaluation | Full metrics across all books |
| 10 | Error Analysis | Categorized failure modes, improvement ideas |
| 11 | Paper Drafting | Introduction, Methods, Results |
| 12 | Paper Refinement | Final draft, professor review |

**Milestone 3 Deliverable:** Paper draft ready for submission

---

## 📊 Evaluation Framework

### Metrics to Report

| Metric | Description | Target |
|--------|-------------|--------|
| **Precision** | Correct extractions / Total extractions | > 0.75 |
| **Recall** | Correct extractions / Total in gold | > 0.70 |
| **F1 Score** | Harmonic mean of P & R | > 0.72 |
| **Cohen's Kappa** | Inter-annotator agreement | > 0.60 |
| **Judge Accuracy** | Judge agrees with gold standard | > 0.80 |

### Comparison Baselines

1. **Naive Co-occurrence** — Simple NER + sliding window co-occurrence
2. **SpaCy NER Only** — No LLM, just entity extraction
3. **Zero-shot GPT-4** — Single powerful LLM, no agentic pipeline
4. **Our System** — Extractor + Judge agentic pipeline

---

## 💰 Resource Requirements

### Compute Resources

| Resource | Provider | Cost | Notes |
|----------|----------|------|-------|
| Extractor LLM | Ollama (local) | $0 | qwen3:8b on local GPU/CPU |
| Judge LLM | Gemini API | $0 | Free tier: 15 RPM, 1M tokens/day |
| Survey Hosting | Streamlit Cloud | $0 | Free tier sufficient |

### Time Investment

| Task | Hours | Who |
|------|-------|-----|
| Code development | 40 | Student |
| Manual annotation (seed) | 10 | Student |
| Survey management | 5 | Student |
| Paper writing | 30 | Student + Advisor |
| **Total** | **85** | |

---

## 🔬 Novel Contributions

For publication, we claim the following contributions:

1. **Agentic LLM Pipeline for Literary NLP**
   - First (to our knowledge) use of judge LLM for validating literary entity extraction
   - Demonstrates improved precision through multi-model validation

2. **Crowdsourced Literary Annotation Framework**
   - Reproducible methodology for building gold-standard literary datasets
   - Open-source survey tool for DH community

3. **Middlemarch Character Network Benchmark**
   - First comprehensive character interaction dataset for this novel
   - Enables future research comparison

4. **Empirical Analysis of LLM Capabilities on 19th-Century Prose**
   - Documented failure modes (archaic language, pronoun resolution, etc.)
   - Practical recommendations for literary NLP

---

## 🚧 Known Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Low survey participation | Medium | High | Partner with multiple DH groups, offer co-authorship |
| Gemini rate limits | Low | Medium | Use sampling (20% judge rate), batch efficiently |
| Low inter-annotator agreement | Medium | High | Provide detailed guidelines, training examples |
| LLM hallucinations | Medium | Medium | Judge layer catches most; human review for edge cases |

---

## 📁 Repository Structure (Updated)

```
GE_LLM/
├── src/
│   ├── character_mapper.py     # Entity resolution
│   ├── data_preprocessor.py    # Text loading
│   ├── graph_manager.py        # Graph construction & analysis
│   ├── llm_client.py           # Ollama extractor client
│   ├── prompt_manager.py       # Prompt engineering
│   ├── schemas.py              # Pydantic models
│   ├── settings.py             # Configuration
│   ├── gemini_judge.py         # NEW: Judge LLM integration
│   └── evaluation_metrics.py   # NEW: Precision/Recall/F1/Kappa
│
├── run_llm_extraction.py       # Main extraction pipeline
├── run_judge_pipeline.py       # NEW: Judge validation pipeline
├── build_graph.py              # Graph artifact builder
├── analyze_graph.py            # Analysis & visualization
├── annotation_survey_app.py    # NEW: Streamlit annotation app
│
├── gold_annotations/           # NEW: Human-annotated ground truth
├── crowd_annotations/          # NEW: Crowdsourced annotations
├── config.yaml                 # Configuration (extended)
└── research_plan.md            # This document
```

---

## ✅ Next Steps (Immediate)

1. **[ ] Get Gemini API Key**
   - Visit: https://makersuite.google.com/app/apikey
   - Set: `export GEMINI_API_KEY='your-key'`

2. **[ ] Test Judge Pipeline**
   ```bash
   uv sync
   uv run run_judge_pipeline.py book_1 --sample-rate 0.1
   ```

3. **[ ] Manually Annotate 1 Chapter**
   - Create `gold_annotations/book_1_chapter_001.json`
   - Use same schema as LLM output

4. **[ ] Deploy Survey App**
   ```bash
   uv pip install streamlit
   streamlit run annotation_survey_app.py
   ```

5. **[ ] Schedule Meeting with Professor**
   - Show working pipeline
   - Discuss publication strategy
   - Get feedback on research questions

---

## 📚 References

1. Zheng et al. (2023). "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena." NeurIPS.
2. Elson et al. (2010). "Extracting Social Networks from Literary Fiction." ACL.
3. Underwood (2019). "Distant Horizons: Digital Evidence and Literary Change." UChicago Press.
4. Bamman et al. (2014). "A Bayesian Mixed Effects Model of Literary Character." ACL.

---

*Document created: December 2024*
*Last updated: December 15, 2024*
