# Pharmacogene Functional-Status Prediction

A two-phase machine-learning pipeline for predicting CPIC-style pharmacogene variant functional status (No function / Decreased function / Normal function) from variant-level genomic features, with SHAP explainability and semi-supervised self-training to address label scarcity.

> **Author:** Dharshana-Karthikeyan  
> **License:** MIT  
> **Status:** Research-grade tool. Not validated for clinical use.

---

## What This Tool Does

Standard pathogenicity tools (REVEL, CADD) classify variants as pathogenic or benign; a binary, disease-focused question. Pharmacogenomics needs something different: predicting *how much* a variant disrupts enzyme function, using the CPIC multi-class framework that clinicians use for drug dosing decisions.

This pipeline:
- Pulls pharmacogene variants from ClinVar and CPIC/ClinPGx
- Annotates them with VEP consequences, gnomAD allele frequencies, and evolutionary conservation scores (phyloP, GERP)
- Trains and compares three classifiers (Logistic Regression, Random Forest, XGBoost)
- Explains predictions using SHAP feature importance
- Extends labeled training data via semi-supervised self-training on unlabeled ClinVar variants
- Benchmarks predictions against REVEL and CADD using Spearman correlation

---

## Gene Panel

The pipeline covers 9 pharmacogenes with CPIC star-allele functional annotations:

`CYP2D6` · `CYP2C19` · `CYP2C9` · `CYP2B6` · `CYP3A5` · `SLCO1B1` · `TPMT` · `NUDT15` · `UGT1A1`

---

## Results Summary

**Phase 1   Supervised baseline (N=73 gold-standard labeled variants):**

| Model | Accuracy | Macro F1 |
|---|---|---|
| Logistic Regression | 0.51 ± 0.04 | 0.44 ± 0.04 |
| **Random Forest (selected)** | **0.63 ± 0.05** | **0.50 ± 0.09** |
| XGBoost | 0.59 ± 0.07 | 0.40 ± 0.10 |

**Phase 2   Semi-supervised self-training:**

| | Phase 1 (supervised) | Phase 2 (semi-supervised) |
|---|---|---|
| Gold-standard variants | 73 | 73 |
| Pseudo-labeled added | 0 | 824 |
| Accuracy | 0.63 | **0.74** |
| Macro F1 | 0.50 | **0.66** |
| F1   Decreased function | 0.19 | **0.53** |
| F1   No function | 0.77 | **0.83** |
| F1   Normal function | 0.54 | **0.63** |

**External validation:** REVEL ρ = −0.545 (p < 0.001), CADD ρ = −0.616 (p < 0.001)

**Sensitivity analysis (N=99, relaxed crosswalk):** Label quality outweighs quantity at this sample size   the strict ambiguity exclusion criterion was justified. See project log for full details.

---

## Installation

**Requirements:** Python 3.10 or higher

```bash
git clone https://github.com/Dharshana-Karthikeyan/pharmacogene-ml.git
cd pharmacogene-ml
pip install -r requirements.txt
```

No local databases or reference genome downloads required   all external data is pulled via REST APIs (Ensembl VEP, UCSC, MyVariant.info) at runtime.

---

## Web App

A browser-based interface is available for researchers who prefer not to use the command line.

### Run locally

```bash
streamlit run app.py
```

Opens automatically at `http://localhost:8501`.

### What it does

- Upload a pre-annotated variant CSV (columns: Name, GeneSymbol, most_severe_consequence, vep_impact, gnomad_af, phylop100way, gerp_score)
- Returns predicted functional class + confidence score per variant
- Displays SHAP feature importance chart
- Download results as CSV

> **Note:** Variants must be pre-annotated using pipeline steps 5–6 before uploading. The app performs prediction only   it does not call external APIs.

---

## Usage

### Run the full pipeline

From the project root (`pharmacogene-ml/`):

```bash
python scripts/run_pipeline.py
```

This runs steps 1–9 in order:
1. Downloads ClinVar variant summary (~440 MB, cached after first run)
2. Pulls ClinPGx allele definitions and functional labels
3. Builds the labeled dataset via variant–function crosswalk
4. Annotates variants with VEP, gnomAD AF, phyloP, and GERP
5. Trains and compares three models via Stratified 5-Fold CV
6. Generates SHAP explainability plots
7. Benchmarks against REVEL and CADD

**Note:** Steps involving API calls (VEP, conservation scores) take 2–5 minutes for the full dataset. Progress is printed to the console.

### Run Phase 2 (semi-supervised extension)

After the full pipeline completes:

```bash
cd scripts
python step10_annotate_unlabeled.py   # ~25 minutes   annotates 1224 unlabeled variants
python step11_self_training.py         # ~5 minutes    self-training + honest CV evaluation
python step12_shap_phase2.py           # ~3 minutes    SHAP on Phase 2 model
```

### Run individual steps

Each step can be run independently from `scripts/`:

```bash
cd scripts
python step7_model_comparison.py       # model comparison only
python step8_shap_explainability.py    # SHAP only (requires step 7 outputs)
python step9_external_benchmark.py     # REVEL/CADD benchmark only
```

---

## Output Files

All outputs are saved to `scripts/data/processed/`:

| File | Description |
|---|---|
| `labeled_dataset_with_conservation.csv` | 73-variant gold-standard labeled dataset (primary) |
| `labeled_dataset_expanded.csv` | 99-variant expanded dataset (sensitivity analysis) |
| `step7_model_comparison_results.csv` | Phase 1 model comparison (LR vs RF vs XGBoost) |
| `shap_summary_bar.png` | Overall SHAP feature importance (Phase 1) |
| `shap_beeswarm_*.png` | Per-class SHAP beeswarm plots (Phase 1) |
| `shap_feature_importance.csv` | Raw mean SHAP values per feature per class |
| `step9_external_benchmark_results.csv` | REVEL/CADD scores per variant |
| `step9_correlation_summary.csv` | Spearman correlation results |
| `unlabeled_pool_annotated.csv` | 1224 annotated unlabeled variants (Phase 2) |
| `step11_self_training_results.csv` | Phase 1 vs Phase 2 honest comparison |
| `step11_iteration_log.csv` | Per-iteration pseudo-label counts |
| `step11_confusion_matrix_phase2.png` | Confusion matrix   Phase 2 model |
| `shap_phase2_summary_bar.png` | Overall SHAP feature importance (Phase 2) |
| `shap_phase2_beeswarm_*.png` | Per-class SHAP beeswarm plots (Phase 2) |
| `shap_phase2_feature_importance.csv` | Raw mean SHAP values (Phase 2) |
| `phase2_rf_model.joblib` | Saved Phase 2 trained model (for web app) |

---

## Project Structure

```
pharmacogene-ml/
├── app.py                           # Streamlit web app
├── requirements.txt
├── LICENSE
├── README.md
├── CODE_AVAILABILITY.md
└── scripts/
    ├── config.py                    # constants, paths, gene panels, API URLs
    ├── run_pipeline.py              # orchestrator (steps 1–9)
    ├── step2_clinvar.py             # ClinVar download + panel filter
    ├── step3_clinpgx.py             # ClinPGx allele definitions + function labels
    ├── step4_merge.py               # variant–function crosswalk + labeled dataset
    ├── step5_features.py            # VEP annotation
    ├── step6_conservation.py        # phyloP + GERP conservation scores
    ├── step7_model_comparison.py    # 3-model Stratified CV comparison
    ├── step8_shap_explainability.py # SHAP on Phase 1 Random Forest
    ├── step9_external_benchmark.py  # REVEL/CADD via MyVariant.info
    ├── step10_annotate_unlabeled.py # VEP + conservation for unlabeled pool
    ├── step11_self_training.py      # semi-supervised self-training + evaluation
    ├── step12_shap_phase2.py        # SHAP on Phase 2 model
    └── data/
        ├── raw/                     # ClinVar download + allele definition files
        └── processed/               # all output CSVs and figures
```

---

## Key Design Decisions

- **Pooled multi-gene model**   gene identity excluded as a feature to avoid gene-identity shortcut learning and improve generalizability
- **ClinVar used as benchmark reference only**   never as a training label source (label leakage prevention)
- **Stratified 5-Fold CV**   chosen over train/test split given small labeled N (73 usable variants)
- **Honest semi-supervised evaluation**   CV test folds drawn from gold-standard labeled variants only; pseudo-labeled variants appear in training folds only
- **No local bigWig libraries**   GERP pulled via UCSC REST API after GRCh38→GRCh37 liftover (avoids pyBigWig Windows incompatibility)
- **Increased function excluded from modeling**   n=2, insufficient for stratified CV
- **Ambiguity exclusion criterion**   (gene, rsID) pairs mapping to multiple function terms are excluded rather than majority-voted; sensitivity analysis confirms this was justified

---

## Limitations

- Covers 9 genes only ; ABCG2, VKORC1, DPYD use named-variant rather than star-allele nomenclature; CYP2A6 and CYP4F2 lack populated function terms at source
- Increased function class not modeled (n=2 in labeled dataset)
- Decreased function class remains challenging (Phase 2 F1 = 0.53) due to small labeled sample size
- All training data derived from ClinVar/ClinPGx   not validated on prospectively collected patient variants
- REVEL/CADD comparison is correlation-based (Phase 1); full threshold-based classifier benchmarking is scoped as future work
- Not intended for clinical prescribing decisions without formal validation

---

## External APIs Used

All API calls are made at runtime; no accounts or API keys required:

| API | Purpose |
|---|---|
| [Ensembl VEP REST](https://rest.ensembl.org) | Variant consequence + gnomAD allele frequency |
| [UCSC REST API](https://api.genome.ucsc.edu) | phyloP100way (hg38) + GERP (hg19) conservation scores |
| [Ensembl Map REST](https://rest.ensembl.org) | GRCh38 → GRCh37 liftover for GERP |
| [MyVariant.info](https://myvariant.info) | REVEL and CADD phred scores (dbNSFP) |

---

## Citation

If you use this pipeline in your research, please cite:

> Dharshana-Karthikeyan. *Pharmacogene Functional-Status Prediction via Explainable Machine Learning and Semi-Supervised Self-Training.* 2026. GitHub: https://github.com/Dharshana-Karthikeyan/pharmacogene-ml

---

## License

MIT License; see [LICENSE](LICENSE) for full text.
