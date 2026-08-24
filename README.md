## Use and permissions

This repository and its contents, including code, processed data, metadata,
analysis outputs, figures, and documentation, accompany a manuscript currently
under review.

Reuse, redistribution, modification, or use of these materials for independent
or derivative research, publications, presentations, or other scholarly or
commercial outputs requires prior written permission from the copyright holders.

**All rights reserved. See `LICENSE` for the full terms.**

# Depth-Associated Soil Bacterial Phyla — Analysis Pipeline

Code and processed data associated with the manuscript:

**"Group-Aware Machine Learning Identifies Depth-Associated Soil Bacterial Phyla Consistent Across Plant Systems and Amendments"**  
Zahra et al. Manuscript under review at *Current Research in Microbial Sciences*.

This repository provides the code and processed data used for the analyses reported in the accompanying manuscript, including 16S rRNA sequence processing, phylum-level compositional analysis, principal component analysis (PCA), treatment-adjusted differential abundance analysis (ANCOM-BC), and group-aware supervised classification using Random Forest, XGBoost, and support vector machines (SVM).

Raw sequence data are **not** stored in this repository. They are deposited at NCBI under **BioProject PRJNA1450284**. Processed data required for the downstream analyses are provided under `data/`.

---

## Repository structure

```text
.
├── README.md
├── requirements.txt              # Python dependencies
├── LICENSE
│
├── data/
│   ├── metadata.tsv              # Sample metadata
│   ├── Allmerged_16Slevel-2.xlsx # Phylum-level feature table (221 samples)
│   └── ancombc_output/           # ANCOM-BC result slices
│       ├── lfc_slice.csv
│       ├── se_slice.csv
│       ├── w_slice.csv
│       ├── p_val_slice.csv
│       └── q_val_slice.csv
│
├── scripts/
│   ├── 01_cutadapt_primer_trim.sh
│   ├── 01_qiime2_processing.sh
│   ├── 02_prepare_phylum_table.sh
│   ├── 03_pca_ordination.py
│   ├── 04_ancombc_qiime2.sh
│   ├── 05_plot_ancombc_figure4.py
│   ├── 06_ml_classification.py
│   ├── 07_rf_feature_extraction.py
│   ├── 08_rf_taxa_boxplots.py
│   ├── 09_stratified_robustness.py
│   └── 10_relative_abundance_donut.py
│
├── output/
│   ├── pca/
│   ├── ml/
│   │   ├── log10_ra/
│   │   └── clr/
│   ├── rf_features/
│   ├── boxplots/
│   └── stratified/
│
└── figures/                      # Final analysis figures
```

---

## Pipeline overview

| Step | Script | Input | Main output |
|---|---|---|---|
| 1. Primer trimming | `01_cutadapt_primer_trim.sh` | Raw FASTQ | Primer-trimmed FASTQ |
| 2. Sequence processing | `01_qiime2_processing.sh` | Primer-trimmed FASTQ | DADA2 feature table, taxonomy, rarefied table |
| 3. Phylum table preparation | `02_prepare_phylum_table.sh` | Rarefied feature table + taxonomy | `data/Allmerged_16Slevel-2.xlsx` |
| 4. PCA ordination | `03_pca_ordination.py` | `data/Allmerged_16Slevel-2.xlsx` | Figure 5 + PCA scores/loadings |
| 5. Differential abundance | `04_ancombc_qiime2.sh` | Phylum-level QIIME 2 table + metadata | `data/ancombc_output/*.csv` |
| 6. ANCOM-BC visualization | `05_plot_ancombc_figure4.py` | `data/ancombc_output/*.csv` | Figure 4 |
| 7. ML classification | `06_ml_classification.py` | `data/Allmerged_16Slevel-2.xlsx` | Tables 3–4, ROC outputs, fitted RF model |
| 8. RF feature extraction | `07_rf_feature_extraction.py` | Phylum table + fitted RF outputs | Figure 6a + RF feature tables |
| 9. RF taxa boxplots | `08_rf_taxa_boxplots.py` | Phylum table + RF feature table | Figure 6b + Upper/Lower statistics |
| 10. Stratified robustness | `09_stratified_robustness.py` | `data/Allmerged_16Slevel-2.xlsx` | Figure 7 + stratified performance tables |
| 11. Relative-abundance composition | `10_relative_abundance_donut.py` | `data/Allmerged_16Slevel-2.xlsx` | Figure 3 |

---

## Key analysis choices

The main analytical choices follow those reported in the manuscript:

- **Rarefaction:** feature table rarefied to an even sequencing depth of **12,000 reads per sample**.
- **Taxonomy:** SILVA 138 taxonomy, aggregated to the **phylum level** for downstream analyses.
- **PCA:** the 25 most abundant bacterial phyla were retained, converted to per-sample relative abundance, and transformed as `log10(relative abundance + 1e-6)` before PCA.
- **Primary ML representation:** `log10(relative abundance + 1e-6)`.
- **ML sensitivity analysis:** centered log-ratio (CLR) transformation.
- **Held-out evaluation:** group-aware train/test splitting at the field-plot level.
- **Cross-validation:** group-aware `StratifiedGroupKFold` with `K = 10`.
- **Grouping variable:** field plots defined by `Species × Amendment × Block`.
- **Leakage control:** samples originating from the same field plot, including paired soil depths and repeated sampling periods, remain together during splitting and cross-validation.
- **CV uncertainty:** Nadeau–Bengio variance correction was used to account for dependence among cross-validation folds.
- **Classifiers:** Random Forest, XGBoost, and SVM were evaluated using complementary performance metrics.
- **Feature prioritization:** Random Forest was used for downstream feature prioritization based on its overall held-out performance and interpretability.
- **ANCOM-BC:** soil profile was evaluated while adjusting for plant species, amendment, and time period.
- **Differential-abundance significance:** Benjamini–Hochberg adjusted `q < 0.05`.
- **Stratified robustness analysis:** the Random Forest model was retrained within individual plant systems and amendment treatments to evaluate the consistency of the depth-classification signal across experimental conditions.

---

## Setup

### Python

Create and activate a Python virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

On Windows:

```bash
venv\Scripts\activate
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Python scripts are used for PCA, figure generation, machine-learning classification, Random Forest feature extraction, taxa boxplots, stratified robustness analysis, and relative-abundance visualization.

---

### QIIME 2

Two QIIME 2 amplicon environments were used during the analysis:

- **Sequence processing / DADA2:** `qiime2-2024.2-amplicon`
- **ANCOM-BC:** `qiime2-amplicon-2024.10`

ANCOM-BC was run through the QIIME 2 `q2-composition` plugin using:

```bash
qiime composition ancombc
```

No separate R installation is required for the QIIME 2 ANCOM-BC workflow used in this analysis.

---

## Reproducing the analysis

Run the following commands from the repository root.

### 1. Primer trimming and sequence processing

With the appropriate QIIME 2 environment active:

```bash
bash scripts/01_cutadapt_primer_trim.sh
bash scripts/01_qiime2_processing.sh
```

### 2. Prepare the phylum-level feature table

```bash
bash scripts/02_prepare_phylum_table.sh
```

The processed phylum-level table used for the downstream analyses is also provided directly as:

```text
data/Allmerged_16Slevel-2.xlsx
```

Therefore, users interested only in reproducing the downstream statistical and machine-learning analyses can begin from this processed table.

---

### 3. PCA ordination

```bash
python scripts/03_pca_ordination.py
```

The PCA workflow uses:

```text
counts
→ phylum-level aggregation
→ per-sample relative abundance
→ top 25 most abundant phyla
→ log10(relative abundance + 1e-6)
→ PCA
```

PCA tables and diagnostic outputs are written to:

```text
output/pca/
```

The main PCA figure is written to:

```text
figures/
```

---

### 4. ANCOM-BC differential abundance

With the QIIME 2 ANCOM-BC environment active:

```bash
bash scripts/04_ancombc_qiime2.sh
```

Generate the corresponding ANCOM-BC figure with:

```bash
python scripts/05_plot_ancombc_figure4.py
```

---

### 5. Machine-learning classification

Run the primary log10-relative-abundance analysis:

```bash
python scripts/06_ml_classification.py
```

The primary ML representation is:

```text
counts
→ per-sample relative abundance
→ log10(relative abundance + 1e-6)
→ group-aware machine learning
```

Random Forest, XGBoost, and SVM are evaluated under the same group-aware validation framework.

To reproduce the CLR sensitivity analysis, set:

```python
NORMALIZATION = "CLR"
```

in `06_ml_classification.py` and rerun the script.

ML models, performance tables, cross-validation diagnostics, and reproducibility outputs are written under:

```text
output/ml/
```

---

### 6. Random Forest feature extraction

After running the primary `LOG10_RA` ML analysis:

```bash
python scripts/07_rf_feature_extraction.py
```

This script loads the fitted Random Forest model and identifies the highest-ranking predictive bacterial phyla.

Random Forest feature importance measures predictive contribution but does not itself encode the direction of association with soil depth. Upper- and Lower-soil labels are therefore assigned separately according to the transformed abundance difference between the two soil-depth groups.

Outputs are written to:

```text
output/rf_features/
```

and the corresponding figure is written to:

```text
figures/
```

---

### 7. Relative-abundance boxplots of RF-prioritized taxa

```bash
python scripts/08_rf_taxa_boxplots.py
```

This script visualizes the relative abundances of Random-Forest-prioritized taxa across Upper and Lower soil.

The boxplot comparisons are descriptive follow-up analyses of the RF-prioritized taxa and are separate from the treatment-adjusted ANCOM-BC differential-abundance analysis.

Statistics are written to:

```text
output/boxplots/
```

and the figure is written to:

```text
figures/
```

---

### 8. Stratified robustness analysis

```bash
python scripts/09_stratified_robustness.py
```

The Random Forest classifier is retrained within individual plant systems and amendment treatments using group-aware cross-validation to assess whether the depth-classification signal remains detectable across experimental conditions.

Outputs are written to:

```text
output/stratified/
```

and the corresponding figure is written to:

```text
figures/
```

---

### 9. Relative-abundance composition figure

```bash
python scripts/10_relative_abundance_donut.py
```

The resulting composition figure is written to:

```text
figures/
```

---

## Output organization

Final manuscript figures are written to:

```text
figures/
```

Intermediate tables, fitted models, PCA outputs, machine-learning diagnostics, feature rankings, and other reproducibility outputs are written under:

```text
output/
```

These generated outputs can be recreated by running the corresponding analysis scripts.

---

## Data availability

- **Raw 16S rRNA sequence data:** NCBI BioProject **PRJNA1450284**.
- **Processed phylum-level abundance table:** `data/Allmerged_16Slevel-2.xlsx`.
- **Sample metadata:** `data/metadata.tsv`.
- **ANCOM-BC result slices:** `data/ancombc_output/`.

The processed data and analysis code provided in this repository support reproduction of the analyses reported in the accompanying manuscript.


---
Code and processed data associated with the manuscript:

**"Group-Aware Machine Learning Identifies Depth-Associated Soil Bacterial Phyla Consistent Across Plant Systems and Amendments"**  
Zahra et al.


---
## Citation

This repository accompanies the following manuscript currently under review:

**Zahra et al. "Group-Aware Machine Learning Identifies Depth-Associated Soil Bacterial Phyla Consistent Across Plant Systems and Amendments."**

A formal journal citation will be added after publication.

No Zenodo DOI is currently associated with this repository.

---

## License

 See `LICENSE`.
