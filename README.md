# Depth-Associated Soil Bacterial Phyla — Analysis Pipeline

Code and processed data for the manuscript:

**"Group-Aware Machine Learning Identifies Depth-Associated Soil Bacterial Phyla Consistent Across Plant Systems and Amendments"**
Zahra et al., *Current Research in Microbial Sciences*.

This repository reproduces the full analysis: 16S rRNA sequence processing, phylum-level
compositional analysis, ordination (PCA), treatment-adjusted differential abundance
(ANCOM-BC), and group-aware supervised classification (Random Forest, XGBoost, SVM),
together with the figures reported in the paper.

Raw sequence data are **not** stored here; they are deposited at NCBI under
**BioProject PRJNA1450284**. This repository begins from the processed feature table.

---

## Repository structure

```
.
├── README.md
├── requirements.txt              # Python dependencies
├── LICENSE
├── data/
│   ├── metadata.tsv              # sample metadata: Species, Amendment, Block, SoilProfile, Time Period
│   ├── Allmerged_16Slevel-2.xlsx # phylum-level (SILVA level-2) feature table (221 samples)
│   └── ancombc_output/           # ANCOM-BC result slices
│       ├── lfc_slice.csv
│       ├── se_slice.csv
│       ├── w_slice.csv
│       ├── p_val_slice.csv
│       └── q_val_slice.csv
├── scripts/
│   ├── 01_cutadapt_primer_trim.sh           # Cutadapt 3.5 primer removal on raw FASTQ
│   ├── 01_qiime2_processing.sh              # import→DADA2→merge→SILVA→filter→rarefy 12,000
│   ├── 02_prepare_phylum_table.sh           # collapse rarefied table to phylum (level 2), export
│   ├── 03_pca_ordination.py                 # PCA (top-25 phyla, log10), scores + loadings (Figure 5)
│   ├── 04_ancombc_qiime2.sh                 # treatment-adjusted ANCOM-BC (QIIME 2) → result slices
│   ├── 05_plot_ancombc_figure4.py           # ANCOM-BC log-fold-change bar plot (Figure 4)
│   ├── 06_ml_classification.py              # RF / XGBoost / SVM, group-aware CV, diagnostics (Tables 3–4)
│   ├── 07_rf_feature_extraction.py          # RF feature importances by depth (Figure 6a)
│   ├── 08_rf_taxa_boxplots.py               # relative-abundance boxplots of RF taxa (Figure 6b)
│   ├── 09_stratified_robustness.py          # retrain RF within plant type / amendment (Figure 7)
│   └── 10_relative_abundance_donut.py       # phylum relative-abundance donut (Figure 3)
└── figures/                                 # output figures
```

---

## Pipeline overview

| Step | Script | Input | Output |
|------|--------|-------|--------|
| 1. Sequence processing | `01_qiime2_processing.sh` | Raw FASTQ (NCBI PRJNA1450284) | Rarefied feature table, taxonomy |
| 2. Phylum table | `02_prepare_phylum_table.sh` | Rarefied table, taxonomy | phylum table -> `Allmerged_16Slevel-2.xlsx` |
| 3. Ordination (PCA) | `03_pca_ordination.py` | `phylum_table.xlsx` | Figure 5 |
| 4. Differential abundance | `04_ancombc_qiime2.sh` | Phylum table (.qza), metadata | `ancombc_output/*.csv` |
| 5. ANCOM-BC figure | `05_plot_ancombc_figure4.py` | `ancombc_output/*.csv` | Figure 4 |
| 6. ML classification | `06_ml_classification.py` | phylum table | Tables 3–4, ROC, saved RF model |
| 7. Feature importance | `07_rf_feature_extraction.py` | phylum table + step 6 output | Figure 6a |
| 8. Taxa boxplots | `08_rf_taxa_boxplots.py` | phylum table + step 7 output | Figure 6b |
| 9. Stratified robustness | `09_stratified_robustness.py` | phylum table | Figure 7, `stratified_results.csv` |
| 10. Composition figure | `10_relative_abundance_donut.py` | phylum table | Figure 3 |

Key analysis choices (as reported in the manuscript):

- **Rarefaction** to an even depth of **12,000 reads per sample**.
- **Taxonomy**: SILVA 138 (V3–V4), collapsed to **phylum level**.
- **Normalization**: log10-transformed relative abundances (primary) and CLR (parallel).
- **Cross-validation**: group-aware `StratifiedGroupKFold` (K = 10), groups defined by
  Species × Amendment × Block (field plot), so all samples from a plot stay in one fold.
- **CV confidence intervals**: Nadeau–Bengio correction for dependent folds.
- **ANCOM-BC**: soil profile with plant species, amendment, and time period as covariates;
  significance at Benjamini–Hochberg q < 0.05.

---

## Setup

### Python (steps 2, 3, 5, 6, 7, 8)

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### QIIME 2 (steps 1 and 4)

Two QIIME 2 **amplicon** environments were used (install per https://docs.qiime2.org):

- **Step 1 (sequence processing / DADA2):** `qiime2-2024.2-amplicon`, run on an HPC
  cluster via SLURM (see the header of `01_qiime2_processing.sh`).
- **Step 4 (ANCOM-BC):** `qiime2-amplicon-2024.10`, run locally.

ANCOM-BC is invoked through QIIME 2's `q2-composition` plugin
(`qiime composition ancombc`); no separate R installation is required.

---

## Reproducing the analysis

Run the scripts in order:

```bash
# Steps 1 and 4 (QIIME 2 environment active)
bash scripts/01_qiime2_processing.sh
bash scripts/04_ancombc_qiime2.sh

# Steps 2, 3, 5–8 (Python venv active)
bash scripts/02_prepare_phylum_table.sh
python scripts/03_pca_ordination.py
python scripts/05_plot_ancombc_figure4.py
python scripts/06_ml_classification.py
python scripts/07_rf_feature_extraction.py
python scripts/08_rf_taxa_boxplots.py
python scripts/09_stratified_robustness.py
python scripts/10_relative_abundance_donut.py
```

Figures are written to `figures/`.

---

## Data availability

- **Raw 16S rRNA sequences**: NCBI BioProject **PRJNA1450284**.
- **Processed phylum table and ANCOM-BC output**: this repository (`data/`).

## Citation

If you use this code or data, please cite the manuscript above and this repository
(release DOI, once minted via Zenodo).

## License

Released under the MIT License (see `LICENSE`).
