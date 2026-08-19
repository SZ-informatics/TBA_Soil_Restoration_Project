#!/usr/bin/env bash
# Step 4 — Treatment-adjusted differential abundance (ANCOM-BC via QIIME 2 q2-composition).
# Run inside the QIIME 2 amplicon environment:  conda activate qiime2-amplicon-2024.10
#
# Inputs:
#   table_phylum_matched.qza   phylum-level feature table, samples matched to metadata
#   Merged4_Metadata.txt       sample metadata (QIIME 2 format)
# Model:
#   SoilProfile + Species + Amendment + TimePeriod, with Upper (U) as the reference
#   level -> a positive log-fold change means enrichment in the Lower (15-30 cm) profile.
# Outputs:
#   differential_depth_phylum_all/differentials.qza
#   differential_depth_phylum_all_export/{lfc,se,w,p_val,q_val}_slice.csv (+ datapackage.json)
#   -> these slice CSVs feed 05_plot_ancombc_figure4.py (Figure 4)

set -euo pipefail

qiime composition ancombc \
  --i-table table_phylum_matched.qza \
  --m-metadata-file Merged4_Metadata.txt \
  --p-formula "SoilProfile + Species + Amendment + TimePeriod" \
  --p-reference-levels "SoilProfile::U" \
  --output-dir differential_depth_phylum_all

qiime tools export \
  --input-path differential_depth_phylum_all/differentials.qza \
  --output-path differential_depth_phylum_all_export
