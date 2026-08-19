#!/usr/bin/env bash
# Step 2 - Collapse the rarefied ASV table to phylum level (SILVA level 2).
# Run in the QIIME 2 environment (qiime2-amplicon-2024.10 or 2024.2).
#
# Input:
#   rarefied-12000-table.qza  (from 01_qiime2_processing.sh)
#   taxonomy.qza              (SILVA V3-V4 classification from step 1)
# Output:
#   table_phylum.qza          phylum-level feature table (level 2)
#   table_phylum.tsv          exported table
#
# The exported phylum table was then transposed (samples as rows, phyla as columns)
# and merged with the sample metadata (Species, Amendment, Block, SoilProfile,
# Time Period, and the Group1/Group2 plot identifiers) to produce
#   data/Allmerged_16Slevel-2.xlsx
# which is the input for the Python analysis scripts (03, 06-10).

set -euo pipefail

qiime taxa collapse \
  --i-table rarefied-12000-table.qza \
  --i-taxonomy taxonomy.qza \
  --p-level 2 \
  --o-collapsed-table table_phylum.qza

qiime tools export \
  --input-path table_phylum.qza \
  --output-path table_phylum_export

biom convert \
  -i table_phylum_export/feature-table.biom \
  -o table_phylum.tsv \
  --to-tsv
