#!/bin/bash
#SBATCH --export=NONE
#SBATCH --job-name=qiime2_16S
#SBATCH --time=0-04:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=360G
#SBATCH --output=stdout.%j
#SBATCH --error=stderr.%j
#
# Step 1 - QIIME 2 sequence processing (16S rRNA V3-V4).
# Environment: qiime2-2024.2-amplicon (amplicon distribution 2024.2), run on an HPC cluster.
# Raw reads are on NCBI BioProject PRJNA1450284.
#
# Samples were sequenced in several batches (e.g., Spring2023, Fall2023, Spring2024,
# Fall2024). Each batch was primer-trimmed and denoised separately, then the per-batch
# tables and representative sequences were merged before filtering. For clarity the
# commands below use one consistent set of names; substitute your per-batch file names.

module purge
module load Anaconda3/2023.07-2
source activate /sw/hprc/sw/Anaconda3/2023.07-2/envs/qiime2-2024.2-amplicon

# (1) Primer removal is done first by 01_cutadapt_primer_trim.sh (Cutadapt 3.5) on the
#     raw FASTQ files. The trimmed reads are then imported into QIIME 2:
#         qiime tools import --type 'SampleData[PairedEndSequencesWithQuality]' \
#           --input-format CasavaOneEightSingleLanePerSampleDirFmt \
#           --input-path <trimmed_reads_dir> --output-path demux.qza

# (2) DADA2 denoising (per batch): forward truncated at 270 bp, reverse at 220 bp
qiime dada2 denoise-paired \
  --p-n-threads 48 \
  --i-demultiplexed-seqs demux.qza \
  --o-table table.qza \
  --o-representative-sequences rep-seqs.qza \
  --p-trim-left-f 0 --p-trim-left-r 0 \
  --p-trunc-len-f 270 --p-trunc-len-r 220 \
  --o-denoising-stats denoising-stats.qza

# (3) Merge the per-batch outputs (repeat --i-tables / --i-data for each batch)
qiime feature-table merge \
  --i-tables table.qza \
  --o-merged-table merged-table.qza
qiime feature-table merge-seqs \
  --i-data rep-seqs.qza \
  --o-merged-data merged-rep-seqs.qza

# (4) Taxonomy: SILVA 138 99% V3-V4 Naive Bayes classifier (as in Methods 2.3).
#     The classifier was trained on the V3-V4 region extracted from the SILVA 138 99%
#     reference (Primer v34 = 515F/806R), e.g.:
#         qiime feature-classifier extract-reads \
#           --i-sequences silva-138-99-seqs.qza \
#           --p-f-primer CCTACGGGNGGCWGCAG --p-r-primer GACTACHVGGGTATCTAATCC \
#           --o-reads silva-138-Primerv34-extracted-seqs.qza
#         qiime feature-classifier fit-classifier-naive-bayes \
#           --i-reference-reads silva-138-Primerv34-extracted-seqs.qza \
#           --i-reference-taxonomy silva-138-99-tax.qza \
#           --o-classifier silva-138-Primerv34-classifier.qza
qiime feature-classifier classify-sklearn \
  --p-n-jobs 48 \
  --i-classifier silva-138-Primerv34-classifier.qza \
  --i-reads merged-rep-seqs.qza \
  --o-classification taxonomy.qza

# (5) Feature filtering: keep features present in >=3 samples with total frequency >=3
qiime feature-table filter-features \
  --i-table merged-table.qza \
  --p-min-samples 3 \
  --p-min-frequency 3 \
  --o-filtered-table filt-table.qza

# (6) Taxonomy filtering: keep only features classified to class level (c__) or finer
qiime taxa filter-table \
  --i-table filt-table.qza \
  --i-taxonomy taxonomy.qza \
  --p-include c__ \
  --o-filtered-table filt-class-table.qza

# (7) Alpha-rarefaction curve to confirm an even depth of 12,000 (inspect the plateau)
qiime diversity alpha-rarefaction \
  --i-table filt-class-table.qza \
  --p-max-depth 12000 \
  --m-metadata-file metadata.tsv \
  --o-visualization alpha-rarefaction.qzv

# (8) Rarefy to an even depth of 12,000 reads/sample (drops samples below 12,000)
qiime feature-table rarefy \
  --i-table filt-class-table.qza \
  --p-sampling-depth 12000 \
  --o-rarefied-table rarefied-12000-table.qza

# The phylum-level (SILVA level-2) table used for downstream analysis is produced from
# rarefied-12000-table.qza in step 2 (qiime taxa collapse --p-level 2).
