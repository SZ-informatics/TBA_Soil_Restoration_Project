#!/bin/bash
#SBATCH --export=NONE
#SBATCH --job-name=16S_trim
#SBATCH --time=01:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=60G
#SBATCH --output=stdout.%x.%j
#SBATCH --error=stderr.%x.%j
#
# Step 1a - Primer removal (Cutadapt 3.5), run on the raw paired-end FASTQ files.
# Removes the 16S V3-V4 primers 515F (CCTACGGGNGGCWGCAG) and 806R
# (GACTACHVGGGTATCTAATCC); read pairs without both primers are discarded.
# Run once per sequencing batch. Trimmed reads are then imported into QIIME 2
# (qiime tools import, Casava/manifest format) to make the demux.qza used in
# 01_qiime2_processing.sh. Raw reads are on NCBI BioProject PRJNA1450284.

module load GCCcore/11.2.0 cutadapt/3.5

output_dir="trimmed_16S2024"
mkdir -p "$output_dir"

# Sample prefixes = the unique part of each file name before _L001_R1_001.fastq.gz
ls *_L001_R1_001.fastq.gz | sed -E 's/_L001_R1_001.fastq.gz//' | sort | uniq > sampleprefix.txt

for x in $(cat sampleprefix.txt); do
  cutadapt \
    -g CCTACGGGNGGCWGCAG \
    -G GACTACHVGGGTATCTAATCC \
    --discard-untrimmed \
    -o ${x}_out.1.fastq.gz \
    -p ${x}_out.2.fastq.gz \
    ${x}_L001_R1_001.fastq.gz \
    ${x}_L001_R2_001.fastq.gz
done
