#!/bin/bash
#SBATCH --job-name=score_new
#SBATCH --partition=normal
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/scratch/users/anaennis/aion-tokenizer-diagnostic/adapt-test/logs/score_new_%j.log
# Entry 19: score vitl_sat, vitl_sat_ckpt, vitl_web (slot 0); settings as $SCRATCH/jobs/score3b.sbatch
set -euo pipefail
ml purge; ml python/3.12.1
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 AION_DATA=$SCRATCH/aion-tokenizer-diagnostic/data HF_HUB_OFFLINE=1
source $HOME/.venvs/aion-cpu/bin/activate
[ -e /scratch/users/anaennis/aion-tokenizer-diagnostic/adapt-test/score_new/p3_results_new.json ] && { echo "STOP: score_new/ output exists"; exit 2; }
hostname; date
python /scratch/users/anaennis/aion-tokenizer-diagnostic/adapt-test/p3_score_new.py
date
