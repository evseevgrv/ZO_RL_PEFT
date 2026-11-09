#!/bin/bash
# One whole H200 runs one job list with PAR concurrent runs.
# usage: sbatch scripts/ipavlov_bc.sbatch scripts/jobs/BC_light_00.txt 6 [wandb_tags]
#SBATCH -p gpu
#SBATCH --gres=gpu:h200:1
#SBATCH -c 16
#SBATCH --mem=40G
#SBATCH --time=3-00:00:00
#SBATCH --qos=brainlab-batch
#SBATCH -o /home/andrey_veprikov/zorl_runs/slurm-%j.out
JOB=$1 PAR=$2
cd "$HOME/ZO_RL_PEFT/llm_ft"
export PATH="$HOME/venvs/zorl/bin:$PATH" HF_HOME="$HOME/hf_cache" TMPDIR="$HOME/tmp"
export WANDB_TAGS="${3:-rebuttal}"
mkdir -p "$TMPDIR"
nvidia-smi -L
xargs -P "$PAR" -L 1 bash scripts/run_rebuttal.sh < "$JOB"
