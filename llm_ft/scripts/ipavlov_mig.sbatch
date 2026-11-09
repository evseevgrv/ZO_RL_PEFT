#!/bin/bash
# One H200 split into 4 MIG parts; each part runs its share of the job list
# sequentially, one run at a time (no time-slicing between processes).
# usage: sbatch scripts/ipavlov_mig.sbatch scripts/jobs/MIG_00.txt [wandb_tags]
#SBATCH -p gpu
#SBATCH --gres=shard:4
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --qos=brainlab-batch
#SBATCH -o /home/andrey_veprikov/zorl_runs/slurm-%j.out
JOB=$1
cd "$HOME/ZO_RL_PEFT/llm_ft"
export PATH="$HOME/venvs/zorl/bin:$PATH" HF_HOME="$HOME/hf_cache" TMPDIR="$HOME/tmp"
export WANDB_TAGS="${2:-rebuttal}" OMP_NUM_THREADS=4
mkdir -p "$TMPDIR"
IFS=',' read -ra DEVS <<< "$CUDA_VISIBLE_DEVICES"
echo "devices: ${DEVS[*]}"
for w in "${!DEVS[@]}"; do
  (
    export CUDA_VISIBLE_DEVICES="${DEVS[$w]}"
    awk -v n="${#DEVS[@]}" -v w="$w" '(NR-1) % n == w' "$JOB" | while read -r line; do
      NAME=$(echo "$line" | awk '{print $NF}')
      [ -f "$HOME/zorl_runs/$NAME/DONE" ] && continue   # resume after requeue
      bash scripts/run_rebuttal.sh $line || echo "FAILED $NAME"
    done
  ) &
done
wait
