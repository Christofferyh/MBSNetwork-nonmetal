#!/bin/bash
#SBATCH --job-name=mbs_global_alignment
#SBATCH --output=logs/global_alignment_%j.out
#SBATCH --error=logs/global_alignment_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1G
#SBATCH --time=00:30:00

set -euo pipefail

cd ~/MBSNetwork

source .venv/bin/activate

set -a
source .env.local
set +a
export PYTHONPATH=src

# Keep Simensen's original reference output safe before we overwrite it
cp -n data/alignments/preliminary/global_registration.npy \
      data/alignments/preliminary/global_registration.original.npy

python3 -m alignment.global
