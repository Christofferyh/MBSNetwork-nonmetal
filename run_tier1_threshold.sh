#!/bin/bash
#SBATCH --job-name=tier1_threshold
#SBATCH --output=logs/tier1_threshold_%j.out
#SBATCH --error=logs/tier1_threshold_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=11
#SBATCH --mem-per-cpu=1G
#SBATCH --time=01:00:00

set -euo pipefail
cd ~/MBSNetwork

source .venv/bin/activate

set -a
source .env.local
set +a
export PYTHONPATH=src

python3 -m network.tier1_threshold
