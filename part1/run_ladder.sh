#!/bin/bash
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --constraint=cpu
#SBATCH --qos=shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=06:00:00
#SBATCH --job-name=sqb_ladder
#SBATCH --output=sqb_ladder_%j.out

# Part 1: the noise ladder, split attachment.
#   sbatch run_ladder.sh split
# Put your own account in the header. About two hours.

module load python
source $SCRATCH/sqb-env/bin/activate
export OMP_NUM_THREADS=1
export SQB_ATTACH=${1:-split}
export SQB_FEATURE=${2:-published}
cd $SLURM_SUBMIT_DIR
echo "SQB_ATTACH = $SQB_ATTACH, SQB_FEATURE = $SQB_FEATURE"
srun python noise_ladder.py
