#!/bin/bash
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --constraint=cpu
#SBATCH --qos=shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --job-name=sqb_part1
#SBATCH --output=sqb_part1_%j.out

# Part 1: inference, check placement and surface code for one attachment of the rails.
#   sbatch run_part1.sh one      # first attachment
#   sbatch run_part1.sh split    # split attachment
# Put your own account in the header. About ten minutes once the training set is saved.

module load python
source $SCRATCH/sqb-env/bin/activate
export OMP_NUM_THREADS=1
export SQB_ATTACH=${1:-one}
cd $SLURM_SUBMIT_DIR
echo "SQB_ATTACH = $SQB_ATTACH"
srun python dual_rail_structured_bath.py
