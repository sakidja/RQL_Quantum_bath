#!/bin/bash
#SBATCH -A YOUR_ACCOUNT
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -t 03:00:00
#SBATCH -J sqb_part2_check
#SBATCH -o sqb_part2_check_%j.out

# Part 2: check placement and surface code from the baths of the training run.
# No training is repeated. Put your own account in the header.

module load python
source $SCRATCH/sqb-env/bin/activate
export OMP_NUM_THREADS=1
cd $SLURM_SUBMIT_DIR
python free_link_check_placement.py --gamma_sys 0.0 \
       --results results_free_links_final16h_gamma_sys_0.0 \
       --t_clock 1 2 3 4 6 12 25 50
