#!/bin/bash
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --time=06:00:00
#SBATCH --job-name=sqb_part2_final
#SBATCH --output=sqb_part2_final_%j.out

# Part 2: training, bounds and fit with every link free.
# Put your own account in the header. About ninety minutes on one node.

module load python
source $SCRATCH/sqb-env/bin/activate
export OMP_NUM_THREADS=1
cd $SLURM_SUBMIT_DIR
python free_link_training.py --gamma_sys 0.0 --n_train 4000 --n_test 49 \
       --n_pc 30 --workers 120 --full_window --n_starts 16 \
       --vertical_range 0.1 0.5 \
       --outdir results_free_links_final16h_gamma_sys_0.0
