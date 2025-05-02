#!/bin/bash
# ---------------------------------------------------------------------
# SLURM script for a job on a Compute Canada cluster. 
# 1h < job duration < 168h
# limit of 1000 jobs (queued and running, per user)
# ---------------------------------------------------------------------
#SBATCH --mem-per-cpu=8000   # memory; default unit is megabytes; aim for 120%
#SBATCH --time=2-00:00      # time (DD-HH:MM); aim for 120%
#SBATCH --output=%x-%j.out  # output/error of job: %x is jobname, %j is job ID number; can also specify location
#SBATCH --mail-user=nadia.ghernaout@grtgaz.com
#SBATCH --mail-type=ALL
# ---------------------------------------------------------------------
echo "Current working directory: `pwd`"
echo "Starting run at: `date`"
# ---------------------------------------------------------------------
module purge
module load python/3.11.5 scipy-stack/2023b mycplex/12.10.0

python scheduling.py  

# ---------------------------------------------------------------------
echo "Job finished with exit code $? at: `date`"
# ----------------------------------------------------