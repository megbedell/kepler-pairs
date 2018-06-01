#! /bin/sh
module load slurm
module load gcc
module load openmpi
mpirun -npernode 1 python find_pairs_mpi.py
