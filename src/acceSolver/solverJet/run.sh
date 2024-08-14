#!/usr/bin/bash
#SBATCH -J expname        # Job name
#SBATCH -p all          # job partition
#SBATCH -N 1       # Run all processes on a single node
#SBATCH -c 1              # cores per MPI rank
#SBATCH -n 64    # Run a single task

module purge

module load Intel-oneAPI-2022.1
module load compiler mpi mkl
module load petsc/3.14.0 pnetcdf/1.12.2

export I_MPI_FABRICS=shm
ulimit -s unlimited

make clean
make
currentDate=`date +"%Y-%m-%d-%H-%M-%S"`

mpirun -np 64 ./test.exe >> log$currentDate.txt
newDate=`date +"%Y-%m-%d %H:%M:%S"`
echo "--------------- FINISHED --------------" >> log$currentDate.txt
echo $newDate >> log$currentDate.txt
#mpirun -np 4 ./test.exe 
