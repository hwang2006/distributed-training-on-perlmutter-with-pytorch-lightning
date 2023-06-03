#!/bin/bash
#SBATCH -A dasrepo_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 2:00:00
#SBATCH -N 2
#SBATCH -c 32
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4 
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err

export SLURM_CPU_BIND="cores"

module load  cudnn/8.3.2  nccl/2.17.1-ofi  evp-patch
source ~/.bashrc
conda activate craympi-hvd

#srun python pt_bert_nsmc_lightning.py --num_nodes 2
srun python distributed-training-on-perlmutter-with-pytorch-lightning/src/pt_bert_nsmc_lightning.py --num_nodes 2
