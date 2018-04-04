#!/bin/bash

#PBS -l nodes=1:ppn=8
#PBS -l walltime=1:00:00
#PBS -N Multiplex

module load intel
module load gcc
module load anaconda3/5.0.1

python3 train.py