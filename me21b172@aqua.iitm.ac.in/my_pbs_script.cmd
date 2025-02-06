#!/bin/bash
#PBS -o logfile.log
#PBS -e errorfile.err
#PBS -l walltime=02:00:00
#PBS -l select=1:ncpus=8

tpdir=`echo $PBS_JOBID | cut -f 1 -d .`
tempdir=$HOME/scratch/job$tpdir
mkdir -p $tempdir

cd $tempdir

cp -R $PBS_O_WORKDIR/* .

source /lfs/sware/anaconda3_2023/etc/profile.d/conda.sh

conda activate /lfs/usrhome/btech/me21b172/laser_sintering/my_env

nohup python $PBS_O_WORKDIR/Cluster/aqua_sim.py > output.txt 2>&1&

mv -R $tempdir/* $PBS_O_WORKDIR/

rm -rf $tempdir

