#!/bin/bash
#SBATCH --job-name=xx
#SBATCH --time=xx
#SBATCH --nodes=xx
#SBATCH --ntasks=xx
#SBATCH --ntasks-per-node=xx
#SBATCH --mem=xx                                    
#SBATCH -o job.o%J
#SBATCH -e job.e%J

#---------------------- sys setup ----------------------#
module purge
MODULE LOAD
ENV LOAD
PATH LOAD
echo 'JOBID:' $SLURM_JOBID
WORKDIR=`pwd`
#---------------------- job setup ----------------------#
cd ${WORKDIR}
log=${WORKDIR}/job.t$SLURM_JOBID
echo "job id:    "$SLURM_JOBID > ${log}
echo "job start: "`date` >> ${log}
echo "work directory: "$WORKDIR >> $log
topdir=`pwd`

#---------------------- job ----------------------------#



wait

echo "job end:   "`date` >> ${log}

exit

