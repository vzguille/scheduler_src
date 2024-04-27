#!/bin/bash
#SBATCH --job-name=SQS
#SBATCH --time=72:00:00                                             
#SBATCH --nodes=1
#SBATCH --ntasks=xx
#SBATCH --ntasks-per-node=xx
#SBATCH --mem=30G                                     
#SBATCH -o job.o%J
#SBATCH -e job.e%J

#---------------------- sys setup ----------------------#
module purge
ml load intel/2020a
export PATH="/scratch/group/arroyave_lab/guillermo.vazquez/ATAT_mod/bin_atat_mod:$PATH"
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

