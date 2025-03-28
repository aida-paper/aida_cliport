#!/bin/sh

#####################################
# TODO: Specify SBATCH requirements #
#####################################

AIDA=True

cd ${AIDA_ROOT}
source .venv/bin/activate

task_list="packing-seen-google-objects-seq packing-seen-google-objects-group packing-seen-shapes put-block-in-bowl-seen-colors" 
task_i=$(($SLURM_ARRAY_TASK_ID / 10 + 1))
task=$(echo $task_list | cut -d ' ' -f $task_i)

srun python ${AIDA_ROOT}/src/aida_cliport/train_interactive.py \
    iteration=$(($SLURM_ARRAY_TASK_ID % 10)) \
    train_interactive.pier=$AIDA \
    relabeling_demos=$AIDA validation_demos=$AIDA \
    train_interactive_task=$task model_task=$task