#!/bin/sh

#####################################
# TODO: Specify SBATCH requirements #
#####################################

AIDA=True

cd ${AIDA_ROOT}
source .venv/bin/activate

task_list="packing-seen-google-objects-seq packing-seen-google-objects-group packing-seen-shapes put-block-in-bowl-seen-colors"
eval_task_list="packing-unseen-google-objects-seq packing-unseen-google-objects-group packing-unseen-shapes put-block-in-bowl-unseen-colors"
task_i=$(($SLURM_ARRAY_TASK_ID / 60 + 1))
task=$(echo $task_list | cut -d ' ' -f $task_i)
eval_task=$(echo $eval_task_list | cut -d ' ' -f $task_i)

srun python ${AIDA_ROOT}/src/aida_cliport/eval.py \
    iteration=$((($SLURM_ARRAY_TASK_ID / 6) % 10)) \
    eval.pier=$AIDA \
    relabeling_demos=$AIDA validation_demos=$AIDA \
    eval_task=$eval_task model_task=$task \
    interactive_demos=$((50 + 50*($SLURM_ARRAY_TASK_ID % 6)))
