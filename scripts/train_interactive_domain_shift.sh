#!/bin/sh

#####################################
# TODO: Specify SBATCH requirements #
#####################################

PIER=True
task="put-block-in-bowl-seen-colors" 

srun python ${AIDA_ROOT}/src/aida_cliport/train_interactive_domain_shift.py \
    iteration=$SLURM_ARRAY_TASK_ID \
    train_interactive.pier=${PIER} \
    relabeling_demos=True validation_demos=True \
    train_interactive_task=$task model_task=$task \
    exp_folder=exps_domain_shift \
    interactive_demos=450
