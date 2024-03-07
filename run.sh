#!/bin/bash
export WORLD_SIZE=$(nvidia-smi -L | wc -l)
export SINGLE_RUN=False # True or False

#export WANDB_API_KEY= # Hey this is yours.
export WANDB_PROJECT="muphead_widthsweep"

if [ "$SINGLE_RUN" = "True" ]; then
    # single run test
    deepspeed --num_gpus $WORLD_SIZE \
    run_trainer.py \
    --learning_rate 5e-4 \
    --head_width 16 \
    --run_name "single_node_test" \
    --print_profile_results
else
    # recursive run
    lrs=(1e-3 2e-3 4e-3 8e-3 2e-2 4e-2)
    head_widths=(16 32 64 128)

    for lr in "${lrs[@]}"; do
        for head_width in "${head_widths[@]}"; do
            deepspeed --num_gpus $WORLD_SIZE run_trainer.py --learning_rate $lr --head_width $head_width
        done
    done
fi
