#!/bin/bash
export WORLD_SIZE=$(nvidia-smi -L | wc -l)
export SINGLE_RUN=True # True or False

if [ "$SINGLE_RUN" = "True" ]; then
    # single run test
    deepspeed --num_gpus $WORLD_SIZE \
    run_trainer.py \
    --lr 5e-4 \
    --width 16 \
    --run_name "single_node_test" \
    --print_profile_results
else
    # recursive run
    lrs=(5e-4, 1e-4)
    widths=(16 32 64 128)

    for lr in "${lrs[@]}"; do
        for width in "${widths[@]}"; do
            run_name="lr_${lr}_width_${width}"
            deepspeed --num_gpus $WORLD_SIZE run_trainer.py --lr $lr --width $width --run_name $run_name
        done
    done
fi
