export CUDA_VISIBLE_DEVICES=4
export WORLD_SIZE=1
#!/bin/bash

lrs=(4e-4 8e-4 2e-3 4e-3)
widths=(8 16 32 256)

for lr in "${lrs[@]}"; do
    for width in "${widths[@]}"; do
        deepspeed --num_gpus 4 run_trainer.py --lr $lr --width $width
    done
done
