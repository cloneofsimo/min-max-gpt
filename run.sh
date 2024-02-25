export WORLD_SIZE=$(nvidia-smi -L | wc -l)
#!/bin/bash

lrs=(4e-4 8e-4 2e-3 4e-3)
widths=(8 16 32 256)

for lr in "${lrs[@]}"; do
    for width in "${widths[@]}"; do
        deepspeed --num_gpus $WORLD_SIZE run_trainer.py --lr $lr --width $width
    done
done
