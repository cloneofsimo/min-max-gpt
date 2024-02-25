export WORLD_SIZE=$(nvidia-smi -L | wc -l)
#!/bin/bash

lrs=(1e-4 2e-4)
widths=(16 32 64)

for lr in "${lrs[@]}"; do
    for width in "${widths[@]}"; do
        run_name="lr_${lr}_width_${width}"
        deepspeed --num_gpus $WORLD_SIZE run_trainer.py --lr $lr --width $width --run_name $run_name
    done
done
