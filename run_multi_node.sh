#!/bin/bash
HOST_FILE_PATH=$1
NGPU_PER_NODE=$2
NUM_NODES=$3
SINGLE_RUN=$4 # True or False
export NGPU=$((NGPU_PER_NODE*NUM_NODES))
export MASTER_PORT=$((NGPU+12345))

echo "HOST_FILE_PATH=$HOST_FILE_PATH"
echo "NGPU_PER_NODE=$NGPU_PER_NODE"
echo "NUM_NODES=$NUM_NODES"
echo "NGPU=$NGPU"
echo "MASTER_PORT=$MASTER_PORT"
echo "SINGLE_RUN=$SINGLE_RUN"

if [ "$SINGLE_RUN" = "True" ]; then
    # single run test
    deepspeed --master_port $MASTER_PORT --hostfile $HOST_FILE_PATH --num_nodes ${NUM_NODES} --num_gpus ${NGPU} \
    run_trainer.py \
    --lr 5e-4 \
    --width 16 \
    --run_name "multi_node_test" \
    --print_profile_results
else
    # recursive run
    lrs=(5e-4, 1e-4)
    widths=(16 32 64 128)

    for lr in "${lrs[@]}"; do
        for width in "${widths[@]}"; do
            run_name="lr_${lr}_width_${width}"
            deepspeed --master_port $MASTER_PORT --hostfile $HOST_FILE_PATH --num_nodes ${NUM_NODES} --num_gpus ${NGPU} \
            run_trainer.py \
            --lr $lr \
            --width $width \
            --run_name $run_name
        done
    done
fi