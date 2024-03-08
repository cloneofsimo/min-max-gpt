# min-max-gpt : minGPT that scales

So you've looked at [minGPT](https://github.com/karpathy/minGPT). Now its time to scale.

<p align="center">
  <img src="mmgpt.png">
</p>


This codebase provides..
* muP-initialization & learning rate settings, with tweakable gpt code.
* mixed precision trianing, FSDP with deepspeed zero-3
* ZERO-HUGGINGFACE : If you are interested in not-relying-on-huggingface for training, this is it. This doesn't use accelerate, transformer, etc, so you have maximal control. (still uses `default_data_collator` and `get_scheduler`, but its really easy to get rid of)
* Good Deepspeed codebase.

---
# Usage

- Install:

```bash
git clone https://github.com/cloneofsimo/min-max-gpt
cd min-max-gpt
pip install -r requirements.txt
```

- Single Node training:

```bash
ROOT_DIR=/path/to/dir/min-max-gpt
cd $ROOT_DIR
export WORLD_SIZE=$(nvidia-smi -L | wc -l)
deepspeed --num_gpus $WORLD_SIZE run_trainer.py --learning_rate 1e-4 --head_width 32 --run_name "test"
```

- Multi-node training:

You should create hostfile and specify environemnt variables.
Nodes must be able to access each other with passwordless SSH.

```bash
$ cat /path/to/dir/min-max-gpt/hostfile
xxx.xxx.xxx.xxx slots=8 #node0 ip, #num_gpus
xxx.xxx.xxx.xxx slots=8 #node1 ip, #num_gpus
```

```bash
ROOT_DIR=/path/to/dir/min-max-gpt
cd $ROOT_DIR
export HOST_FILE_PATH=$ROOT_DIR/hostfile
export NGPU_PER_NODE=?? # e.g. 8
export NUM_NODES=?? # e.g. 2
export SINGLE_RUN=True
./run_multi_node.sh $HOST_FILE_PATH $NGPU_PER_NODE $NUM_NODES $SINGLE_RUN
```

---

## Current Test Status:

* Tested on 8x 80GB A100 GPUs, can train 20B models.


---

todo : 
- [ ] multi-node training
- [ ] mu-scaling plot
- [ ] test on llama 