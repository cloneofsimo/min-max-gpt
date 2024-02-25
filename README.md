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

todo : 
llama, test multi-node training, mu-scaling plot, etc.
