# very much based on deepspeed-examples.
# https://github.com/microsoft/DeepSpeedExamples/blob/master/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/main.py

import json
import math

import click
import deepspeed
import matplotlib.pyplot as plt
import torch
from datasets import load_dataset
from deepspeed import get_accelerator
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from deepspeed.utils import logger
from torch.utils.data import (DataLoader, Dataset, RandomSampler,
                              SequentialSampler)
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, default_data_collator, get_scheduler

import wandb
from memory_profile_utils import print_memory_with_message
from tweakablegpt import GPTConfig, GPTModel


class WikiTextDataset(Dataset):
    def __init__(self, tokenizer, type_path="train", max_length=512, debug=False):
        if debug:
            vernum = 2
        else:
            vernum = 103
        self.vernum = vernum
        self.dataset = load_dataset(
            "wikitext", f"wikitext-{vernum}-raw-v1", split=type_path
        )
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return int(len(self.dataset) * 0.1) if (self.vernum == 103) else 32

    def __getitem__(self, idx):
        text = self.dataset[idx]["text"]
        # logger.info(text)
        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {"input_ids": inputs.input_ids.squeeze()}


def train(ds_engine, train_loader, device, print_profile_results):
    ds_engine.train()
    total_loss = 0
    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device)

        outputs = ds_engine(input_ids)
        loss = outputs["loss"]
        total_loss += loss.item()
        if torch.distributed.get_rank() == 0:
            logger.info(f"loss : {loss.item()}")
            wandb.log({"trainloss": loss.item()})

        ds_engine.backward(loss)
        ds_engine.step()
        if print_profile_results:
            print_memory_with_message(torch.distributed.get_rank(), device)

        get_accelerator().empty_cache()

    avg_loss = total_loss / len(train_loader)
    return avg_loss


def get_all_reduce_mean(tensor):
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    tensor = tensor / torch.distributed.get_world_size()
    return tensor


def validate(model, val_loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            input_ids = batch["input_ids"].to(device)
            outputs = model(input_ids)
            loss = outputs["loss"]
            total_loss += loss.float()

    losses = total_loss / len(val_loader)

    try:
        losses = get_all_reduce_mean(losses)
    except:
        pass

    try:
        perplexity = torch.exp(losses).item()
    except OverflowError:
        perplexity = float("inf")

    model.train()

    return losses, perplexity


import os


def _z3_params_to_fetch(param_list):
    return [
        p
        for p in param_list
        if hasattr(p, "ds_id") and p.ds_status == ZeroParamStatus.NOT_AVAILABLE
    ]


def save_zero_three_model(model_ema, global_rank, save_dir, zero_stage=0):
    zero_stage_3 = zero_stage == 3
    os.makedirs(save_dir, exist_ok=True)
    WEIGHTS_NAME = "pytorch_model.bin"
    output_model_file = os.path.join(save_dir, WEIGHTS_NAME)

    model_to_save = model_ema.module if hasattr(model_ema, "module") else model_ema
    if not zero_stage_3:
        if global_rank == 0:
            torch.save(model_to_save.state_dict(), output_model_file)
    else:
        output_state_dict = {}
        for k, v in model_to_save.named_parameters():
            if hasattr(v, "ds_id"):
                with deepspeed.zero.GatheredParameters(
                    _z3_params_to_fetch([v]), enabled=zero_stage_3
                ):
                    v_p = v.data.cpu()
            else:
                v_p = v.cpu()
            if global_rank == 0 and "lora" not in k:
                output_state_dict[k] = v_p
        if global_rank == 0:
            torch.save(output_state_dict, output_model_file)
        del output_state_dict


@click.command()
@click.option("--num_warmup_steps", default=0, help="Number of warmup steps")
@click.option("--seed", default=42, help="Random seed")
@click.option(
    "--gradient_checkpointing", default=True, help="Use gradient checkpointing"
)
@click.option("--zero_stage", default=3, help="Zero stage for gradient checkpointing")
@click.option("--output_dir", default="output", help="Output directory")
@click.option("--offload", default=True, help="Offload computation")
@click.option("--debug", is_flag=True, help="Debug mode")
@click.option("--run_name", default=None, help="Run name")
@click.option("--print_profile_results", is_flag=True, help="Print Deepspeed Profiling")
@click.option("--local_rank", default=-1, help="Local rank")
@click.option("--n_head", default=4, help="Number of heads")
@click.option("--n_layer", default=12, help="Number of layers")
@click.option(
    "--head_width",
    default=2,
    help="Width of the Head, total dim is head_width * n_head",
)
@click.option(
    "--per_device_train_batch_size", default=32, help="Per device training batch size"
)
@click.option("--train_batch_size", default=2048, help="Total training batch size")
@click.option("--learning_rate", default=1e-3, help="Learning rate")
@click.option("--weight_decay", default=0.1, help="Weight decay for optimization")
@click.option("--num_train_epochs", default=1, help="Number of training epochs")
@click.option(
    "--lr_scheduler_type", default="linear", help="Type of learning rate scheduler"
)
def main(
    num_warmup_steps,
    seed,
    gradient_checkpointing,
    zero_stage,
    output_dir,
    offload,
    debug,
    run_name,
    local_rank,
    print_profile_results,
    n_head,
    n_layer,
    head_width,
    per_device_train_batch_size,
    train_batch_size,
    learning_rate,
    weight_decay,
    num_train_epochs,
    lr_scheduler_type,
):
    if run_name is None:
        run_name = f"LR:{learning_rate}_HeadWidth:{head_width}_TotalBS:{train_batch_size}_Nhead:{n_head}_NLayer:{n_layer}"

    if head_width > 32:
        per_device_train_batch_size = 32

    if local_rank == -1:
        device = torch.device(get_accelerator().device_name())
    else:
        get_accelerator().set_device(local_rank)
        device = torch.device(get_accelerator().device_name(), local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        deepspeed.init_distributed()

    # offload?
    offload_device = "cpu" if offload else "none"

    ds_config = {
        "train_micro_batch_size_per_gpu": per_device_train_batch_size,
        "train_batch_size": train_batch_size,
        "zero_optimization": {
            "stage": zero_stage,
            "offload_param": {"device": offload_device},
            "offload_optimizer": {"device": offload_device},
            "stage3_param_persistence_threshold": 1e4,
            "stage3_max_live_parameters": 3e7,
            "stage3_prefetch_bucket_size": 3e7,
            "memory_efficient_linear": False,
        },
        "bfloat16": {"enabled": True},
        "gradient_clipping": 1.0,
        "wall_clock_breakdown": print_profile_results,
    }

    torch.distributed.barrier()
    global_rank = torch.distributed.get_rank()

    # Initialize WANDB
    if global_rank == 0:
        wandb.init(
            name=run_name,
            config={
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "num_train_epochs": num_train_epochs,
                "lr_scheduler_type": lr_scheduler_type,
                "num_warmup_steps": num_warmup_steps,
                "seed": seed,
                "gradient_checkpointing": gradient_checkpointing,
                "zero_stage": zero_stage,
                "output_dir": output_dir,
                "offload": offload,
                "head_width": head_width,
                "Nhead": n_head,
                "NLayer": n_layer,
            },
        )

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    config = GPTConfig(
        vocab_size=len(tokenizer),
        max_position_embeddings=512,
        n_head=n_head,
        n_layer=n_layer,
        n_embd=n_head * head_width,
    )

    # zero-init
    with deepspeed.zero.Init():
        model = GPTModel(config)

    model.train()

    # print params: should be 0 if zero-3
    total_params = sum(p.numel() for p in model.parameters())
    size_in_bytes = total_params * 4
    size_in_gb = size_in_bytes / (1024**3)
    logger.info(f"Model Size: {size_in_bytes}, {size_in_gb} GB")

    train_dataset = WikiTextDataset(tokenizer, "train", debug=debug)
    val_dataset = WikiTextDataset(tokenizer, "validation", debug=debug)

    train_sampler = (
        RandomSampler(train_dataset)
        if local_rank == -1
        else DistributedSampler(train_dataset, seed=seed)
    )
    eval_sampler = (
        SequentialSampler(val_dataset)
        if local_rank == -1
        else DistributedSampler(val_dataset, seed=seed)
    )

    train_loader = DataLoader(
        train_dataset,
        collate_fn=default_data_collator,
        sampler=train_sampler,
        batch_size=per_device_train_batch_size,
    )
    val_loader = DataLoader(
        val_dataset,
        collate_fn=default_data_collator,
        sampler=eval_sampler,
        batch_size=per_device_train_batch_size * 2,
    )

    # CONFIG DECAY.

    no_decay_name_list = [
        "bias",
        "ln_",
        "ln_f.weight",
    ]

    optimizer_grouped_parameters = []
    final_optimizer_settings = {}

    for n, p in model.named_parameters():
        group_parameters = {}
        if p.requires_grad:
            if any(ndnl in n for ndnl in no_decay_name_list):
                group_parameters["weight_decay"] = 0.0
            else:
                group_parameters["weight_decay"] = weight_decay

            # Define learning rate for specific types of params

            is_embed = "embed" in n
            if "embed" in n or any(ndnl in n for ndnl in no_decay_name_list):
                group_parameters["lr"] = learning_rate * (3.3 if is_embed else 1.0)
            else:
                group_parameters["lr"] = learning_rate * (1 / head_width)

            group_parameters["params"] = [p]
            final_optimizer_settings[n] = {
                "lr": group_parameters["lr"],
                "wd": group_parameters["weight_decay"],
            }
            optimizer_grouped_parameters.append(group_parameters)

    # View the settings, see if anything is wrong.
    with open("./opt_config.json", "w") as json_file:
        json.dump(final_optimizer_settings, json_file, indent=4)

    AdamOptimizer = DeepSpeedCPUAdam if offload else FusedAdam

    optimizer = AdamOptimizer(
        optimizer_grouped_parameters, lr=learning_rate, betas=(0.9, 0.95)
    )

    lr_scheduler = get_scheduler(
        name=lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_train_epochs * math.ceil(len(train_loader)),
    )

    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model, config=ds_config, lr_scheduler=lr_scheduler, optimizer=optimizer
    )

    for epoch in range(num_train_epochs):
        if local_rank == -1:
            pass
        else:
            train_sampler.set_epoch(epoch)

        avg_train_loss = train(
            model_engine, train_loader, model_engine.device, print_profile_results
        )

        logger.info(f"Epoch {epoch+1}, Train Loss: {avg_train_loss}")
        eval_loss, perp = validate(model_engine, val_loader, device=device)
        if global_rank == 0:
            logger.info(f"Eval loss : {eval_loss}")
            wandb.log({"ppl": perp, "loss": eval_loss, "epoch": epoch})

        saving_output_dir = os.path.join(output_dir, f"step_{epoch}_final")

        save_zero_three_model(
            model_engine, global_rank, saving_output_dir, zero_stage=3
        )



if __name__ == "__main__":
    main()
