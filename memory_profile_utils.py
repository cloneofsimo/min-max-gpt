import gc
import subprocess as sp

import psutil
import torch
from deepspeed.utils import logger


def print_memory_with_message(rank, device, message: str = None):
    if rank == 0:
        if None is not None:
            logger.info("=====" * 2 + message + "=====" * 2 + "\n")
        get_nvidia_gpu_memory()
    torch.distributed.barrier()

    gpu_memory_plot_helper(rank, device, message)
    torch.distributed.barrier()


def get_nvidia_gpu_memory():
    output_to_list = lambda x: x.decode("ascii").split("\n")[:-1]
    ACCEPTABLE_AVAILABLE_MEMORY = 1024
    COMMAND = "nvidia-smi --query-gpu=memory.used --format=csv"
    try:
        memory_use_info = output_to_list(
            sp.check_output(COMMAND.split(), stderr=sp.STDOUT)
        )[1:]
    except sp.CalledProcessError as e:
        raise RuntimeError(
            "command '{}' return with error (code {}): {}".format(
                e.cmd, e.returncode, e.output
            )
        )
    memory_use_values = [int(x.split()[0]) for i, x in enumerate(memory_use_info)]
    logger.info(
        f"""
    nvidia-smi info
    {memory_use_values}"""
    )


def gpu_memory_plot_helper(
    rank,
    device,
    use_deepspeed_memory_usage_helper: bool = False,
):
    """
    DeepSpeed has its own memory profiler, but we add this for customization in the future.
    https://github.com/microsoft/DeepSpeed/blob/ff7d5275f2aa916cb5f320e0d817154e96f9cdb6/deepspeed/runtime/utils.py#L793
    """
    gc.collect()
    torch.cuda.synchronize(
        device
    )  # https://pytorch.org/docs/stable/generated/torch.cuda.synchronize.html

    allocated = torch.cuda.memory_allocated(device) / (1024**2)
    max_allocated = torch.cuda.max_memory_allocated(device) / (1024**2)
    reserved = torch.cuda.memory_reserved(device) / (1024**2)
    max_reserved = torch.cuda.max_memory_reserved(device) / (1024**2)

    vm_stats = psutil.virtual_memory()
    used_GB = round(((vm_stats.total - vm_stats.available) / (1024**3)), 2)

    logger.info(
        f"""
    rank: {rank} / device: {device}
    CPU Virtual Memory:  used = {used_GB} GB, percent = {vm_stats.percent}%
    Allocated / Reserved: {allocated:.2f}MB / {reserved:.2f}MB
    Max Allocated / Max Reserved: {max_allocated:.2f}MB / {max_reserved:.2f}MB"""
    )
    torch.cuda.reset_peak_memory_stats(
        device
    )  # https://pytorch.org/docs/stable/generated/torch.cuda.reset_peak_memory_stats.html
