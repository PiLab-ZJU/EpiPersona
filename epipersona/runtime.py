"""Training helpers shared by the reward-model entry points."""

from __future__ import annotations

import hashlib
import os
import random
import time
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch

from epipersona.paths import cache_root, output_root


DS_CONFIG = {
    "train_batch_size": "auto",
    "gradient_accumulation_steps": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "fp16": {"enabled": "auto"},
    "bf16": {"enabled": "auto"},
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {"device": "cpu", "pin_memory": True},
        "overlap_comm": True,
        "contiguous_gradients": True,
        "allgather_partitions": True,
        "allgather_bucket_size": 5e8,
        "reduce_bucket_size": 5e8,
    },
    "optimizer": {
        "type": "AdamW",
        "params": {"lr": "auto", "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.01},
    },
    "scheduler": {
        "type": "WarmupLR",
        "params": {"warmup_min_lr": 0, "warmup_max_lr": "auto", "warmup_num_steps": "auto"},
    },
    "gradient_clipping": 1.0,
    "steps_per_print": 10,
    "zero_allow_untested_optimizer": True,
}


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True, warn_only=True)


def initialize_distributed() -> None:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1 or torch.distributed.is_initialized():
        return
    if not torch.cuda.is_available():
        raise RuntimeError("Distributed training requires CUDA")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    timeout = timedelta(seconds=int(os.environ.get("EPIPERSONA_DIST_TIMEOUT_SEC", "10800")))
    init_kwargs = dict(backend="nccl", init_method="env://", timeout=timeout)
    try:
        torch.distributed.init_process_group(
            device_id=torch.device("cuda", local_rank),
            **init_kwargs,
        )
    except TypeError:
        torch.distributed.init_process_group(**init_kwargs)


def is_main_process() -> bool:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    return int(os.environ.get("RANK", "0")) == 0


def _flag_path() -> Path:
    job = os.environ.get("EPIPERSONA_DIST_SYNC_ID")
    if not job:
        job = f"{os.environ.get('MASTER_PORT', '0')}_{os.environ.get('WORLD_SIZE', '1')}"
    return cache_root() / ".dist_sync" / f"{job}.done"


@contextmanager
def main_process_first():
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        yield
        return
    rank = torch.distributed.get_rank()
    flag = _flag_path()
    timeout = int(os.environ.get("EPIPERSONA_DIST_TIMEOUT_SEC", "10800"))
    if rank == 0:
        flag.parent.mkdir(parents=True, exist_ok=True)
        if flag.exists():
            flag.unlink()
        try:
            yield
        finally:
            flag.write_text("done\n", encoding="utf-8")
            torch.distributed.barrier()
            flag.unlink(missing_ok=True)
    else:
        start = time.time()
        while not flag.exists():
            if time.time() - start > timeout:
                raise TimeoutError(f"Timed out waiting for cache flag {flag}")
            time.sleep(5)
        try:
            yield
        finally:
            torch.distributed.barrier()


def run_directory(model_name: str, config_payload: dict) -> Path:
    material = str(sorted(config_payload.items()))
    digest = hashlib.md5(material.encode()).hexdigest()
    directory = (
        output_root()
        / "b"
        / f"{model_name}.prism.win{config_payload['window_size']}.co{config_payload['codebook_size']}.{digest}"
    )
    return directory


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d%H%M%S")
