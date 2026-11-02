"""Checkpoint loading for PyTorch and safetensors shards."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file


def run_dir_of(checkpoint_dir: str | Path) -> Path:
    path = Path(checkpoint_dir).expanduser().resolve()
    return path.parent if path.name.startswith("checkpoint-") else path


def load_run_config(checkpoint_dir: str | Path) -> dict:
    directory = run_dir_of(checkpoint_dir)
    path = directory / "exp_args.json"
    if not path.is_file():
        raise FileNotFoundError(f"No exp_args.json in {directory}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_state_dict(checkpoint_dir: str | Path) -> dict:
    directory = Path(checkpoint_dir).expanduser().resolve()
    if not directory.is_dir():
        raise NotADirectoryError(f"Checkpoint directory does not exist: {directory}")
    state = None
    for filename in ("model.safetensors", "pytorch_model.bin"):
        path = directory / filename
        if path.is_file():
            if path.suffix == ".safetensors":
                state = load_file(str(path), device="cpu")
            else:
                try:
                    state = torch.load(path, map_location="cpu", weights_only=True)
                except TypeError:
                    state = torch.load(path, map_location="cpu")
            break
    if state is None:
        for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
            index_path = directory / index_name
            if not index_path.is_file():
                continue
            weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
            state = {}
            for shard_name in sorted(set(weight_map.values())):
                shard = directory / shard_name
                if shard.suffix == ".safetensors":
                    state.update(load_file(str(shard), device="cpu"))
                else:
                    state.update(torch.load(shard, map_location="cpu"))
            break
    if state is None:
        raise FileNotFoundError(f"No model weights found in {directory}")
    if state and all(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    return state
