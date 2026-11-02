"""Experiment configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml

from epipersona.paths import repo_root


TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


@dataclass
class RunConfig:
    name: str
    model_id: str
    model_name: str
    data_name: str = "prism"
    belief_name: str = "qwen3-235b-a22b-instruct-2507"
    window_size: int = 30
    codebook_size: int = 128
    codebook_dim: int = 32
    vq_mode: str = "now"
    stable_mode: str = "pooling"
    max_turn: int = 1
    num_train_epochs: int = 4
    learning_rate: float = 5e-6
    commit_weight: float = 1.0
    diversity_weight: float = 0.1
    diversity_temp: float = 0.5
    hard_usage_weight: float = 0.0
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 2
    num_gpus: int = 4
    logging_steps: int = 10
    eval_steps: int = 150
    save_steps: int = 150
    save_total_limit: int = 16
    warmup_steps: int = 100
    weight_decay: float = 0.01
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    seed: int = 42
    mask_mode: str = "only_test"
    stat_max_length: int = 1600
    dyn_max_length: int = 1024
    resp_max_length: int = 1024
    inference_batch_size: int = 32
    master_port: int = 29531
    bf16: bool = True

    def validate(self) -> None:
        if self.data_name != "prism":
            raise ValueError("data_name must be 'prism'")
        if self.vq_mode != "now":
            raise ValueError("vq_mode must be 'now'")
        if self.stable_mode != "pooling":
            raise ValueError("stable_mode must be 'pooling'")
        if self.mask_mode != "only_test":
            raise ValueError("mask_mode must be 'only_test'")
        if self.num_gpus < 1:
            raise ValueError("num_gpus must be positive")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["target_modules"] = list(TARGET_MODULES)
        return payload


def load_config(path: str | Path) -> RunConfig:
    config_path = Path(path).expanduser()
    if not config_path.is_absolute():
        config_path = repo_root() / config_path
    with config_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    valid = {item.name for item in fields(RunConfig)}
    unknown = sorted(set(raw) - valid)
    if unknown:
        raise ValueError(f"Unknown config keys in {config_path}: {unknown}")
    config = RunConfig(**raw)
    config.validate()
    return config
