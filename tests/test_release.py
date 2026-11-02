"""Checks that do not download models or start training."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.chdir(tempfile.gettempdir())


def check_configs():
    from epipersona.config import load_config

    shared = dict(
        data_name="prism",
        belief_name="qwen3-235b-a22b-instruct-2507",
        window_size=30,
        codebook_size=128,
        codebook_dim=32,
        vq_mode="now",
        stable_mode="pooling",
        max_turn=1,
        learning_rate=5e-6,
        commit_weight=1.0,
        diversity_weight=0.1,
        diversity_temp=0.5,
        hard_usage_weight=0.0,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        num_gpus=4,
        logging_steps=10,
        eval_steps=150,
        save_steps=150,
        warmup_steps=100,
        weight_decay=0.01,
        lora_r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        seed=42,
        mask_mode="only_test",
        stat_max_length=1600,
        dyn_max_length=1024,
        resp_max_length=1024,
        bf16=True,
    )
    small = load_config("configs/llama3b.yaml")
    large = load_config("configs/llama8b.yaml")
    for config in (small, large):
        for key, value in shared.items():
            assert getattr(config, key) == value, (config.name, key, getattr(config, key))
    assert small.model_id == "meta-llama/Llama-3.2-3B-Instruct"
    assert small.model_name == "Llama-3.2-3B-Instruct"
    assert small.num_train_epochs == 4
    assert small.per_device_eval_batch_size == 8
    assert small.inference_batch_size == 32
    assert small.save_total_limit == 16
    assert small.master_port == 29531
    assert large.model_id == "meta-llama/Meta-Llama-3.1-8B-Instruct"
    assert large.model_name == "Meta-Llama-3.1-8B-Instruct"
    assert large.num_train_epochs == 5
    assert large.per_device_eval_batch_size == 4
    assert large.inference_batch_size == 8
    assert large.save_total_limit == 20
    assert large.master_port == 29532


def check_prompts():
    from epipersona.dataset import load_kept_ids
    from epipersona.prompts import stable_preference_prompt

    context = ["turn {not-a-field}"]
    chosen = {"role": "model", "content": "yes {x}"}
    rejected = {"role": "model", "content": "no"}
    stable = stable_preference_prompt(context, chosen, rejected)
    assert stable.startswith("\nYour task: Identify the stable preference")
    assert '{"stable_preferences": "<preference>"}' in stable
    assert "turn {not-a-field}" in stable
    assert "yes {x}" in stable
    assert len(load_kept_ids()) == 16705


def check_metrics():
    from epipersona.metrics import binary_metrics

    metrics = binary_metrics([0.6, 0.4])
    assert metrics["n"] == 2
    assert metrics["accuracy"] == 0.5
    assert metrics["n_wrong"] == 1
    assert metrics["n_near_half"] == 0
    assert metrics["n_confident_wrong"] == 0


def check_paths():
    from epipersona.paths import cache_root, data_root, repo_root

    assert repo_root() == ROOT
    assert data_root() == (ROOT / "data").resolve()
    override = ROOT / "cache" / "selfcheck-override"
    os.environ["EPIPERSONA_CACHE_ROOT"] = "cache/selfcheck-override"
    assert cache_root() == override.resolve()
    os.environ.pop("EPIPERSONA_CACHE_ROOT", None)


def check_forward_keys():
    import torch
    import torch.nn as nn

    from epipersona.model.reward import RewardModel

    class Stub(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(32, 16)
            self.config = type("Config", (), {"hidden_size": 16})()

        def get_input_embeddings(self):
            return self.embed

        def forward(self, inputs_embeds, attention_mask, use_cache=False, output_hidden_states=True, **kwargs):
            del attention_mask, use_cache, output_hidden_states, kwargs
            return type("Output", (), {"hidden_states": (inputs_embeds,)})()

    torch.manual_seed(42)
    model = RewardModel(Stub(), codebook_size=16, codebook_dim=8, diversity_temp=0.5, hard_usage_weight=0.0)
    model.eval()
    batch = 2
    output = model(
        input_ids=torch.randint(0, 32, (batch, 4)),
        attention_mask=torch.ones(batch, 4, dtype=torch.long),
        resp2_input_ids=torch.randint(0, 32, (batch, 4)),
        resp2_attention_mask=torch.ones(batch, 4, dtype=torch.long),
        allocated_embedding_stat=torch.randn(batch, 3, 16),
        allocated_embedding_stat_mask=torch.tensor([[1, 1, 0], [1, 0, 0]]),
    )
    assert list(output) == [
        "loss",
        "logits",
        "score_resp1",
        "score_resp2",
        "vq_commit_loss",
        "vq_diversity_loss",
        "vq_indices",
    ]
    assert torch.isfinite(output["loss"])
    assert output["logits"].shape == (batch,)


def main():
    check_configs()
    check_prompts()
    check_metrics()
    check_paths()
    check_forward_keys()
    print("selfcheck ok")


if __name__ == "__main__":
    main()
