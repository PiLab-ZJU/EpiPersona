"""Score a saved reward-model checkpoint on the PRISM test rows."""

from __future__ import annotations

import argparse
import os
from collections import defaultdict

import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq

from epipersona.checkpoint import load_run_config, load_state_dict, run_dir_of
from epipersona.config import TARGET_MODULES
from epipersona.dataset import RewardDataset, prepare_tokenizer
from epipersona.io_utils import save_json
from epipersona.metrics import binary_metrics, write_summary
from epipersona.model.reward import RewardModel
from epipersona.runtime import seed_everything


def main():
    parser = argparse.ArgumentParser(description="Run reward-model inference on PRISM test")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--model-path", default=None)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Inference requires a CUDA device with bfloat16 support")
    seed_everything(42)
    saved = load_run_config(args.checkpoint)
    model_source = args.model_path or os.environ.get("EPIPERSONA_MODEL_PATH") or saved.get("base_model_path")
    if not model_source:
        raise ValueError("Pass --model-path or set EPIPERSONA_MODEL_PATH")
    model_name = saved["model_name"]
    batch_size = args.batch_size or int(saved.get("inference_batch_size", 32))
    print(f"checkpoint {args.checkpoint}")
    print(f"model {model_name} batch_size {batch_size}")

    tokenizer = prepare_tokenizer(
        AutoTokenizer.from_pretrained(model_source, use_fast=True),
        model_name,
    )
    base_model = AutoModelForCausalLM.from_pretrained(model_source, torch_dtype=torch.bfloat16)
    dataset = RewardDataset(
        split="test",
        model_name=model_name,
        model_path=model_source,
        belief_name=saved.get("belief_name", "qwen3-235b-a22b-instruct-2507"),
        tokenizer=tokenizer,
        model=base_model,
        window_size=int(saved["window_size"]),
        max_turn=int(saved.get("max_turn", 1)),
        resp_max_length=int(saved.get("resp_max_length", 1024)),
        stat_max_length=int(saved.get("stat_max_length", 1600)),
        mask_mode=saved.get("mask_mode", "only_test"),
    )
    print(f"test_samples {len(dataset)}")
    base_model.enable_input_require_grads()
    base_model = get_peft_model(
        base_model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=saved.get("target_modules", TARGET_MODULES),
            inference_mode=False,
            r=int(saved.get("lora_r", 8)),
            lora_alpha=int(saved.get("lora_alpha", 32)),
            lora_dropout=float(saved.get("lora_dropout", 0.1)),
        ),
    )
    wrapper = RewardModel(
        base_model=base_model,
        codebook_size=int(saved["codebook_size"]),
        codebook_dim=int(saved.get("codebook_dim", 32)),
        diversity_temp=float(saved.get("diversity_temp", 0.5)),
        hard_usage_weight=float(saved.get("hard_usage_weight", 0.0)),
    )
    state = load_state_dict(args.checkpoint)
    missing, unexpected = wrapper.load_state_dict(state, strict=False)
    if os.environ.get("EPIPERSONA_DEBUG") == "1":
        print(f"missing_keys {len(missing)} unexpected_keys {len(unexpected)}")
    codebook = getattr(wrapper.soft_prompt_generator.vq, "_codebook", None)
    if codebook is not None:
        std = codebook.embed.float().std().item()
        if std < 1e-6:
            raise RuntimeError("VQ codebook was not loaded from the checkpoint")
    wrapper.to(device="cuda", dtype=torch.bfloat16)
    wrapper.eval()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=False),
    )
    probabilities = []
    indices = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="inference"):
            batch = {
                key: value.to("cuda") if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            batch["allocated_embedding_stat"] = batch["allocated_embedding_stat"].to(dtype=torch.bfloat16)
            output = wrapper.forward(**batch)
            probabilities.append(output["logits"].float().cpu().numpy())
            if output.get("vq_indices") is not None:
                indices.append(output["vq_indices"].detach().cpu())
    probs = np.concatenate(probabilities)
    metrics = binary_metrics(probs)
    run_dir = run_dir_of(args.checkpoint)
    ckpt_name = os.path.basename(os.path.normpath(args.checkpoint))
    prediction_path = run_dir / f"inference_predictions.{ckpt_name}.json"
    indices_path = run_dir / f"user_quantized_indices.{ckpt_name}.pt"
    summary_path = run_dir / f"metrics.{ckpt_name}.json"
    if indices:
        stacked = torch.cat(indices, dim=0)
        valid = stacked[stacked >= 0]
        unique = torch.unique(valid)
        metrics["codebook_used"] = int(unique.numel())
        metrics["codebook_size"] = int(saved["codebook_size"])
        by_user = defaultdict(list)
        for index, user_id in enumerate(dataset.user_ids):
            by_user[user_id].append(stacked[index])
        torch.save({user_id: torch.stack(rows) for user_id, rows in by_user.items()}, indices_path)
        print(f"codebook_used {metrics['codebook_used']}/{metrics['codebook_size']}")
        print(f"indices {indices_path}")
    save_json(
        {
            "ckpt_dir": str(args.checkpoint),
            "metrics": {"accuracy": metrics["accuracy"], "difference": metrics["accuracy"] - metrics["ci95_low"]},
            "probs": probs.tolist(),
            "uniq_ids": dataset.uniq_ids,
        },
        prediction_path,
    )
    print(f"predictions {prediction_path}")
    write_summary(metrics, summary_path)


if __name__ == "__main__":
    main()
