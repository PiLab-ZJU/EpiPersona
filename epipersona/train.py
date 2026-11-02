"""Train the PRISM reward model."""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model
from scipy.stats import bootstrap
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    logging as transformers_logging,
)

from epipersona.config import TARGET_MODULES, load_config
from epipersona.dataset import RewardDataset, prepare_tokenizer
from epipersona.io_utils import save_json
from epipersona.model.reward import RewardModel
from epipersona.runtime import (
    DS_CONFIG,
    initialize_distributed,
    is_main_process,
    main_process_first,
    run_directory,
    seed_everything,
    timestamp,
)


def compute_metrics(eval_pred):
    probs = eval_pred.predictions[0]
    correct = (probs > 0.5).astype(int)
    accuracy = float(np.mean(correct))
    if len(correct) < 2:
        difference = 0.0
    else:
        interval = bootstrap((correct,), np.mean, confidence_level=0.95, method="basic")
        difference = accuracy - float(interval.confidence_interval.low)
    return {"accuracy": accuracy, "difference": difference}


class MetricsRecorderCallback(TrainerCallback):
    def __init__(self, output_dir: str):
        super().__init__()
        os.makedirs(output_dir, exist_ok=True)
        self.eval_file = os.path.join(output_dir, "eval_metrics.jsonl")
        self.train_file = os.path.join(output_dir, "train_metrics.jsonl")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not state.is_world_process_zero or metrics is None:
            return
        record = {"step": state.global_step, "epoch": state.epoch, **metrics}
        with open(self.eval_file, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero or logs is None:
            return
        if any(key.startswith("eval_") for key in logs):
            return
        record = {"step": state.global_step, "epoch": state.epoch, **logs}
        with open(self.train_file, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")


class RewardTrainer(Trainer):
    def __init__(self, commit_weight: float, diversity_weight: float, *args, **kwargs):
        self.commit_weight = commit_weight
        self.diversity_weight = diversity_weight
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if self.label_smoother is not None and "labels" in inputs:
            inputs.pop("labels")
        outputs = model.forward(**inputs)
        loss = outputs["loss"]
        if (
            self.args.average_tokens_across_devices
            and self.model_accepts_loss_kwargs
            and num_items_in_batch is not None
        ):
            loss *= self.accelerator.num_processes
        loss = (
            loss
            + outputs["vq_commit_loss"].squeeze() * self.commit_weight
            + outputs["vq_diversity_loss"].squeeze() * self.diversity_weight
        )
        return (loss, outputs) if return_outputs else loss


def resolve_model_source(config) -> str:
    override = os.environ.get("EPIPERSONA_MODEL_PATH", "").strip()
    return override or config.model_id


def main():
    parser = argparse.ArgumentParser(description="Train a PRISM reward model")
    parser.add_argument("--config", required=True)
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()
    config = load_config(args.config)
    transformers_logging.set_verbosity_error()
    initialize_distributed()
    seed_everything(config.seed)

    model_source = resolve_model_source(config)
    if is_main_process():
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        print(
            f"train {config.model_name} epochs={config.num_train_epochs} "
            f"gpus={config.num_gpus} source={model_source}"
        )
        if visible:
            print(f"cuda_visible_devices {visible}")

    tokenizer = prepare_tokenizer(
        AutoTokenizer.from_pretrained(model_source, use_fast=True),
        config.model_name,
    )
    base_model = AutoModelForCausalLM.from_pretrained(model_source, torch_dtype=torch.bfloat16)
    payload = config.to_dict()
    payload["base_model_path"] = model_source
    payload["run_datetime"] = timestamp()
    output_dir = run_directory(config.model_name, payload)
    payload["output_dir"] = str(output_dir)
    if is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
        save_json(payload, output_dir / "exp_args.json")
        print(f"output_dir {output_dir}")

    dataset_kwargs = dict(
        model_name=config.model_name,
        model_path=model_source,
        belief_name=config.belief_name,
        tokenizer=tokenizer,
        model=base_model,
        window_size=config.window_size,
        max_turn=config.max_turn,
        resp_max_length=config.resp_max_length,
        stat_max_length=config.stat_max_length,
        mask_mode=config.mask_mode,
    )
    with main_process_first():
        train_dataset = RewardDataset(split="train", **dataset_kwargs)
        test_dataset = RewardDataset(split="test", **dataset_kwargs)
    if is_main_process():
        print(f"samples train={len(train_dataset)} test={len(test_dataset)}")

    base_model.enable_input_require_grads()
    base_model = get_peft_model(
        base_model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=TARGET_MODULES,
            inference_mode=False,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
        ),
    )
    trainable = sum(parameter.numel() for parameter in base_model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in base_model.parameters())
    if is_main_process():
        print(f"trainable_parameters {trainable}/{total}")

    wrapper = RewardModel(
        base_model=base_model,
        codebook_size=config.codebook_size,
        codebook_dim=config.codebook_dim,
        diversity_temp=config.diversity_temp,
        hard_usage_weight=config.hard_usage_weight,
    )
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_train_epochs,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        deepspeed=DS_CONFIG,
        learning_rate=config.learning_rate,
        bf16=config.bf16,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps,
        eval_strategy="steps",
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        eval_steps=config.eval_steps,
        save_safetensors=False,
        save_strategy="steps",
        report_to=[],
        log_level="error",
        log_level_replica="error",
    )
    trainer = RewardTrainer(
        model=wrapper,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset={"test": test_dataset},
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=False),
        compute_metrics=compute_metrics,
        commit_weight=config.commit_weight,
        diversity_weight=config.diversity_weight,
        callbacks=[MetricsRecorderCallback(str(output_dir))],
    )
    trainer.train()
    trainer.evaluate(metric_key_prefix="final")
    if is_main_process():
        print(f"finished {output_dir}")


if __name__ == "__main__":
    main()
