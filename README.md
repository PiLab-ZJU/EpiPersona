History embeddings are quantized with a cosine codebook, mean-pooled into a persona prototype, and scored against the chosen and rejected responses. The loss is InfoNCE plus the VQ commitment loss and a codebook diversity penalty.

| Config | Base model | Epochs | Eval batch | Infer batch |
|---|---|---:|---:|---:|
| `configs/llama3b.yaml` | `meta-llama/Llama-3.2-3B-Instruct` | 4 | 8 | 32 |
| `configs/llama8b.yaml` | `meta-llama/Meta-Llama-3.1-8B-Instruct` | 5 | 4 | 8 |

Both use a history window of 30, codebook size 128, codebook dimension 32, LoRA r=8 / alpha=32, learning rate 5e-6, and seed 42. Training uses 4 GPUs, bfloat16, and global batch 16 (`2 x 4 x 2`). ZeRO-2 offloads Adam states to CPU. A row is correct when `sigmoid(score_chosen - score_rejected) > 0.5`.

Python 3.11, PyTorch 2.5.1+cu121, transformers 4.57.1, DeepSpeed 0.18.6. Accept the Llama Community License before downloading the base models. Third-party terms are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Copy `.env.example` to `.env` to override paths or set `HF_TOKEN`. Process environment variables take priority. `EPIPERSONA_GPU_IDS` must contain 4 devices. If the CUDA compiler is newer than the PyTorch build, the scripts set `DS_SKIP_CUDA_CHECK=1`.

## Data

```bash
bash scripts/prepare_prism.sh
```

This downloads the `train`, `validation`, and `test` splits of [MichaelR207/prism_personalized_0125](https://huggingface.co/datasets/MichaelR207/prism_personalized_0125) to `data/prism_personalized_0125`. Row ids are `{hf_split}.{conversation_id}.{user_id}.{row_index}`. Those splits are concatenated in that order. The `split` column (`train`, `val`, `test`) is the label: 7,748 train, 3,233 val, and 5,724 test rows, 723 users. A train row uses the same user's other train rows as history. A test row uses that user's train and validation rows. History is capped at 30. A row is used only when its id is listed in `epipersona/resources/prism_row_ids.txt`.

## Train

```bash
bash scripts/train.sh configs/llama3b.yaml
bash scripts/train.sh configs/llama8b.yaml
```

Checkpoints are saved every 150 steps under `outputs/b/<model>.prism.win30.co128.<hash>/checkpoint-N/pytorch_model.bin`. Logs go to `logs/`.

## Evaluation

```bash
bash scripts/infer.sh outputs/b/<run>/checkpoint-1650          # 3B
bash scripts/infer.sh outputs/b/<run>/checkpoint-1200          # 8B
bash scripts/evaluate.sh outputs/b/<run>/inference_predictions.checkpoint-1650.json
```

Inference writes `inference_predictions.<checkpoint>.json`, `user_quantized_indices.<checkpoint>.pt`, and `metrics.<checkpoint>.json` next to the run directory.
