<!-- <p align="center">
  <img src="assets/logo.png" alt="BAO Logo" width="200"/>
</p> -->

# Pushing Forward Pareto Frontiers of Proactive Agents with Behavioral Agentic Optimization

## Overview

- `verl/` — RL training framework (built on [VERL](https://github.com/volcengine/verl)), with multi-turn GRPO and behavior-regularized reward shaping.
- `gyms/` — proactive-agent environments from the [UserRL](https://github.com/SalesforceAIResearch/UserRL) benchmark (Function / Turtle / Telepathy / Travel / Search / Tau / Intention / Persuade / Alfworld).
- `examples/functiongym/` — ready-to-run training scripts and configs for Function-Gym.
- `data/function_multiturn/` — Function-Gym train/val data.
- `notebook/` — `plot_pareto_frontier.ipynb` for plotting the user-engagement vs. task-performance frontier.

## Installation

**Prerequisites:** Python 3.12, one or more CUDA GPUs. Training below assumes a single node with 8 GPUs.

```bash
# 1. Create the environment
conda create -n BAO python=3.12 -y
conda activate BAO

# 2. Install the RL stack (vLLM + SGLang + FlashAttention/FlashInfer)
#    Set USE_MEGATRON=0 to skip the (slow) Megatron/TransformerEngine build; the
#    scripts below use FSDP + SGLang and do not need it.
USE_MEGATRON=0 bash install_vllm_sglang_mcore.sh
pip install -e . --no-deps

# 3. Install the gym environments
bash install_gyms.sh
```

> If you hit uvloop errors, run `pip install uvloop==0.21`.

Verify the install (run on a GPU node):

```bash
python -c "import torch, vllm, sglang, flash_attn, verl, functiongym; \
print('cuda', torch.cuda.is_available(), 'gpus', torch.cuda.device_count())"
```

## Training (RL on Function-Gym)

**1. Download the SFT checkpoints** (behavior-enhanced warm-start models):

```bash
git clone https://huggingface.co/YYY-45/BAO-SFT-ckpt
```

**2. Launch training.** Point `SFT_CKPT_DIR` at the downloaded Function-Gym checkpoint and
provide a Weights & Biases key, then run:

```bash
export SFT_CKPT_DIR=/path/to/BAO-SFT-ckpt/BAO-functiongym-sft-4B-ckpt/checkpoint-final
export WANDB_API_KEY=<your key from https://wandb.ai/authorize>

bash examples/functiongym/4B_train_functiongym-BAO-CR.sh
```

The script auto-detects the repo root, so no other paths need editing. Every option
(`SFT_CKPT_DIR`, `EXP_NAME`, `CUDA_VISIBLE_DEVICES`, ...) can be overridden from the
environment. To run without Weights & Biases, set `trainer.logger=[console]` in the script.

## Evaluation

Use `notebook/plot_pareto_frontier.ipynb` to draw the Pareto frontier between user
engagement and task performance from the trajectories saved during training.

## Acknowledgments

Built on top of:
- [UserRL](https://github.com/SalesforceAIResearch/UserRL) — user-centric proactive-agent benchmark
- [VERL](https://github.com/volcengine/verl) — RL framework for LLMs
- [SGLang](https://github.com/sgl-project/sglang) — efficient LLM serving
