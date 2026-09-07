# Copilot instructions for UserRL

Purpose: give AI coding agents the minimal, high-value context to be productive in this repo.

- **Big picture:** Core RL framework lives in `verl/` (trainer, workers, models, utils). Environments (gyms) live in `gyms/` and example training configs live in `examples/`. Data and validation artifacts are under `data/` and `validation_trajectory` / `training_rollout_trajectory` paths referenced by examples.

- **How training is launched:** examples call the trainer module directly. Typical invocation:

  python3 -m verl.trainer.main_ppo \
    data.train_files=$PROJECT_DIR/data/... \
    trainer.n_gpus_per_node=8 trainer.nnodes=1 trainer.total_epochs=30

  See example scripts in `examples/functiongym_ablation/` and `examples/sglang_multiturn/train.sh` for complete parameter patterns.

- **Configs & overrides:** Most runtime settings are YAMLs under `verl/trainer/config/` and frequently overridden on the CLI using `key=value` style (see example scripts). When editing behaviour, prefer changing YAMLs and use CLI overrides for experiments.

- **Distributed & model integrations:** This repo integrates custom model loaders and parallel state for SGLang/vLLM/MeGaTron in `verl/third_party/*` and `verl/models/*`. Key files to read for model conversion/loading are in `verl/models/mcore/` (config_converter, weight_converter, registry).

- **Environment / reward hooks:** Gym environments live in `gyms/*`. Reward and PPO hooks are under `verl/trainer/ppo/` (search for `reward`, `rollout`, `main_ppo`, `main_eval`). Look at `verl/protocol.py` to understand dataflow between dataset loader, rollout, and trainer.

- **Developer workflows / common commands:**
  - Install dependencies / env bootstrap: `bash install_gyms.sh` and consult `INSTALL&RUN.md`.
  - Run an example training (single command): see `examples/*/train.sh` or invoke `python3 -m verl.trainer.main_ppo` with the example's CLI overrides.
  - Use `bv_scripts/` wrappers for cluster/batch launches (examples call `bv_scripts/bv_normal.sh`).

- **Conventions & patterns to follow:**
  - CLI overrides use dot-separated keys (e.g. `trainer.n_gpus_per_node`, `data.train_batch_size`).
  - Experiments store rollouts/validation under `training_rollout_trajectory/<env>/<exp>` and `validation_trajectory/<env>/<exp>` — scripts assume these directories exist.
  - Logging uses `trainer.logger` (console + wandb) and `trainer.project_name`/`experiment_name` keys.

- **Where to make common changes:**
  - Add/modify gym environments: `gyms/<GymName>/` and examples under `examples/<GymName>/`.
  - Change trainer behavior: `verl/trainer/*` and YAMLs in `verl/trainer/config/`.
  - Modify model IO/conversion: `verl/models/mcore/`.

- **Search tips for agents:**
  - Look for `python3 -m verl.trainer.main_ppo` in `examples/` for runnable patterns.
  - Grep for `trainer.` keys to find configurable knobs (save_freq, test_freq, n_gpus_per_node).

- **Quick useful file references:**
  - Repo overview: [README.md](README.md)
  - Run & install notes: [INSTALL&RUN.md](INSTALL&RUN.md)
  - Example train scripts: [examples/functiongym_ablation](examples/functiongym_ablation)
  - Trainer entrypoints & configs: [verl/trainer](verl/trainer)
  - Model conversion & registry: [verl/models/mcore](verl/models/mcore)

If any section needs more detail (e.g., exact command for a particular GPU topography or how to convert a new HF model), tell me which area and I will expand with concrete examples.
