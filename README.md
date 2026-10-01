# Latency-Aware Adaptive Speculative Decoding

## Authors

- Dor Lugasi
- Samiha Alem


Final project implementation for adaptive speculative decoding under changing runtime conditions.

The project compares fixed speculative drafting lengths with adaptive LinUCB policies that choose
`k ∈ {1, 3, 5, 8}` before each speculative verification round.

The repository contains the implementation, tests, experiment scripts, and analysis utilities
required to reproduce the experiments. Experimental result files are intentionally **not**
included in the repository.

## Main experiment variants

- `fixed_1`, `fixed_3`, `fixed_5`, `fixed_8`
- `entropy`
- `acceptance`
- `latency`
- `combined`

The adaptive policies use `ThroughputRateReward(beta=0.0)`. Training and test prompts are
separate. In the static experiment, each adaptive policy is trained once per repetition,
frozen, and then the same policy is evaluated under all requested concurrency levels.

## Repository structure

```text
specdecode_final_project/
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── specdecode/
│   ├── __init__.py
│   ├── cache_utils.py
│   ├── experiments.py
│   ├── harness.py
│   ├── instrumentation.py
│   ├── models.py
│   ├── policy.py
│   ├── rejection_sampling.py
│   ├── reward.py
│   └── speculative.py
├── scripts/
│   ├── analyze_results.py
│   ├── check_correctness_oracle.py
│   ├── run_dynamic_load.py
│   └── run_experiments.py
└── tests/
    ├── _helpers.py
    ├── conftest.py
    ├── test_cache_utils.py
    ├── test_harness.py
    ├── test_instrumentation.py
    ├── test_policy.py
    ├── test_rejection_sampling.py
    ├── test_reward.py
    ├── test_speculative.py
    └── test_timing.py
```

### Main directories

- **`specdecode/`** contains the actual implementation: model loading, speculative decoding,
  KV-cache utilities, rejection sampling, runtime harness, instrumentation, LinUCB policies,
  reward functions, and shared experiment helpers.
- **`scripts/`** contains the final entry points used to reproduce and analyze the experiments.
- **`tests/`** contains the automated test suite for the implementation.
- **`requirements.txt`** lists all Python packages required for experiments, testing, CUDA
  utilization metrics, and analysis.
- **`pyproject.toml`** defines the local `specdecode` Python package.
- **`.gitignore`** keeps generated results, caches, environments, model files, and credentials
  out of Git.

The repository should contain source code only. CSV result files, logs, generated plots,
policy snapshots, run metadata, downloaded models, and caches are generated locally and are
not part of the submitted repository.

## Requirements

The code requires Python 3.9 or newer. The final experiments were executed on an NVIDIA
RTX 2080 Ti using CUDA.

The complete dependency list used by this project is in `requirements.txt`:

- PyTorch
- Hugging Face Transformers
- pytest
- nvidia-ml-py
- pandas
- matplotlib

The Llama experiments additionally require Hugging Face access to the gated Meta Llama models.

## Setup on a fresh machine/server


All commands below should be run from the repository root unless stated otherwise.

### 1. Clone the repository

```bash
git clone <REPOSITORY_URL>
cd specdecode_final_project
```

Replace `<REPOSITORY_URL>` with the submitted Git repository URL.

### 2. Request a GPU resource

The final experiments should be run on a CUDA-capable GPU.

On the cluster used for this project, an interactive RTX 2080 Ti allocation was requested with:

```bash
srun -p all --nodelist=lambda2 --gres=gpu:2080ti:1 --pty bash
```

This command is specific to the cluster used during development. On another SLURM cluster,
the partition, node name, and GPU resource request must be changed according to that cluster's
configuration. On a local CUDA machine this step is not required.

After entering the allocated node, verify the GPU:

```bash
nvidia-smi
```

### 3. Create and activate a Python environment

A separate Conda environment is recommended:

```bash
conda create -n specdecode python=3.11 -y
conda activate specdecode
```

For an already-created environment:

```bash
conda activate specdecode
```

### 4. Install the project and dependencies

Install all dependencies listed by the project:

```bash
pip install -r requirements.txt
```

Then install the local `specdecode` package in editable mode:

```bash
pip install -e .
```

The editable installation allows the scripts to import the local `specdecode` package while
keeping the source code directly editable.

### 5. Hugging Face authentication for Llama

The Llama experiments use:

- draft: `meta-llama/Llama-3.2-1B`
- target: `meta-llama/Llama-3.2-3B`

Access to these gated models must first be approved for the Hugging Face account used on the
server. Authenticate on the machine that will run the experiment.

For a Hugging Face CLI installation that provides the current `hf` command:

```bash
hf auth login
```

If an older Hugging Face CLI is installed, the equivalent login command may be:

```bash
huggingface-cli login
```

Do not store a Hugging Face access token in the repository, scripts, README, or Git history.

The GPT-2 experiments (`gpt2` / `gpt2-medium`) do not require access to the gated Llama models.

### 6. Verify PyTorch and CUDA

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

For the final CUDA experiments, `CUDA available` must be `True`.

### 7. Run the automated tests

```bash
pytest -q
```

The full test suite should pass before running the final experiments.

### 8. Optional smoke test

Before starting the long final experiments, a small GPT-2 run can verify that model loading,
CUDA execution, package imports, and CSV output work correctly:

```bash
python scripts/run_experiments.py \
  --draft gpt2 \
  --target gpt2-medium \
  --device cuda \
  --users 1 \
  --variants fixed_3 \
  --repetitions 1 \
  --max-new-tokens 10 \
  --train-tokens 10 \
  --out smoke_results.csv \
  --steps-out smoke_steps.csv
```

Because this smoke test uses a fixed policy, its `--train-tokens` value does not affect policy
training. The generated CSV files are local outputs and should not be committed.

### 9. Running long experiments remotely

The final experiments can take a long time. When working through SSH, `tmux` is recommended:

```bash
tmux new -s specdecode
```

Start the experiment inside the session. Detach with `Ctrl-b`, then `d`.

Reconnect with:

```bash
tmux attach -t specdecode
```

The SLURM allocation itself must remain active for the GPU experiment to continue.

## Static experiment

The final static protocol uses:

- training: 150 generated tokens per training load
- evaluation: up to 100 new tokens per request
- evaluation concurrency: 1, 2, 4, 6, and 8 users
- repetitions: 5
- adaptive-policy training loads: 1, 2, 4, 6, and 8 users
- reward beta: 0.0
- greedy decoding (`do_sample=False` in the experiment code)

Each adaptive policy is trained once for a repetition and then frozen. That exact frozen policy
is reused across all evaluation loads for that repetition.

### Llama static experiment

```bash
python scripts/run_experiments.py \
  --draft meta-llama/Llama-3.2-1B \
  --target meta-llama/Llama-3.2-3B \
  --device cuda \
  --users 1 2 4 6 8 \
  --repetitions 5 \
  --max-new-tokens 100 \
  --train-tokens 150 \
  --train-users 1 2 4 6 8 \
  --reward-beta 0.0 \
  --out experiment_results_llama.csv \
  --steps-out experiment_steps_llama.csv
```

### GPT-2 static experiment

```bash
python scripts/run_experiments.py \
  --draft gpt2 \
  --target gpt2-medium \
  --device cuda \
  --users 1 2 4 6 8 \
  --repetitions 5 \
  --max-new-tokens 100 \
  --train-tokens 150 \
  --train-users 1 2 4 6 8 \
  --reward-beta 0.0 \
  --out experiment_results_gpt2.csv \
  --steps-out experiment_steps_gpt2.csv
```

If `--variants` is omitted, the script evaluates all eight variants:
`fixed_1 fixed_3 fixed_5 fixed_8 entropy acceptance latency combined`.

The script also supports the LinUCB parameters `--alpha`, `--epsilon`, and `--ridge-lambda`.
Their defaults in the checked-in code are `0.8`, `0.0`, and `1.0`, respectively.

### Static command parameters

| Parameter | Meaning |
|---|---|
| `--draft` | Hugging Face model used as the smaller draft model that proposes speculative tokens. |
| `--target` | Hugging Face target model used to verify the draft proposals and determine the final output. |
| `--device` | PyTorch device. The final experiments use `cuda`. |
| `--users` | Evaluation concurrency levels. Each value specifies the number of concurrent decoding streams for a measured trial. |
| `--variants` | Optional subset of policies/baselines to evaluate. If omitted, all variants are run. |
| `--repetitions` | Number of independent repetitions for every variant and evaluation load. |
| `--max-new-tokens` | Maximum number of new tokens generated for each evaluation request. |
| `--train-tokens` | Number of generated tokens per training load when training an adaptive policy. |
| `--train-users` | Concurrency levels presented to an adaptive policy during training. |
| `--reward-beta` | `beta` passed to `ThroughputRateReward`. The final protocol uses `0.0`. |
| `--alpha` | LinUCB exploration coefficient. Default: `0.8`. |
| `--epsilon` | Additional epsilon-random exploration probability. Default: `0.0`. |
| `--ridge-lambda` | Ridge regularization value used by LinUCB. Default: `1.0`. |
| `--out` | Trial-level CSV output path. |
| `--steps-out` | Detailed speculative-step CSV output path. |

The static script also writes provenance files next to `--out`: run metadata and policy
snapshots. These are generated experiment artifacts and should not be committed.

## Dynamic-load experiment

The dynamic experiment keeps request workers alive while the target concurrency changes.
When concurrency increases, new workers start immediately. When it decreases, excess workers
finish their current request and stop before the next measured phase begins. Retained workers
continue running, so the workload remains live while the target concurrency changes.

The final dynamic protocol uses:

- adaptive-policy training: 150 generated tokens for each training load
- request length: up to 100 new tokens
- phase duration: 30 seconds
- cycles: 3
- variants: `fixed_3 entropy acceptance latency combined`
- reward beta: 0.0
- schedule per cycle:
  `1 1 2 3 4 5 6 7 6 5 4 3 2 1 1 8 8 6 4 2 1 1 3 6 8 4 1`

The 27-phase schedule is repeated three times, producing 81 measured phases per policy.

### Llama dynamic experiment

```bash
python scripts/run_dynamic_load.py \
  --draft meta-llama/Llama-3.2-1B \
  --target meta-llama/Llama-3.2-3B \
  --device cuda \
  --variants fixed_3 entropy acceptance latency combined \
  --schedule 1 1 2 3 4 5 6 7 6 5 4 3 2 1 1 8 8 6 4 2 1 1 3 6 8 4 1 \
  --cycles 3 \
  --phase-seconds 30 \
  --request-tokens 100 \
  --train-tokens 150 \
  --reward-beta 0.0 \
  --out dynamic_results_llama.csv \
  --steps-out dynamic_steps_llama.csv
```

### GPT-2 dynamic experiment

```bash
python scripts/run_dynamic_load.py \
  --draft gpt2 \
  --target gpt2-medium \
  --device cuda \
  --variants fixed_3 entropy acceptance latency combined \
  --schedule 1 1 2 3 4 5 6 7 6 5 4 3 2 1 1 8 8 6 4 2 1 1 3 6 8 4 1 \
  --cycles 3 \
  --phase-seconds 30 \
  --request-tokens 100 \
  --train-tokens 150 \
  --reward-beta 0.0 \
  --out dynamic_results_gpt2.csv \
  --steps-out dynamic_steps_gpt2.csv
```

### Dynamic command parameters

| Parameter | Meaning |
|---|---|
| `--draft` | Draft model used to propose speculative tokens. |
| `--target` | Target model used to verify proposals. |
| `--device` | Device used for inference; final experiments use `cuda`. |
| `--variants` | Policies/baselines evaluated during the dynamic run. |
| `--schedule` | Ordered target concurrency values for one dynamic cycle. |
| `--cycles` | Number of times the complete schedule is repeated. |
| `--phase-seconds` | Measurement duration of each concurrency phase. |
| `--request-tokens` | Maximum number of new tokens generated by each continuously running request. |
| `--train-tokens` | Number of generated tokens used at each training load for adaptive policies. |
| `--reward-beta` | `beta` passed to `ThroughputRateReward`; final value is `0.0`. |
| `--out` | Phase-level dynamic result CSV. |
| `--steps-out` | Detailed step-level dynamic result CSV. |

In the checked-in dynamic script, adaptive policies are trained on user loads
`1, 2, 4, 6, 8` and then frozen before the measured dynamic workload.

## Correctness / hindsight fixed-k check

`check_correctness_oracle.py` compares greedy speculative generation against direct greedy
generation by the target model for fixed `k ∈ {1, 3, 5, 8}`. It also reports the
hindsight best fixed `k` by measured speculative throughput for each prompt. This is a
best-fixed-k baseline, not a true per-step oracle.

### Llama

```bash
python scripts/check_correctness_oracle.py \
  --draft meta-llama/Llama-3.2-1B \
  --target meta-llama/Llama-3.2-3B \
  --device cuda \
  --max-new-tokens 80 \
  --out oracle_correctness_llama.csv
```

### GPT-2

```bash
python scripts/check_correctness_oracle.py \
  --draft gpt2 \
  --target gpt2-medium \
  --device cuda \
  --max-new-tokens 80 \
  --out oracle_correctness_gpt2.csv
```

The generated correctness CSV files are experiment outputs and should not be committed.

## Analysis

`scripts/analyze_results.py` reads generated CSV files and creates summary CSVs and plots.
It is an analysis utility and is not required by the decoder itself.

Its available inputs are:

```text
--trials     static trial-level CSV
--steps      static step-level CSV
--dynamic    dynamic phase-level CSV
--oracle     correctness CSV
--out-dir    directory for generated summaries and plots
```

Example:

```bash
python scripts/analyze_results.py \
  --trials experiment_results_llama.csv \
  --steps experiment_steps_llama.csv \
  --dynamic dynamic_results_llama.csv \
  --oracle oracle_correctness_llama.csv \
  --out-dir results_llama
```

The analysis output directory is generated locally and should not be committed.

## Recommended reproduction order

1. Clone the repository.
2. Request a CUDA GPU resource if running on a cluster.
3. Create and activate the `specdecode` Conda environment.
4. Install `requirements.txt`.
5. Install the local package with `pip install -e .`.
6. Authenticate with Hugging Face when running the Llama pair.
7. Verify CUDA.
8. Run `pytest -q`.
9. Optionally run the short smoke test.
10. Run the static experiments.
11. Run the dynamic-load experiments.
12. Run the correctness check.
13. Analyze the locally generated CSV files.

## Reproducibility notes

- Draft and target models must use the same tokenizer vocabulary; the experiment scripts
  explicitly check this before running.
- The model loader uses `float16` on CUDA and `float32` on CPU by default.
- The final experiments use greedy decoding.
- Adaptive policies are frozen before measured evaluation.
- Static train and test prompt sets are separate.
- The static script records run metadata and policy snapshots for local provenance.
- Generated outputs are intentionally excluded from the submitted repository.
