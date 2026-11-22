# JSpaceDoNOT

JSpaceDoNOT experiments with **Qwen3-4B** on **GSM8K, MATH-500, and RACE-H**. The repository includes model and corpus preparation, full J++ lens fitting, stopping-head training and evaluation, and interpretation of the fitted LR/MLP heads.

Start with [`Run_JSpaceDoNOT.ipynb`](Run_JSpaceDoNOT.ipynb). All notebook instructions and comments are in English; the implementation is encapsulated in `jspace_donot/` and `scripts/`.

## Installation

Use Python 3.12 and a CUDA-enabled PyTorch installation with BF16 support. Choose the PyTorch build appropriate for your GPU and driver, then install the project dependencies in the same environment as your Jupyter kernel:

```bash
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
```

`requirements.txt` lists required packages; it is **not an exact environment lock**. Keep an existing working GPU environment rather than changing its PyTorch, CUDA, or Transformers versions merely to match another machine. Each completed experiment records its package versions in `environment.json`.

Preparation fetches the official J++ implementation at commit `d7efc0de372e20589e899188c0cfe8434acc63ec` from [safety-research/jpp_lens](https://github.com/safety-research/jpp_lens). Its `jlens`, `workspace_lens`, and `lens_evals` source is made available to fitting and experiment subprocesses. The default preparation flow **does not install, upgrade, or downgrade dependencies**.

If fitting requires a separate existing environment, set `FIT_PYTHON` in the notebook or pass `--fit-python` to `prepare.py`.

## Complete workflow

Run the main notebook in order, or use:

```bash
python prepare.py --root .
python run.py --dataset gsm8k
python run.py --dataset math500
python run.py --dataset raceh
```

`prepare.py` performs these steps:

1. Download missing official Qwen3-4B model/tokenizer files, or reuse an existing model directory.
2. Fetch the pinned J++ source.
3. Download WikiText-103 raw TRAIN and create the original J++ prompt pool.
4. Reuse a completed compatible J++ lens, or run the full fit when none exists.
5. Download and process the selected task datasets.
6. Save the resolved asset paths in `preparation_manifest.json`.

To prepare and run GSM8K in one command:

```bash
python run.py --dataset gsm8k --prepare
```

To prepare only its task data while retaining the same model/lens preparation:

```bash
python prepare.py --root . --datasets gsm8k
```

### Reuse existing assets

```bash
python prepare.py --root . --datasets gsm8k \
    --model-dir /path/to/Qwen3-4B \
    --lens-path /path/to/completed/jpp_fit
```

Preparation accepts either a completed lens `.pt` file or a directory containing exactly one completed checkpoint and its manifest. Model files outside the root are reused through a directory link. Existing conflicting model directories are not overwritten.

With `lens_path=None`, preparation searches under `jlens_outputs/qwen3_4b/jpp_fit/`; it fits the lens if no completed checkpoint exists. Multiple completed checkpoints require an explicit selection. Checkpoint and model-asset SHA-256 checks reject incompatible files. Use `--no-fit` if a missing completed lens should raise an error instead of starting fitting.

The experiment runner itself expects a completed `.pt` file when `lens_path` is explicit. `prepare_assets` returns that resolved file automatically:

```python
from pathlib import Path
from jspace_donot import prepare_assets, ExperimentConfig, run_experiment

assets = prepare_assets(
    root=Path("."),
    datasets=("gsm8k", "math500", "raceh"),
    model_dir=None,
    lens_path=None,
    fit_if_missing=True,
)

result = run_experiment(ExperimentConfig(
    dataset="gsm8k",
    root=assets.root,
    lens_path=assets.lens_path,
))
```

## WikiText download and J++ fitting

These stages are included in `prepare.py` and the main notebook. They can also be run separately:

```bash
python scripts/download_model.py --root .
python scripts/setup_jpp.py --root . --source-only
python scripts/prepare_wikitext103.py --root .
python scripts/fit_jpp.py --root .
```

The model is the official unquantized [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B). The download script accepts `--revision` and records the resolved model revision.

### Fitting corpus

The corpus is **`Salesforce/wikitext`, `wikitext-103-raw-v1`, TRAIN**, derived from Wikipedia. The preparation script downloads the dataset, materializes TRAIN under `data/wikitext103_raw/train/`, and records the dataset revision.

Prompt selection follows the original J++ recipe exactly:

- Keep the original dataset order.
- Take the first **1,000** records satisfying `len(text.strip()) >= 600`.
- Preserve the original text without shuffling or stripping the saved prompt.
- Use all 1,000 prompts for routers and the first **64** for expert Jacobians.

The saved pool is:

```text
jlens_outputs/qwen3_4b/jpp_fit/wikitext103_raw_train_first_1000_minchars600.json
```

`prepare_wikitext103.py --stream` optionally retrieves only enough records to create the pool. The default downloads and saves the full training split. The plain J-lens shuffled prompt pool is not used for J++ fitting.

### Full fitting recipe

The fitter executes the official implementation:

1. Collect source activations and fit PCA/K-means routers.
2. Estimate LRP-filtered expert Jacobians.
3. Fit signed expert weights on the readout development subset.
4. Combine experts into one fixed FP32 matrix per source layer and export the final lens.

| Fitting setting | Default |
|---|---|
| Seed | **0** |
| Source blocks / target block | 0–34 / 35 |
| Router / Jacobian prompts | 1,000 / 64 |
| Maximum sequence length / skipped initial positions | 128 / 16 |
| Experts / router PCA dimension | 8 / 64 |
| Layers per group / Jacobian rows per pass | 4 / 8 |

Qwen weights remain frozen. Task TRAIN/TEST examples are not used for lens fitting. Signed expert weights use the original readout-development protocol. The router is used during fitting; the combined lens applies a fixed matrix at inference.

Full fitting is an expensive GPU operation. The fitter saves and resumes its router, expert, and weight stages using identity checks. Reducing task sample counts does not reduce the full lens-fitting job. To reduce peak fitting memory without changing prompt counts, fitted layers, or seed:

```bash
python prepare.py --root . --layers-per-group 2 --jacobian-rows-per-pass 4
```

Completed exports are discovered recursively:

```text
jlens_outputs/qwen3_4b/jpp_fit/**/qwen3_4b_jpp_lens.pt
jlens_outputs/qwen3_4b/jpp_fit/**/qwen3_4b_jpp_manifest.json
```

A plain J-lens checkpoint cannot substitute for this J++ checkpoint.

## Task datasets

Preparation downloads these sources and writes JSON lists consumed directly by the task adapters:

| Task | Calibration pool | Held-out TEST | Output files |
|---|---|---|---|
| GSM8K | `openai/gsm8k`, `main`, 7,473 TRAIN rows | 1,319 rows | `data/gsm8k_official_train.json`, `data/gsm_original_test.json` |
| MATH-500 | Seven `EleutherAI/hendrycks_math` TRAIN subjects, 7,500 rows before cleaning | `HuggingFaceH4/MATH-500`, 500 rows | `data/math500/math_original_train.json`, `data/math500/math500_test.json` |
| RACE-H | `ehovy/race`, `high`, 62,445 TRAIN rows | 3,498 rows | `data/race/race_high_{train,validation,test}.json` |

Individual entry points:

```bash
python scripts/prepare_gsm8k.py --root .
python scripts/prepare_math500.py --root .
python scripts/prepare_raceh.py --root .
```

Resolved dataset revisions and provenance are recorded. GSM8K and RACE-H accept `--revision`; MATH accepts `--train-revision` and `--test-revision`.

GSM8K records preserve `question` and the original rationale in `answer`, adding normalized decimal `gold` from the terminal `####` label. Model prompts do not include the rationale. To verify existing files against the complete official ordered question/gold sequences without replacing them:

```bash
python scripts/prepare_gsm8k.py --root . --verify-existing
```

MATH preparation extracts the final balanced boxed TRAIN answer, removes duplicate/conflicting questions and TRAIN/TEST overlaps, and preserves all 500 TEST rows. RACE-H preserves passage/question/options formatting and official duplicate instances. Its 3,451-row validation split is downloaded and audited but is not used to fit or tune stop heads. RACE folds are passage-disjoint; overlapping TRAIN/TEST passages are excluded from the calibration pool using text only.

Existing different dataset files are not silently overwritten. A custom filtered GSM8K test file should be kept distinct from the official 1,319-row split.

## Experiment configuration

The main notebook runs all three tasks. For an individual prepared task:

```python
from jspace_donot import ExperimentConfig, run_experiment

result = run_experiment(ExperimentConfig(
    dataset="gsm8k",
    schedule="wait",          # default: wait | token
    every_n_tokens=256,        # used only in token mode
    cache_mode="resume",      # resume | read_only | off
))
result.summary
```

```bash
python run.py --dataset gsm8k --schedule wait
python run.py --dataset math500 --schedule token --every-n-tokens 256
python run.py --dataset raceh --schedule wait
python run.py --dataset gsm8k --dry-run
```

The minimum stopping position is 256 tokens. `max_monitor_tokens` optionally limits monitoring; its default is `None`. Wait scheduling uses the original case-insensitive word-plus-delimiter rule and exact byte-BPE alignment. It scores the state at the beginning of the cue; detecting that cue requires lookahead, which is included in logical-token accounting.

### Experimental settings

| Setting | GSM8K | MATH-500 | RACE-H |
|---|---|---|---|
| Stop-head seed / TRAIN questions | **2026 / 256** | **2026 / 256** | **2026 / 256** |
| TEST | Complete supplied TEST file | All 500 | All 3,498 |
| Four OOF folds | Question-disjoint KFold | Question-disjoint KFold | Passage-disjoint GroupKFold |
| Natural generation | Fixed batches, requested size 4 | Per-question serial | Per-question serial |
| Natural token budget | 8,192 | 8,192 | 8,192 |
| Sampling | Temperature .6, top-p .95, top-k 20 | Same | Same |
| Greedy forced-answer budget | 64 | 128 | 128 |
| Prefix guard | 12,000 | 14,000 | 14,000 |
| Correctness | Last boxed numeric answer | Boxed-expression `math_verify` equivalence | Strict last boxed A–D |

The head reads **16 raw J++ log-scores**: four candidate tokens across four selected source blocks, in layer-major order. Each block proposes its top eight vocabulary IDs; the union is ranked by later-half mean J++ log-score, retaining four candidates. Normally the selected blocks are 0, 11, 23, and 34. Candidate IDs, positions, temporal deltas, gold labels, and scalar diagnostics do not enter the classifier.

LR uses StandardScaler and question-balanced LogisticRegression with C in `.1, 1, 10`. MLP retains both training normalizations, `Linear(16,32) → Tanh → Linear(32,1)`, and 80 full-batch Adam steps with learning rate .01, weight decay .01, and question-balanced BCE. A single-class training fold uses the original constant-classifier fallback.

Cutoffs `.2, .4, .6, .75, .85, .95` and up to six Pareto points per family are selected on TRAIN OOF predictions. Final heads are fitted on TRAIN before evaluating their frozen policies on TEST. The runner fits only J++ LR/MLP heads. Full forced-answer results are retained separately for protocol and token-cost auditing.

`wait` is the requested default. To use the source notebooks' active token schedule, select `schedule="token", every_n_tokens=256`.

### Caches and historical generation

| Mode | Behavior |
|---|---|
| `resume` | Reuse compatible natural CoTs, readouts, and answer branches; compute missing entries. Existing historical GSM8K source holes still fail rather than regrouping examples. |
| `read_only` | Require existing compatible CoTs, readouts, and answer branches. Misses fail before their expensive computation. Heads are refitted and reports are still saved. |
| `off` | Use a temporary intermediate cache during the run, then remove it. Keep reports and interpretation artifacts. |

Temporary within-run reuse keeps one sampled CoT per question. These settings do not remove model files, prepared datasets, or the fitted lens.

For GSM8K, `source_mode="auto"` reuses a present homogeneous historical source cache. With no source cache, it generates fixed batches in the recorded selected-TRAIN-then-TEST order. `historical` requires the original source records; `fresh` uses a separate fresh-source namespace. Use `cache_root` to isolate different cohorts.

Exact historical replay requires the original generation records, batch membership, and compatible model/tokenizer assets. The same seed and declared sampling settings alone do not guarantee identical sampled CoTs across different batches or library versions. The original record-integrity fallback is retained and identified in `source_cache_protocol.json` when an old source signature cannot be independently recovered.

Each run gets a new report directory unless `output` is supplied. Reusing a report directory requires the same method/source protocol; use a new output for a changed schedule.

## Outputs and interpretation

Results are saved under `jlens_outputs/qwen3_4b/<task>_JSpaceDoNOT_<timestamp>/` by default:

| File | Contents |
|---|---|
| `test_summary.csv` | J++ LR/MLP policy results |
| `test_per_question_j.csv` | Per-question decisions and token accounting |
| `full_reference.csv` | Forced-answer Full reference |
| `frozen_train_policies.csv` | TRAIN-selected policies |
| Calibration/test manifests and protocol JSONs | Cohorts, assets, readout, scheduling, and source provenance |
| `cache_audit.json`, `environment.json` | Cache activity and package versions |
| `analysis_bundle.pkl` | Actual trained CPU heads and TRAIN/TEST J features |
| `checkpoint_context.json`, `generation_context.json` | Candidate tokens, evaluated outcomes, and exact CoT boundaries |

Only load pickle artifacts you trust. Interpretation runs on CPU from these saved objects without refitting heads or invoking Qwen:

```python
from jspace_donot.interpretation import interpret_run, StopInterpreter

interpret_run(result.output, linear_C=.1, mlp_C=1., cutoff=.85,
              max_ig_examples=300, ig_steps=48, find_cases=True)

explainer = StopInterpreter.load(result.output)
cases = explainer.find_stop_flips(cutoff=.85)
if not cases.empty:
    case = cases.iloc[0]
    raw, context = explainer.checkpoint(case.question_id, int(case.step))
    table, _ = explainer.group_ablation(
        case.question_id, int(case.step),
        groups={"L3 ranks 1+2": [8, 9], "L4 ranks 1+2": [12, 13]},
    )
    sweep = explainer.sweep(case.question_id, int(case.step), indices=[8, 9])
```

Exports include exact LR logit contributions, midpoint MLP Integrated Gradients through both normalizations, IG completeness residuals, TRAIN-median feature replacement, and PDF/SVG/PNG publication panels. IG uses the TRAIN raw-score mean as its baseline. Group interventions retain candidate order; score sweeps include the original scores and exactly zero. Flat indices use `layer_slot * 4 + rank_slot`, with zero-based slots.

A constant fallback head has no LR coefficients or MLP gradients; interpretation reports that limitation. Case searches may return no STOP-to-CONTINUE examples. These interventions measure the stopping predictor's sensitivity, not causal effects within Qwen.

## Small-sample validation

The separately supplied `JSpaceDoNOT_GSM8K_Smoke_Validation.ipynb` uses the same package and includes complete model/WikiText/lens preparation. Its default task run uses **8 TRAIN questions and 2 TEST questions**, a 2,048-token natural budget, and token checks at 256 and 512. Missing lenses still require the full original J++ fit.

It checks cohort sizes, cached readout-to-feature reconstruction, saved-head predictions, policy exits, answer scoring, token accounting, summary aggregation, and a read-only replay. Reduced cohorts, truncation, absent Wait cues, and constant heads limit what such a run covers; small-sample accuracy is not a full-benchmark result.

## Verification

```bash
python -m compileall -q jspace_donot scripts prepare.py run.py
python -m unittest discover -s tests -v
```

The verification record includes **29 passing CPU tests**, real preparation of all task datasets, a real WikiText download, and exact equality of all 1,000 prepared prompts with the pinned official loader. Both notebooks have validated schemas and compiled code cells. The tiny-backbone readout check requires the upstream `jlens` source to be importable; fitting and experiment workers set their source path automatically.

See [`docs/VALIDATION.md`](docs/VALIDATION.md), [`docs/source_function_parity.json`](docs/source_function_parity.json), [`docs/fit_recipe_parity.json`](docs/fit_recipe_parity.json), and [`docs/wikitext_validation.json`](docs/wikitext_validation.json) for checks and provenance.

A full Qwen3-4B GPU fit and the complete GPU experiments have **not** been reproduced in this workspace. Evaluation remains the original **offline trajectory replay**: natural CoTs are generated/loaded before checkpoint observations and forced-answer branches. Reported token savings are logical-token accounting, not measured online wall-clock speedups.
