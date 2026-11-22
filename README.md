# JSpaceDoNOT

Qwen3-4B experiments on **GSM8K, MATH-500 and RACE-H**, with reusable method and interpretation APIs and one notebook: [`Run_JSpaceDoNOT.ipynb`](Run_JSpaceDoNOT.ipynb).

This package continues the November 17 `JSpaceDoNoT_GitHub.zip`. The uploaded November 21 notebooks are the authority for experimental settings. Only the J++ stop heads (LR and MLP) are fitted and evaluated. Other-method code and comparison plots are removed. Full forced-answer results remain in a separate `full_reference.csv` for protocol auditing and token-cost context.

## Complete workflow from a clean checkout

Open **`Run_JSpaceDoNOT.ipynb`** and run in order. Its preparation cell now explicitly performs the whole sequence:

1. Download missing official Qwen3-4B weights (or reuse `MODEL_DIR`).
2. Fetch the pinned official J++ source, without installing/downgrading your environment.
3. Download **WikiText-103 RAW TRAIN**, save it locally, and create the exact original J++ 1,000-prompt pool.
4. If no completed matching J++ lens is available, run the original full router / LRP expert-Jacobian / signed-weight fitting and export `.pt` + manifest.
5. Download/process the selected task datasets.
6. Pass the resolved lens file directly into the three JSpaceDoNOT experiments and interpretation.

Equivalent commands:

```bash
python prepare.py --root .
python run.py --dataset gsm8k
python run.py --dataset math500
python run.py --dataset raceh
```

Or prepare and run a single dataset together:

```bash
python run.py --dataset gsm8k --prepare
```

`prepare.py --datasets gsm8k` prepares only that task. It still performs the **full original J++ fit** when the lens is missing; reducing task sample counts does not make lens fitting cheaper. Existing completed lens files or their containing folders can be supplied with `--lens-path`; the selected lens and model checksums are validated. `--model-dir` reuses model files outside the preparation root. `--fit-python /path/to/jpp-env/bin/python` runs fitting in an existing separate environment; evaluation continues in the current environment.

For independent WikiText preparation:

```bash
python scripts/prepare_wikitext103.py --root .
```

This uses `Salesforce/wikitext`, **`wikitext-103-raw-v1`**, TRAIN. It is a Wikipedia-derived corpus, not a download of the whole Wikipedia site. The full training dataset is materialized under `data/wikitext103_raw/train/`; the resolved revision is recorded. The pool is written to `jlens_outputs/qwen3_4b/jpp_fit/wikitext103_raw_train_first_1000_minchars600.json`. Selection exactly follows the uploaded J++ recipe: original row order, first 1,000 records with `len(text.strip()) >= 600`, original text retained, no shuffle; Jacobian fitting uses the first 64. `--stream` optionally downloads only enough data for the pool. Do not substitute the ordinary J-lens notebook's shuffled seed43, 64-prompt corpus.

Fitting keeps seed **0**, all source blocks 0–34, final target 35, eight experts, sequence length 128 and skip-first 16. Only `--layers-per-group` and `--jacobian-rows-per-pass` are exposed for memory tuning; their defaults remain 4 and 8. Stopping experiments retain seed **2026**. The preparation API reuses completed fits; it never forces a full refit just because a new report folder is created.

The standalone smoke notebook also contains this complete asset-preparation stage. It can reuse your explicit existing lens, or fit a missing one before the 8-TRAIN/2-TEST task validation.

## Install

Use **Python 3.12** and CUDA-capable PyTorch supporting BF16. Install the PyTorch build appropriate for your GPU first, then:

```bash
pip install -r requirements.txt
python scripts/setup_jpp.py --root . --source-only
pip install --no-deps -e .
```

The setup script checks out the fitting recipe's upstream commit `d7efc0de372e20589e899188c0cfe8434acc63ec` from [safety-research/jpp_lens](https://github.com/safety-research/jpp_lens). This source provides `jlens`, `workspace_lens`, and `lens_evals`; fitting and experiment subprocesses use it directly. If you explicitly omit `--source-only`, the setup script installs it with `--no-deps` because the upstream project's declared environment differs from the supplied experiment environment; it must not replace that environment silently. The recipe imports and a tiny CPU Qwen3 readout are checked separately; a complete GPU fit remains to be run here. `requirements.txt` is a base dependency list, not a validated lock for every GPU environment. Its earlier unverified Transformers `<5` restriction has been removed; preserve your working environment and export exact versions after a successful GPU run. Existing experimental environments can keep their installed versions; each completed run exports `environment.json`.

## 1. Download Qwen3-4B

```bash
python scripts/download_model.py --root .
```

Downloads official unquantized [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) to `base_models/Qwen3-4B`. `--revision COMMIT` selects a model revision; the resolved revision is recorded in `model_download_manifest.json`. To reproduce an existing lens, **reuse the model/tokenizer bytes that lens was fitted on**: inference checks the fitting manifest's SHA-256 hashes and rejects mismatches.

## 2. Prepare datasets

```bash
python scripts/prepare_gsm8k.py --root .
python scripts/prepare_math500.py --root .
python scripts/prepare_raceh.py --root .
```

| Experiment | TRAIN source | TEST source | Saved paths |
|---|---|---|---|
| GSM8K | `openai/gsm8k`, `main`, 7,473 rows | Official test, 1,319 rows | `data/gsm8k_official_train.json`, `data/gsm_original_test.json` |
| MATH-500 | Seven `EleutherAI/hendrycks_math` train subjects, 7,500 rows before cleaning | `HuggingFaceH4/MATH-500`, 500 rows | `data/math500/math_original_train.json`, `data/math500/math500_test.json` |
| RACE-H | `ehovy/race`, `high`, 62,445 rows | Official test, 3,498 rows | `data/race/race_high_{train,validation,test}.json` |

Preparation records resolved dataset revisions. GSM8K and RACE-H accept `--revision`; MATH accepts `--train-revision` and `--test-revision`. Existing different data are not silently replaced.

**GSM8K format.** The supplied experiment reads a JSON **list**, with `question` and a numeric gold extracted from `gold`, `target`, `final_answer`, or the original `answer` (including a terminal `#### NUMBER`). The downloader preserves `question` and the original rationale in `answer`, and adds normalized decimal `gold`. Question IDs use the original SHA-256 of whitespace-collapsed question text. No rationale is included in model prompts. The file names and loader identify the intended official train/test protocol, but the original JSON data themselves were not supplied with this request, so their actual contents cannot be certified from notebooks alone. Compare existing files without replacing them:

```bash
python scripts/prepare_gsm8k.py --root . --verify-existing
```

This checks the complete ordered question/gold sequences against official GSM8K. A custom filtered/augmented test file must not be relabelled as the official 1,319-question test split.

MATH processing retains the uploaded balanced-box TRAIN label extraction and deduplication/conflicting-gold/TEST-overlap exclusions; MATH-500 stays intact. RACE processing retains passage/question/options formatting, A–D golds, passage group IDs, official row order and duplicates. The uploaded RACE downloader wrote directly under `data/` while its experiment reads `data/race/`; the wrapper corrects that path mismatch. Official validation (3,451 rows) is downloaded and audited, but is not used for stop-head training or tuning.

## 3. Prepare the fitted J++ lens

Reuse your existing completed **J++** checkpoint and matching manifest whenever available:

```text
jlens_outputs/qwen3_4b/jpp_fit/**/qwen3_4b_jpp_lens.pt
jlens_outputs/qwen3_4b/jpp_fit/**/qwen3_4b_jpp_manifest.json
```

Exactly one pair must be discoverable, or pass `lens_path` / `--lens-path` explicitly. A plain J-lens from `Qwen3_4B_download_fit_JLens.ipynb` is a different asset and cannot replace the J++ lens loaded by these experiments.

For a new fit, the official J++ recipe recovered from the earlier package is now an executable entry point:

```bash
python scripts/fit_jpp.py --root .
```

It reads the explicitly prepared WikiText-103 raw prompt JSON using the same selection as the original `jlens.examples.load_wikitext_prompts`, fits activation routers and LRP expert Jacobians, fits signed expert weights on the original readout development items, and saves combined FP32 maps plus the exact model-asset manifest. Its original defaults are retained: router prompts 1,000; Jacobian prompts 64; sequence length 128; skip-first 16; 8 experts; router PCA dimension 64; all source blocks 0–34 to final block 35; fitting seed **0**. Stop-head experiments separately use seed **2026**. The fitting script resumes completed stages using their original identity guards. This is an expensive GPU job; skip it when you already have the matching completed lens. No GSM8K/MATH/RACE examples are used to fit that lens.

## 4. Run the three experiments

Open **`Run_JSpaceDoNOT.ipynb` from the repository root**. Configuration and execution are short calls; implementation lives inside `jspace_donot/`. Each experiment runs in a separate process so Qwen GPU memory is released before the next dataset.

```python
from pathlib import Path
from jspace_donot import ExperimentConfig, run_experiment

result = run_experiment(ExperimentConfig(
    dataset="gsm8k", root=Path("."),
    schedule="wait",                 # default
    every_n_tokens=256,               # used only when schedule="token"
    cache_mode="resume",             # resume | read_only | off
))
result.summary
```

Equivalent CLI:

```bash
python run.py --dataset gsm8k
python run.py --dataset math500 --schedule token --every-n-tokens 256
python run.py --dataset raceh --schedule wait
python run.py --dataset gsm8k --dry-run
```

`--min-stop-tokens` defaults to 256 for either schedule. `--max-monitor-tokens` optionally limits monitoring. Wait markers default to `("Wait",)` and use the original case-insensitive word-plus-delimiter matching and exact byte-BPE token alignment. **The scored state is at the start of the cue, and detecting that cue needs lookahead.** Logical token accounting includes that lookahead; do not call it free online detection.

### Preserved settings

| Setting | GSM8K | MATH-500 | RACE-H |
|---|---|---|---|
| Stop-head seed / TRAIN sample | 2026 / 256 | 2026 / 256 | 2026 / 256 |
| TEST | Entire supplied test file | All 500 | All 3,498, duplicates retained |
| Four OOF folds | Question-disjoint KFold | Question-disjoint KFold | Passage-disjoint GroupKFold |
| Natural generation | Historical fixed batches, requested batch size 4 | Original per-question serial implementation | Original per-question serial implementation |
| Natural budget | 8,192 | 8,192 | 8,192 |
| Sampling | temperature .6, top-p .95, top-k 20 | Same | Same |
| Greedy forced-answer budget | 64 | 128 | 128 |
| Prefix context guard | 12,000 | 14,000 | 14,000 |
| Correctness | Last boxed numeric answer | Original `math_verify` boxed-expression scoring | Strict last boxed A–D |

All tasks retain four evenly spaced fitted source blocks (normally 0, 11, 23, 34), a pool from top 8 vocabulary IDs per source layer, J-only proposals, and the top 4 candidates ranked by later-half mean J log-score. The head receives **16 raw log-scores in layer-major order**. No candidate IDs, positions, temporal deltas, gold labels or auxiliary scalar diagnostics enter the predictor. Original readout auxiliary LM/hidden fields remain solely for observation/cache compatibility; no LM/hidden stop head is fitted.

LR: StandardScaler + question-balanced LogisticRegression, C in `.1, 1, 10`. MLP: StandardScaler, the original second mean/std normalization, `16→32→Tanh→1`, 80 full-batch Adam steps, learning rate .01 and weight decay .01, question-balanced BCE. A single-class fold retains the original constant-classifier fallback. Such a head has no LR coefficients or MLP gradients, and interpretation reports that explicitly.

Thresholds `.2, .4, .6, .75, .85, .95` and at most six Pareto points per head family are selected using TRAIN OOF predictions. Final heads are fitted on TRAIN and frozen before TEST outcomes at selected exits. Changing the source notebooks' active `token` default to **`wait` is intentional**; use `schedule="token", every_n_tokens=256` to match those uploaded runs' schedule.

### GSM8K historical versus fresh sources

The November 17 package had replaced batch-size-4 generation with a different serial generator. This release restores the supplied historical loader with its prompt, batch journal, member key, RNG seed, and source-signature validation.

`source_mode="auto"` reuses a present historical cache without filling its holes by regrouping examples. If no source exists, it creates fixed batches of four in the recorded selected-TRAIN-then-TEST order. `source_mode="historical"` requires the original sources. `source_mode="fresh"` creates a separate `fresh_fixed_batch_v1` namespace and freezes its cohort/order. Use `cache_root` to isolate a new cohort. Fresh batching has the same declared sampling settings but cannot guarantee identical historical CoTs without their original batch membership, model/library fingerprints and RNG trajectory. For **exact historical replay**, use `source_mode="historical"` and retain all original batch members/journals.

The historical notebook optionally accepts record-integrity-only provenance when the old signature cannot be recovered. That original behavior is retained and identified in `source_cache_protocol.json`; it is not an independent proof of old model/sampling bytes.

### Cache options

| `cache_mode` | Behavior |
|---|---|
| `resume` (default) | Reuse exact compatible natural trajectories, answer branches and vocabulary readouts; compute missing entries. Historical GSM8K source holes still raise an error. |
| `read_only` | Require existing natural trajectories, answer branches and readouts. Cache misses fail before their expensive computation; reports and refitted heads are still saved. |
| `off` | Use a temporary intermediate cache for this run, then remove it. No persistent generation/readout/branch cache is reused or created. Reports and interpretation artifacts remain available. |

Within-run temporary reuse is necessary to keep one sampled trajectory per question. `off` does not mean regenerating a different CoT at every checkpoint. Model downloads, prepared datasets and the fitted lens are durable inputs, independent of these cache modes.

Use `output=Path(...)` to resume a report folder with the **same** method/source protocol. Use a new output for a changed schedule. Source/readout cache identities and asset hashes remain checked. `calibration`, `test_limit`, and `max_new_tokens` are exposed for smoke runs; reduced values are not the full experimental setting.

### Outputs

Each task saves `test_summary.csv` (J++ LR/MLP only), `test_per_question_j.csv`, `full_reference.csv`, calibration/test manifests, OOF-selected policies, method/schedule/readout/source provenance, cache audit and environment versions. `analysis_bundle.pkl` stores the actual trained CPU heads and TRAIN/TEST J features. `checkpoint_context.json` plus `generation_context.json` preserve candidate tokens and exact CoT boundaries for later case studies without retaining Qwen in memory. Only load your own pickle artifacts.

These are the original **offline trajectory-replay experiments**: natural CoTs are generated/loaded first, then readouts and forced answer branches are evaluated. Logical-token savings are not measured wall-clock inference savings. Refactoring does not convert the experiment into an online stopping service.

## 5. Interpret the actual LR/MLP

```python
from jspace_donot.interpretation import interpret_run, StopInterpreter

interpret_run(result.output, linear_C=.1, cutoff=.85,
              max_ig_examples=300, ig_steps=48, find_cases=True)

explainer = StopInterpreter.load(result.output)
cases = explainer.find_stop_flips(cutoff=.85)
```

This runs on CPU after restart and never refits the heads or invokes Qwen. It preserves the supplied analysis's exact LR logit decomposition, MLP midpoint Integrated Gradients through both normalizations (TRAIN raw mean baseline), and feature replacement by the TRAIN median. CSVs, PNG/PDF heatmaps and numerical IG completeness residuals are exported. The original rounded-tile publication panels and LR/MLP sensitivity appendix are also retained (PDF/SVG/PNG) for the original 4×4 feature layout. `interpretation_plots.plot_score_sweep` exports the actual-score intervention curve and threshold crossings. LR contributions reconstruct the logit exactly; IG is numerical and its residual is reported.

Generic case APIs replace the original notebook's undefined helpers and fixed `CASE_INDEX=2` / hardcoded token assumptions:

```python
# Select any actual checkpoint returned by the case search.
case = cases.iloc[0]   # guard cases.empty first
raw, context = explainer.checkpoint(case.question_id, int(case.step))
print(context["token_strings"], context["cot_prefix"])
print(context["outcome"])  # None if this TEST exit was not evaluated; no answer is invented.

# Flat indices: source-layer slot * 4 + candidate-rank slot (both zero-based).
table, context = explainer.group_ablation(
    case.question_id, int(case.step),
    groups={"L3 ranks 1+2": [8, 9], "L4 ranks 1+2": [12, 13],
            "Both": [8, 9, 12, 13]}, cutoff=.85)
sweep = explainer.sweep(case.question_id, int(case.step), indices=[8, 9])
```

Group replacement keeps the original candidate order and feeds modified scores into the fitted predictor. `sweep` scales actual selected raw log-scores jointly, including the original values and exactly zero. Token semantics and paper-specific group definitions remain choices of the analyst; the API does not assume every new case contains `total`, `Total`, `5` or `4`. A STOP→CONTINUE search can return no cases, which is a valid result. These interventions establish predictor sensitivity, not causal effects inside Qwen.

## Verification and provenance

```bash
python -m compileall -q jspace_donot scripts run.py
python -m unittest discover -s tests -v
```

[`docs/source_function_parity.json`](docs/source_function_parity.json) records AST equality against the uploaded notebooks for scheduling, candidate proposals/readouts, raw features, model training/prediction, stopping, answer branches, scoring and dataset loaders. Hidden-control dispatcher wrappers are compared at the original J++ implementation they call. [`docs/VALIDATION.md`](docs/VALIDATION.md) records checks actually completed and their limits. No full Qwen3-4B GPU experiment or lens fit is claimed as reproduced in this workspace.
