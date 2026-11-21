# Validation record

Completed on November 21, 2026. This record describes checks actually run for this refactor.

- Recovered and reused the November 17 `JSpaceDoNoT_GitHub.zip`; updated it against the eight supplied notebooks.
- **25 CPU tests passed**, including a real randomly initialized tiny Qwen3 backbone, answer-boundary activation hooks, identity-lens transport and independent direct-forward log-score comparison. CUDA synchronization alone was mocked for this CPU test; the backbone and readout math were real.
- **57 key function AST comparisons across the three tasks match the uploaded original J++ implementations**: scheduling and exact token alignment, candidate proposal/readout, raw feature construction, fitting/prediction, stopping and cost accounting, answer branches, parsing/scoring, and dataset loaders. The retained original implementation is used before unrelated hidden-baseline dispatch wrappers. See `source_function_parity.json`.
- Tested question-balanced LR/MLP fitting, raw feature layer-major ordering and temporal independence, STOP/fallback decisions, Wait lookahead, fixed-batch seed/member identity and left padding, strict read-only misses, and refusal to regroup a changed fresh cohort.
- Verified exact LR logit reconstruction, MLP predictions through both fitted normalizations, midpoint IG completeness, saved-head CPU interpretation, group replacement and score-sweep APIs, and original PDF/SVG/PNG publication panel exports.
- Downloaded and processed **all three real datasets** using the delivered scripts. GSM8K: 7,473 TRAIN and 1,319 TEST; verification mode passed. MATH: 7,500 original TRAIN, 4 unboxed skipped and one duplicate extra row removed, yielding 7,495 usable TRAIN; all 500 TEST kept. RACE-H: 62,445 / 3,451 / 3,498 official rows, validation unused. Dataset revisions and content SHA-256 are in the accompanying source manifests.
- Executed each task's actual configuration, parser, dataset loader, overlap audits, and seed-fixed cohort selection on those downloaded outputs, up to model loading. Each selected 256 TRAIN questions and the complete official TEST split. RACE-H retained duplicate TRAIN instances and detected no shared TRAIN/TEST passages. Cohort identities are in `dataset_validation.json`.
- Cloned the exact upstream J++ commit `d7efc0de372e20589e899188c0cfe8434acc63ec`; validated fitting-recipe imports and the frozen unembedding adapter with Transformers 4.57.6. Upstream source was installed without replacing the experiment environment.
- All packaged Python files compile. Notebook schema validates and every code cell compiles. CLI defaults/preflight and fitting CLI help execute. Static undefined-name checks passed.

## Deliberate changes and limits

- Default scheduling changes from the uploaded notebooks' active `token` to requested `wait`. Set `schedule="token", every_n_tokens=256` to reproduce their scheduling setting.
- Restored the uploaded GSM8K historical fixed-batch loader instead of the earlier ZIP's serial replacement. Fresh generation uses requested batch size 4 and records its own cohort and batch membership. It cannot promise the same historical CoTs unless the original batch/RNG/assets are available. Historical source validation remains as strict as the supplied notebook, including its explicitly audited record-integrity fallback.
- Removed other-method execution and assets. Full is saved separately as an answer-protocol/cost reference. Removed raw-hidden-only fields from RACE's fitting identity; resulting policy IDs can differ while the J++ fit and evaluation math remain unchanged.
- Fixed the RACE downloader's `data/` versus experiment's `data/race/` mismatch. No RACE prompt or label transformation changed.
- Interpretation uses the saved actual heads rather than replaying/retraining a source notebook. The supplied advanced cells referenced missing helpers and fixed tokens/case indices; generic explicit APIs now supply those operations. Original global interpretation math and rounded-tile publication panels are retained.
- The environment has **no CUDA GPU**, original Qwen3-4B weights, or the user's completed lens. No complete 4B GPU experiment or official J++ fit was run; GPU numerical results, memory use and runtime have not been reproduced. The original user dataset JSON files were also absent, so downloaded official files cannot certify the identity of a custom historical test file.
- The experiment remains an offline trajectory replay. Token savings are logical accounting, not measured online wall-clock speedups.

## Python 3.11 source-parity compatibility fix

The reference AST hashes were created with Python 3.12. Python 3.11 omits the empty `type_params` field on function/class nodes. For `decoded_bounds`, omitting that field exactly reproduces the reported `082ec41d...` hash instead of the reference `aa47738c...`; this is not evidence of a changed method. The test now adds that empty field only when absent, preserving all original reference hashes. All 57 reference functions passed a simulated Python 3.11 missing-field normalization check, and the 25-test CPU suite was rerun. Full tests under the user's actual Python 3.11 installation still need to be rerun there. No inference, training, dataset or cache code was changed by this fix.
