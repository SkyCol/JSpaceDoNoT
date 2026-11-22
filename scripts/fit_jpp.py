"""Official J++ fitting recipe recovered from the November 17 source package.
The defaults, including fitting seed 0, are preserved independently of stop-head seed 2026.
"""
import argparse

def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',default='.')
    parser.add_argument('--repo',default=None)
    parser.add_argument('--model-dir',default=None)
    parser.add_argument('--prompts-json',default=None)
    parser.add_argument('--layers-per-group',type=int,default=4)
    parser.add_argument('--jacobian-rows-per-pass',type=int,default=8)
    args=parser.parse_args(argv)
    # Reference extraction of Qwen3_4B_fit_official_JPlusPlus_Lens (1).ipynb; requires notebook environment and external dependencies.

    # =============== Cell 2 ===============
    from pathlib import Path

    ROOT = Path(args.root).resolve()  # Set your project folder explicitly when needed.
    REPO_DIR = Path(args.repo).resolve() if args.repo else ROOT / 'external' / 'jpp_lens'
    OFFICIAL_COMMIT = 'd7efc0de372e20589e899188c0cfe8434acc63ec'
    MODEL_ID = 'Qwen/Qwen3-4B'
    MODEL_DIR = Path(args.model_dir).expanduser().resolve() if args.model_dir else ROOT / 'base_models' / 'Qwen3-4B'
    ARTIFACT_ROOT = ROOT / 'jlens_outputs' / 'qwen3_4b' / 'jpp_fit'
    EXPERIMENT_TAG = 'official_qwen3_4b_v1'

    SOURCE_BLOCKS = None       # None = 0...34, target = final block 35.
    LAYERS_PER_GROUP = args.layers_per_group       # Increase for speed if HOST RAM allows.
    JACOBIAN_ROWS_PER_PASS = args.jacobian_rows_per_pass # Lower to 4 or 2 for lower GPU memory.
    CPU_THREADS = 8
    N_ROUTER_PROMPTS = 1000
    N_JACOBIAN_PROMPTS = 64
    MAX_SEQ_LEN = 128
    SKIP_FIRST = 16
    N_EXPERTS = 8
    PCA_DIM = 64
    MIN_EXPERT_POSITIONS = 50
    SEED = 0
    WIKITEXT_PROMPTS_JSON = args.prompts_json  # Optional existing JSON list of >=1,000 strings.

    DEVELOPMENT_FRACTION = 0.20
    READOUT_MAX_SEQ_LEN = 512
    WEIGHT_STEPS = 200
    WEIGHT_LR = 0.05
    WEIGHT_PASS_K = 10
    WEIGHT_TOP_N = 3
    WEIGHT_TEMPERATURE = 0.1
    WEIGHT_EXACT_EVERY = 5

    # =============== Cell 3 ===============
    import gc
    import hashlib
    import importlib.metadata
    import json
    import logging
    import os
    import subprocess
    import sys
    import time
    from dataclasses import asdict, replace
    from datetime import datetime, timezone
    actual_commit=subprocess.check_output(['git','-C',str(REPO_DIR),'rev-parse','HEAD'],text=True).strip()
    if actual_commit!=OFFICIAL_COMMIT:raise RuntimeError('Official J++ checkout differs from the fitting recipe commit')
    # Explicit data-preparation step before loading a GPU model; same prompt list as the source notebook.
    if WIKITEXT_PROMPTS_JSON is None:
        subprocess.run([sys.executable,str(Path(__file__).with_name('prepare_wikitext103.py')),
            '--root',str(ROOT),'--n-prompts',str(N_ROUTER_PROMPTS)],check=True)
        WIKITEXT_PROMPTS_JSON = str(ARTIFACT_ROOT / f'wikitext103_raw_train_first_{N_ROUTER_PROMPTS}_minchars600.json')
    sys.path.insert(0, str(REPO_DIR / 'src'))
    sys.path.insert(0, str(REPO_DIR / 'scripts'))

    from IPython.display import display
    import numpy as np
    import pandas as pd
    import torch
    import transformers
    import jlens
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from workspace_lens.config import LensConfig
    from workspace_lens.routing.cluster import collect_valid_position_activations
    from workspace_lens.routing.router import ActivationRouterCollection
    from workspace_lens.fitting.relp_fitting import ExpertJacobianRelPTrainer
    from workspace_lens.fitting.condense_experts import CondenseConfig, ExpertJacobians, ExpertWeighting, ExpertWeightingFitter
    from workspace_lens.fitting.utils import expert_checkpoint_filename
    from workspace_lens.lenses.base_lens import BaseLens
    from workspace_lens.lenses.jacobian_lens import JacobianLens
    from workspace_lens.lrp import apply_lrp_rules_for_mode
    from workspace_lens.utils import vocab_size_of
    from lens_evals.readout_evals.readout_eval_items import (
        MACRO_EVALS, load_readout_eval_items, filter_items_by_correctness,
        non_semantic_token_ids, split_fitting_items,
    )
    from lens_evals.readout_evals.readout_evals import ReadoutEvalRunner
    assert 0 < N_JACOBIAN_PROMPTS <= N_ROUTER_PROMPTS
    assert 0 <= SKIP_FIRST < MAX_SEQ_LEN - 1
    assert LAYERS_PER_GROUP > 0 and JACOBIAN_ROWS_PER_PASS > 0
    assert 1 <= WEIGHT_TOP_N <= N_EXPERTS
    torch.set_num_threads(CPU_THREADS)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s', stream=sys.stdout, force=True)
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)

    def digest(obj):
        return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()

    def file_sha(path):
        value = hashlib.sha256()
        with Path(path).open('rb') as stream:
            for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
                value.update(chunk)
        return value.hexdigest()

    def atomic_json(path, value):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + '.tmp')
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        temporary.replace(path)

    def atomic_save(path, writer):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + '.tmp')
        writer(str(temporary))
        temporary.replace(path)

    def guard_identity(directory, identity):
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / 'identity.json'
        if path.exists() and json.loads(path.read_text(encoding='utf-8')) != identity:
            raise RuntimeError(f'Artifact identity mismatch: {directory}')
        if not path.exists():
            atomic_json(path, identity)

    def release_memory():
        gc.collect()
        torch.cuda.empty_cache()

    print('GPU:', torch.cuda.get_device_name(), flush=True)

    # =============== Cell 5 ===============
    asset_paths = sorted(set(MODEL_DIR.glob('*.safetensors')) | set(MODEL_DIR.glob('pytorch_model*.bin')))
    if not asset_paths:
        raise FileNotFoundError('No local model weight files found.')
    asset_paths += sorted(p for p in MODEL_DIR.glob('*.json') if p not in asset_paths)
    for name in ('tokenizer.model', 'vocab.txt', 'merges.txt'):
        if (MODEL_DIR / name).is_file(): asset_paths.append(MODEL_DIR / name)
    model_assets = {}
    for path in asset_paths:
        print('Hashing model asset:', path.name, flush=True)
        model_assets[path.name] = dict(bytes=path.stat().st_size, sha256=file_sha(path))
    environment = {name: importlib.metadata.version(name) for name in
                   ('torch', 'transformers', 'numpy', 'pandas', 'datasets', 'huggingface_hub')}
    base_identity = dict(version='qwen3-official-jpp-fit-v1', tag=EXPERIMENT_TAG,
                         model_id=MODEL_ID, assets=model_assets,
                         environment=environment, attention='eager', dtype='bfloat16')
    BASE_DIR = ARTIFACT_ROOT / EXPERIMENT_TAG / digest(base_identity)[:16]
    guard_identity(BASE_DIR, base_identity)

    pool_path = Path(WIKITEXT_PROMPTS_JSON) if WIKITEXT_PROMPTS_JSON else (
        ARTIFACT_ROOT / f'wikitext103_raw_train_first_{N_ROUTER_PROMPTS}_minchars600.json')
    if pool_path.is_file():
        prompts = json.loads(pool_path.read_text(encoding='utf-8'))
        print('REUSE prompt pool:', pool_path, flush=True)
    else:
        if WIKITEXT_PROMPTS_JSON:
            raise FileNotFoundError(pool_path)
        from jlens.examples import load_wikitext_prompts
        print(f'Loading {N_ROUTER_PROMPTS} WikiText-103 raw training prompts.', flush=True)
        prompts = load_wikitext_prompts(N_ROUTER_PROMPTS)
        atomic_json(pool_path, prompts)
    if not isinstance(prompts, list) or len(prompts) < N_ROUTER_PROMPTS or not all(isinstance(p, str) for p in prompts):
        raise ValueError('Prompt pool must be a JSON list of enough strings.')
    ROUTER_PROMPTS = prompts[:N_ROUTER_PROMPTS]
    FIT_PROMPTS = ROUTER_PROMPTS[:N_JACOBIAN_PROMPTS]
    atomic_json(BASE_DIR / 'router_prompts.json', ROUTER_PROMPTS)
    atomic_json(BASE_DIR / 'jacobian_prompts.json', FIT_PROMPTS)
    print('Prompt identities frozen. No GSM8K data is used.', flush=True)

    # =============== Cell 6 ===============
    print('Loading local Qwen3-4B in BF16.', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    hf_model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, local_files_only=True, dtype=torch.bfloat16,
        attn_implementation='eager').to('cuda').eval()
    if hf_model.config.model_type != 'qwen3':
        raise RuntimeError('Expected the dense Qwen3 model architecture.')
    if MODEL_ID == 'Qwen/Qwen3-4B' and (hf_model.config.num_hidden_layers, hf_model.config.hidden_size) != (36, 2560):
        raise RuntimeError('MODEL_DIR does not have Qwen3-4B layer count and residual width.')
    hf_model.gradient_checkpointing_disable()
    # Stable canonical name for the official compatibility checker; actual local bytes
    # are independently bound by model_assets above, rather than trusting this name.
    hf_model.config._name_or_path = MODEL_ID
    model = jlens.from_hf(hf_model, tokenizer, compile=False)
    TARGET_BLOCK = model.n_layers - 1
    SOURCE_LAYERS = list(range(TARGET_BLOCK)) if SOURCE_BLOCKS is None else sorted(set(map(int, SOURCE_BLOCKS)))
    if not SOURCE_LAYERS or not 0 <= min(SOURCE_LAYERS) <= max(SOURCE_LAYERS) < TARGET_BLOCK:
        raise ValueError('Source blocks must be nonempty and strictly before the final block.')
    if PCA_DIM > model.d_model:
        raise ValueError('PCA_DIM exceeds the residual width.')
    GROUPS = [SOURCE_LAYERS[i:i + LAYERS_PER_GROUP] for i in range(0, len(SOURCE_LAYERS), LAYERS_PER_GROUP)]
    with apply_lrp_rules_for_mode(model, 'rlens') as patch_report:
        if patch_report is None or not patch_report.ln_rule or not patch_report.mlp_rule:
            raise RuntimeError('Official Qwen3 LRP patch did not cover RMSNorm and gated MLP modules.')
    atomic_json(BASE_DIR / 'lrp_patch_modules.json', asdict(patch_report))

    router_identity = dict(base=digest(base_identity), prompts=digest(ROUTER_PROMPTS),
        source_layers=SOURCE_LAYERS, groups=GROUPS, n_experts=N_EXPERTS, pca_dim=PCA_DIM,
        seed=SEED, max_seq_len=MAX_SEQ_LEN, skip_first=SKIP_FIRST)
    ROUTER_DIR = BASE_DIR / ('routers_' + digest(router_identity)[:16])
    guard_identity(ROUTER_DIR, router_identity)
    expert_identity = dict(router=digest(router_identity), prompts=digest(FIT_PROMPTS),
        lrp_mode='rlens', jacobian_rows_per_pass=JACOBIAN_ROWS_PER_PASS,
        min_expert_positions=MIN_EXPERT_POSITIONS, target_offset=-1)
    EXPERT_DIR = BASE_DIR / ('experts_' + digest(expert_identity)[:16])
    guard_identity(EXPERT_DIR, expert_identity)
    print('Source blocks (zero-based):', SOURCE_LAYERS, '| target:', TARGET_BLOCK, flush=True)
    print('Groups:', GROUPS, '| residual width:', model.d_model, flush=True)

    # =============== Cell 8 ===============
    DATA_DIR = REPO_DIR / 'data' / 'jlens' / 'evaluations'
    raw_items = load_readout_eval_items(str(DATA_DIR), slugs=MACRO_EVALS)
    items_identity = digest([asdict(item) for item in raw_items])
    dev_identity = dict(base=digest(base_identity), items=items_identity, fraction=DEVELOPMENT_FRACTION,
        split_seed=SEED, readout_max_seq_len=READOUT_MAX_SEQ_LEN,
        selection='official Qwen3-4B greedy-next-token correctness then scoreable stratified split')
    DEV_DIR = BASE_DIR / ('development_' + digest(dev_identity)[:16])
    guard_identity(DEV_DIR, dev_identity)
    runner = ReadoutEvalRunner(model, {}, layers=[SOURCE_LAYERS[0]], max_seq_len=READOUT_MAX_SEQ_LEN)
    correctness_path = DEV_DIR / 'qwen3_4b_correctness.csv'
    if not correctness_path.is_file():
        # Save completed item batches so interruptions do not regrade the whole set.
        parts_dir = DEV_DIR / 'correctness_parts'; parts_dir.mkdir(exist_ok=True)
        frames = []
        for start in range(0, len(raw_items), 25):
            path = parts_dir / f'{start:05d}.csv'
            if path.is_file():
                frame = pd.read_csv(path)
            else:
                print(f'Qwen3-4B development filter {start + 1}/{len(raw_items)}', flush=True)
                frame = runner.grade_model_correctness(raw_items[start:start+25], hf_model_name=MODEL_ID)
                atomic_save(path, lambda temporary: frame.to_csv(temporary, index=False))
            frames.append(frame)
        atomic_save(correctness_path, lambda temporary: pd.concat(frames, ignore_index=True).to_csv(temporary, index=False))
    else:
        print('REUSE Qwen3-4B development correctness labels.', flush=True)
    filtered_items = filter_items_by_correctness(raw_items, hf_model_name=MODEL_ID, correctness_csv=str(correctness_path))
    scoreable_items = runner.scoreable_items(filtered_items)
    FIT_ITEMS, HELD_OUT_ITEMS = split_fitting_items(scoreable_items, seed=SEED, fit_fraction=DEVELOPMENT_FRACTION)
    if not FIT_ITEMS:
        raise RuntimeError('No development items survived the official filters.')
    fit_keys = {(item.eval_slug, item.name) for item in FIT_ITEMS}
    held_out_keys = {(item.eval_slug, item.name) for item in HELD_OUT_ITEMS}
    assert not fit_keys & held_out_keys
    atomic_json(DEV_DIR / 'selection.json', dict(
        fit=[asdict(item) for item in FIT_ITEMS], held_out=[asdict(item) for item in HELD_OUT_ITEMS],
        counts=dict(raw=len(raw_items), correctness_kept=len(filtered_items), scoreable=len(scoreable_items))))
    # Candidate exceptions follow the official mask and do not use outcomes or ranks.
    EXCLUDED_TOKEN_IDS = non_semantic_token_ids(tokenizer, vocab_size=vocab_size_of(model),
        items=load_readout_eval_items(str(DATA_DIR)))
    atomic_json(DEV_DIR / 'excluded_token_ids.json', EXCLUDED_TOKEN_IDS)
    print('Expert-weight fitting items:', len(FIT_ITEMS), '| held out, NOT evaluated:', len(HELD_OUT_ITEMS), flush=True)
    display(pd.DataFrame([dict(eval=i.eval_slug, subset='fit') for i in FIT_ITEMS]).value_counts().rename('n'))
    del runner

    # =============== Cell 10 ===============
    def group_name(group):
        return f'blocks_{group[0]:02d}_{group[-1]:02d}'

    for group in GROUPS:
        name = group_name(group); path = ROUTER_DIR / f'{name}.pt'
        if path.is_file():
            router = ActivationRouterCollection.load(str(path))
            if sorted(router.layer_routers_L_dict) != group:
                raise RuntimeError('Saved router layer mismatch.')
            print('REUSE router:', name, flush=True)
            del router
            continue
        print('FIT router:', name, flush=True)
        chunks = {layer: [] for layer in group}
        for start in range(0, len(ROUTER_PROMPTS), 50):
            values = collect_valid_position_activations(model, ROUTER_PROMPTS[start:start+50], group,
                skip_first_n_positions=SKIP_FIRST, max_seq_len=MAX_SEQ_LEN)
            for layer in group: chunks[layer].append(values[layer])
            del values
            print(f'  Router activations {min(start+50, len(ROUTER_PROMPTS))}/{len(ROUTER_PROMPTS)}', flush=True)
        activations = {layer: torch.cat(chunks.pop(layer), dim=0) for layer in group}
        if any(value.shape[0] < N_EXPERTS for value in activations.values()):
            raise RuntimeError('Too few valid router positions.')
        print('  PCA and K-means on CPU; this may take several minutes.', flush=True)
        router = ActivationRouterCollection(N_EXPERTS, PCA_DIM, seed=SEED)
        router.fit(activations)
        atomic_save(path, router.save)
        del router, activations, chunks
        release_memory()
    print('All requested routers ready.', flush=True)

    # =============== Cell 12 ===============
    expert_paths = {}
    for group in GROUPS:
        name = group_name(group); directory = EXPERT_DIR / name
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / 'experts.pt'; expert_paths[name] = path
        if path.is_file():
            print('REUSE expert Jacobians:', name, flush=True)
            continue
        router = ActivationRouterCollection.load(str(ROUTER_DIR / f'{name}.pt'))
        config_path = directory / 'fit_config.json'
        if config_path.is_file():
            config = LensConfig.from_dict(json.loads(config_path.read_text(encoding='utf-8')))
        else:
            config = LensConfig(hf_model_name=MODEL_ID, checkpoint_name=name,
                artifacts_base_dir=str(EXPERT_DIR / 'checkpoints'), source_layers=group,
                relative_end_transport_layer=-1, jacobian_rows_per_pass=JACOBIAN_ROWS_PER_PASS,
                max_seq_len=MAX_SEQ_LEN, skip_first_n_positions=SKIP_FIRST,
                checkpoint_every_n_prompts=1, lrp_mode='rlens')
            atomic_json(config_path, config.to_dict())
        checkpoint = Path(config.checkpoint_path) / expert_checkpoint_filename(N_EXPERTS)
        if checkpoint.is_file():
            print('RESUME expert fit:', name, '|', checkpoint, flush=True)
            trainer = ExpertJacobianRelPTrainer.from_checkpoint_dir(
                config.checkpoint_path, FIT_PROMPTS,
                router_collections_K_dict={N_EXPERTS: router}, model=model)
        else:
            print('FIT expert Jacobians:', name, flush=True)
            trainer = ExpertJacobianRelPTrainer(config, model, FIT_PROMPTS,
                router_collections_K_dict={N_EXPERTS: router})
        sums = trainer.fit()[N_EXPERTS]
        if trainer.next_prompt_idx != len(FIT_PROMPTS):
            raise RuntimeError('Expert trainer did not finish all requested prompts.')
        experts = ExpertJacobians.from_sums(sums, trainer.config.with_router(
            num_clusters=N_EXPERTS, projection_dim=PCA_DIM), min_kept_positions=MIN_EXPERT_POSITIONS)
        # Exact pooled FP32 map from sufficient statistics; not a fitted J++ lens.
        pooled = experts.pooled_lens(checkpoint_name=name + '_pooled_rlens')
        atomic_save(directory / 'pooled_rlens.pt', lambda temporary, fitted=pooled: fitted.save(temporary, dtype=torch.float32))
        counts = [dict(layer=layer, expert=e,
            positions=int(experts.position_counts_L_dict_E[layer][e]),
            pooled_fallback=bool(experts.fallback_L_dict_Bool_E[layer][e]))
            for layer in group for e in range(N_EXPERTS)]
        atomic_save(directory / 'expert_counts.csv', lambda temporary: pd.DataFrame(counts).to_csv(temporary, index=False))
        atomic_save(path, experts.save)
        del trainer, sums, experts, pooled, router
        release_memory()
    print('All requested expert Jacobians ready.', flush=True)

    # =============== Cell 14 ===============
    condense = CondenseConfig(pass_k=WEIGHT_PASS_K, top_n=WEIGHT_TOP_N, steps=WEIGHT_STEPS,
        learning_rate=WEIGHT_LR, temperature=WEIGHT_TEMPERATURE, exact_every=WEIGHT_EXACT_EVERY)
    print('Hashing completed expert files for the weight-fit identity.', flush=True)
    expert_hashes = {name: file_sha(path) for name, path in expert_paths.items()}
    weight_identity = dict(expert=digest(expert_identity), expert_files=expert_hashes, development=digest(dev_identity),
        fitting_items=digest([asdict(item) for item in FIT_ITEMS]),
        filter=digest(EXCLUDED_TOKEN_IDS), optimizer=asdict(condense), seed=SEED)
    OUT = EXPERT_DIR / ('weights_' + digest(weight_identity)[:16])
    guard_identity(OUT, weight_identity)
    atomic_json(OUT / 'excluded_token_ids.json', EXCLUDED_TOKEN_IDS)
    estimated = len(FIT_ITEMS) * N_EXPERTS * vocab_size_of(model) * 4 / 1024**3
    print(f'Official per-layer expert logits tensor approximately {estimated:.2f} GiB, plus model/workspace.', flush=True)
    layer_paths = {}
    for group in GROUPS:
        name = group_name(group)
        missing = [layer for layer in group if not (OUT / f'layer_{layer:02d}.done.json').is_file()]
        for layer in group: layer_paths[layer] = OUT / f'layer_{layer:02d}.pt'
        if not missing:
            print('REUSE all expert weights:', name, flush=True)
            continue
        experts = ExpertJacobians.load(str(expert_paths[name]))
        residual_path = DEV_DIR / f'{name}_{digest(group)[:8]}_residuals.pt'
        runner = ReadoutEvalRunner(model, {}, layers=group, max_seq_len=READOUT_MAX_SEQ_LEN)
        residuals = runner.record_residuals(FIT_ITEMS, cache_path=str(residual_path), hf_model_name=MODEL_ID)
        if residuals.num_items != len(FIT_ITEMS):
            raise RuntimeError('Development recording unexpectedly dropped fitting items.')
        fitter = ExpertWeightingFitter(model, experts, residuals, excluded_token_ids=EXCLUDED_TOKEN_IDS)
        for layer in group:
            done = OUT / f'layer_{layer:02d}.done.json'
            if done.is_file():
                saved = json.loads(done.read_text(encoding='utf-8'))
                if saved['weight_identity'] != digest(weight_identity) or file_sha(layer_paths[layer]) != saved['sha256']:
                    raise RuntimeError(f'Completed layer artifact changed: {layer}')
                print('REUSE signed expert weights:', layer, flush=True)
                continue
            print('FIT signed expert weights:', layer, flush=True)
            torch.manual_seed(SEED + layer)
            weighting = fitter.fit([layer], condense_config=condense)
            lens = experts.combine(weighting.weights_L_dict_E, checkpoint_name=f'qwen3_4b_jpp_layer_{layer}')
            atomic_save(OUT / f'layer_{layer:02d}_weights.json', weighting.save)
            atomic_save(layer_paths[layer], lambda temporary, fitted=lens: fitted.save(temporary, dtype=torch.float32))
            atomic_json(done, dict(weight_identity=digest(weight_identity), sha256=file_sha(layer_paths[layer])))
            del weighting, lens
            release_memory()
        del fitter, residuals, runner, experts
        release_memory()
    print('All requested signed expert weights fitted.', flush=True)

    # =============== Cell 16 ===============
    jacobians = {}
    component_hashes = {}
    template_config = None
    for layer in SOURCE_LAYERS:
        path = layer_paths[layer]
        done = json.loads((OUT / f'layer_{layer:02d}.done.json').read_text(encoding='utf-8'))
        if done['weight_identity'] != digest(weight_identity) or done['sha256'] != file_sha(path):
            raise RuntimeError(f'Layer {layer} completion identity mismatch.')
        component = BaseLens.load(str(path))
        if component.source_layers != [layer]:
            raise RuntimeError('Combined layer file has unexpected source layers.')
        matrix = component.jacobians_L_dict_FN[layer].cpu().float()
        if matrix.shape != (model.d_model, model.d_model) or not torch.isfinite(matrix).all():
            raise RuntimeError(f'Invalid combined matrix at layer {layer}')
        jacobians[layer] = matrix
        component_hashes[str(layer)] = done['sha256']
        template_config = component.config
        del component
    final_config = replace(template_config, checkpoint_name='qwen3_4b_jpp',
                           source_layers=SOURCE_LAYERS, num_prompts_trained_on=N_JACOBIAN_PROMPTS)
    final_lens = JacobianLens(jacobians, config=final_config)
    LENS_PATH = OUT / 'qwen3_4b_jpp_lens.pt'
    atomic_save(LENS_PATH, lambda temporary: final_lens.save(temporary, dtype=torch.float32))
    loaded = BaseLens.load(str(LENS_PATH))
    assert loaded.source_layers == SOURCE_LAYERS
    for layer in SOURCE_LAYERS:
        if not torch.equal(loaded.jacobians_L_dict_FN[layer], jacobians[layer]):
            raise RuntimeError('Final FP32 lens did not round-trip exactly.')
    atomic_json(OUT / 'qwen3_4b_jpp_manifest.json', dict(
        method='J++ Lens', official_repository='https://github.com/safety-research/jpp_lens',
        base=base_identity, router=router_identity, experts=expert_identity,
        weights=weight_identity, component_sha256=component_hashes,
        source_layers=SOURCE_LAYERS, target_block=TARGET_BLOCK, d_model=model.d_model,
        lens_sha256=file_sha(LENS_PATH), filter_sha256=file_sha(OUT / 'excluded_token_ids.json'),
        fitting_items=len(FIT_ITEMS), held_out_items_not_evaluated=len(HELD_OUT_ITEMS),
        finished_utc=datetime.now(timezone.utc).isoformat()))
    print('FITTING COMPLETE', flush=True)
    print('J++ lens:', LENS_PATH.resolve(), flush=True)
    print('Manifest:', (OUT / 'qwen3_4b_jpp_manifest.json').resolve(), flush=True)
    print('Source layers:', loaded.source_layers, '| target:', TARGET_BLOCK, flush=True)
    print('Original J-lens and generation caches were not changed.', flush=True)

    # =============== Cell 18 ===============


    # =============== Cell 19 ===============


    # =============== Cell 20 ===============


if __name__=="__main__":main()
