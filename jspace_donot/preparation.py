"""Complete asset preparation: model, WikiText, fitted J++ lens and task datasets."""
from dataclasses import dataclass
from pathlib import Path
import hashlib,json,os,subprocess,sys

@dataclass(frozen=True)
class PreparedAssets:
    root: Path
    model_dir: Path
    lens_path: Path
    wikitext_prompts: Path


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024**2),b''):h.update(chunk)
    return h.hexdigest()


def completed_lenses(path):
    path=Path(path)
    candidates=[path] if path.is_file() else sorted(path.rglob('qwen3_4b_jpp_lens.pt'))
    return [p.resolve() for p in candidates if p.is_file() and p.with_name('qwen3_4b_jpp_manifest.json').is_file()]


def validate_lens(path,model_dir):
    path=Path(path);manifest=json.loads(path.with_name('qwen3_4b_jpp_manifest.json').read_text())
    assets=manifest['base'].get('assets',{})
    if not assets or 'config.json' not in assets:raise ValueError('Incomplete J++ model-asset manifest')
    for name,record in assets.items():
        source=Path(model_dir)/name
        if not source.is_file() or source.stat().st_size!=record['bytes'] or file_sha(source)!=record['sha256']:
            raise ValueError(f'J++ lens/model asset mismatch: {source}')
    if manifest.get('lens_sha256') and file_sha(path)!=manifest['lens_sha256']:
        raise ValueError('J++ lens checksum mismatch')
    return path


def prepare_assets(root=Path('.'),datasets=('gsm8k','math500','raceh'),model_dir=None,lens_path=None,
                   fit_if_missing=True,download_wikitext=True,layers_per_group=4,jacobian_rows_per_pass=8,
                   fit_python=None,setup_source=True):
    """Prepare a clean checkout, reusing a matching completed lens when present.

    No dependencies are installed/downgraded. The pinned source is placed on
    subprocess PYTHONPATH. A new full J++ fit runs only when no completed lens
    was supplied/found. Original 1000/64 prompt counts and fitting seed 0 stay fixed.
    fit_python may name a separate existing J++ Python environment.
    """
    root=Path(root).expanduser().resolve();root.mkdir(parents=True,exist_ok=True)
    scripts=Path(__file__).resolve().parents[1]/'scripts'
    if not scripts.is_dir():raise FileNotFoundError('Run preparation from the complete source checkout (scripts/ required)')
    if isinstance(datasets,str):datasets=(datasets,)
    if not set(datasets)<={'gsm8k','math500','raceh'}:raise ValueError('Unknown dataset')
    if layers_per_group<1 or jacobian_rows_per_pass<1:raise ValueError('Positive fitting memory parameters required')
    model=Path(model_dir).expanduser().resolve() if model_dir else root/'base_models/Qwen3-4B'
    link=root/'base_models/Qwen3-4B'
    fitting_python=str(Path(fit_python).expanduser().resolve()) if fit_python else sys.executable
    repo=root/'external/jpp_lens'
    env=os.environ.copy()
    env['PYTHONPATH']=str(repo/'src')+os.pathsep+str(repo/'scripts')+os.pathsep+env.get('PYTHONPATH','')
    def run(script,*args,python=sys.executable):
        subprocess.run([str(python),str(scripts/script),'--root',str(root),*map(str,args)],env=env,check=True)
    if not (model/'config.json').is_file() or not any(model.glob('*.safetensors')):
        print('PREPARE: download Qwen3-4B',flush=True)
        run('download_model.py','--model-dir',model)
    else:print('REUSE: Qwen3-4B',model,flush=True)
    if link.resolve()!=model:
        if link.exists() or link.is_symlink():raise RuntimeError('Root contains another model directory; choose another root')
        link.parent.mkdir(parents=True,exist_ok=True)
        link.symlink_to(model,target_is_directory=True)
    if setup_source:
        print('PREPARE: pinned official J++ source (no package installation)',flush=True)
        run('setup_jpp.py','--source-only')
    prompts=root/'jlens_outputs/qwen3_4b/jpp_fit/wikitext103_raw_train_first_1000_minchars600.json'
    if download_wikitext:
        print('PREPARE: WikiText-103 RAW TRAIN + original prompt pool',flush=True)
        run('prepare_wikitext103.py')
    search=Path(lens_path).expanduser().resolve() if lens_path else root/'jlens_outputs/qwen3_4b/jpp_fit'
    matches=completed_lenses(search)
    if len(matches)>1:raise RuntimeError(f'Multiple completed J++ fits; choose lens_path explicitly: {matches}')
    if lens_path is not None and not matches:raise FileNotFoundError(f'No completed lens at explicit lens_path: {search}')
    if not matches:
        if not fit_if_missing:raise FileNotFoundError('No completed J++ lens. Set fit_if_missing=True to perform the full original fit.')
        if not prompts.is_file():run('prepare_wikitext103.py')
        print('FIT: official J++ routers -> LRP expert Jacobians -> signed weights -> combined lens',flush=True)
        run('fit_jpp.py','--model-dir',model,'--prompts-json',prompts,
            '--layers-per-group',layers_per_group,'--jacobian-rows-per-pass',jacobian_rows_per_pass,python=fitting_python)
        matches=completed_lenses(root/'jlens_outputs/qwen3_4b/jpp_fit')
        if len(matches)!=1:raise RuntimeError(f'Expected one completed fit after fitting, found: {matches}')
    lens=validate_lens(matches[0],model)
    print('READY: fitted J++ lens',lens,flush=True)
    for task in datasets:
        print('PREPARE: task dataset',task,flush=True)
        if task=='gsm8k' and all((root/f'data/{name}').is_file() for name in ('gsm8k_official_train.json','gsm_original_test.json')):
            run('prepare_gsm8k.py','--verify-existing')
        else:run(f'prepare_{task}.py')
    result=PreparedAssets(root,model,lens,prompts)
    (root/'preparation_manifest.json').write_text(json.dumps(dict(model_dir=str(model),lens_path=str(lens),
        lens_sha256=file_sha(lens),wikitext_prompts=str(prompts),datasets=list(datasets),
        fit_seed=0,router_prompts=1000,jacobian_prompts=64,fit_if_missing=fit_if_missing),indent=2))
    return result
