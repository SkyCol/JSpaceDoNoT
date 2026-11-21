"""Run a task in an isolated process, freeing GPU memory between experiments."""
from dataclasses import dataclass,asdict
from pathlib import Path
import json,os,subprocess,sys
from .config import ExperimentConfig

@dataclass(frozen=True)
class ExperimentResult:
    output: Path
    @property
    def summary(self):
        import pandas as pd
        return pd.read_csv(self.output/'test_summary.csv')
    def interpret(self,**kwargs):
        from .interpretation import interpret_run
        return interpret_run(self.output,**kwargs)

def required_paths(config):
    config.validate();root=Path(config.root).resolve()
    files={'gsm8k':['data/gsm8k_official_train.json','data/gsm_original_test.json'],
        'math500':['data/math500/math_original_train.json','data/math500/math500_test.json'],
        'raceh':['data/race/race_high_train.json','data/race/race_high_validation.json','data/race/race_high_test.json']}[config.dataset]
    paths=[root/'base_models/Qwen3-4B/config.json']+[root/f for f in files]
    if config.dataset=='gsm8k':
        paths=[(root/'data/gsm8k'/p.name if not p.exists() and (root/'data/gsm8k'/p.name).exists() else p) for p in paths]
    missing=[str(p) for p in paths if not p.is_file()]
    candidates=([Path(config.lens_path)] if config.lens_path else list((root/'jlens_outputs/qwen3_4b/jpp_fit').rglob('qwen3_4b_jpp_lens.pt')))
    valid=[p for p in candidates if p.is_file() and p.with_name('qwen3_4b_jpp_manifest.json').is_file()]
    if len(valid)!=1:missing.append(f'Exactly one J++ lens + manifest required (found {len(valid)})')
    return missing

def run_experiment(config:ExperimentConfig):
    config.validate()
    missing=required_paths(config)
    if missing:raise FileNotFoundError('\n'.join(missing))
    from datetime import datetime,timezone
    root=Path(config.root).resolve()
    output=Path(config.output).resolve() if config.output else root/'jlens_outputs/qwen3_4b'/f'{config.dataset}_JSpaceDoNOT_{datetime.now(timezone.utc):%Y%m%d_%H%M%S_%f}'
    output.mkdir(parents=True,exist_ok=True)
    package_root=Path(__file__).resolve().parents[1]
    env=os.environ.copy()
    env['PYTHONPATH']=str(package_root)+os.pathsep+env.get('PYTHONPATH','')
    payload={k:str(v) if isinstance(v,Path) else v for k,v in asdict(config).items()}
    payload['root']=str(root);payload['output']=str(output)
    subprocess.run([sys.executable,'-m','jspace_donot.worker',json.dumps(payload)],env=env,check=True)
    return ExperimentResult(output)
