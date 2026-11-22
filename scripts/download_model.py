"""Download official unquantized Qwen3-4B and record the resolved revision."""
import argparse,json,hashlib
from pathlib import Path

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('.'));p.add_argument('--revision',default='main');p.add_argument('--model-dir',type=Path);a=p.parse_args(argv)
    from huggingface_hub import HfApi,snapshot_download
    revision=HfApi().model_info('Qwen/Qwen3-4B',revision=a.revision).sha
    root=a.root.resolve();dest=a.model_dir.expanduser().resolve() if a.model_dir else root/'base_models/Qwen3-4B'
    snapshot_download(repo_id='Qwen/Qwen3-4B',revision=revision,local_dir=str(dest))
    manifest=dict(model='Qwen/Qwen3-4B',revision=revision,config_sha256=hashlib.sha256((dest/'config.json').read_bytes()).hexdigest())
    path=root/'model_download_manifest.json';path.write_text(json.dumps(manifest,indent=2));print(dest)

if __name__=='__main__':main()
