"""Download WikiText-103 RAW training data and freeze the original J++ prompt pool.

Identical selection to jlens.examples.load_wikitext_prompts: original order,
first 1000 records with len(text.strip()) >= 600; keep original text unchanged.
"""
import argparse,hashlib,json
from pathlib import Path
SOURCE='Salesforce/wikitext'
CONFIG='wikitext-103-raw-v1'


def select_prompts(records,n_prompts=1000,min_chars=600):
    if n_prompts<1 or min_chars<1:raise ValueError('Positive prompt count and minimum length required')
    prompts=[]
    for record in records:
        text=record['text']
        if len(text.strip())>=min_chars:
            prompts.append(text)
            if len(prompts)==n_prompts:return prompts
    raise ValueError(f'Only {len(prompts)} eligible WikiText prompts; required {n_prompts}')


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp');temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');temporary.replace(path)


def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('.'))
    p.add_argument('--revision',default='main')
    p.add_argument('--n-prompts',type=int,default=1000)
    p.add_argument('--min-chars',type=int,default=600)
    p.add_argument('--stream',action='store_true',help='Download only enough records for the prompt pool instead of materializing full TRAIN')
    a=p.parse_args(argv);root=a.root.resolve()
    from datasets import load_dataset,load_from_disk
    from huggingface_hub import HfApi
    data_root=root/'data/wikitext103_raw';dataset_dir=data_root/'train'
    source_manifest=data_root/'source_manifest.json'
    if not a.stream and source_manifest.is_file() and dataset_dir.is_dir():
        source=json.loads(source_manifest.read_text())
        if source.get('dataset')!=SOURCE or source.get('config')!=CONFIG:raise ValueError('Wrong cached WikiText source')
        revision=source['revision']
        if a.revision!='main' and a.revision!=revision:raise ValueError('Cached WikiText revision differs; use another root')
        records=load_from_disk(str(dataset_dir));print('REUSE WikiText RAW TRAIN:',dataset_dir,flush=True)
    else:
        revision=HfApi().dataset_info(SOURCE,revision=a.revision).sha
        print('DOWNLOAD WikiText RAW TRAIN:',SOURCE,CONFIG,revision,flush=True)
        records=load_dataset(SOURCE,CONFIG,split='train',revision=revision,streaming=a.stream,
            cache_dir=str(root/'data/hf_cache'))
        if not a.stream:
            if dataset_dir.exists():raise FileExistsError(f'Incomplete/unidentified WikiText directory: {dataset_dir}')
            dataset_dir.parent.mkdir(parents=True,exist_ok=True);records.save_to_disk(str(dataset_dir))
            write_json(source_manifest,dict(dataset=SOURCE,config=CONFIG,split='train',revision=revision,rows=len(records)))
    prompts=select_prompts(records,a.n_prompts,a.min_chars)
    path=root/'jlens_outputs/qwen3_4b/jpp_fit'/f'wikitext103_raw_train_first_{a.n_prompts}_minchars{a.min_chars}.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8'))!=prompts:
        raise RuntimeError('Existing J++ prompt pool differs from the downloaded source. Keep it and use a separate root for the new fit.')
    if not path.exists():write_json(path,prompts)
    metadata=dict(dataset=SOURCE,config=CONFIG,split='train',revision=revision,
        selection='original row order; first N records with len(text.strip()) >= min_chars; original text preserved',
        n_prompts=a.n_prompts,min_chars=a.min_chars,default_jacobian_prompts=64,
        prompts_sha256=hashlib.sha256(json.dumps(prompts,ensure_ascii=False).encode()).hexdigest(),
        file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),full_training_download=not a.stream)
    write_json(path.with_suffix('.manifest.json'),metadata)
    print('J++ router prompt pool:',path,flush=True)
    return path

if __name__=='__main__':main()
