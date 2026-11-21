"""Prepare official GSM8K main train/test for JSpaceDoNoT.

Usage:
    python scripts/prepare_gsm8k.py --root .

The original GSM8K pipeline expects JSON lists with `question` and `gold`.
Official `answer` fields include a rationale followed by `#### NUMBER`.
This script retains the full original answer and adds the normalized gold.
"""
import argparse
import hashlib
import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path

SOURCE = 'openai/gsm8k'
CONFIG = 'main'
NUMERIC = re.compile(r'^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$')
FINAL = re.compile(r'####\s*([^\r\n]+)\s*$')


def question_id(question):
    return hashlib.sha256(re.sub(r'\s+', ' ', question).strip().encode('utf-8')).hexdigest()


def gold_from_official_answer(answer):
    if not isinstance(answer, str):
        raise ValueError('Official answer must be a string.')
    match = FINAL.search(answer)
    if not match:
        raise ValueError('Official GSM8K answer lacks a terminal #### numeric label.')
    label = match.group(1).strip().replace(',', '')
    if not NUMERIC.fullmatch(label):
        raise ValueError(f'Not a plain numeric label: {label!r}')
    try:
        value = Decimal(label)
    except InvalidOperation as exc:
        raise ValueError(f'Invalid numeric gold: {label!r}') from exc
    if not value.is_finite():
        raise ValueError(f'Nonfinite numeric gold: {label!r}')
    return format(value.normalize(), 'f')


def prepare_rows(dataset, split):
    output, seen = [], set()
    for i, row in enumerate(dataset):
        question = row.get('question')
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f'{split} index {i}: missing question')
        qid = question_id(question)
        if qid in seen:
            raise ValueError(f'{split}: duplicate normalized question at index {i}; stop rather than silently drop')
        seen.add(qid)
        answer = row.get('answer')
        gold = gold_from_official_answer(answer)
        output.append({'question': question, 'answer': answer, 'gold': gold})
    return output, seen


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Download and verify official GSM8K main dataset')
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--revision', default='main')
    parser.add_argument('--verify-existing',action='store_true',help='Compare existing question/gold sequences to official splits without overwriting')
    args = parser.parse_args(argv)
    from datasets import load_dataset
    from huggingface_hub import HfApi
    revision=HfApi().dataset_info(SOURCE,revision=args.revision).sha
    train_ds = load_dataset(SOURCE, CONFIG, split='train',revision=revision)
    test_ds = load_dataset(SOURCE, CONFIG, split='test',revision=revision)
    train, train_ids = prepare_rows(train_ds, 'train')
    test, test_ids = prepare_rows(test_ds, 'test')
    overlap = train_ids & test_ids
    if overlap:
        raise ValueError(f'Official GSM8K has {len(overlap)} normalized question overlaps across splits; refusing leakage')
    if len(train) != 7473 or len(test) != 1319:
        raise ValueError(f'Unexpected GSM8K main sizes: train={len(train)}, test={len(test)} (expected 7473/1319)')
    dest = args.root.resolve() / 'data'
    paths = {'train': dest / 'gsm8k_official_train.json', 'test': dest / 'gsm_original_test.json'}
    for split, rows in [('train', train), ('test', test)]:
        if args.verify_existing:
            existing=json.loads(paths[split].read_text(encoding='utf-8-sig'))
            def pair(r):
                label=r.get('gold',r.get('target',r.get('final_answer')))
                if label is None:label=gold_from_official_answer(r['answer'])
                return r['question'],format(Decimal(str(label).replace(',','')).normalize(),'f')
            if [pair(r) for r in existing]!=[pair(r) for r in rows]:
                raise ValueError(f'{split}: existing data differ from official question/gold sequence; files untouched')
        elif paths[split].exists():
            existing=json.loads(paths[split].read_text(encoding='utf-8-sig'))
            if existing!=rows:raise FileExistsError(f'{paths[split]} exists with different content. Use --verify-existing to compare question/gold without overwriting.')
        else:save_json(paths[split], rows)
    manifest = {'source': SOURCE, 'revision':revision, 'config': CONFIG, 'splits': {'train': len(train), 'test': len(test)},
                'format': 'question, answer (original reasoning with #### label), gold (normalized decimal)',
                'normalization': 'SHA256 of whitespace-collapsed question', 'overlap': 0,
                'sha256': {split: hashlib.sha256(path.read_bytes()).hexdigest() for split, path in paths.items()}}
    if not args.verify_existing:save_json(dest / 'gsm8k_source_manifest.json', manifest)
    print(json.dumps({'files': {k: str(v) for k,v in paths.items()}, 'manifest': manifest}, indent=2))


if __name__ == '__main__':
    main()
