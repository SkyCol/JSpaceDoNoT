import argparse

def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",default=".")
    parser.add_argument('--train-revision',default='main')
    parser.add_argument('--test-revision',default='main')
    args=parser.parse_args(argv)
    from pathlib import Path
    import hashlib, json, re, unicodedata
    from datasets import load_dataset
    from huggingface_hub import HfApi
    import pandas as pd
    from IPython.display import display
    ROOT=Path(args.root).resolve();DEST=ROOT/'data/math500';DEST.mkdir(parents=True,exist_ok=True)
    SUBJECTS=['algebra','counting_and_probability','geometry','intermediate_algebra','number_theory','prealgebra','precalculus']
    def qhash(s):return hashlib.sha256(re.sub(r'\s+',' ',unicodedata.normalize('NFKC',s)).strip().encode()).hexdigest()
    def last_box(s):
        start=list(re.finditer(r'\\boxed\s*\{',s))
        if not start:return None
        a=start[-1].end(); depth=1; i=a
        while i<len(s) and depth:
            if s[i]=='{':depth+=1
            elif s[i]=='}':depth-=1
            i+=1
        return s[a:i-1].strip() if depth==0 else None
    def parse_level(raw):
        match=re.search(r'(?<!\d)[1-5](?!\d)',str(raw))
        return int(match.group()) if match else None
    def digest_file(p):return hashlib.sha256(p.read_bytes()).hexdigest()
    assert last_box(r'\boxed{\frac{1}{2}}')==r'\frac{1}{2}'
    train_revision=HfApi().dataset_info('EleutherAI/hendrycks_math',revision=args.train_revision).sha
    test_revision=HfApi().dataset_info('HuggingFaceH4/MATH-500',revision=args.test_revision).sha
    train=[];unboxed=[];unknown_training_levels=[]
    for sub in SUBJECTS:
        ds=load_dataset('EleutherAI/hendrycks_math',sub,split='train',revision=train_revision)
        for r in ds:
            gold=last_box(r['solution'])
            if not gold:
                unboxed.append(dict(subject=sub,question=r['problem']))
                continue
            level=parse_level(r.get('level'))
            if level is None:
                unknown_training_levels.append(dict(subject=sub,raw_level=str(r.get('level')),question=r['problem']))
            train.append(dict(question=r['problem'],gold=gold,subject=sub,level=level))
        print(sub,len(ds),flush=True)
    test=[]
    for r in load_dataset('HuggingFaceH4/MATH-500',split='test',revision=test_revision):
        test.append(dict(question=r['problem'],gold=r['answer'],subject=r['subject'],
            level=int(r['level']),source_id=r['unique_id']))
    assert len(train)+len(unboxed)==7500 and len(test)==500,(len(train),len(test))

    # Keep the published 500-row TEST intact. Inspect each overlap condition separately.
    test_ids=[qhash(r['question']) for r in test]
    if len(set(test_ids))!=len(test_ids):
        raise ValueError(f'MATH-500 contains {len(test_ids)-len(set(test_ids))} duplicate questions; '
                         'inspect the test source before evaluation.')

    def clean_training_pool(extractable_train,test_questions):
        # Question-level deduplication before calibration and OOF folds.
        groups={}
        for row in extractable_train:
            groups.setdefault(qhash(row['question']),[]).append(row)
        test_hashes={qhash(r['question']) for r in test_questions}
        kept=[];duplicates=[];conflicts=[];overlaps=[]
        for question_hash,rows in groups.items():
            if len(rows)>1:
                duplicates.extend(rows[1:])
                # An identical prompt with different gold labels cannot be used
                # as a well-defined correctness target.
                if len({r['gold'].strip() for r in rows})>1:
                    conflicts.extend(rows)
                    continue
            if question_hash in test_hashes:
                overlaps.extend(rows)
                continue
            kept.append(rows[0])
        return kept,duplicates,conflicts,overlaps

    extractable_train_n=len(train)
    train,duplicate_train_rows,conflicting_train_rows,train_test_overlap=clean_training_pool(train,test)
    assert len({qhash(r['question']) for r in train})==len(train)
    assert not ({qhash(r['question']) for r in train}&set(test_ids))
    print('Extractable TRAIN:',extractable_train_n,
          '| duplicate extra rows:',len(duplicate_train_rows),
          '| conflicting-label group rows excluded:',len(conflicting_train_rows),
          '| overlapping TEST rows excluded:',len(train_test_overlap),
          '| usable TRAIN:',len(train),flush=True)
    if len(train)<256:raise RuntimeError('Fewer than 256 unique, non-overlapping training questions remain.')
    for name,rows in [('duplicate_train_rows',duplicate_train_rows),
                      ('conflicting_train_rows',conflicting_train_rows),
                      ('train_test_overlap',train_test_overlap)]:
        if rows:pd.DataFrame(rows).to_csv(DEST/f'{name}.csv',index=False)
    for name,rows in [('math_original_train',train),('math500_test',test)]:
        path=DEST/f'{name}.json'
        if path.exists() and json.loads(path.read_text())!=rows:raise FileExistsError(f'Existing data differ: {path}; preserve the old files or choose a separate root')
        path.write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    manifest=dict(train_revision=train_revision,test_revision=test_revision,train_source='EleutherAI/hendrycks_math: train, seven subjects',
        test_source='HuggingFaceH4/MATH-500: test',train_n=len(train),test_n=len(test),
        original_train_n=7500,unboxed_training_skipped=len(unboxed),
        unknown_training_levels=len(unknown_training_levels),overlap=0,
        train_sha256=digest_file(DEST/'math_original_train.json'),
        test_sha256=digest_file(DEST/'math500_test.json'),
        gold='last balanced boxed expression in TRAIN solution; original MATH-500 answer field in TEST')
    (DEST/'source_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    display(pd.DataFrame(test).groupby(['subject','level']).size().unstack(fill_value=0))
    if unknown_training_levels:
        pd.DataFrame(unknown_training_levels).to_csv(DEST/'unknown_training_levels.csv',index=False)
        print('Training levels unavailable:',len(unknown_training_levels),flush=True)
    if unboxed:
        pd.DataFrame(unboxed).to_csv(DEST/'unboxed_training_audit.csv',index=False)
        print('Skipped unboxed training solutions:',len(unboxed),flush=True)
    print('Saved:',DEST.resolve(), '\n',manifest)

if __name__=="__main__":main()
