"""GSM8K source preparation, preserving the historical batch key/member schema."""
from pathlib import Path
import json


def prepare_gsm8k_source(ns, source_mode='auto'):
    """Reuse a homogeneous historical cache, or create a fresh fixed-batch cohort.

    auto never fills holes in an existing historical cache by regrouping questions.
    fresh uses selected TRAIN then TEST order, recorded in a cohort manifest.
    """
    if source_mode not in {'auto','historical','fresh'}:
        raise ValueError(source_mode)
    root=ns['MODULAR_CACHE_ROOT']
    existing=any((root/'generations').glob('*/*.json'))
    if source_mode=='historical' or (source_mode=='auto' and existing):return
    if ns['NATURAL_CACHE_POLICY']=='read_only':
        raise FileNotFoundError('No compatible source cache; read_only cannot generate CoTs.')
    if source_mode=='fresh':
        root=root/'fresh_fixed_batch_v1'
        ns['MODULAR_CACHE_ROOT']=root
    import torch
    rows=list(ns['CAL_DATA'])+([] if ns['CALIBRATION_ONLY'] else list(ns['TEST_DATA']))
    prompts=[ns['expected_prompt'](r['question']) for r in rows]
    cohort=dict(version='gsm8k-fixed-batch-v1',question_ids=[r['question_id'] for r in rows],
        prompts=prompts,model=ns['MODEL_SIGNATURE'],seed=ns['SEED'],batch_size=ns['SOURCE_GEN_BATCH_SIZE'],
        sampling=ns['SOURCE_SAMPLING'],budget=ns['OLD_TOKEN_BUDGET'])
    cohort_path=root/'fresh_cohort.json'
    if cohort_path.exists() and json.loads(cohort_path.read_text())!=cohort:
        raise RuntimeError('Fresh source cohort changed. Use another cache root; never silently regroup sampled CoTs.')
    ns['atomic_json'](cohort_path,cohort)
    tokenizer=ns['tokenizer'];size=ns['SOURCE_GEN_BATCH_SIZE']
    for lo in range(0,len(rows),size):
        group=rows[lo:lo+size];ps=prompts[lo:lo+size]
        key=ns['digest'](dict(engine=ns['SOURCE_GEN_ENGINE_VERSION'],model=ns['MODEL_SIGNATURE'],mode='cot',
            requested_batch_size=size,prompts=ps,seed=ns['SEED'],sampling=ns['SOURCE_SAMPLING'],
            budget=ns['OLD_TOKEN_BUDGET'],revision=ns['SOURCE_GENERATION_REVISION']))
        journal=root/'generation_batches'/key[:2]/f'{key}.json'
        if journal.exists():
            pack=json.loads(journal.read_text());runs=pack['runs']
            if pack.get('key')!=key:raise RuntimeError('Batch identity mismatch')
        else:
            width=max(map(len,ps))
            if width+ns['OLD_TOKEN_BUDGET']>min(ns['MAX_PREFIX_TOKENS'],ns['base_model'].config.max_position_embeddings):
                raise ValueError('GSM8K natural generation exceeds context guard')
            ids=torch.full((len(ps),width),tokenizer.pad_token_id,dtype=torch.long,device=ns['DEVICE'])
            mask=torch.zeros_like(ids)
            for i,p in enumerate(ps):
                ids[i,-len(p):]=torch.tensor(p,device=ids.device);mask[i,-len(p):]=1
            seed=(ns['SEED']+int(key[:8],16))%(2**31)
            torch.manual_seed(seed)
            with torch.inference_mode():
                out=ns['base_model'].generate(input_ids=ids,attention_mask=mask,
                    generation_config=ns['GenerationConfig'](do_sample=True,
                        temperature=ns['SOURCE_SAMPLING']['temperature'],top_p=ns['SOURCE_SAMPLING']['top_p'],
                        top_k=ns['SOURCE_SAMPLING']['top_k'],min_p=ns['SOURCE_SAMPLING']['min_p'],
                        max_new_tokens=ns['OLD_TOKEN_BUDGET'],pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=ns['EOS_IDS'],use_cache=True))
            runs=[]
            for i,(row,p) in enumerate(zip(group,ps)):
                tokens=out[i,width:].tolist()
                # Strip only padding following the first actual EOS.
                for end,tid in enumerate(tokens):
                    if tid in ns['EOS_IDS']:tokens=tokens[:end+1];break
                member_key=ns['digest'](dict(engine=ns['SOURCE_GEN_ENGINE_VERSION'],batch=key,member=i))
                runs.append(dict(key=member_key,question=row['question'],mode='cot',prompt_ids=p,
                    generated_ids=tokens,decoded=tokenizer.decode(tokens,skip_special_tokens=False,
                    clean_up_tokenization_spaces=False),seed=seed,batch_key=key,batch_member=i,
                    batch_size=len(group),generation_engine=ns['SOURCE_GEN_ENGINE_VERSION'],requested_batch_size=size))
            ns['atomic_json'](journal,dict(key=key,runs=runs));del out,ids,mask
        for run in runs:
            path=root/'generations'/run['key'][:2]/f"{run['key']}.json"
            if not path.exists():ns['atomic_json'](path,run)
        print(f'GSM8K source batches: {min(lo+size,len(rows))}/{len(rows)}',flush=True)
