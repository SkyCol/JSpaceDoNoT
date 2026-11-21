"""Internal execution worker. Not imported by the notebook or package API."""
import json,os,pickle,runpy,sys,tempfile
from pathlib import Path

def main():
    cfg=json.loads(sys.argv[1]);output=Path(cfg['output'])
    values={'ROOT':cfg['root'],'OUTPUT':cfg['output'],'SCHEDULE':cfg['schedule'],
        'CHECK_EVERY':cfg['every_n_tokens'],'WAIT_MARKERS':','.join(cfg['wait_markers']),
        'MIN_STOP':cfg['min_stop_tokens'],'CALIBRATION':cfg['calibration'],
        'MAX_NEW_TOKENS':cfg['max_new_tokens'],'CACHE_MODE':'resume' if cfg['cache_mode']=='off' else cfg['cache_mode'],
        'SOURCE_MODE':cfg['source_mode']}
    for key,field in [('TEST_LIMIT','test_limit'),('MAX_MONITOR','max_monitor_tokens'),('CACHE_ROOT','cache_root'),('LENS_PATH','lens_path')]:
        if cfg[field] is not None:values[key]=cfg[field]
    # Clear inherited optional knobs so runs cannot contaminate each other.
    for key in list(os.environ):
        if key.startswith('JSPACE_'):del os.environ[key]
    with tempfile.TemporaryDirectory(prefix='jspace_intermediates_') as temporary:
        if cfg['cache_mode']=='off':values['CACHE_ROOT']=temporary
        os.environ.update({'JSPACE_'+k:str(v) for k,v in values.items()})
        state=runpy.run_path(str(Path(__file__).with_name(cfg['dataset']+'_pipeline.py')))
        # CPU-only self-contained interpretation input; no backbone pickles.
        bundle=dict(version=1,heads=state['FULL_HEAD'],train=state['TRAIN_X'][['question_id','step','correct','j_vector']].copy(),
            test=state['TEST_X'][['question_id','step','correct','j_vector']].copy(),
            layers=state['DYN_LAYERS'],top=state['TOP'],seed=state['SEED'],config=cfg)
        temp=output/'analysis_bundle.pkl.tmp'
        with temp.open('wb') as stream:pickle.dump(bundle,stream,pickle.HIGHEST_PROTOCOL)
        temp.replace(output/'analysis_bundle.pkl')
        contexts=[];generations={}
        for split,rows,profiles in [('train',state['CAL_DATA'],state['TRAIN_PROFILES']),('test',state['TEST_DATA'],state['TEST_PROFILES'])]:
            for row in rows:
                run=state['load_old'](row)
                decoded,bounds=state['decoded_bounds'](run['generated_ids'])
                generations[split+':'+row['question_id']]=dict(text=decoded,bounds=bounds)
                for event in profiles[row['question_id']]['trace']:
                    if not event['usable']:continue
                    contexts.append(dict(split=split,question_id=row['question_id'],step=event['step'],
                        detected=event['detected'],question=row['question'],gold=row['gold'],
                        token_ids=event['token_ids'],token_strings=event['token_strings'],
                        generation_ref=split+':'+row['question_id'],
                        outcome=profiles[row['question_id']]['outcomes'].get(str(event['step']))))
        state['atomic_json'](output/'checkpoint_context.json',contexts)
        state['atomic_json'](output/'generation_context.json',generations)
        state['atomic_json'](output/'run_config.json',cfg)
        state['atomic_json'](output/'cache_audit.json',dict(counts=dict(state['COUNTS']),entries=state['CACHE_AUDIT']))
        import importlib.metadata
        packages={}
        for name in ['torch','transformers','numpy','pandas','scikit-learn','math-verify','jlens']:
            try:packages[name]=importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:pass
        state['atomic_json'](output/'environment.json',packages)

if __name__=='__main__':main()
