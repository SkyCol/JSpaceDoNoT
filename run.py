"""CLI matching the public notebook API."""
import argparse
from pathlib import Path
from jspace_donot import ExperimentConfig,run_experiment,required_paths

def make_parser():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',required=True,choices=['gsm8k','math500','raceh'])
    p.add_argument('--root',type=Path,default=Path('.'))
    p.add_argument('--schedule',choices=['wait','token'],default='wait')
    p.add_argument('--check-every','--every-n-tokens',dest='check_every',type=int,default=256)
    p.add_argument('--wait-markers',default='Wait')
    p.add_argument('--min-stop-tokens',type=int,default=256)
    p.add_argument('--max-monitor-tokens',type=int)
    p.add_argument('--calibration',type=int,default=256)
    p.add_argument('--test-limit',type=int)
    p.add_argument('--max-new-tokens',type=int,default=8192)
    p.add_argument('--cache-mode',choices=['resume','read_only','off'],default='resume')
    p.add_argument('--cache-root',type=Path)
    p.add_argument('--source-mode',choices=['auto','historical','fresh'],default='auto')
    p.add_argument('--output',type=Path)
    p.add_argument('--lens-path',type=Path)
    p.add_argument('--prepare',action='store_true',help='Download model/WikiText, fit missing J++ lens, and prepare this dataset first')
    p.add_argument('--dry-run',action='store_true')
    return p

def to_config(a):
    return ExperimentConfig(dataset=a.dataset,root=a.root,schedule=a.schedule,every_n_tokens=a.check_every,
        wait_markers=tuple(x.strip() for x in a.wait_markers.split(',') if x.strip()),
        min_stop_tokens=a.min_stop_tokens,max_monitor_tokens=a.max_monitor_tokens,
        calibration=a.calibration,test_limit=a.test_limit,max_new_tokens=a.max_new_tokens,
        cache_mode=a.cache_mode,cache_root=a.cache_root,source_mode=a.source_mode,output=a.output,lens_path=a.lens_path)

def validate(a):return to_config(a).validate()

def main(argv=None):
    args=make_parser().parse_args(argv);cfg=to_config(args).validate()
    if args.dry_run:
        missing=required_paths(cfg)
        print(cfg)
        print('\n'.join(missing) if missing else 'Prerequisites found; model not loaded.')
        return int(bool(missing))
    if args.prepare:
        from dataclasses import replace
        from jspace_donot import prepare_assets
        assets=prepare_assets(root=cfg.root,datasets=(cfg.dataset,),lens_path=cfg.lens_path)
        cfg=replace(cfg,lens_path=assets.lens_path)
    result=run_experiment(cfg);print('Results:',result.output)
    return 0

if __name__=='__main__':raise SystemExit(main())
