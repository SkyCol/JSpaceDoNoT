"""One command: download model and WikiText, fit J++ if missing, prepare task datasets."""
import argparse
from pathlib import Path
from jspace_donot.preparation import prepare_assets

def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('.'))
    p.add_argument('--datasets',nargs='+',choices=['gsm8k','math500','raceh'],default=['gsm8k','math500','raceh'])
    p.add_argument('--model-dir',type=Path)
    p.add_argument('--lens-path',type=Path,help='Completed lens file or directory; omit to fit if missing')
    p.add_argument('--no-fit',action='store_true')
    p.add_argument('--layers-per-group',type=int,default=4)
    p.add_argument('--jacobian-rows-per-pass',type=int,default=8)
    p.add_argument('--fit-python',type=Path,help='Optional existing dedicated J++ environment Python executable')
    a=p.parse_args(argv)
    print(prepare_assets(root=a.root,datasets=a.datasets,model_dir=a.model_dir,lens_path=a.lens_path,
        fit_if_missing=not a.no_fit,layers_per_group=a.layers_per_group,
        jacobian_rows_per_pass=a.jacobian_rows_per_pass,fit_python=a.fit_python))

if __name__=='__main__':main()
