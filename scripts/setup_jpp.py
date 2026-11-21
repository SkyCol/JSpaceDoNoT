"""Fetch the exact upstream J++ implementation used by the recovered fit recipe.

Install with --no-deps to avoid replacing the existing evaluation environment.
"""
import argparse,subprocess,sys
from pathlib import Path
COMMIT='d7efc0de372e20589e899188c0cfe8434acc63ec'
URL='https://github.com/safety-research/jpp_lens.git'

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('.'));a=p.parse_args(argv)
    dest=a.root.resolve()/'external/jpp_lens'
    dest.parent.mkdir(parents=True,exist_ok=True)
    if not dest.exists():subprocess.run(['git','clone',URL,str(dest)],check=True)
    current=subprocess.check_output(['git','-C',str(dest),'rev-parse','HEAD'],text=True).strip()
    if current!=COMMIT:
        dirty=subprocess.check_output(['git','-C',str(dest),'status','--porcelain'],text=True).strip()
        if dirty:raise RuntimeError('Upstream checkout has local changes; refusing to replace them')
        subprocess.run(['git','-C',str(dest),'checkout',COMMIT],check=True)
    subprocess.run([sys.executable,'-m','pip','install','--no-deps','-e',str(dest)],check=True)
    print('Official J++ source ready:',dest,COMMIT)

if __name__=='__main__':main()
