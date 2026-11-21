"""Scientific regression checks: feature/training parity and interpretation math."""
import ast,json,pathlib,sys,unittest,tempfile,pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
import torch
from torch import nn
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from jspace_donot.interpretation import lr_decomposition,integrated_gradients,mlp_logits_raw,stop_probability,StopInterpreter,interpret_run
from test_pipeline import extracted_function


def functions(path):
    text=path.read_text();tree=ast.parse(text)
    # Python 3.12 adds an empty type_params field to function/class ASTs.
    # Reproduce the original 3.12 reference representation on Python 3.11.
    for node in ast.walk(tree):
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
            if 'type_params' not in node._fields:
                node._fields=(*node._fields,'type_params')
                node.type_params=[]
    return {n.name:ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,ast.FunctionDef)}


class ScientificParity(unittest.TestCase):
    def test_checked_in_source_function_parity(self):
        manifest=json.loads((ROOT/'docs/source_function_parity.json').read_text())
        for task,report in manifest.items():
            actual=functions(ROOT/'jspace_donot'/f'{task}_pipeline.py')
            for name,record in report.items():
                if name.startswith('_'):continue
                import hashlib
                self.assertEqual(hashlib.sha256(actual[name].encode()).hexdigest(),record['packaged_ast_sha256'])
                self.assertTrue(record['equal_to_uploaded'],(task,name))
    def test_raw_feature_layer_major_order_and_no_temporal_inputs(self):
        f=extracted_function('gsm8k','raw_only_vectors',dict(np=np,DYN_LAYERS=[0,11,23,34],TOP=4,MIN_CANDIDATES=2))
        readout=dict(eligible=[f'tok:{i}' for i in range(5)],scores={f'tok:{i}':dict(mean_logp=[-10+i,-9+i,-8+i,-7+i],lm_mean_logp=-i) for i in range(5)})
        out=f(readout,None);other=f(readout,{'winner':'tok:1','scores':{}})
        expected=np.array([[-6,-7,-8,-9],[-5,-6,-7,-8],[-4,-5,-6,-7],[-3,-4,-5,-6]]).reshape(-1)
        np.testing.assert_array_equal(out['j_vector'],expected)
        np.testing.assert_array_equal(out['j_vector'],other['j_vector'])
    def test_training_heads_with_question_balancing(self):
        rng=np.random.default_rng(2026);x=rng.normal(size=(30,16)).astype(np.float32)
        frame=pd.DataFrame({'question_id':[f'q{i//3}' for i in range(30)],'step':np.arange(30),'correct':(x[:,0]>0).astype(int),'j_vector':list(x)})
        scope=dict(np=np,torch=torch,nn=nn,raw_x=lambda f,r:np.stack(f.j_vector),preprocess_for_head=lambda x,r:StandardScaler(),
            StandardScaler=StandardScaler,DummyClassifier=DummyClassifier,make_pipeline=make_pipeline,LogisticRegression=LogisticRegression,SEED=2026,MLP_HIDDEN=32,MLP_EPOCHS=80)
        f=extracted_function('gsm8k','fit_classifier',scope)
        for kind in ['linear','MLP']:
            h=f(frame,'J++',kind,.1 if kind=='linear' else 1.)
            scores=stop_probability(h,x)
            self.assertEqual(scores.shape,(30,));self.assertTrue(np.isfinite(scores).all())
            if kind=='MLP':self.assertEqual(h['model'][0].in_features,16)

class Interpretation(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(2026);self.x=rng.normal(size=(30,16)).astype(np.float32)
        self.y=(self.x[:,0]>0).astype(int)
        self.lr=dict(kind='linear',model=make_pipeline(StandardScaler(),LogisticRegression(C=.1)).fit(self.x,self.y))
        pre=StandardScaler().fit(self.x);tx=pre.transform(self.x)
        torch.manual_seed(2026)
        self.mlp=dict(kind='MLP',preprocessor=pre,mean=tx.mean(0),std=tx.std(0).clip(1e-6),model=nn.Sequential(nn.Linear(16,32),nn.Tanh(),nn.Linear(32,1)).eval())
    def test_exact_lr_decomposition(self):
        c,v,l=lr_decomposition(self.lr,self.x)
        np.testing.assert_allclose(l,self.lr['model'].decision_function(self.x),atol=1e-6)
    def test_mlp_both_standardizations(self):
        expected=self.mlp['model'](torch.from_numpy((self.mlp['preprocessor'].transform(self.x)-self.mlp['mean'])/self.mlp['std'])).detach().numpy().ravel()
        np.testing.assert_allclose(mlp_logits_raw(self.mlp,self.x).detach().numpy(),expected,atol=2e-7)
    def test_ig_completeness(self):
        base=self.x.mean(0);ig=integrated_gradients(self.mlp,self.x[:5],base,steps=96)
        expected=(mlp_logits_raw(self.mlp,self.x[:5])-mlp_logits_raw(self.mlp,base[None])).detach().numpy()
        np.testing.assert_allclose(ig.sum(1),expected,atol=2e-4)
    def test_saved_bundle_cpu_interpretation_and_case_apis(self):
        frame=pd.DataFrame(dict(question_id=[f'q{i//3}' for i in range(30)],step=list(range(30)),correct=self.y,j_vector=list(self.x)))
        bundle=dict(heads={('J++','linear',.1):self.lr,('J++','MLP',1.):self.mlp},train=frame,test=frame,layers=[0,11,23,34],top=4,seed=2026)
        obj=StopInterpreter(bundle)
        table,_=obj.group_ablation('q0',0,{'L3':[8,9]})
        self.assertEqual(len(table),2)
        sweep=obj.sweep('q0',0,[8,9]);self.assertIn(0.,sweep.scale);self.assertIn(1.,sweep.scale)
        with tempfile.TemporaryDirectory() as folder:
            with open(pathlib.Path(folder)/'analysis_bundle.pkl','wb') as f:pickle.dump(bundle,f)
            out=interpret_run(folder,max_ig_examples=4,ig_steps=16)
            self.assertTrue((out/'lr_checkpoint_attribution.csv').exists())
            self.assertTrue((out/'mlp_ig_abs.pdf').exists())

if __name__=='__main__':unittest.main()
