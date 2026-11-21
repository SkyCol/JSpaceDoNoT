"""Batch membership, left-padding, source reuse and strict cache controls."""
import hashlib,json,pathlib,tempfile,types,unittest,sys
import torch
from transformers import GenerationConfig
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from jspace_donot.generation import prepare_gsm8k_source
from jspace_donot import ExperimentConfig

class GenerationCache(unittest.TestCase):
    def ns(self,root):
        def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()
        def save(p,value):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value))
        class Tokenizer:
            pad_token_id=0
            def decode(self,tokens,**kwargs):return ','.join(map(str,tokens))
        class Model:
            config=types.SimpleNamespace(max_position_embeddings=100)
            def __init__(self):self.calls=0;self.batches=[]
            def generate(self,input_ids,attention_mask,**kwargs):
                self.calls+=1;self.batches.append((input_ids.clone(),attention_mask.clone()))
                return torch.cat([input_ids,torch.tensor([[10,63,0]]*len(input_ids))],dim=1)
        return dict(MODULAR_CACHE_ROOT=pathlib.Path(root),NATURAL_CACHE_POLICY='resume',CALIBRATION_ONLY=False,
            CAL_DATA=[dict(question=str(i),question_id=f'q{i}',split='train') for i in range(4)],
            TEST_DATA=[dict(question='4',question_id='q4',split='test')],expected_prompt=lambda q,mode='cot':[1]*(int(q)+1),
            MODEL_SIGNATURE='model',SEED=2026,SOURCE_GEN_BATCH_SIZE=4,SOURCE_SAMPLING=dict(temperature=.6,top_p=.95,top_k=20,min_p=0),
            OLD_TOKEN_BUDGET=8,atomic_json=save,digest=digest,tokenizer=Tokenizer(),base_model=Model(),DEVICE='cpu',
            SOURCE_GEN_ENGINE_VERSION='hf-leftpad-fixed-batch-shared-rng-v1',SOURCE_GENERATION_REVISION='vanilla-boxed-v1',
            MAX_PREFIX_TOKENS=100,GenerationConfig=GenerationConfig,EOS_IDS=[63])
    def test_batch_seed_member_keys_padding_and_auto_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            ns=self.ns(tmp);prepare_gsm8k_source(ns)
            model=ns['base_model'];self.assertEqual(model.calls,2)
            ids,mask=model.batches[0];self.assertEqual(ids[0].tolist(),[0,0,0,1]);self.assertEqual(mask[0].tolist(),[0,0,0,1])
            journals=list((pathlib.Path(tmp)/'generation_batches').glob('*/*.json'))
            for path in journals:
                pack=json.loads(path.read_text());key=pack['key']
                for i,run in enumerate(pack['runs']):
                    self.assertEqual(run['key'],ns['digest'](dict(engine=ns['SOURCE_GEN_ENGINE_VERSION'],batch=key,member=i)))
                    self.assertEqual(run['seed'],(2026+int(key[:8],16))%(2**31))
                    self.assertEqual(run['generated_ids'],[10,63])
            prepare_gsm8k_source(ns);self.assertEqual(model.calls,2)
    def test_read_only_miss_fails_before_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            ns=self.ns(tmp);ns['NATURAL_CACHE_POLICY']='read_only'
            with self.assertRaises(FileNotFoundError):prepare_gsm8k_source(ns)
            self.assertEqual(ns['base_model'].calls,0)
    def test_changed_fresh_cohort_does_not_regroup(self):
        with tempfile.TemporaryDirectory() as tmp:
            ns=self.ns(tmp);prepare_gsm8k_source(ns,'fresh')
            # New worker starts from the original base, never nested fresh namespaces.
            other=self.ns(tmp);other['CAL_DATA']=list(reversed(other['CAL_DATA']))
            with self.assertRaisesRegex(RuntimeError,'cohort changed'):prepare_gsm8k_source(other,'fresh')
    def test_defaults_and_incompatible_off_historical(self):
        self.assertEqual(ExperimentConfig('gsm8k').validate().schedule,'wait')
        self.assertEqual(ExperimentConfig('gsm8k').every_n_tokens,256)
        with self.assertRaises(ValueError):ExperimentConfig('gsm8k',cache_mode='off',source_mode='historical').validate()

if __name__=='__main__':unittest.main()
