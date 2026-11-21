"""Optional real tiny Qwen3 check for the extracted answer-boundary readout.
No weights or tokenizer are downloaded. CUDA synchronization is mocked on CPU.
"""
import pathlib,sys,unittest,time
from unittest.mock import patch
import numpy as np
import torch
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from test_pipeline import extracted_function

class TinyQwenReadout(unittest.TestCase):
    def test_real_backbone_hooks_lens_transport_and_score_readout(self):
        try:
            import jlens
            from transformers import Qwen3Config,Qwen3ForCausalLM
        except ImportError:self.skipTest('Optional transformers / official jlens source not installed')
        torch.manual_seed(2026)
        model=Qwen3ForCausalLM(Qwen3Config(vocab_size=64,hidden_size=32,intermediate_size=64,
            num_hidden_layers=6,num_attention_heads=4,num_key_value_heads=2,head_dim=8)).eval()
        layers=model.model.layers
        lens_model=jlens.from_hf(model,None,compile=False)
        scope=dict(torch=torch,base_model=model,layers=layers,DEVICE=torch.device('cpu'),DYN_LAYERS=[0,1,3,4],MAX_PREFIX_TOKENS=100)
        capture=extracted_function('gsm8k','ac_capture',scope)
        class IdentityLens:
            def transport(self,h,layer):return h
        top=extracted_function('gsm8k','top_allowed_ids',dict(np=np))
        pool=extracted_function('gsm8k','token_candidate_pool',dict(PROPOSAL_TOP_PER_LAYER=8,PROPOSAL_INCLUDE_LM=False,top_allowed_ids=top))
        ns=dict(scope,np=np,time=time,MAX_PREFIX_TOKENS=100,STOP_IDS=[5,6],ac_capture=capture,
            lens=IdentityLens(),lens_model=lens_model,token_candidate_pool=pool,ALLOWED_PROPOSAL_IDS=np.arange(64))
        compute=extracted_function('gsm8k','compute_token_readout',ns)
        run=dict(prompt_ids=[1,2,3],generated_ids=[4,7,8])
        with patch('torch.cuda.synchronize'):
            report=compute(run,2)
        self.assertEqual(report['proposal']['forwards'],1)
        self.assertEqual(report['proposal']['replayed_tokens'],7)
        # Independent direct forward establishes the layer/output-normalization convention.
        with torch.no_grad():
            result=model(torch.tensor([[1,2,3,4,7,5,6]]),output_hidden_states=True)
            for idx,layer in enumerate([0,1,3,4]):
                state=result.hidden_states[layer+1][0,-1]
                expected=lens_model.unembed(state).float().log_softmax(-1).numpy()
                for token in report['proposal']['token_ids']:
                    np.testing.assert_allclose(report['scores'][f'tok:{token}']['mean_logp'][idx],expected[token],atol=2e-5)

if __name__=='__main__':unittest.main()
