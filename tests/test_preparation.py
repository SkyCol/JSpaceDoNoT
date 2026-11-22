"""Preparation sequencing and the exact WikiText prompt-selection contract."""
import hashlib,importlib.util,json,pathlib,sys,tempfile,unittest
from unittest.mock import patch
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from jspace_donot.preparation import prepare_assets,completed_lenses,validate_lens
spec=importlib.util.spec_from_file_location('prepare_wikitext103',ROOT/'scripts/prepare_wikitext103.py')
wikitext=importlib.util.module_from_spec(spec);spec.loader.exec_module(wikitext)

class WikiTextPrompts(unittest.TestCase):
    def test_original_order_threshold_and_unstripped_text(self):
        records=[{'text':'  short  '},{'text':'  abcdef  '},{'text':'second long'},{'text':'third long'}]
        self.assertEqual(wikitext.select_prompts(records,2,6),['  abcdef  ','second long'])
    def test_insufficient_prompts_fail(self):
        with self.assertRaises(ValueError):wikitext.select_prompts([{'text':'short'}],2,6)

class FullPreparation(unittest.TestCase):
    def test_missing_lens_triggers_original_fitting_after_wikitext(self):
        with tempfile.TemporaryDirectory() as folder:
            root=pathlib.Path(folder);calls=[]
            def run(command,**kwargs):
                script=pathlib.Path(command[1]).name;calls.append((script,command))
                if script=='download_model.py':
                    model=root/'base_models/Qwen3-4B';model.mkdir(parents=True)
                    (model/'config.json').write_text('{}');(model/'model.safetensors').write_bytes(b'model-fixture')
                if script=='prepare_wikitext103.py':
                    p=root/'jlens_outputs/qwen3_4b/jpp_fit/wikitext103_raw_train_first_1000_minchars600.json'
                    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(['text']*1000))
                if script=='fit_jpp.py':
                    self.make_lens(root)
            with patch('jspace_donot.preparation.subprocess.run',side_effect=run):
                assets=prepare_assets(root,datasets=('gsm8k',))
                order=[s for s,c in calls]
                self.assertEqual(order,['download_model.py','setup_jpp.py','prepare_wikitext103.py','fit_jpp.py','prepare_gsm8k.py'])
                fit=calls[3][1]
                self.assertEqual(fit[fit.index('--layers-per-group')+1],'4')
                self.assertEqual(fit[fit.index('--jacobian-rows-per-pass')+1],'8')
                self.assertTrue(assets.lens_path.is_file())
                self.assertIn('--source-only',calls[1][1])
                calls.clear()
                prepare_assets(root,datasets=(),lens_path=assets.lens_path)
                self.assertNotIn('fit_jpp.py',[s for s,c in calls])
    @staticmethod
    def make_lens(root):
        model=root/'base_models/Qwen3-4B'
        lens=root/'jlens_outputs/qwen3_4b/jpp_fit/test-completed/qwen3_4b_jpp_lens.pt';lens.parent.mkdir(parents=True,exist_ok=True);lens.write_bytes(b'lens-fixture')
        assets={p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in [model/'config.json',model/'model.safetensors']}
        lens.with_name('qwen3_4b_jpp_manifest.json').write_text(json.dumps(dict(base=dict(assets=assets),lens_sha256=hashlib.sha256(lens.read_bytes()).hexdigest())))
        return lens
    def test_mismatched_model_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=pathlib.Path(folder);model=root/'base_models/Qwen3-4B';model.mkdir(parents=True)
            (model/'config.json').write_text('{}');(model/'model.safetensors').write_bytes(b'model')
            lens=self.make_lens(root);(model/'model.safetensors').write_bytes(b'changed-model')
            with self.assertRaisesRegex(ValueError,'asset mismatch'):validate_lens(lens,model)

if __name__=='__main__':unittest.main()
