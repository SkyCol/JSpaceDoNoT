"""CPU checks: extracted original functions without importing CUDA pipelines."""
import ast
import importlib.util
import pathlib
import sys
import unittest

ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from run import make_parser,validate


def extracted_function(dataset,name,environment):
    path=ROOT/'jspace_donot'/f'{dataset}_pipeline.py'
    tree=ast.parse(path.read_text())
    matches=[node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name==name]
    assert len(matches)==1,(dataset,name,len(matches))
    fn=matches[0]
    fn.decorator_list=[]
    scope=environment.copy()
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),scope)
    return scope[name]


class CLI(unittest.TestCase):
    def test_token_schedule(self):
        p=make_parser().parse_args(['--dataset','gsm8k','--schedule','token','--check-every','512'])
        validate(p)
        self.assertEqual(p.check_every,512)
    def test_wait_schedule(self):
        p=make_parser().parse_args(['--dataset','math500','--schedule','wait','--wait-markers','Wait,Hold'])
        validate(p)
        self.assertEqual(p.schedule,'wait')
    def test_zero_interval_rejected(self):
        p=make_parser().parse_args(['--dataset','raceh','--check-every','0'])
        with self.assertRaises(ValueError):validate(p)


class Schedules(unittest.TestCase):
    def test_every_n_checks(self):
        f=extracted_function('math500','scheduled_events',{
            'thinking_end':lambda run:run['end'], 'MAX_MONITOR_TOKENS':None,
            'SCHEDULE':'token','CHECK_EVERY':256,
        })
        self.assertEqual([e['step'] for e in f({'end':900})],[256,512,768])
    def test_wait_cue_boundary_and_lookahead(self):
        import bisect,re
        # Exact bytes/token alignments are delegated to the notebook's
        # decoded_bounds implementation; fixture provides a known alignment.
        s='Thinking. Wait, perhaps revise.'
        f=extracted_function('math500','marker_events',{
            'thinking_end':lambda run:len(s),
            'decoded_bounds':lambda ids:(s,list(range(len(s)+1))),
            'bisect':bisect,'re':re,
        })
        hits=f({'generated_ids':list(s)},('Wait',))
        self.assertEqual(len(hits),1)
        self.assertEqual(hits[0]['step'],s.index('Wait'))
        self.assertGreater(hits[0]['detected'],hits[0]['step'])
    def test_no_fake_wait_substring(self):
        import bisect,re
        s='Awaiting results, no cue.'
        f=extracted_function('math500','marker_events',{
            'thinking_end':lambda run:len(s),
            'decoded_bounds':lambda ids:(s,list(range(len(s)+1))),
            'bisect':bisect,'re':re,
        })
        self.assertEqual(f({'generated_ids':list(s)},('Wait',)),[])


class Policies(unittest.TestCase):
    def test_stop_only_when_usable_and_above_cutoff(self):
        f=extracted_function('math500','stop_decision',{'STOP_IDS':[1,2]})
        row={'question_id':'q'}
        trace=[dict(step=s,detected=s,usable=u,candidate_tokens=0,candidate_replay_tokens=0,
                    proposal_replay_tokens=100,j_readout_seconds=0.1)
               for s,u in [(256,False),(512,True),(768,True)]]
        p={'trace':trace,'natural_end':1024}
        score={('q',256):.99,('q',512):.4,('q',768):.9}
        out=f(row,p,{'cutoff':.8},score)
        self.assertEqual(out['step'],768)
        self.assertTrue(out['early_exit'])
        self.assertEqual(out['checks'],3)
    def test_fallback_full(self):
        f=extracted_function('math500','stop_decision',{'STOP_IDS':[1]})
        out=f({'question_id':'q'},{'trace':[],'natural_end':23},{'cutoff':.5},{})
        self.assertEqual(out['step'],23)
        self.assertFalse(out['early_exit'])
    def test_representations_only_j(self):
        for dataset in ('gsm8k','math500','raceh'):
            txt=(ROOT/'jspace_donot'/f'{dataset}_pipeline.py').read_text()
            self.assertIn("REPRESENTATIONS=('J++',)",txt)
            self.assertNotIn('RAW_TEST_X=raw_hidden_table',txt)
            self.assertNotIn("family='Budget'",txt)


if __name__=='__main__':unittest.main()
