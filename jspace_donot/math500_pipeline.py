# GENERATED FROM USER-PROVIDED NOTEBOOKS. Source cells are identified inline.
# Private task adapter; use the public run_experiment API. Only J++ heads are fitted.
# Requires a completed official J++ lens and a CUDA GPU.

# ===== Source notebook Qwen3_MATH500_JSpaceDoNOT.ipynb / cell 1 =====
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
from decimal import Decimal, InvalidOperation
import hashlib, json, re, time, gc, os, bisect, codecs
import numpy as np
import pandas as pd
from IPython.display import display
ROOT=Path(os.environ.get('JSPACE_ROOT', '.')).resolve()

# Main experiment settings
SEED=2026  # One root seed for sampling, folds, model initialization and cache provenance.
N_CALIBRATION=int(os.environ.get('JSPACE_CALIBRATION', '256'))  # Parent TRAIN cohort; PILOT_TRAIN_N optionally limits it.
PILOT_TRAIN_N=None  # None keeps the entire parent TRAIN cohort.
MAX_TEST_QUESTIONS=None  # None selects the entire official TEST parent cohort.
PILOT_TEST_N=int(os.environ['JSPACE_TEST_LIMIT']) if os.environ.get('JSPACE_TEST_LIMIT') else None  # None keeps the entire parent TEST cohort.
CV_FOLDS=4
N_SOURCE_BLOCKS=4
SOURCE_BLOCKS=None             # explicit zero-based fitted blocks override count
CHECK_EVERY=int(os.environ.get('JSPACE_CHECK_EVERY', '256'))
MIN_STOP_TOKENS=int(os.environ.get('JSPACE_MIN_STOP', '256'))
MAX_MONITOR_TOKENS=int(os.environ['JSPACE_MAX_MONITOR']) if os.environ.get('JSPACE_MAX_MONITOR') else None
J_WAIT_MARKERS=tuple(m.strip() for m in os.environ.get('JSPACE_WAIT_MARKERS','Wait').split(',') if m.strip())

# Display controls only: do not affect training, thresholds or saved evaluation rows.
PLOT_MIN_TOKENS=0  # E.g. 512 hides plotted points with mean logical-token cost <512.
# The upper display limit remains Full's mean logical-token cost.

# Operating-point search. Same defaults as the uploaded notebook.
MAX_FRONTIER_POINTS=6  # Maximum TRAIN Pareto points per family; None retains all.
LEARNED_CUTOFFS=(.2,.4,.6,.75,.85,.95)
LINEAR_C=(.1,1.,10.)
MLP_HIDDEN=32
MLP_EPOCHS=80

# Full-vocabulary proposal settings (no text extraction or lexical whitelist).
SCORE_FEATURES='raw'  # raw | full: use full for the matched feature ablation.
PROPOSAL_TOP_PER_LAYER=8   # Candidate pool: union of top 8 from each selected J++ layer.
PROPOSAL_INCLUDE_LM=False   # J-only proposals; LM remains a separate scoring baseline.
EXCLUDE_SPECIAL_PROPOSALS=True  # Only tokenizer control tokens; punctuation remains eligible.
LEARNED_TOP_CANDIDATES=4   # Per-head feature slots, independent of pool union size.
MIN_CANDIDATES=2
# Raw-hidden MLP: trained only on TRAIN questions; no PCA or J candidate filter.

# Paths
MODEL_ID='Qwen/Qwen3-4B'
MODEL_DIR=ROOT/'base_models'/'Qwen3-4B'
LENS_DIR=ROOT/'jlens_outputs'/'qwen3_4b'
LENS_TAG='qwen3_4b_wikitext_64_seq128'
JPP_LENS_PATH=os.environ.get('JSPACE_LENS_PATH')  # None finds exactly one completed checkpoint
JPP_SEARCH_ROOT=LENS_DIR/'jpp_fit'
TRAIN_PATH=ROOT/'data/math500/math_original_train.json'
TEST_PATH=ROOT/'data/math500/math500_test.json'
MODULAR_CACHE_ROOT=LENS_DIR/'math500_boxed_cache'

# Existing source-generation protocol: must match the cached job
OLD_TOKEN_BUDGET=int(os.environ.get('JSPACE_MAX_NEW_TOKENS','8192'))
OLD_INSTRUCTION=("Solve the problem. " r"Give only the final answer in \boxed{}.")
SOURCE_GEN_BACKEND='hf_batched'  # hf_batched | hf_serial
SOURCE_GEN_BATCH_SIZE=1
SOURCE_GEN_ENGINE_VERSION='hf-leftpad-fixed-batch-shared-rng-v1'
SOURCE_GENERATION_REVISION='math500-boxed-v1'
SOURCE_SAMPLING=dict(do_sample=True,temperature=.6,top_p=.95,top_k=20,min_p=0.)

# Answer protocol and cache controls
ANSWER_SUFFIX='\n</think>\n\nAnswer: \\boxed{'
BOX_PREFIX=r'\boxed{'
ANSWER_PROTOCOL_VERSION='math500-symbolic-branch-v1'
ANSWER_BUDGET=128
MAX_PREFIX_TOKENS=14000
ALLOW_NEW_ANSWER_PROBES=True
VALIDATION_SPLITS_FILE=None  # optional OLD question_splits.csv; use only its val questions
CALIBRATION_ONLY=False
RESUME_DIR=os.environ.get('JSPACE_OUTPUT')  # previous calibrated experiment output to resume TEST reports
REBUILD_GENERATION_INDEX=False
# Retain historical CoT generations when a library-version fingerprint changed.
# If the original modular run's protocol.json exists, its signature is checked exactly.
# Otherwise check every stored batch/member/journal/prompt and record the limitation.
ALLOW_RECORD_VERIFIED_SOURCE=True
PRIMARY_METRIC='math_verify_boxed_v1'  # last balanced numerical box, matching source notebook
RUN_METHODS=['JspaceDoNOT']  # enables the existing lens loader only
# SCHEDULE='wait'  # J++ and matched LM/hidden controls inspect Wait transitions.
SCHEDULE=os.environ.get('JSPACE_SCHEDULE', 'wait')
PERSISTENCE=1  # one accepted check is sufficient; T itself uses two observed states

# Derived directories and run state.
PILOT_RAW_ROOT=LENS_DIR/'math500_boxed_branches_v1'
PROBE_CACHE=PILOT_RAW_ROOT
INDEX_ROOT=LENS_DIR/'math500_generation_indices'
BOUNDARY_ROOT=LENS_DIR/'math500_boundary_signals_v1'
TRAJECTORY_CACHE=LENS_DIR/'math500_vocab_boundary_readouts_v1'
OUT=(Path(RESUME_DIR) if RESUME_DIR is not None else
     LENS_DIR/('MATH500_JSpaceDoNOT_RawScores_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')))
COUNTS=Counter()
CACHE_AUDIT=[]
for path in (OUT,PROBE_CACHE,INDEX_ROOT,BOUNDARY_ROOT,TRAJECTORY_CACHE):
    path.mkdir(parents=True,exist_ok=True)
assert SOURCE_GEN_BACKEND in {'hf_batched','hf_serial'}
assert SCHEDULE in {'token','wait'}
assert N_CALIBRATION>=CV_FOLDS>=2 and PERSISTENCE>=1
assert CHECK_EVERY>0 and MIN_STOP_TOKENS>0
assert MAX_MONITOR_TOKENS is None or MAX_MONITOR_TOKENS>=MIN_STOP_TOKENS
assert J_WAIT_MARKERS and all(isinstance(m,str) and m for m in J_WAIT_MARKERS)
assert PLOT_MIN_TOKENS>=0
assert MAX_FRONTIER_POINTS is None or (isinstance(MAX_FRONTIER_POINTS,int) and MAX_FRONTIER_POINTS>=1)
assert SCORE_FEATURES in {'raw','full'}
print('Feature mode:',SCORE_FEATURES,'| LM proposals:',PROPOSAL_INCLUDE_LM,flush=True)
print('Output:',OUT,flush=True)

NATURAL_CACHE_POLICY=os.environ.get('JSPACE_CACHE_MODE','resume')

# API-selected cache roots; report files always remain in OUT.
if os.environ.get('JSPACE_CACHE_ROOT'):
    _cache_base=Path(os.environ['JSPACE_CACHE_ROOT'])
    MODULAR_CACHE_ROOT=_cache_base/MODULAR_CACHE_ROOT.name
    PILOT_RAW_ROOT=_cache_base/PILOT_RAW_ROOT.name
    PROBE_CACHE=PILOT_RAW_ROOT
    INDEX_ROOT=_cache_base/INDEX_ROOT.name
    BOUNDARY_ROOT=_cache_base/BOUNDARY_ROOT.name
    TRAJECTORY_CACHE=_cache_base/TRAJECTORY_CACHE.name
    for _p in (PILOT_RAW_ROOT,INDEX_ROOT,BOUNDARY_ROOT,TRAJECTORY_CACHE):
        _p.mkdir(parents=True,exist_ok=True)
ALLOW_NEW_ANSWER_PROBES=NATURAL_CACHE_POLICY=='resume'

# ===== Source notebook Qwen3_MATH500_JSpaceDoNOT.ipynb / cell 2 =====
def final_region(text):
    if '<think>' in text and '</think>' not in text:return ''
    return text.rsplit('</think>',1)[-1] if '</think>' in text else text

def parse_boxed(text):
    text=final_region(str(text)); starts=list(re.finditer(r'\\boxed\s*\{',text))
    if not starts:return dict(value=None,status='missing_box',span=None)
    start=starts[-1].end();depth=1;pos=start
    while pos<len(text) and depth:
        if text[pos]=='{':depth+=1
        elif text[pos]=='}':depth-=1
        pos+=1
    if depth:return dict(value=None,status='unfinished_last_box',span=None)
    raw=text[start:pos-1].strip()
    return dict(value=raw if raw else None,status='boxed_expression' if raw else 'empty_box',span=(start,pos-1))
assert parse_boxed(r'\boxed{\frac{1}{2}}')['value']==r'\frac{1}{2}'

# ===== Source notebook Qwen3_MATH500_JSpaceDoNOT.ipynb / cell 3 =====
from math_verify import parse as mv_parse, verify as mv_verify, LatexExtractionConfig
from functools import lru_cache

def gold_of(row):
    gold=row.get('gold',row.get('answer'))
    return str(gold).strip() if gold is not None and str(gold).strip() else None

@lru_cache(maxsize=32768)
def math_equivalent(gold,candidate):
    if not gold or not candidate:return False
    cfg=[LatexExtractionConfig(boxed_match_priority=0)]
    gold_expr=mv_parse(r'\boxed{'+gold+'}',extraction_config=cfg,parsing_timeout=3)
    candidate_expr=mv_parse(r'\boxed{'+candidate+'}',extraction_config=cfg,parsing_timeout=3)
    if not gold_expr or not candidate_expr:return False
    return bool(mv_verify(gold_expr,candidate_expr))

def score_probe(probe,gold):
    p=parse_boxed(probe['text']);correct=bool(p['value'] and math_equivalent(gold,p['value']))
    return dict(prediction=p['value'],correct=correct,
                completed_correct=bool(correct and probe['ended']),ended=probe['ended'])
assert score_probe(dict(text=r'\boxed{\frac{1}{2}}',ended=True),r'\frac{2}{4}')['correct']

# ===== Source notebook Qwen3_MATH500_JSpaceDoNOT.ipynb / cell 4 =====
def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()

def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()

def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp.json')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    tmp.replace(path)

def qid(question):
    return hashlib.sha256(re.sub(r'\s+',' ',question).strip().encode()).hexdigest()

def read_dataset(path,split):
    raw=json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if not isinstance(raw,list):raise TypeError('Expected list of question records.')
    rows=[];seen=set()
    for idx,r in enumerate(raw):
        q=r.get('question');gold=gold_of(r)
        if not isinstance(q,str) or gold is None:raise ValueError(f'{split}: missing question/gold at index {idx}.')
        key=qid(q)
        if key in seen:raise ValueError(f'{split}: duplicate normalized question. Resolve explicitly.')
        seen.add(key);rows.append(dict(dataset_idx=idx,question_id=key,question=q,gold=gold,split=split,pilot_seen=False))
    return rows

train_rows=read_dataset(TRAIN_PATH,'calibration')
TEST_DATA=read_dataset(TEST_PATH,'test')
if {r['question_id'] for r in train_rows}&{r['question_id'] for r in TEST_DATA}:
    raise ValueError('Normalized question overlap between official training and test files.')
if VALIDATION_SPLITS_FILE is not None:
    prior=pd.read_csv(VALIDATION_SPLITS_FILE,dtype=str)
    allowed={qid(q) for q in prior.loc[prior.split.eq('val'),'question']}
    train_rows=[r for r in train_rows if r['question_id'] in allowed]
if len(train_rows)<N_CALIBRATION:raise RuntimeError('Insufficient TRAIN-derived calibration pool.')
pool=sorted(train_rows,key=lambda r:r['question_id'])
rng=np.random.default_rng(SEED)
CAL_DATA=[pool[int(i)] for i in rng.choice(len(pool),N_CALIBRATION,replace=False)]
n_test=len(TEST_DATA) if MAX_TEST_QUESTIONS is None else MAX_TEST_QUESTIONS
if not isinstance(n_test,int) or not 1<=n_test<=len(TEST_DATA):
    raise ValueError('MAX_TEST_QUESTIONS must be a positive integer within dataset size, or None.')
test_rng=np.random.default_rng(SEED)
TEST_DATA=[TEST_DATA[int(i)] for i in sorted(test_rng.choice(len(TEST_DATA),n_test,replace=False))]
# Slice only after the predeclared parent cohorts have been drawn; this is
# deterministic and independent of generation availability or correctness.
CAL_DATA=CAL_DATA[:PILOT_TRAIN_N]
TEST_DATA=TEST_DATA[:PILOT_TEST_N]
TRAIN_SHA=file_sha(TRAIN_PATH);TEST_SHA=file_sha(TEST_PATH)
pd.DataFrame(CAL_DATA).to_csv(OUT/'calibration_manifest.csv',index=False)
pd.DataFrame(TEST_DATA).to_csv(OUT/'test_manifest.csv',index=False)
print('Calibration:',len(CAL_DATA),'official TRAIN questions | Test:',len(TEST_DATA),'| overlap: 0',flush=True)

# Stable fingerprints exclude notebook filenames/line numbers.
def implementation_fingerprint(*functions):
    import types,inspect
    def serialize(value):
        if isinstance(value,types.CodeType):
            return dict(bytecode=value.co_code.hex(),constants=[serialize(x) for x in value.co_consts],
                        names=list(value.co_names),locals=list(value.co_varnames))
        if isinstance(value,(tuple,list)):return [serialize(x) for x in value]
        if isinstance(value,(set,frozenset)):return sorted([serialize(x) for x in value],key=str)
        if isinstance(value,(str,int,float,bool)) or value is None:return value
        if isinstance(value,bytes):return value.hex()
        return type(value).__name__+':'+str(value)
    return digest([serialize(inspect.unwrap(fn).__code__) for fn in functions])

# ===== Source notebook Qwen3_MATH500_JSpaceDoNOT.ipynb / cell 5 =====
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

if not torch.cuda.is_available(): raise RuntimeError('Select a CUDA GPU session.')
DEVICE = torch.device('cuda')
DTYPE = torch.bfloat16
if not torch.cuda.is_bf16_supported():raise RuntimeError('Source cache used bfloat16; use a compatible GPU.')
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR, torch_dtype=DTYPE, local_files_only=True, attn_implementation='sdpa').to(DEVICE).eval()
for p in base_model.parameters(): p.requires_grad_(False)
if base_model.config.model_type != 'qwen3': raise RuntimeError('This pilot is for Qwen3.')
if tokenizer.pad_token_id is None: tokenizer.pad_token_id = tokenizer.eos_token_id
layers = base_model.model.layers
EOS_IDS = base_model.generation_config.eos_token_id
EOS_IDS = set(EOS_IDS if isinstance(EOS_IDS, list) else [EOS_IDS]) - {None}
if tokenizer.eos_token_id is not None: EOS_IDS.add(tokenizer.eos_token_id)
EOS_IDS = sorted(EOS_IDS)
CLOSE_IDS = tokenizer.encode('</think>', add_special_tokens=False)
if len(CLOSE_IDS) != 1: raise RuntimeError('Expected one Qwen3 closing-think token.')
CLOSE_ID = CLOSE_IDS[0]
STOP_IDS = tokenizer.encode(ANSWER_SUFFIX, add_special_tokens=False)

weights=[dict(name=p.name,size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns)
    for p in sorted(MODEL_DIR.glob('*.safetensors'))]
import jlens
lens_model=jlens.from_hf(base_model,tokenizer,compile=False)  # use only frozen unembed
def resolve_jpp_checkpoint(explicit, root):
    if explicit is not None:
        path=Path(explicit)
        if not path.is_file():raise FileNotFoundError(path)
        return path
    paths=sorted(Path(root).rglob('qwen3_4b_jpp_lens.pt'))
    paths=[p for p in paths if p.with_name('qwen3_4b_jpp_manifest.json').is_file()]
    if len(paths)!=1:
        raise RuntimeError('Set JPP_LENS_PATH to one completed fit. Candidates: '+str([str(p) for p in paths]))
    return paths[0]

class SavedCombinedJpp:
    # Exact official JacobianLens transport: row-vector h @ J.T, no expert routing.
    def __init__(self, path):
        payload=torch.load(str(path),map_location='cpu',weights_only=True)
        if not isinstance(payload,dict) or 'config' not in payload:
            raise ValueError('Expected completed official J++ combined lens, not a trainer checkpoint.')
        self.config=payload['config']
        if 'parameters' in payload:
            matrices=payload['parameters'].get('jacobians')
        else:matrices=payload.get('J')
        if not isinstance(matrices,dict) or not matrices:
            raise ValueError('Combined J++ checkpoint has no Jacobian matrices.')
        self.jacobians_L_dict_FN={int(k):v.detach().cpu().float().contiguous() for k,v in matrices.items()}
        self.source_layers=sorted(self.jacobians_L_dict_FN)
        declared=payload.get('source_layers',self.config.get('source_layers',self.source_layers))
        if sorted(map(int,declared))!=self.source_layers:raise ValueError('J++ source-layer metadata mismatch.')
        self.d_model=base_model.config.hidden_size
        if self.config.get('relative_end_transport_layer',-1)!=-1:
            raise ValueError('J++ must transport to the final block.')
        for layer,matrix in self.jacobians_L_dict_FN.items():
            if not 0<=layer<len(layers)-1 or matrix.shape!=(self.d_model,self.d_model):
                raise ValueError(f'Invalid J++ layer/shape: {layer}, {matrix.shape}')
            if not torch.isfinite(matrix).all():raise ValueError('Nonfinite J++ matrix.')
    def transport(self, hidden, layer):
        matrix=self.jacobians_L_dict_FN[layer].to(hidden.device,dtype=hidden.dtype)
        return hidden @ matrix.T

JPP_LENS_PATH=resolve_jpp_checkpoint(JPP_LENS_PATH,JPP_SEARCH_ROOT)
JPP_MANIFEST_PATH=JPP_LENS_PATH.with_name('qwen3_4b_jpp_manifest.json')
jpp_manifest=json.loads(JPP_MANIFEST_PATH.read_text(encoding='utf-8'))
assets=jpp_manifest['base'].get('assets',{})
if not assets or 'config.json' not in assets:raise ValueError('Missing J++ fitting asset hashes.')
for name,record in assets.items():
    path=MODEL_DIR/name
    if not path.is_file() or path.stat().st_size!=record['bytes'] or file_sha(path)!=record['sha256']:
        raise ValueError(f'J++ was fitted to different model/tokenizer bytes: {name}')
if 'lens_sha256' in jpp_manifest and file_sha(JPP_LENS_PATH)!=jpp_manifest['lens_sha256']:
    raise RuntimeError('J++ lens checksum differs from manifest.')
jpp_lens=SavedCombinedJpp(JPP_LENS_PATH)
if sorted(map(int,jpp_manifest['source_layers']))!=jpp_lens.source_layers:
    raise ValueError('J++ manifest source layers disagree with checkpoint.')

lens=jpp_lens
SOURCE_LAYERS=sorted(set(map(int,jpp_lens.source_layers)))
if SOURCE_BLOCKS is not None:DYN_LAYERS=sorted(set(map(int,SOURCE_BLOCKS)))
elif N_SOURCE_BLOCKS is None:DYN_LAYERS=SOURCE_LAYERS
else:
    if not 4<=N_SOURCE_BLOCKS<=len(SOURCE_LAYERS):raise ValueError('Choose at least four fitted J++ layers')
    DYN_LAYERS=[SOURCE_LAYERS[i] for i in np.unique(np.linspace(0,len(SOURCE_LAYERS)-1,N_SOURCE_BLOCKS).round().astype(int))]
if len(DYN_LAYERS)<4 or not set(DYN_LAYERS)<=set(SOURCE_LAYERS):
    raise ValueError('Select four or more J++ fitted source blocks')
atomic_json(OUT/'jpp_readout_manifest.json',dict(checkpoint=str(JPP_LENS_PATH),
    checkpoint_sha256=file_sha(JPP_LENS_PATH),manifest=str(JPP_MANIFEST_PATH),
    manifest_sha256=file_sha(JPP_MANIFEST_PATH),source_layers=DYN_LAYERS))
MODEL_SIGNATURE = digest(dict(model=MODEL_ID,config=file_sha(MODEL_DIR/'config.json'), weights=weights,
    tokenizer=tokenizer.get_vocab(), template=tokenizer.chat_template,
    torch=torch.__version__, transformers=transformers.__version__, dtype=str(DTYPE), attention='sdpa'))
print('GPU:', torch.cuda.get_device_name(), '| source blocks (zero-based):', DYN_LAYERS, flush=True)

def thinking_end(run):
    return next((i for i, token in enumerate(run['generated_ids'])
                 if token == CLOSE_ID or token in EOS_IDS), len(run['generated_ids']))

def scheduled_events(run):
    end = thinking_end(run)
    limit = end - 1 if MAX_MONITOR_TOKENS is None else min(end - 1, MAX_MONITOR_TOKENS)
    if SCHEDULE == 'token':
        return [dict(step=s, detected=s) for s in range(CHECK_EVERY, limit + 1, CHECK_EVERY)]
    if SCHEDULE == 'wait':
        # j_wait_events is defined in the next cell, before this function is called.
        return [e for e in j_wait_events(run) if e['step'] <= limit]
    raise ValueError(f'Unknown schedule: {SCHEDULE}')

# ===== Source notebook Qwen3_MATH500_JSpaceDoNOT.ipynb / cell 6 =====
def decoded_bounds(ids):
    # Exact incremental byte-BPE alignment, verified against tokenizer.decode.
    codes=list(range(ord('!'),ord('~')+1))+list(range(ord('¡'),ord('¬')+1))+list(range(ord('®'),ord('ÿ')+1))
    chars=codes.copy(); extra=0
    for c in range(256):
        if c not in codes: codes.append(c);chars.append(256+extra);extra+=1
    decoder_map={chr(c):b for b,c in zip(codes,chars)}
    decoder=codecs.getincrementaldecoder('utf-8')('replace')
    special=set(tokenizer.all_special_ids); parts=[];bounds=[0]
    for tid,token in zip(ids,tokenizer.convert_ids_to_tokens(ids)):
        if tid in special:
            text=decoder.decode(b'',final=True)+token
            decoder=codecs.getincrementaldecoder('utf-8')('replace')
        else: text=decoder.decode(bytes(decoder_map[c] for c in token),final=False)
        parts.append(text);bounds.append(bounds[-1]+len(text))
    tail=decoder.decode(b'',final=True)
    if tail: parts[-1]+=tail;bounds[-1]+=len(tail)
    raw=''.join(parts)
    expected=tokenizer.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
    if raw!=expected: raise RuntimeError('Token/text boundary reconstruction failed; do not approximate checkpoint alignment.')
    return raw,bounds


def marker_events(run,markers):
    end=thinking_end(run)
    raw,bounds=decoded_bounds(run['generated_ids'][:end])
    pattern=r'(?i)\b(?:'+'|'.join(re.escape(x) for x in markers)+r')(?P<delimiter>\s|[,.:;!?])'
    result=[];used=set()
    for m in re.finditer(pattern,raw):
        step=max(0,bisect.bisect_right(bounds,m.start())-1)
        detected=bisect.bisect_left(bounds,m.end())
        if 0<step<end and detected<=end and step not in used:
            result.append(dict(step=step,detected=detected));used.add(step)
    return result


def j_wait_events(run):
    return marker_events(run,J_WAIT_MARKERS)


@torch.inference_mode()
def ac_capture(prefix, positions):
    """Capture only requested states, not [sequence, vocabulary] logits."""
    if not prefix or max(positions) >= len(prefix) or min(positions) < 0:
        raise ValueError('Invalid answer prediction positions.')
    if len(prefix) > MAX_PREFIX_TOKENS:
        raise MemoryError('Answer readout exceeds context guard.')
    states = {}; handles = []
    def make_hook(layer):
        def hook(module, args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            states[layer] = hidden[0, positions].detach().cpu()
        return hook
    try:
        for layer in DYN_LAYERS:
            handles.append(layers[layer].register_forward_hook(make_hook(layer)))
        ids = torch.tensor([prefix], device=DEVICE)
        out = base_model.model(input_ids=ids, attention_mask=torch.ones_like(ids),
                               use_cache=False, return_dict=True)
        states['final_normalized'] = out.last_hidden_state[0, positions].detach().cpu()
        del out, ids
    finally:
        for handle in handles: handle.remove()
    return states

# ===== Source notebook Qwen3_MATH500_JSpaceDoNOT.ipynb / cell 7 =====
# Fresh MATH generation cache. No GSM8K source keys or old batch-size constraints.
GENERATION_VERSION='math500-hf-serial-per-question-v1'
MODULAR_MODEL_SIGNATURE=MODEL_SIGNATURE
NATURAL_ROOT=MODULAR_CACHE_ROOT/'generations'
NATURAL_ROOT.mkdir(parents=True,exist_ok=True)
def expected_prompt(question,mode='cot'):
    assert mode=='cot'
    chat=tokenizer.apply_chat_template([dict(role='user',content=OLD_INSTRUCTION+'\n\nProblem: '+question)],
        tokenize=False,add_generation_prompt=True,enable_thinking=True)
    return tokenizer.encode(chat,add_special_tokens=False)
def generation_key(row):
    return digest(dict(version=GENERATION_VERSION,model=MODEL_SIGNATURE,prompt_ids=expected_prompt(row['question']),
        sampling=SOURCE_SAMPLING,seed=SEED,budget=OLD_TOKEN_BUDGET))
@torch.inference_mode()
def load_old(row,mode='cot'):
    if mode!='cot':raise ValueError('Only MATH thinking generations are requested.')
    key=generation_key(row);path=NATURAL_ROOT/key[:2]/f'{key}.json';prompt=expected_prompt(row['question'])
    if path.is_file():
        run=json.loads(path.read_text(encoding='utf-8'))
        if run.get('key')!=key or run.get('prompt_ids')!=prompt:raise RuntimeError('Source cache identity mismatch')
        COUNTS['math_generation_hits']+=1
    else:
        if NATURAL_CACHE_POLICY!='resume':raise FileNotFoundError(f'Missing MATH generation: {key}')
        seed=(SEED+int(key[:8],16))%(2**31)
        # 保留每道题原来计算的 seed
        torch.manual_seed(int(seed))
        
        ids = torch.tensor([prompt], dtype=torch.long, device=DEVICE)
        out = base_model.generate(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            max_new_tokens=OLD_TOKEN_BUDGET,
            do_sample=True,
            temperature=SOURCE_SAMPLING['temperature'],
            top_p=SOURCE_SAMPLING['top_p'],
            top_k=SOURCE_SAMPLING['top_k'],
            pad_token_id=tokenizer.pad_token_id,
        )
        generated=out[0,len(prompt):].tolist()
        run=dict(key=key,question=row['question'],mode='cot',prompt_ids=prompt,generated_ids=generated,
            decoded=tokenizer.decode(generated,skip_special_tokens=False,clean_up_tokenization_spaces=False),seed=seed)
        atomic_json(path,run);del ids,out
        COUNTS['math_generation_new']+=1
    decoded=tokenizer.decode(run['generated_ids'],skip_special_tokens=False,clean_up_tokenization_spaces=False)
    if decoded!=run['decoded']:raise RuntimeError('Cached generation does not decode with current tokenizer')
    return run
required=CAL_DATA+([] if CALIBRATION_ONLY else TEST_DATA)
SOURCE_GENERATION_SIGNATURE=digest(dict(version=GENERATION_VERSION,source_keys=[generation_key(r) for r in required],
    train_sha=file_sha(TRAIN_PATH),test_sha=file_sha(TEST_PATH)))
pd.DataFrame([dict(question_id=r['question_id'],split=r['split'],key=generation_key(r))
    for r in required]).to_csv(OUT/'source_generation_manifest.csv',index=False)
print('Natural cache:',NATURAL_ROOT,'| policy:',NATURAL_CACHE_POLICY,flush=True)

# ===== Source notebook Qwen3_MATH500_JSpaceDoNOT.ipynb / cell 8 =====
def modular_probe_signature(run,step):
    prefix=run['prompt_ids']+run['generated_ids'][:step]
    key=digest(dict(engine=ANSWER_PROTOCOL_VERSION,model=MODULAR_MODEL_SIGNATURE,prefix=prefix,
        suffix=STOP_IDS,budget=ANSWER_BUDGET,kind='boxed_answer',greedy=True))
    return key,prefix

def box_status(answer):
    parsed=parse_boxed(answer)
    return parsed,parsed['status'] not in {'missing_box','unfinished_last_box'}

@torch.inference_mode()
def answer_probe(run,step):
    # These probes deliberately use a new cache namespace. Old long-response
    # answers cannot be reused with the new prefix, stop criterion or budget.
    key,prefix=modular_probe_signature(run,step)
    path=PILOT_RAW_ROOT/'math_boxed_answer_v1'/key[:2]/f'{key}.json'
    hit=path.is_file()
    if hit:
        saved=json.loads(path.read_text(encoding='utf-8'))
        if saved.get('key')!=key or saved.get('protocol')!=ANSWER_PROTOCOL_VERSION:
            raise RuntimeError('Boxed-answer cache identity mismatch')
        COUNTS['boxed_answer_hits']+=1
    else:
        if not ALLOW_NEW_ANSWER_PROBES:
            raise FileNotFoundError(f'Missing boxed-answer branch: {key}. Natural generation stays read-only.')
        query=prefix+STOP_IDS
        if len(query)+ANSWER_BUDGET>min(MAX_PREFIX_TOKENS,base_model.config.max_position_embeddings):
            raise ValueError('Boxed-answer branch exceeds context guard')
        ids=torch.tensor([query],device=DEVICE)
        torch.cuda.synchronize();start=time.perf_counter()
        out=base_model.model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=True,return_dict=True)
        logits=base_model.get_output_embeddings()(out.last_hidden_state[:,-1])[0].float()
        past=out.past_key_values;del out,ids
        generated=[];eos=False
        for _ in range(ANSWER_BUDGET):
            tid=int(logits.argmax());generated.append(tid)
            if tid in EOS_IDS:
                eos=True;break
            continuation=tokenizer.decode(generated,skip_special_tokens=False,
                clean_up_tokenization_spaces=False)
            _,closed=box_status(BOX_PREFIX+continuation)
            if closed:break
            out=base_model.model(input_ids=torch.tensor([[tid]],device=DEVICE),past_key_values=past,
                attention_mask=torch.ones((1,len(query)+len(generated)),device=DEVICE,dtype=torch.long),
                use_cache=True,return_dict=True)
            logits=base_model.get_output_embeddings()(out.last_hidden_state[:,-1])[0].float()
            past=out.past_key_values;del out
        continuation=tokenizer.decode(generated,skip_special_tokens=False,
            clean_up_tokenization_spaces=False)
        answer=BOX_PREFIX+continuation
        parsed,closed=box_status(answer)
        torch.cuda.synchronize()
        saved=dict(key=key,protocol=ANSWER_PROTOCOL_VERSION,kind='boxed_answer',
            answer_ids=generated,answer_text=answer,answer_ended=closed,
            eos_ended=eos,suffix_tokens=len(STOP_IDS),parser_status=parsed['status'],
            seconds=time.perf_counter()-start)
        del past,logits
        atomic_json(path,saved);COUNTS['boxed_answer_new']+=1
    continuation=tokenizer.decode(saved['answer_ids'],skip_special_tokens=False,
        clean_up_tokenization_spaces=False)
    answer=BOX_PREFIX+continuation
    parsed,closed=box_status(answer)
    if saved['answer_text']!=answer or saved['answer_ended']!=closed or len(saved['answer_ids'])>ANSWER_BUDGET:
        raise RuntimeError('Boxed-answer cache decoding/completion mismatch')
    if closed and parsed['status']!='boxed_expression':
        # Closed but nonnumeric boxes remain incorrect; never search for gold.
        COUNTS['closed_nonnumeric_answer']+=1
    CACHE_AUDIT.append(dict(kind='boxed_answer_branch',key=key,path=str(path),
        status='hit' if hit else 'new'))
    return dict(key=key,text=answer,ids=saved['answer_ids'],ended=closed,
        eos_ended=saved.get('eos_ended',False),seconds=saved.get('seconds',0.))

def score_answer(text,gold,ended):
    p=parse_boxed(text);ok=bool(p['value'] and math_equivalent(gold,p['value']))
    return dict(prediction=p['value'],numeric_match=ok,extracted_match=ok,
        completed_match=bool(ok and ended),answer_ended=bool(ended),
        correct=ok,parser_status=p['status'])
assert score_answer(r'\boxed{\frac{1}{2}}',r'\frac{2}{4}',True)['correct']
print('MATH symbolic branch and last completed boxed expression ready.',flush=True)

# ===== Source notebook Qwen3_MATH500_JSpaceDoNOT.ipynb / cell 9 =====

from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
import torch.nn as nn

TRAJECTORY_VERSION='full-vocabulary-single-boundary-v1'
# Token identity is the vocabulary ID, not its decoded string or numeric value.
# Permit punctuation, whitespace and arbitrary languages. Only tokenizer special
# control IDs are optionally excluded, without renormalizing vocabulary scores.
_model_vocab=base_model.get_output_embeddings().weight.shape[0]
_excluded=set(tokenizer.all_special_ids) if EXCLUDE_SPECIAL_PROPOSALS else set()
ALLOWED_PROPOSAL_IDS=np.asarray(sorted(set(tokenizer.get_vocab().values())-_excluded),dtype=np.int64)
ALLOWED_PROPOSAL_IDS=ALLOWED_PROPOSAL_IDS[(ALLOWED_PROPOSAL_IDS>=0)&(ALLOWED_PROPOSAL_IDS<_model_vocab)]
if len(ALLOWED_PROPOSAL_IDS)<PROPOSAL_TOP_PER_LAYER:raise ValueError('Insufficient allowed vocabulary IDs.')
if not 2<=LEARNED_TOP_CANDIDATES<=PROPOSAL_TOP_PER_LAYER:raise ValueError('Choose 2 <= feature K <= proposal K.')

def top_allowed_ids(logscores,allowed,n):
    """Stable score-descending, ID-ascending top K over valid vocabulary IDs."""
    a=np.asarray(logscores);allowed=np.asarray(allowed,dtype=np.int64)
    values=a[allowed]
    if not np.isfinite(values).all():raise ValueError('Nonfinite vocabulary scores.')
    k=min(int(n),len(allowed))
    if k<=0:return []
    cutoff=np.partition(values,len(values)-k)[len(values)-k]
    # Include all ties before breaking them by vocabulary ID.
    ids=allowed[values>=cutoff]
    order=np.lexsort((ids,-a[ids]))
    return ids[order[:k]].astype(int).tolist()

def token_candidate_pool(j_logp,lm_logp,allowed):
    per_layer=[top_allowed_ids(s,allowed,PROPOSAL_TOP_PER_LAYER) for s in j_logp]
    head=top_allowed_ids(lm_logp,allowed,PROPOSAL_TOP_PER_LAYER) if PROPOSAL_INCLUDE_LM else []
    union=sorted(set(head).union(*(set(ids) for ids in per_layer)))
    return union,per_layer,head

@torch.inference_mode()
def compute_token_readout(run,step):
    # Exactly one answer-boundary backbone replay. No candidate continuation,
    # trial-answer generation, number extraction, or lexical proposal whitelist.
    prefix=run['prompt_ids']+run['generated_ids'][:step]+STOP_IDS
    if len(prefix)>min(MAX_PREFIX_TOKENS,base_model.config.max_position_embeddings):
        raise ValueError('Vocabulary readout exceeds prefix context guard.')
    torch.cuda.synchronize();started=time.perf_counter()
    states=ac_capture(prefix,[len(prefix)-1])
    hidden=np.concatenate([states[l][0].float().numpy().reshape(-1) for l in DYN_LAYERS]).astype(np.float32)
    j=[]
    for layer in DYN_LAYERS:
        h=states[layer].to(DEVICE,dtype=torch.float32)
        logits=lens_model.unembed(lens.transport(h,layer=layer)).float()[0]
        j.append(logits.log_softmax(-1).cpu().numpy())
    j=np.stack(j).astype(np.float32)
    head=base_model.get_output_embeddings()
    lm=head(states['final_normalized'].to(DEVICE,dtype=head.weight.dtype)).float()[0].log_softmax(-1).cpu().numpy()
    ids,by_layer,by_head=token_candidate_pool(j,lm,ALLOWED_PROPOSAL_IDS)
    scores={}
    for tid in ids:
        name=f'tok:{tid}'
        scores[name]=dict(token_ids=[tid],mean_logp=j[:,tid].astype(float).tolist(),
                          lm_mean_logp=float(lm[tid]))
    torch.cuda.synchronize()
    return dict(step=int(step),eligible=[f'tok:{tid}' for tid in ids],scores=scores,
        proposal=dict(token_ids=ids,layer_top_ids=by_layer,lm_top_ids=by_head,
            hidden_vector=hidden.astype(float).tolist(),forwards=1,replayed_tokens=len(prefix),
            seconds=time.perf_counter()-started),
        candidate_forward_tokens=0,first_token_only=True)

READOUT_SIGNATURE=digest(dict(version=TRAJECTORY_VERSION,model=MODEL_SIGNATURE,lens=file_sha(JPP_LENS_PATH),
    layers=DYN_LAYERS,suffix=STOP_IDS,proposal_top_per_layer=PROPOSAL_TOP_PER_LAYER,
    include_lm=PROPOSAL_INCLUDE_LM,exclude_special=EXCLUDE_SPECIAL_PROPOSALS,
    valid_vocabulary_ids=ALLOWED_PROPOSAL_IDS.tolist(),
    implementation=implementation_fingerprint(top_allowed_ids,token_candidate_pool,compute_token_readout)))

def readout_at(run,row,step):
    # row supplies no candidate content. In particular, question/gold are never inspected here.
    key=digest(dict(readout=READOUT_SIGNATURE,generation=run['key'],step=int(step)))
    path=TRAJECTORY_CACHE/key[:2]/f'{key}.json'
    hit=path.is_file()
    if hit:
        report=json.loads(path.read_text(encoding='utf-8'))
        if report.get('key')!=key or report.get('readout_signature')!=READOUT_SIGNATURE:
            raise RuntimeError('Vocabulary observation cache identity mismatch.')
        COUNTS['vocab_readout_hits']+=1
    else:
        if NATURAL_CACHE_POLICY=='read_only':raise FileNotFoundError(f'Missing readout: {key}')
        report=dict(key=key,readout_signature=READOUT_SIGNATURE,**compute_token_readout(run,int(step)))
        atomic_json(path,report);COUNTS['vocab_readout_new']+=1
    CACHE_AUDIT.append(dict(kind='vocabulary_boundary_readout',key=key,path=str(path),status='hit' if hit else 'new'))
    return report

TOP=LEARNED_TOP_CANDIDATES


_LM_SUPERSET_READOUT_SIGNATURE=digest(dict(version=TRAJECTORY_VERSION,model=MODEL_SIGNATURE,lens=file_sha(JPP_LENS_PATH),
    layers=DYN_LAYERS,suffix=STOP_IDS,proposal_top_per_layer=PROPOSAL_TOP_PER_LAYER,
    include_lm=True,exclude_special=EXCLUDE_SPECIAL_PROPOSALS,
    valid_vocabulary_ids=ALLOWED_PROPOSAL_IDS.tolist(),
    implementation=implementation_fingerprint(top_allowed_ids,token_candidate_pool,compute_token_readout)))


# A previous LM-inclusive pool contains the identical full-vocabulary scores for
# every J-proposed token. Recover that exact subset without another GPU forward.
# This never changes old files or treats LM-only proposals as J proposals.
_readout_without_projection=getattr(readout_at,'_raw_scores_original',readout_at)
def readout_at(run,row,step):
    key=digest(dict(readout=READOUT_SIGNATURE,generation=run['key'],step=int(step)))
    path=TRAJECTORY_CACHE/key[:2]/f'{key}.json'
    if not path.exists() and not PROPOSAL_INCLUDE_LM and NATURAL_CACHE_POLICY!='read_only':
        parent_key=digest(dict(readout=_LM_SUPERSET_READOUT_SIGNATURE,generation=run['key'],step=int(step)))
        parent_path=TRAJECTORY_CACHE/parent_key[:2]/f'{parent_key}.json'
        if parent_path.is_file():
            old=json.loads(parent_path.read_text(encoding='utf-8'))
            if (old.get('key')!=parent_key or old.get('readout_signature')!=_LM_SUPERSET_READOUT_SIGNATURE
                or old.get('step')!=int(step)):
                raise RuntimeError('LM-superset cache identity mismatch.')
            tops=old['proposal']['layer_top_ids']
            if len(tops)!=len(DYN_LAYERS):raise RuntimeError('Cached proposal layers mismatch.')
            ids=sorted(set(int(tid) for layer_ids in tops for tid in layer_ids))
            names=[f'tok:{tid}' for tid in ids]
            for name,tid in zip(names,ids):
                score=old['scores'].get(name)
                if score is None or score.get('token_ids')!=[tid]:
                    raise RuntimeError('Incomplete J token scores in LM-superset cache.')
                if len(score['mean_logp'])!=len(DYN_LAYERS):raise RuntimeError('Cached score layer mismatch.')
            projected=dict(old,key=key,readout_signature=READOUT_SIGNATURE,
                eligible=names,scores={name:old['scores'][name] for name in names},
                proposal=dict(old['proposal'],token_ids=ids,lm_top_ids=[]),
                cache_projection=dict(source_key=parent_key,source_path=str(parent_path),
                    reason='exact J-only subset of previously scored vocabulary pool',
                    seconds_provenance='inherited measurement from the shared source observation'))
            atomic_json(path,projected)
            COUNTS['vocab_J_subset_reused']+=1
    return _readout_without_projection(run,row,step)
readout_at._raw_scores_original=_readout_without_projection

def full_score_vectors(readout, previous):
    names = list(readout['eligible'])
    scores = readout['scores']
    n = len(names)

    if n < MIN_CANDIDATES:
        return dict(
            usable=False,
            winner=None,
            margin=np.nan,
            depth_gain=np.nan,
            late_agreement=np.nan,
            persistent=False,
            j_vector=np.zeros(
                3 * len(DYN_LAYERS) * TOP + 2 * TOP,
                dtype=np.float32,
            ).tolist(),
            lm_vector=np.zeros(
                3 * TOP + 2 * TOP,
                dtype=np.float32,
            ).tolist(),
        )

    j = np.array(
        [scores[a]['mean_logp'] for a in names],
        dtype=np.float32,
    ).T
    lm = np.array(
        [scores[a]['lm_mean_logp'] for a in names],
        dtype=np.float32,
    )

    if (
        j.shape != (len(DYN_LAYERS), n)
        or not np.isfinite(j).all()
        or not np.isfinite(lm).all()
    ):
        raise ValueError(
            'Incomplete/nonfinite J or LM candidate score matrix'
        )

    half = len(DYN_LAYERS) // 2
    early = j[:half].mean(0)
    late = j[half:].mean(0)

    order = np.argsort(-late, kind='stable')
    winner = names[int(order[0])]
    runner = names[int(order[1])]
    advantage = j[:, order[0]] - j[:, order[1]]

    margin = float(late[order[0]] - late[order[1]])
    gain = float(
        advantage[half:].mean() - advantage[:half].mean()
    )
    agree = float(
        np.mean(np.argmax(j[half:], axis=1) == order[0])
    )

    # Shared candidate pool; each representation ranks its own top K.
    # No gold, final-answer probe or future tokens enter.
    selection = order[:TOP]
    selected = [names[int(i)] for i in selection]

    raw = np.full(
        (len(DYN_LAYERS), TOP), -50., dtype=np.float32
    )
    relative = np.full_like(raw, -50.)
    delta = np.zeros_like(raw)

    lm_raw = np.full(TOP, -50., dtype=np.float32)
    lm_relative = np.full(TOP, -50., dtype=np.float32)
    lm_delta = np.zeros(TOP, dtype=np.float32)

    present = np.zeros(TOP, dtype=np.float32)
    previous_present = np.zeros(TOP, dtype=np.float32)

    log_z = np.logaddexp.reduce(j, axis=1)
    lm_z = np.logaddexp.reduce(lm)

    for k, name in enumerate(selected):
        idx = names.index(name)
        raw[:, k] = j[:, idx]
        relative[:, k] = j[:, idx] - log_z
        present[k] = 1.

        if previous is not None and name in previous['scores']:
            old = previous['scores'][name]
            delta[:, k] = j[:, idx] - np.asarray(
                old['mean_logp'], np.float32
            )
            previous_present[k] = 1.

    lm_present = np.zeros(TOP, dtype=np.float32)
    lm_previous_present = np.zeros(TOP, dtype=np.float32)

    for k, idx in enumerate(
        np.argsort(-lm, kind='stable')[:TOP]
    ):
        name = names[int(idx)]
        lm_raw[k] = lm[idx]
        lm_relative[k] = lm[idx] - lm_z
        lm_present[k] = 1.

        if previous is not None and name in previous['scores']:
            lm_delta[k] = lm[idx] - float(
                previous['scores'][name]['lm_mean_logp']
            )
            lm_previous_present[k] = 1.

    return dict(
        usable=True,
        winner=winner,
        runner=runner,
        margin=margin,
        depth_gain=gain,
        late_agreement=agree,
        persistent=(
            previous is not None
            and previous.get('winner') == winner
        ),
        candidate_count=n,
        winner_tokens=len(scores[winner]['token_ids']),
        j_vector=np.r_[
            raw.flatten(),
            relative.flatten(),
            delta.flatten(),
            present,
            previous_present,
        ].tolist(),
        lm_vector=np.r_[
            lm_raw,
            lm_relative,
            lm_delta,
            lm_present,
            lm_previous_present,
        ].tolist(),
    )


def raw_only_vectors(readout, previous=None):
    """Current scores only. No relative scores, deltas or indicator inputs."""
    names=list(readout['eligible']);scores=readout['scores'];n=len(names)
    width_j=len(DYN_LAYERS)*TOP
    # Never feed padding/missing indicators into the score-only classifier.
    if n<max(MIN_CANDIDATES,TOP):
        return dict(usable=False,winner=None,runner=None,margin=np.nan,depth_gain=np.nan,
            late_agreement=np.nan,persistent=False,candidate_count=n,winner_tokens=0,
            j_vector=np.zeros(width_j,np.float32).tolist(),lm_vector=np.zeros(TOP,np.float32).tolist())
    j=np.asarray([scores[name]['mean_logp'] for name in names],dtype=np.float32).T
    lm=np.asarray([scores[name]['lm_mean_logp'] for name in names],dtype=np.float32)
    if j.shape!=(len(DYN_LAYERS),n) or not np.isfinite(j).all() or not np.isfinite(lm).all():
        raise ValueError('Invalid current token-score matrix.')
    half=len(DYN_LAYERS)//2
    late=j[half:].mean(0)
    j_order=np.argsort(-late,kind='stable')
    lm_order=np.argsort(-lm,kind='stable')
    winner=names[int(j_order[0])];runner=names[int(j_order[1])]
    advantage=j[:,j_order[0]]-j[:,j_order[1]]
    # The following scalar annotations are for saved diagnostics only.
    # Classifiers consume exclusively j_vector/lm_vector (or the hidden-state control).
    gain=float(advantage[half:].mean()-advantage[:half].mean()) if half else 0.
    return dict(usable=True,winner=winner,runner=runner,
        margin=float(late[j_order[0]]-late[j_order[1]]),depth_gain=gain,
        late_agreement=float(np.mean(np.argmax(j[half:],axis=1)==j_order[0])),
        persistent=bool(previous is not None and previous.get('winner')==winner),
        candidate_count=n,winner_tokens=1,
        j_vector=j[:,j_order[:TOP]].flatten().tolist(),
        lm_vector=lm[lm_order[:TOP]].tolist())

def raw_vectors(readout, previous):
    if SCORE_FEATURES=='raw':return raw_only_vectors(readout,previous)
    return full_score_vectors(readout,previous)

J_FEATURE_WIDTH=(len(DYN_LAYERS)*TOP if SCORE_FEATURES=='raw' else 3*len(DYN_LAYERS)*TOP+2*TOP)
LM_FEATURE_WIDTH=(TOP if SCORE_FEATURES=='raw' else 5*TOP)
FEATURE_SIGNATURE=digest(dict(readout=READOUT_SIGNATURE,top=TOP,mode=SCORE_FEATURES,
    representation='raw-only-or-original-full-v1',
    code=implementation_fingerprint(raw_only_vectors,full_score_vectors,raw_vectors)))


def question_profile(row):
    run=load_old(row);end=thinking_end(run)
    events=[e for e in scheduled_events(run) if MIN_STOP_TOKENS<=e['step']<end]
    trace=[];previous=None
    for event in events:
        step=int(event['step']);pack=readout_at(run,row,step)
        feature=raw_vectors(pack,previous)
        feature['hidden_vector']=pack['proposal']['hidden_vector']
        if feature['usable']:
            _names=list(pack['eligible'])
            _j=np.asarray([pack['scores'][n]['mean_logp'] for n in _names],dtype=np.float32).T
            _order=np.argsort(-_j[len(DYN_LAYERS)//2:].mean(0),kind='stable')[:TOP]
            feature['token_ids']=[int(pack['scores'][_names[i]]['token_ids'][0]) for i in _order]
            feature['token_strings']=[tokenizer.decode([t],clean_up_tokenization_spaces=False) for t in feature['token_ids']]

        previous=dict(scores=pack['scores'],winner=feature['winner']) if feature['usable'] else None
        trace.append(dict(step=step,detected=event['detected'],**feature,
            eligible_candidates=pack['eligible'],requested_candidates=pack['eligible'],
            candidate_replay_tokens=0,candidate_tokens=0,
            proposal_replay_tokens=pack['proposal']['replayed_tokens'],proposal_forwards=1,
            j_readout_seconds=pack['proposal']['seconds']))
    return dict(question_id=row['question_id'],natural_end=end,trace=trace,outcomes={})

def collect_profiles(rows,label):
    packs={};started=time.perf_counter();before=dict(COUNTS)
    for i,row in enumerate(rows,1):
        # Keep visibility during a long question, too. No per-candidate output.
        print(f'\r{label}: {i}/{len(rows)} | loading checkpoints...'.ljust(155),end='',flush=True)
        packs[row['question_id']]=question_profile(row)
        hit=COUNTS['vocab_readout_hits']-before.get('vocab_readout_hits',0)
        new=COUNTS['vocab_readout_new']-before.get('vocab_readout_new',0)
        print(f'\r{label}: {i}/{len(rows)} | boundary cache hits={hit} new={new} | {(time.perf_counter()-started)/60:.1f} min'.ljust(155),end='',flush=True)
    print(flush=True)
    return packs

SCHEDULE_PROTOCOL=dict(schedule=SCHEDULE,j_wait_markers=list(J_WAIT_MARKERS),
    min_stop_tokens=MIN_STOP_TOKENS,max_monitor_tokens=MAX_MONITOR_TOKENS,
    check_every=CHECK_EVERY if SCHEDULE=='token' else None,
    event_code=implementation_fingerprint(scheduled_events,marker_events,j_wait_events))
SCHEDULE_PROTOCOL['signature']=digest(SCHEDULE_PROTOCOL)
METHOD_PROTOCOL=dict(readout_signature=READOUT_SIGNATURE,feature_signature=FEATURE_SIGNATURE,
    schedule=SCHEDULE_PROTOCOL,source_signature=SOURCE_GENERATION_SIGNATURE,
    candidate_kind='single vocabulary token at common answer boundary',
    proposals='union of each source layer top K and optional final-head top K',
    no_text_candidate_extraction=True,no_candidate_teacher_forcing=True,
    answer_transition_is_task_adapter=True,score_feature_mode=SCORE_FEATURES,
    J_feature_width=J_FEATURE_WIDTH,LM_feature_width=LM_FEATURE_WIDTH)
protocol_path=OUT/'vocabulary_method_protocol.json'
if protocol_path.exists():
    if json.loads(protocol_path.read_text(encoding='utf-8'))!=METHOD_PROTOCOL:
        raise RuntimeError('RESUME_DIR contains another feature/source protocol. Use a new output directory; caches remain reusable.')
elif any((OUT/name).exists() for name in ('test_summary.csv','frozen_train_policies.csv')):
    raise RuntimeError('Old output directory predates vocabulary features. Keep it; use RESUME_DIR=None for the new experiment.')
atomic_json(protocol_path,METHOD_PROTOCOL)
atomic_json(OUT/'jpp_schedule_protocol.json',SCHEDULE_PROTOCOL)
print('Vocabulary pool: top',PROPOSAL_TOP_PER_LAYER,'per layer | include LM:',PROPOSAL_INCLUDE_LM,
      '| classifier K:',TOP,'| feature mode:',SCORE_FEATURES,'| J/LM widths:',J_FEATURE_WIDTH,LM_FEATURE_WIDTH,'| cache:',TRAJECTORY_CACHE,flush=True)
print('Source/answer/readout caches retained; predictors are refitted for the selected feature mode.',flush=True)
TRAIN_PROFILES=collect_profiles(CAL_DATA,'TRAIN vocabulary trajectories')

# ===== Source notebook Qwen3_MATH500_JSpaceDoNOT.ipynb / cell 10 =====
def fill_outcomes(rows,profiles,needed,label):
    for i,row in enumerate(rows,1):
        p=profiles[row['question_id']];run=None
        for step in sorted(set(needed[row['question_id']])):
            if str(step) in p['outcomes']:continue
            if run is None:run=load_old(row)
            raw=answer_probe(run,int(step))
            p['outcomes'][str(step)]=dict(**score_answer(raw['text'],row['gold'],raw['ended']),
                text=raw['text'],answer_tokens=len(raw['ids']))
        if i%8==0 or i==len(rows):print(f'{label} outcomes {i}/{len(rows)}',flush=True)

fill_outcomes(CAL_DATA,TRAIN_PROFILES,{r['question_id']:[TRAIN_PROFILES[r['question_id']]['natural_end']]
    +[e['step'] for e in TRAIN_PROFILES[r['question_id']]['trace']] for r in CAL_DATA},'TRAIN')

def full_audit(rows,profiles,label):
    records=[]
    for row in rows:
        p=profiles[row['question_id']];run=load_old(row)
        full=p['outcomes'][str(p['natural_end'])]
        natural=score_answer(run['decoded'],row['gold'],bool(run['generated_ids'] and
            run['generated_ids'][-1] in EOS_IDS))
        records.append(dict(question_id=row['question_id'],natural_correct=natural['correct'],
            forced_correct=full['correct'],forced_box_closed=full['answer_ended'],
            forced_status=full['parser_status']))
    table=pd.DataFrame(records);table.to_csv(OUT/f'{label}_answer_protocol_audit.csv',index=False)
    display(table.agg(dict(natural_correct='mean',forced_correct='mean',forced_box_closed='mean')))
    if table.forced_box_closed.mean()<.95:print('WARNING: Forced Full box closure below 95%; inspect audit and answer budget.',flush=True)
    return table

TRAIN_FULL=full_audit(CAL_DATA,TRAIN_PROFILES,'train')


def add_offline_diagnostics(rows,profiles,label):
    # Gold is used for this OFFLINE audit only, after candidate readout construction.
    # A matching first token is not a complete answer and is never a correctness label.
    records=[]
    for row in rows:
        gold_ids=tokenizer.encode(ANSWER_SUFFIX+str(row['gold']),add_special_tokens=False)
        stable=gold_ids[:len(STOP_IDS)]==STOP_IDS and len(gold_ids)>len(STOP_IDS)
        first=f'tok:{gold_ids[len(STOP_IDS)]}' if stable else None
        for event in profiles[row['question_id']]['trace']:
            records.append(dict(question_id=row['question_id'],step=event['step'],
                reference_first_token_id=first,reference_token_alignment_ok=bool(stable),
                reference_first_token_in_pool=bool(first is not None and first in event['eligible_candidates']),
                first_token_winner_matches_reference=bool(first is not None and event['winner']==first),
                forced_answer_correct=profiles[row['question_id']]['outcomes'].get(str(event['step']),{}).get('correct'),
                **{k:event[k] for k in ('usable','margin','depth_gain','late_agreement','persistent','candidate_count')}))
    table=pd.DataFrame(records)
    table.to_csv(OUT/f'{label}_readout_audit.csv',index=False)
    if not table.empty:
        print(label,'first-token reference coverage (not answer accuracy):',round(table.reference_first_token_in_pool.mean(),3),flush=True)
    return table

TRAIN_DIAGNOSTICS=add_offline_diagnostics(CAL_DATA,TRAIN_PROFILES,'train')

# ===== Source notebook Qwen3_MATH500_JSpaceDoNOT.ipynb / cell 11 =====
from itertools import product


def feature_table(rows,profiles):
    records=[]
    for row in rows:
        p=profiles[row['question_id']]
        for e in p['trace']:
            if not e['usable']:continue
            records.append(dict(question_id=row['question_id'],step=e['step'],
                correct=int(p['outcomes'][str(e['step'])]['correct']) if str(e['step']) in p['outcomes'] else None,
                j_vector=e['j_vector'],lm_vector=e['lm_vector'],hidden_vector=e['hidden_vector']))
    if not records:raise RuntimeError('No usable answer candidate checkpoints')
    return pd.DataFrame(records)

TRAIN_X=feature_table(CAL_DATA,TRAIN_PROFILES)
train_ids=[r['question_id'] for r in CAL_DATA]
FOLDS=[([train_ids[i] for i in tr],[train_ids[i] for i in va])
    for tr,va in KFold(n_splits=CV_FOLDS,shuffle=True,random_state=SEED).split(train_ids)]

def raw_x(frame,representation):
    if representation!='J++':raise KeyError(representation)
    return np.stack(frame.j_vector.to_list()).astype(np.float32)

REPRESENTATIONS=('J++',)

def preprocess_for_head(x,representation):
    if representation!='J++':raise KeyError(representation)
    return StandardScaler()

def fit_classifier(frame,representation,kind,C=1.):
    x=raw_x(frame,representation);y=frame.correct.to_numpy(int)
    count=frame.groupby('question_id').size()
    w=frame.question_id.map(lambda q:1/count[q]).to_numpy(np.float32)
    if len(np.unique(y))<2:
        model=DummyClassifier(strategy='constant',constant=int(y[0]))
        model.fit(x,y,sample_weight=w)
        return dict(kind='linear',model=model,representation=representation)
    if kind=='linear':
        model=make_pipeline(preprocess_for_head(x,representation),LogisticRegression(C=C,max_iter=1000))
        model.fit(x,y,logisticregression__sample_weight=w)
        return dict(kind=kind,model=model,representation=representation)
    if kind!='MLP':raise KeyError(kind)
    preprocessor=preprocess_for_head(x,representation)
    tx_np=preprocessor.fit_transform(x).astype(np.float32)
    mean=tx_np.mean(0);std=tx_np.std(0).clip(1e-6)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED)
        model=nn.Sequential(nn.Linear(tx_np.shape[1],MLP_HIDDEN),nn.Tanh(),
            nn.Linear(MLP_HIDDEN,1))
        opt=torch.optim.Adam(model.parameters(),lr=.01,weight_decay=.01)
        tx=torch.from_numpy((tx_np-mean)/std)
        ty=torch.from_numpy(y.astype(np.float32)[:,None])
        tw=torch.from_numpy((w/w.sum())[:,None])
        for _ in range(MLP_EPOCHS):
            opt.zero_grad(set_to_none=True)
            loss=(nn.functional.binary_cross_entropy_with_logits(model(tx),ty,reduction='none')*tw).sum()
            loss.backward();opt.step()
    model.eval();return dict(kind=kind,model=model,mean=mean,std=std,
        representation=representation,preprocessor=preprocessor)

def predict_classifier(head,frame):
    if frame.empty:return {}
    x=raw_x(frame,head['representation'])
    if head['kind']=='linear':
        p=head['model'].predict_proba(x);classes=list(head['model'].classes_)
        scores=p[:,classes.index(1)] if 1 in classes else np.zeros(len(x))
    else:
        with torch.no_grad():
            scores=head['model'](torch.from_numpy((head['preprocessor'].transform(x).astype(np.float32)-head['mean'])/head['std'])).sigmoid().flatten().numpy()
    return {(r.question_id,int(r.step)):float(score) for r,score in zip(frame.itertuples(),scores)}

OOF={};FULL_HEAD={}
for representation in REPRESENTATIONS:
    for kind in ('linear','MLP'):
        for C in (LINEAR_C if kind=='linear' else (1.,)):
            key=(representation,kind,C);oof={}
            for tr,va in FOLDS:
                a=TRAIN_X[TRAIN_X.question_id.isin(tr)]
                b=TRAIN_X[TRAIN_X.question_id.isin(va)]
                if len(a) and len(b):
                    oof.update(predict_classifier(fit_classifier(a,representation,kind,C),b))
            OOF[key]=oof
            FULL_HEAD[key]=fit_classifier(TRAIN_X,representation,kind,C)
def stop_decision(row,profile,spec,scores=None):
    observed=0;candidate_forward_tokens=0;replayed=0;proposal_replayed=0;j_seconds=0.
    for e in profile['trace']:
        observed+=1
        candidate_forward_tokens+=e['candidate_tokens']
        replayed+=e['candidate_replay_tokens']
        proposal_replayed+=e['proposal_replay_tokens']
        j_seconds+=e['j_readout_seconds']
        p=(scores or {}).get((row['question_id'],e['step']))
        good=bool(e['usable'] and p is not None and p>=spec['cutoff'])
        if good:
            return dict(step=e['step'],checks=observed,monitor_tokens=(observed-1)*len(STOP_IDS),
                lookahead=max(0,e['detected']-e['step']),early_exit=True,candidate_forward_tokens=candidate_forward_tokens,
                replayed_tokens=replayed+proposal_replayed,proposal_replayed_tokens=proposal_replayed,
                j_readout_seconds=j_seconds)
    return dict(step=profile['natural_end'],checks=observed,monitor_tokens=observed*len(STOP_IDS),
        lookahead=0,early_exit=False,candidate_forward_tokens=candidate_forward_tokens,
        replayed_tokens=replayed+proposal_replayed,proposal_replayed_tokens=proposal_replayed,
        j_readout_seconds=j_seconds)

def outcome_at(profile,decision):
    answer=profile['outcomes'][str(decision['step'])]
    visible=decision['step']+len(STOP_IDS)+answer['answer_tokens']
    return dict(**decision,correct=answer['correct'],visible_tokens=visible,
        logical_tokens=visible+decision['monitor_tokens']+decision['lookahead'])

def metrics(results):
    return dict(accuracy=float(np.mean([r['correct'] for r in results])),
        logical_tokens=float(np.mean([r['logical_tokens'] for r in results])),
        early_exit_rate=float(np.mean([r['early_exit'] for r in results])))

def score_policy(rows,profiles,spec,scores=None):
    vals=[outcome_at(profiles[r['question_id']],
        stop_decision(r,profiles[r['question_id']],spec,scores)) for r in rows]
    return dict(**metrics(vals),candidate_forward_tokens=float(np.mean([v['candidate_forward_tokens'] for v in vals])),
        replayed_tokens=float(np.mean([v['replayed_tokens'] for v in vals])),
        proposal_replayed_tokens=float(np.mean([v['proposal_replayed_tokens'] for v in vals])),
        j_seconds=float(np.mean([v['j_readout_seconds'] for v in vals])))

def pareto(entries,max_points=MAX_FRONTIER_POINTS):
    result=[]
    for family in dict.fromkeys(e['family'] for e in entries):
        group=[e for e in entries if e['family']==family]
        keep=[];seen=set()
        for r in sorted(group,key=lambda z:z['policy_id']):
            point=(r['accuracy'],r['logical_tokens'])
            dominated=any(s['accuracy']>=r['accuracy'] and s['logical_tokens']<=r['logical_tokens'] and
                (s['accuracy']>r['accuracy'] or s['logical_tokens']<r['logical_tokens']) for s in group)
            if not dominated and point not in seen:keep.append(r);seen.add(point)
        keep.sort(key=lambda z:z['logical_tokens'])
        if max_points is not None and len(keep)>max_points:
            indices=np.unique(np.linspace(0,len(keep)-1,max_points).round().astype(int))
            keep=[keep[int(i)] for i in indices]
        result.extend(keep)
    return result

CANDIDATES=[]
for representation in REPRESENTATIONS:
    for kind in ('linear','MLP'):
        for C in (LINEAR_C if kind=='linear' else (1.,)):
            for cutoff in LEARNED_CUTOFFS:
                spec=dict(kind=kind,representation=representation,C=C,cutoff=cutoff)
                key=(representation,kind,C)
                CANDIDATES.append(dict(policy_id=digest(dict(feature_signature=FEATURE_SIGNATURE,spec=spec))[:16],
                    family=f'{representation} {kind}',spec=spec,
                    **score_policy(CAL_DATA,TRAIN_PROFILES,spec,OOF[key])))
FROZEN=pareto(CANDIDATES)
pd.DataFrame(CANDIDATES).to_csv(OUT/'train_candidate_policies.csv',index=False)
pd.DataFrame(FROZEN).to_csv(OUT/'frozen_train_policies.csv',index=False)
display(pd.DataFrame(FROZEN)[['family','policy_id','accuracy','logical_tokens','early_exit_rate']])

# ===== Source notebook Qwen3_MATH500_JSpaceDoNOT.ipynb / cell 14 =====
# TEST is first read after all J heads and TRAIN policies are frozen.
TEST_PROFILES=collect_profiles(TEST_DATA,'TEST JSpaceDoNOT trajectories')
fill_outcomes(TEST_DATA,TEST_PROFILES,{r['question_id']:[TEST_PROFILES[r['question_id']]['natural_end']]
    for r in TEST_DATA},'TEST Full only')
TEST_FULL=full_audit(TEST_DATA,TEST_PROFILES,'test')
TEST_DIAGNOSTICS=add_offline_diagnostics(TEST_DATA,TEST_PROFILES,'test')
TEST_X=feature_table(TEST_DATA,TEST_PROFILES)

TEST_HEAD={key:predict_classifier(head,TEST_X)
    for key,head in FULL_HEAD.items()}

TEST_DECISIONS={}
for policy in FROZEN:
    spec=policy['spec'];score=TEST_HEAD[(spec['representation'],spec['kind'],spec['C'])]
    TEST_DECISIONS[policy['policy_id']]={r['question_id']:stop_decision(r,TEST_PROFILES[r['question_id']],spec,score)
        for r in TEST_DATA}
needed={r['question_id']:{TEST_PROFILES[r['question_id']]['natural_end']} for r in TEST_DATA}
for choices in TEST_DECISIONS.values():
    for qid,d in choices.items():needed[qid].add(d['step'])
fill_outcomes(TEST_DATA,TEST_PROFILES,needed,'TEST frozen J exits')

test_rows=[];detail=[]
for policy in FROZEN:
    vals=[]
    for r in TEST_DATA:
        result=outcome_at(TEST_PROFILES[r['question_id']],TEST_DECISIONS[policy['policy_id']][r['question_id']])
        vals.append(result);detail.append(dict(question_id=r['question_id'],family=policy['family'],
            policy_id=policy['policy_id'],**result))
    test_rows.append(dict(family=policy['family'],policy_id=policy['policy_id'],
        n=len(vals),**metrics(vals),
        candidate_forward_tokens=float(np.mean([v['candidate_forward_tokens'] for v in vals])),
        replayed_tokens=float(np.mean([v['replayed_tokens'] for v in vals])),
        j_seconds=float(np.mean([v['j_readout_seconds'] for v in vals])),
        proposal_replayed_tokens=float(np.mean([v['proposal_replayed_tokens'] for v in vals])),
        train_accuracy=policy['accuracy'],train_logical_tokens=policy['logical_tokens']))
full=[]
for r in TEST_DATA:
    p=TEST_PROFILES[r['question_id']];end=p['natural_end'];answer=p['outcomes'][str(end)]
    full.append(dict(correct=answer['correct'],logical_tokens=end+len(STOP_IDS)+answer['answer_tokens'],early_exit=False))
test_rows.append(dict(family='Full',policy_id='Full',n=len(full),**metrics(full),
    candidate_forward_tokens=0.,replayed_tokens=0.,proposal_replayed_tokens=0.,j_seconds=0.))

_all_results=pd.DataFrame(test_rows)
_all_results[_all_results.family.eq('Full')].to_csv(OUT/'full_reference.csv',index=False)
SUMMARY=_all_results[~_all_results.family.eq('Full')].copy()
SUMMARY.to_csv(OUT/'test_summary.csv',index=False)
pd.DataFrame(detail).to_csv(OUT/'test_per_question_j.csv',index=False)
display(SUMMARY[['family','n','accuracy','logical_tokens','early_exit_rate',
    'candidate_forward_tokens','replayed_tokens','proposal_replayed_tokens','j_seconds']].sort_values(['family','logical_tokens']))
