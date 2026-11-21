"""Interpret the fitted stop predictor in raw J-score coordinates.

LR decomposition is exact. MLP IG uses midpoint integration through BOTH
training standardizations. Interventions affect the predictor input only.
"""
from pathlib import Path
import json,pickle,warnings
import numpy as np
import pandas as pd


def matrix(frame):
    x=np.stack(frame.j_vector.to_list()).astype(np.float32)
    if not np.isfinite(x).all():raise ValueError('Nonfinite J features')
    return x


def mlp_logits_raw(head,raw):
    import torch
    from sklearn.preprocessing import StandardScaler
    if head['kind']!='MLP' or not isinstance(head.get('preprocessor'),StandardScaler):
        raise ValueError('A fitted raw J++ MLP with StandardScaler is required; no substitute is fitted.')
    pre=head['preprocessor']
    x=raw if isinstance(raw,torch.Tensor) else torch.as_tensor(raw,dtype=torch.float32)
    x=x.to(dtype=torch.float32,device='cpu')
    mean=torch.as_tensor(pre.mean_,dtype=x.dtype);scale=torch.as_tensor(pre.scale_,dtype=x.dtype)
    secondmean=torch.as_tensor(head['mean'],dtype=x.dtype);secondstd=torch.as_tensor(head['std'],dtype=x.dtype)
    return head['model'].cpu().eval()(((x-mean)/scale-secondmean)/secondstd).reshape(-1)


def stop_probability(head,raw):
    if head['kind']=='linear':
        classes=list(head['model'].classes_);p=head['model'].predict_proba(np.asarray(raw,dtype=np.float32))
        return p[:,classes.index(1)] if 1 in classes else np.zeros(len(p))
    import torch
    with torch.no_grad():return mlp_logits_raw(head,raw).sigmoid().numpy()


def lr_decomposition(head,x):
    pipe=head['model']
    if not hasattr(pipe,'named_steps') or 'logisticregression' not in pipe.named_steps:
        raise ValueError('This head is a constant classifier: LR coefficients do not exist.')
    scaler=pipe.named_steps['standardscaler'];est=pipe.named_steps['logisticregression']
    standardized=scaler.transform(x);contributions=standardized*est.coef_[0]
    logits=contributions.sum(1)+float(est.intercept_[0])
    if not np.allclose(logits,pipe.decision_function(x),atol=1e-5):raise AssertionError('LR decomposition mismatch')
    return est.coef_[0],contributions,logits


def integrated_gradients(head,inputs,baseline,steps=48,batch=24):
    import torch
    if steps<1 or batch<1:raise ValueError('Positive IG steps and batch required')
    result=[]
    for lo in range(0,len(inputs),batch):
        raw=torch.as_tensor(inputs[lo:lo+batch],dtype=torch.float32)
        base=torch.as_tensor(baseline,dtype=torch.float32).expand_as(raw);total=torch.zeros_like(raw)
        for k in range(steps):
            z=(base+((k+.5)/steps)*(raw-base)).detach().requires_grad_(True)
            total+=torch.autograd.grad(mlp_logits_raw(head,z).sum(),z)[0].detach()
        result.append(((raw-base)*total/steps).numpy())
    return np.concatenate(result)


def replace_features(head,x,reference,indices):
    modified=np.asarray(x,dtype=np.float32).copy()
    modified[...,list(indices)]=np.asarray(reference)[list(indices)]
    return modified,stop_probability(head,np.atleast_2d(modified))


def heatmap(values,path,title,layers,top,signed=False):
    import matplotlib.pyplot as plt
    a=np.asarray(values).reshape(len(layers),top);fig,ax=plt.subplots(figsize=(6.6,4.1))
    opts={}
    if signed:
        limit=max(float(np.abs(a).max()),1e-8);opts=dict(vmin=-limit,vmax=limit)
    im=ax.imshow(a,cmap='coolwarm' if signed else 'viridis',aspect='auto',**opts)
    ax.set_xticks(range(top),[f'Rank {i+1}' for i in range(top)])
    ax.set_yticks(range(len(layers)),[f'Block {i}' for i in layers]);ax.set_title(title)
    for i in range(len(layers)):
        for j in range(top):ax.text(j,i,f'{a[i,j]:+.2f}' if signed else f'{a[i,j]:.3f}',ha='center',va='center',fontsize=8)
    fig.colorbar(im,ax=ax);fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(Path(path).with_suffix('.'+ext),dpi=180)
    plt.close(fig)


class StopInterpreter:
    def __init__(self,bundle,contexts=(),generations=None):
        self.bundle=bundle;self.train=matrix(bundle['train']);self.test=matrix(bundle['test']);self.generations=generations or {}
        self.context={(r['split'],r['question_id'],int(r['step'])):r for r in contexts}
        self.reference=np.median(self.train,axis=0)
    @classmethod
    def load(cls,run_dir):
        run_dir=Path(run_dir)
        # Load only your own run artifacts: pickle contains sklearn/torch objects.
        with (run_dir/'analysis_bundle.pkl').open('rb') as f:bundle=pickle.load(f)
        context=run_dir/'checkpoint_context.json'
        generations=run_dir/'generation_context.json'
        return cls(bundle,json.loads(context.read_text()) if context.exists() else (),
            json.loads(generations.read_text()) if generations.exists() else {})
    def head(self,kind='MLP',C=1.):return self.bundle['heads'][('J++',kind,C)]
    def checkpoint(self,question_id,step,split='test'):
        frame=self.bundle[split];hit=frame[(frame.question_id==question_id)&(frame.step==int(step))]
        if len(hit)!=1:raise KeyError((question_id,step))
        raw=np.asarray(hit.iloc[0].j_vector,dtype=np.float32)
        context=dict(self.context.get((split,question_id,int(step)),{}))
        generation=self.generations.get(context.get('generation_ref'))
        if generation:
            boundary=generation['bounds'][int(step)]
            context.update(cot_prefix=generation['text'][:boundary],cot_suffix=generation['text'][boundary:])
        return raw,context
    def group_ablation(self,question_id,step,groups,kind='MLP',C=1.,cutoff=.85,split='test'):
        """groups maps names to flat score indices, e.g. [8,9] for L3 ranks 1,2."""
        x,context=self.checkpoint(question_id,step,split);head=self.head(kind,C)
        rows=[dict(group='original',probability=float(stop_probability(head,x[None])[0]))]
        for name,indices in groups.items():
            _,p=replace_features(head,x,self.reference,indices)
            rows.append(dict(group=name,probability=float(p[0])))
        table=pd.DataFrame(rows);table['stop']=table.probability>=cutoff
        return table,context
    def sweep(self,question_id,step,indices,scales=None,kind='MLP',C=1.,split='test'):
        """Joint proportional raw log-score changes; scale=0 sets selected scores to 0."""
        x,_=self.checkpoint(question_id,step,split)
        scales=np.unique(np.r_[np.linspace(2,0,201) if scales is None else scales,0.,1.])
        values=np.repeat(x[None],len(scales),axis=0);values[:,indices]=x[indices]*scales[:,None]
        result=pd.DataFrame({'scale':scales,'probability':stop_probability(self.head(kind,C),values)})
        for i in indices:result[f'feature_{i}_raw_score']=values[:,i]
        return result
    def find_stop_flips(self,cutoff=.85,kind='MLP',C=1.,max_examples=None):
        """Search each question's first threshold STOP; no reordering/reranking of scores."""
        import itertools
        head=self.head(kind,C);frame=self.bundle['test'].copy();frame['probability']=stop_probability(head,self.test)
        first=frame[frame.probability>=cutoff].sort_values('step').groupby('question_id',sort=False).head(1)
        if max_examples is not None:first=first.iloc[:max_examples]
        layers=len(self.bundle['layers']);top=self.bundle['top'];rows=[]
        layer_groups={'all':tuple(range(layers)),'last_half':tuple(range(layers//2,layers))}
        layer_groups.update({f'L{i+1}':(i,) for i in range(layers)})
        for r in first.itertuples():
            x=np.asarray(r.j_vector,dtype=np.float32)
            for size in range(1,min(3,top)+1):
                for ranks in itertools.combinations(range(top),size):
                    for label,ls in layer_groups.items():
                        indices=[l*top+j for l in ls for j in ranks];_,p=replace_features(head,x,self.reference,indices)
                        if p[0]<cutoff:rows.append(dict(question_id=r.question_id,step=int(r.step),layer_group=label,
                            ranks=','.join(str(j+1) for j in ranks),original_probability=r.probability,
                            modified_probability=float(p[0]),drop=float(r.probability-p[0])))
        columns=['question_id','step','layer_group','ranks','original_probability','modified_probability','drop']
        return pd.DataFrame(rows,columns=columns).sort_values('drop',ascending=False)


def interpret_run(run_dir,output=None,linear_C=.1,mlp_C=1.,cutoff=.85,max_ig_examples=300,ig_steps=48,find_cases=False):
    if not 0<=cutoff<=1 or max_ig_examples<1:raise ValueError('Invalid cutoff/sample count')
    obj=StopInterpreter.load(run_dir);bundle=obj.bundle
    out=Path(output) if output else Path(run_dir)/'interpretation';out.mkdir(parents=True,exist_ok=True)
    lr=obj.head('linear',linear_C);mlp=obj.head('MLP',mlp_C);x=obj.test
    coef,contributions,logits=lr_decomposition(lr,x)
    layers=bundle['layers'];top=bundle['top']
    if x.shape[1]!=len(layers)*top:raise ValueError('Feature layout mismatch')
    pd.DataFrame(coef.reshape(len(layers),top),index=layers).to_csv(out/'lr_standardized_coefficients.csv')
    table=bundle['test'][['question_id','step','correct']].copy()
    table['lr_probability']=stop_probability(lr,x);table['lr_logit']=logits;table['lr_stop_at_cutoff']=table.lr_probability>=cutoff
    for j in range(x.shape[1]):table[f'lr_contribution_L{j//top+1}_R{j%top+1}']=contributions[:,j]
    table.to_csv(out/'lr_checkpoint_attribution.csv',index=False)
    heatmap(coef,out/'lr_coefficients','LR standardized coefficients',layers,top,True)
    rng=np.random.default_rng(bundle['seed'])
    indices=np.sort(rng.choice(len(x),min(len(x),max_ig_examples),replace=False))
    baseline=obj.train.mean(0).astype(np.float32)
    ig=integrated_gradients(mlp,x[indices],baseline,steps=ig_steps)
    import torch
    with torch.no_grad():
        delta=(mlp_logits_raw(mlp,x[indices])-mlp_logits_raw(mlp,baseline[None])).numpy()
    residual=delta-ig.sum(1)
    if np.max(np.abs(residual))>.1:warnings.warn('IG residual exceeds .1; increase ig_steps')
    igframe=bundle['test'].iloc[indices][['question_id','step','correct']].copy().reset_index(drop=True)
    igframe['mlp_probability']=stop_probability(mlp,x[indices]);igframe['mlp_stop_at_cutoff']=igframe.mlp_probability>=cutoff
    igframe['ig_completeness_residual']=residual
    for j in range(x.shape[1]):igframe[f'IG_L{j//top+1}_R{j%top+1}']=ig[:,j]
    igframe.to_csv(out/'mlp_ig_checkpoint_attribution.csv',index=False)
    heatmap(np.abs(ig).mean(0),out/'mlp_ig_abs','MLP mean absolute IG',layers,top)
    heatmap(ig.mean(0),out/'mlp_ig_signed','MLP mean signed IG',layers,top,True)
    # Same RNG stream and training median reference as the uploaded analysis.
    chosen=np.sort(rng.choice(len(x),min(500,len(x)),replace=False));raw=x[chosen];impact=[]
    original_lr=stop_probability(lr,raw);original_mlp=stop_probability(mlp,raw)
    for j in range(x.shape[1]):
        _,a=replace_features(lr,raw,obj.reference,[j]);_,b=replace_features(mlp,raw,obj.reference,[j])
        impact.append(dict(layer=1+j//top,rank=1+j%top,lr_mean_abs_delta_p=float(np.abs(a-original_lr).mean()),mlp_mean_abs_delta_p=float(np.abs(b-original_mlp).mean())))
    impact=pd.DataFrame(impact);impact.to_csv(out/'feature_replacement_dependence.csv',index=False)
    for kind in ['lr','mlp']:heatmap(impact[f'{kind}_mean_abs_delta_p'],out/f'{kind}_replacement',f'{kind.upper()} median replacement',layers,top)
    if len(layers)==4 and top==4:
        from .interpretation_plots import export_original_panels
        export_original_panels(coef,np.abs(ig).mean(0),impact,out)
    if find_cases:obj.find_stop_flips(cutoff=cutoff,C=mlp_C).to_csv(out/'stop_to_continue_cases.csv',index=False)
    (out/'interpretation_protocol.json').write_text(json.dumps(dict(split='test',linear_C=linear_C,mlp_C=mlp_C,cutoff=cutoff,
        ig_steps=ig_steps,ig_examples=len(indices),ig_baseline='TRAIN raw feature mean',replacement='TRAIN feature median',
        max_abs_ig_residual=float(np.max(np.abs(residual))),predictor_interventions_only=True),indent=2))
    return out
