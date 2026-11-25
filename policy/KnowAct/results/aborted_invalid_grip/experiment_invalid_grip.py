"""Small actual G05Policy ICL experiment with held-out 3-way permutations.
Train/query outputs from the real native FSQ+rule codec, no memory/head changes.
Historical command records use the same 4/8 prefix padding as deployment.
"""
import argparse, itertools, json, random, time
from pathlib import Path
import torch
from native_policy_fixture import make_policy,turn
from g05.models.g05 import native_context as n
from g05.data.native_episodes import hold_pad

SYMBOLS=['left','right','same']
PERMS=list(itertools.permutations(range(3)))
TRAIN=[p for i,p in enumerate(PERMS) if i not in (1,4)]
TEST=[PERMS[i] for i in (1,4)]

def make_turn(symbol, label, gen, history=False):
    x=turn(SYMBOLS[symbol]);s=x['samples'][0];a=s['action']['value'].clone()
    # Three distinct rule sequences, with distinguishable executed four-step prefixes.
    grip=[[-1]*8,[1]*8,[-1,1,-1,1,-1,1,-1,1]][label]
    a[:,-1]=torch.tensor(grip,dtype=a.dtype)
    if history:a=hold_pad(a,4);x['supervise']=False
    s['action']['value']=a;x['action']=a[None]
    # nuisance changes independent of query and mapping (not extra target hints)
    x['pixel_values']['image'] += .005*torch.randn(x['pixel_values']['image'].shape,generator=gen)
    s['proprio']['value'] += .005*torch.randn(s['proprio']['value'].shape,generator=gen)
    return x

def example(mapping, query, order, gen, history=True):
    old=[make_turn(i,mapping[i],gen,True) for i in order] if history else []
    for i,t in enumerate(old):
        if i:t['executed_steps']=4
    now=make_turn(query,mapping[query],gen);now['executed_steps']=4 if old else None
    return old+[now]

@torch.no_grad()
def evaluate(p, mappings, history, seed, n_orders=2):
    p.eval();gen=torch.Generator().manual_seed(seed);rng=random.Random(seed);rows=[]
    for mapping in mappings:
        for query in range(3):
            for _ in range(n_orders):
                order=list(range(3));rng.shuffle(order);ep=example(mapping,query,order,gen,history)
                # Real observations + known demonstrations. Use exact completed
                # history, not the target query nor ground truth at future times.
                st=None
                for t in ep[:-1]:
                    samp=n.receipt(t['samples'][0],t.get('executed_steps'),executed_only=True)
                    ids,_,mask,emb,grid,pix=n.encode(p,samp,t['pixel_values']);st,_=n.append(p,st,ids,mask,emb,grid,pix)
                now=ep[-1];sample=n.receipt(now['samples'][0],now.get('executed_steps'),executed_only=True)
                ids,_,mask,emb,grid,pix=n.encode(p,sample,now['pixel_values'],prefix=True);st,_=n.append(p,st,ids,mask,emb,grid,pix)
                pred=p.generate_action(st,now['samples'],action_source='ar')['decoded_action_tokens'][0].tolist()
                gt=p.action_tokenizer(now['action'],encode_kwargs={'action_op_mask':torch.ones(1,5,dtype=torch.bool),'action_dim_is_pad':torch.zeros(1,5,dtype=torch.bool)})[0]
                ar=p.model.ar_helper;idx=[i for i in range(len(gt)) if 'fixed_token' not in ar._bar_position_constraints[i]]
                c=ar._bar_joint_token_constraints[0];off=c['offset'];end=off+c['num_tokens']
                rows.append(dict(mapping=mapping,query=query,order=order,rule_correct=pred[off:end]==gt[off:end],exact=pred==gt,variable_acc=sum(pred[i]==gt[i] for i in idx)/len(idx)))
    return {k:sum(r[k] for r in rows)/len(rows) for k in ('rule_correct','exact','variable_acc')}|{'n':len(rows),'rows':rows}

def run(seed,mode,steps):
    p=make_policy(seed);p.config.native_replan_steps=4;p.config.native_mask_fixed_action_targets=(mode=='mask');p.train()
    rng=random.Random(seed+700);gen=torch.Generator().manual_seed(seed+300);opt=torch.optim.AdamW(p.model.parameters(),lr=.002,weight_decay=0.);losses=[];start=time.monotonic();history=mode!='current'
    for step in range(steps):
        episodes=[]
        for j in range(2):
            order=list(range(3));rng.shuffle(order);episodes.append(example(rng.choice(TRAIN),rng.randrange(3),order,gen,history))
        opt.zero_grad(set_to_none=True);loss,_=p({'native_episodes':episodes});loss.backward();torch.nn.utils.clip_grad_norm_(p.model.parameters(),1.);opt.step();losses.append(float(loss.detach()))
        if (step+1)%50==0: print(seed,mode,step+1, sum(losses[-50:])/50,flush=True)
    out=dict(seed=seed,mode=mode,steps=steps,train=evaluate(p,TRAIN,history,12345,1),heldout=evaluate(p,TEST,history,12345,3),reset=evaluate(p,TEST,False,12345,3),losses=losses,seconds=time.monotonic()-start)
    print('RESULT',seed,mode,{k:{a:b for a,b in out[k].items() if a!='rows'} for k in ('train','heldout','reset')},flush=True);return out

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);ap.add_argument('--mode',choices=['history','mask','current'],required=True);ap.add_argument('--steps',type=int,default=240);ap.add_argument('--output',required=True);a=ap.parse_args();Path(a.output).write_text(json.dumps(run(a.seed,a.mode,a.steps),indent=2))
