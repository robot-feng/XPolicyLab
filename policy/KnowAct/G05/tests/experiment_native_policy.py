"""Small real G05Policy/ActionCodec training comparison; not a robot benchmark.
Balanced identical current observations. Only prior instruction predicts grip.
Match supervised target count, optimizer updates; history uses more input compute.
"""
import argparse,json,time
from pathlib import Path
import torch
from native_policy_fixture import make_policy,turn


def example(label, history):
    now=turn('same'); a=now['samples'][0]['action']['value'].clone(); a[:,-1]=1. if label else -1.; now['samples'][0]['action']['value']=a
    now['action']=a[None]; now['executed_steps']=8
    if not history:return [now]
    old=turn('right' if label else 'left'); old['supervise']=False
    return [old,now]

def run(seed,history,steps):
    torch.manual_seed(seed); p=make_policy(seed); p.train(); opt=torch.optim.AdamW(p.model.parameters(),lr=.002,weight_decay=0.)
    losses=[]; start=time.monotonic()
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        loss,_=p({'native_episodes':[example(0,history),example(1,history)]}); loss.backward(); torch.nn.utils.clip_grad_norm_(p.model.parameters(),1.); opt.step()
        losses.append(float(loss.detach()))
    p.eval(); outputs=[]
    with torch.no_grad():
        for label in (0,1):
            ep=example(label,history); state=None
            if history:
                old=ep[0]; state=p.append_context(None,old['samples'],old['pixel_values'])
                # Known demonstration (not generated history). Append the exact original
                # encoded action suffix, as supplied in a demonstration prompt.
                from g05.models.g05 import native_context as native
                ids,_,mask,_,_,_=native.encode(p,old['samples'][0],old['pixel_values']); start_idx=state.input_ids.size(1)
                from itertools import groupby
                for k,g in groupby(zip(ids[0,start_idx:].tolist(),mask[0,start_idx:].tolist()),key=lambda z:z[1]):
                    state,_=native.append_ids(p,state,[x for x,_ in g],[k])
            now=ep[-1]; state=p.append_context(state,now['samples'],now['pixel_values'],executed_steps=8)
            r=p.generate_action(state,now['samples'],action_source='ar')
            gt=p.action_tokenizer(now['action'],encode_kwargs={'action_op_mask':torch.ones(1,5,dtype=torch.bool),'action_dim_is_pad':torch.zeros(1,5,dtype=torch.bool)})[0]
            pred=r['decoded_action_tokens'][0].tolist(); rule=p.model.ar_helper._bar_joint_token_constraints[0]; off=rule['offset']; n=rule['num_tokens']
            outputs.append({'label':label,'rule_correct':pred[off:off+n]==gt[off:off+n],'exact_codes':pred==gt,'code_accuracy':sum(a==b for a,b in zip(pred,gt))/len(gt),'gripper_values':r['action'][0,:,-1].tolist(),'target_rule':gt[off:off+n],'generated_rule':pred[off:off+n]})
    result={'seed':seed,'history':history,'steps':steps,'first_loss':losses[0],'last_loss':losses[-1],'rule_accuracy':sum(o['rule_correct'] for o in outputs)/2,'exact_accuracy':sum(o['exact_codes'] for o in outputs)/2,'code_accuracy':sum(o['code_accuracy'] for o in outputs)/2,'seconds':time.monotonic()-start,'outputs':outputs,'losses':losses}
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--steps',type=int,default=100);ap.add_argument('--output',default='native_policy_experiment.json');args=ap.parse_args(); results=[]
    for seed in (11,22,33):
        for history in (False,True):
            r=run(seed,history,args.steps);results.append(r);Path(args.output).write_text(json.dumps(results,indent=2));print({k:v for k,v in r.items() if k not in ('outputs','losses')},flush=True)
