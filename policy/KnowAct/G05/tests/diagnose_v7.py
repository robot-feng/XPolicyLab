import torch, time,json, argparse
from native_policy_fixture import make_policy
from experiment_native_policy import example
from g05.models.g05 import native_context as n
from itertools import groupby

def evaluate(p,E):
 p.eval(); rows=[]
 with torch.no_grad():
  for ep in E:
   enc=[n.encode(p,n.receipt(t['samples'][0],t.get('executed_steps')),t['pixel_values']) for t in ep]; ar=p.model.ar_helper
   ids,lab,mask,emb=[torch.cat([e[j] for e in enc],dim=1) for j in range(4)]
   h,_,_=p.model.forward_context(emb,mask,sum([e[4] for e in enc],[])); z=p.model.vlm.decode(h[:,:-1]); l=lab[:,1:]; valid=l!=-100
   old=ep[0]; s=p.append_context(None,old['samples'],old['pixel_values']); i=s.input_ids.size(1)
   for k,g in groupby(zip(enc[0][0][0,i:].tolist(),enc[0][2][0,i:].tolist()),key=lambda z:z[1]):s,_=n.append_ids(p,s,[x for x,_ in g],[k])
   now=ep[-1]; s=p.append_context(s,now['samples'],now['pixel_values'],executed_steps=8)
   groups=n.action_layout(p,now['samples'][0]); codes=[];tf=[];off=s.input_ids.size(1); st=s; logits=None; maxerr=0.; slot_loss=[]; learnable=[]
   for k,g in groups:
    a=[i for i,_ in g]; bar=int(k)==3 and k!=3
    if bar and logits is not None:
     true=ids[0,off:off+len(a)].tolist(); target=lab[0,off-len(a)+1:off+1].tolist(); assert true==target,(true,target)
     for j,c in enumerate([ar._bar_position_constraints[len(codes)+j] for j in range(len(a))]):
      tf.append(int(logits[0,j,c['token_lower']:c['token_upper']].argmax())+c['token_lower'] if 'token_lower' in c else c['fixed_token']); slot_loss.append(float(-logits[0,j].log_softmax(-1)[true[j]]));learnable.append('fixed_token' not in c)
     codes.extend(true); a=true
    st,hh=n.append_ids(p,st,a,[k]*len(a)); maxerr=max(maxerr,float((hh-h[:,off:off+len(a)]).abs().max()));off+=len(a)
    if bar:logits=p.model.vlm.decode(hh)
   pred=p.generate_action(s,now['samples'],action_source='ar')['decoded_action_tokens'][0].tolist(); fixed=[i for i,v in enumerate(learnable) if not v]; variable=[i for i,v in enumerate(learnable) if v]
   rule=ar._bar_joint_token_constraints[0]; ri=list(range(rule['offset'],rule['offset']+rule['num_tokens']))
   rows.append(dict(ce=float(torch.nn.functional.cross_entropy(z[valid],l[valid])),tf_acc=sum(tf[i]==codes[i] for i in variable)/len(variable),free_acc=sum(pred[i]==codes[i] for i in variable)/len(variable),exact=pred==codes[:len(pred)],rule_correct=all(pred[i]==codes[i] for i in ri),fixed_ce=sum(slot_loss[i] for i in fixed)/len(fixed),variable_ce=sum(slot_loss[i] for i in variable)/len(variable),maxerr=maxerr,gt=codes,tf=tf,pred=pred))
 p.train();return rows

def run(seed,steps):
 p=make_policy(seed); p.train(); opt=torch.optim.AdamW(p.model.parameters(),lr=.002,weight_decay=0.); E=[example(i,True) for i in (0,1)]; out=[]
 for step in range(steps+1):
  if step in (0,100,steps):
   rows=evaluate(p,E); out.append(dict(step=step,rows=rows)); print(seed,step,[{k:v for k,v in r.items() if k not in ('gt','tf','pred')} for r in rows],flush=True)
  if step==steps:break
  opt.zero_grad(); loss,_=p({'native_episodes':E});loss.backward();torch.nn.utils.clip_grad_norm_(p.model.parameters(),1.);opt.step()
 return out
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,default=11);ap.add_argument('--steps',type=int,default=300);ap.add_argument('--output',required=True);a=ap.parse_args();json.dump(run(a.seed,a.steps),open(a.output,'w'),indent=2)
