"""Actual reduced G0.5 source model, native encoders, BARBuilder, CE.
Synthetic code IDs only; not an ActionCodec or robot benchmark.
"""
from native_context_fixture import *
from pathlib import Path
import json,time
WIDTH=16
B=8
STEPS=120

def encoded(m,classes):
    b=len(classes)
    history=torch.tensor([[14,20+int(c),15,17] for c in classes]); hm=torch.full(history.shape,4.)
    obs=torch.tensor([[9,7,10,8,19]]*b); om=torch.tensor([[4.,1.,4.,2.,4.]]*b)
    h,hg=m.encode_context_segment(history,hm)
    o,og=m.encode_context_segment(obs,om,{'front':torch.zeros(b,1,3,8,8)},torch.zeros(b,1,2))
    rows=[bar(WIDTH).build_io_from_ids([30+int(c)*4+(j%4) for j in range(WIDTH)]) for c in classes]
    ids=torch.tensor([r[0] for r in rows]); mask=torch.tensor([r[1] for r in rows]); labels=torch.tensor([r[2] for r in rows]);labels[:,0]=-100
    a,ag=m.encode_context_segment(ids,mask)
    labels=torch.cat([torch.full_like(history,-100),torch.full_like(obs,-100),labels],1)
    return h,hm,o,om,a,mask,hg+og+ag,labels,ids

@torch.no_grad()
def evaluate(m):
    m.eval(); classes=torch.arange(4).repeat(4)
    h,hm,o,om,a,am,g,labels,ids=encoded(m,classes)
    n=1+WIDTH
    context=torch.cat([h,o,a[:,:n]],1); cm=torch.cat([hm,om,am[:,:n]],1)
    y,c,p=m.forward_context(context,cm,g)
    codes=m.vlm.decode(y[:,-WIDTH:]).argmax(-1); expected=ids[:,n:n+WIDTH]
    e=m.vlm.embed(codes); cm2=torch.cat([cm,am[:,n:n+WIDTH]],1)
    z,c2,_=m.forward_context(e,cm2,g,c); eos=m.vlm.decode(z).argmax(-1)
    classes2=(classes+1)%4; h2,hm2,*_=encoded(m,classes2)
    cf,_,_=m.forward_context(torch.cat([h2,o,a[:,:n]],1),cm,g)
    cf_codes=m.vlm.decode(cf[:,-WIDTH:]).argmax(-1)
    expected2=torch.tensor([[30+int(c)*4+(j%4) for j in range(WIDTH)] for c in classes2])
    no,_,_=m.forward_context(torch.cat([o,a[:,:n]],1),torch.cat([om,am[:,:n]],1),g)
    no_codes=m.vlm.decode(no[:,-WIDTH:]).argmax(-1)
    return dict(action_exact=float((codes==expected).all(-1).float().mean()),eos_exact=float((eos==101).all(-1).float().mean()),paired_counterfactual=float(((codes==expected).all(-1)&(cf_codes==expected2).all(-1)).float().mean()),without_history_exact=float((no_codes==expected).all(-1).float().mean()))

if __name__=='__main__':
    results=[]; start=time.monotonic()
    for seed in [11,22,33]:
        m=tiny(seed);opt=torch.optim.AdamW(m.parameters(),lr=.003);before=evaluate(m);m.train();gen=torch.Generator().manual_seed(seed+100);losses=[]
        for i in range(STEPS):
            c=torch.randint(0,4,(B,),generator=gen)
            h,hm,o,om,a,am,g,labels,_=encoded(m,c)
            hid,_,_=m.forward_context(torch.cat([h,o,a],1),torch.cat([hm,om,am],1),g)
            loss,_=m.ar_helper.cal_ce_loss(hid,labels,m)
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.0);opt.step();losses.append(float(loss.detach()))
        after=evaluate(m);row=dict(seed=seed,steps=STEPS,initial=before,final=after,first_loss=losses[0],last_loss=losses[-1],vlm_vision_proprio_params=sum(p.numel() for k,p in m.named_parameters() if not k.startswith('action_expert.')));results.append(row);print(json.dumps(row),flush=True)
    output=dict(results=results,wall_seconds=time.monotonic()-start,notes='Random-init reduced source model; synthetic code IDs, not real codec or robot actions.')
    Path('tests/results').mkdir(exist_ok=True);Path('tests/results/source_train_smoke.json').write_text(json.dumps(output,indent=2))
