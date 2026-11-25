"""Native code dependence: exact rule-based production settings; random FSQ toy."""
import sys,json,itertools
from pathlib import Path
import numpy as np,torch
from native_policy_fixture import make_policy
from g05.tokenizer.models.binary_sequence import ConstrainedSequenceTokenizer
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_action_codes import paired_audit

@torch.no_grad()
def run(seed=11):
    p=make_policy(seed);g=torch.Generator().manual_seed(seed+17);t=torch.linspace(0,1,8)[None,:,None];N=64
    base=torch.rand(N,1,4,generator=g)*.6-.3
    amp=torch.rand(N,1,4,generator=g)*.5+.3
    motion=torch.sin(t*torch.pi/2)*amp
    a=torch.cat([base+motion,-torch.ones(N,8,1)],-1)
    b=torch.cat([base-motion,torch.ones(N,8,1)],-1)
    both=torch.stack([a,b],1).reshape(N*2,8,5)
    codes=np.asarray(p.action_tokenizer(both,encode_kwargs={'action_op_mask':torch.ones(N*2,5,dtype=torch.bool),'action_dim_is_pad':torch.zeros(N*2,5,dtype=torch.bool)}),dtype=np.int64).reshape(N,2,-1)
    r=p.model.ar_helper._bar_joint_token_constraints[0]
    # Exclude rule differences for the neural-code-only diagnostic. Keep serialized positions.
    motion_codes=codes.copy();off=r['offset'];motion_codes[:,1,off:off+r['num_tokens']]=motion_codes[:,0,off:off+r['num_tokens']]
    result={'seed':seed,'full':[paired_audit(codes,w) for w in (1,8,16,32)],'motion_only':[paired_audit(motion_codes,w) for w in (1,8,16,32)]}
    # Change a late neural code in a batch and measure influence on decoded first four steps.
    ar=p.model.ar_helper;index=max(i for i,c in ar._bar_position_constraints.items() if i<codes.shape[-1] and 'token_lower' in c and 'gripper' not in c['name']);c=ar._bar_position_constraints[index]
    original=torch.as_tensor(codes[:,0]);changed=original.clone();changed[:,index]=c['token_lower']+(changed[:,index]-c['token_lower']+1)%(c['token_upper']-c['token_lower'])
    def dec(x):
        return torch.stack(p.processor.decode_ar(list(x),8,5,generated_ids_are_action_only=True)[0])
    da,db=dec(original),dec(changed);diff=(da[:,:4]-db[:,:4]).abs().flatten(1).amax(1)
    result['prefix_probe']={'changed_code_offset':index,'cases':N,'changed_prefix_count':int((diff>1e-8).sum()),'max_abs_diff':float(diff.max()),'mean_abs_max':float(diff.mean()),'values':diff.tolist()}
    return result,codes

if __name__=='__main__':
    out=Path(sys.argv[1]);r=ConstrainedSequenceTokenizer(32,1,4096);a=r.encode([0]*32,strict=True);b=r.encode([1]*32,strict=True);pairs=list(itertools.product([a[0],b[0]],[a[1],b[1]]));valid=[]
    for pair in pairs:
        try:r.decode(list(pair),safe=False);valid.append(True)
        except ValueError:valid.append(False)
    result={'rule_production_config':{'horizon':32,'base':4096,'num_tokens':r.num_tokens,'valid_sequences':r.num_valid_sequences,'cartesian_size':4096**r.num_tokens,'uniform_valid_fraction':r.num_valid_sequences/(4096**r.num_tokens),'modes':[a,b],'independent_pairs':pairs,'valid_pairs':valid,'valid_mass':sum(valid)/len(valid),'exact_mode_mass':sum(list(pair) in [a,b] for pair in pairs)/len(pairs)},'small_random_fsq':[]}
    for seed in (11,22,33):
        res,codes=run(seed);result['small_random_fsq'].append(res);np.save(out.parent/f'codec_pairs_{seed}.npy',codes)
    out.write_text(json.dumps(result,indent=2));print('production rule',result['rule_production_config']);
    for r in result['small_random_fsq']:
        print(r['seed'],[(x['width'],x['distinct_contexts'],x['mean_independent_mode_mass']) for x in r['motion_only']],{k:v for k,v in r['prefix_probe'].items() if k!='values'})
