"""Run unchanged source or patched source with G05_TEST_SOURCE=..."""
from native_context_fixture import *
import json
with torch.no_grad():
    m=tiny(); m.eval(); layer=m.vlm.layers[0].linear_attn
    x=torch.randn(1,27,32)
    def run(parts):
        cache=SparseKVCache(last_linear_layer=0); out=[]; start=0
        for n in parts:
            out.append(layer(x[:,start:start+n],cache_params=cache,cache_position=start));start+=n
        return torch.cat(out,1),cache
    full,c=run([27]); stream,cs=run([7,16,4])
    helper=m.mask_helper
    a=torch.full((1,7),4.); helper.build_vlm_mask(torch.zeros(1,3,dtype=torch.long),a[:,:3])
    mask,_=helper.build_vlm_mask(torch.zeros(1,4,dtype=torch.long),a,kv_len=3)
    proc=BuiltinActionProcessor(bar(),False)
    ids,lab,attn=proc.process(list(range(30,48)),True,True,None,{})
    print(json.dumps(dict(gdn_hidden_error=float((full-stream).abs().max()),gdn_conv_error=float((c.conv_states[0]-cs.conv_states[0]).abs().max()),gdn_state_error=float((c.recurrent_states[0]-cs.recurrent_states[0]).abs().max()),incremental_future_edges=int((mask[0,0,:,3:]==0).triu(1).sum()),historical_supervised_labels=sum(y!=-100 for y in lab),length=len(ids)),indent=2))
