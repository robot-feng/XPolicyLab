"""Source-level CPU tests using the actual G0.5 classes, reduced dimensions."""
import unittest, copy, itertools
from native_context_fixture import *


def join(segs):
    return torch.cat([s[0] for s in segs],1),torch.cat([s[1] for s in segs],1),sum([s[2] for s in segs],[])

def error(a,b): return float((a-b).abs().max().detach())

def run_chunks(m,segs):
    cache=None; out=[]; masks=[]; grids=[]
    for e,a,g in segs:
        masks.append(a); grids+=g
        h,cache,p=m.forward_context(e,torch.cat(masks,1),grids,cache)
        out.append(h)
    return torch.cat(out,1),cache,p

class TestSource(unittest.TestCase):
    def setUp(self): self.m=tiny().eval()

    def test_qwen_still_image_patch_layout(self):
        # Distinct values in every channel and spatial position expose axis swaps.
        image = torch.arange(3 * 8 * 8, dtype=torch.float32).view(1, 1, 3, 8, 8)
        patches = []
        hook = self.m.vision_tower.register_forward_pre_hook(lambda _, args: patches.append(args[0]))
        try:
            self.m._forward_vision({"front": image})
            self.m.cfg.vision.hf_image_patch_layout = True
            self.m._forward_vision({"front": image})
        finally:
            hook.remove()
        expected = torch.stack([image[0, 0, :, y:y+4, x:x+4]
                                for y in (0, 4) for x in (0, 4)])
        legacy, corrected = [p.view(4, 6, 4, 4) for p in patches]
        torch.testing.assert_close(legacy, expected[:, [0, 1, 0, 2, 1, 2]])
        torch.testing.assert_close(corrected, expected[:, [0, 0, 1, 1, 2, 2]])

    def test_gdn_multichunk(self):
        for b,k in itertools.product([1,2],[1,2,4,7]):
            c=config().vlm; c.linear_conv_kernel_dim=k
            layer=Qwen3_5GatedDeltaNet(c,0).eval()
            x=torch.randn(b,33,32)
            with torch.no_grad():
                def run(ns):
                    cache=SparseKVCache(last_linear_layer=0); out=[]; i=0
                    for n in ns:
                        out.append(layer(x[:,i:i+n],cache_params=cache,cache_position=i));i+=n
                    return torch.cat(out,1),cache
                ref,c0=run([33])
                for parts in ([1]*33,[3,16,14],[1,31,1],[17,16]):
                    got,c1=run(parts)
                    torch.testing.assert_close(ref,got,atol=3e-6,rtol=3e-5)
                    torch.testing.assert_close(c0.recurrent_states[0],c1.recurrent_states[0],atol=3e-6,rtol=3e-5)
                    torch.testing.assert_close(c0.conv_states[0],c1.conv_states[0],atol=3e-6,rtol=3e-5)
    def test_legacy_causal_mask(self):
        helper=self.m.mask_helper
        a=torch.full((1,7),4.)
        helper.build_vlm_mask(torch.zeros(1,3,dtype=torch.long),a[:,:3])
        mask,_=helper.build_vlm_mask(torch.zeros(1,4,dtype=torch.long),a,kv_len=3)
        self.assertEqual(int((mask[0,0,:,3:]==0).triu(1).sum()),0)
        mask,_=helper.build_vlm_mask(torch.zeros(1,4,dtype=torch.long),a,kv_len=3,is_action_block=True)
        self.assertTrue(bool((mask==0).all()))
    def test_legacy_prefill_layout(self):
        segs=sample_segments(self.m); _,a,g=join(segs)
        helper=self.m.mask_helper; helper._image_grid_thw=g
        mask,p=helper.build_vlm_mask(torch.zeros_like(a,dtype=torch.long),a)
        m2,p2=helper.build_context_layout(a,g)
        self.assertTrue(torch.equal(mask,m2)); self.assertTrue(torch.equal(p,p2))
    def test_full_stream_vision_state_bar(self):
        for seed in [11,22,33]:
            m=tiny(seed).eval()
            with torch.no_grad():
                segs=sample_segments(m); E,A,G=join(segs)
                full,c,p=m.forward_context(E,A,G)
                stream,cs,ps=run_chunks(m,segs)
                self.assertTrue(torch.equal(p,ps)); self.assertEqual(c.num_items(),cs.num_items())
                torch.testing.assert_close(full,stream,atol=3e-5,rtol=3e-5)
                torch.testing.assert_close(m.vlm.decode(full),m.vlm.decode(stream),atol=2e-4,rtol=3e-5)
                for name in ['key_cache','value_cache','conv_states','recurrent_states']:
                    for k,v in getattr(c,name).items(): torch.testing.assert_close(v,getattr(cs,name)[k],atol=3e-5,rtol=3e-5)
    def test_future_isolation(self):
        with torch.no_grad():
            segs=sample_segments(self.m); E,A,G=join(segs); h,_,_=self.m.forward_context(E,A,G)
            E2=E.clone(); E2[:,-6:]+=torch.randn_like(E2[:,-6:])*3
            h2,_,_=self.m.forward_context(E2,A,G)
            torch.testing.assert_close(h[:,:-6],h2[:,:-6],atol=1e-6,rtol=0)
    def test_branch_isolation_and_replay(self):
        with torch.no_grad():
            segs=sample_segments(self.m); e,a,g=join(segs[:3]); _,c,_=self.m.forward_context(e,a,g)
            saved={name:{k:v.clone() for k,v in getattr(c,name).items()} for name in ['key_cache','value_cache','conv_states','recurrent_states']}
            e2,a2,g2=segs[3]; mask=torch.cat([a,a2],1)
            h1,c1,_=self.m.forward_context(e2,mask,g+g2,c)
            h2,c2,_=self.m.forward_context(e2+2,mask,g+g2,c)
            self.assertGreater(error(h1,h2),1e-3); self.assertEqual(c.num_items(),a.size(1))
            for name,values in saved.items():
                for k,v in values.items(): self.assertTrue(torch.equal(v,getattr(c,name)[k]))
            hf,_,_=self.m.forward_context(torch.cat([e,e2],1),mask,g+g2)
            torch.testing.assert_close(h1,hf[:,-e2.size(1):],atol=3e-5,rtol=3e-5)
    def test_read_only_gdn(self):
        layer=self.m.vlm.layers[0].linear_attn; cache=SparseKVCache(last_linear_layer=0)
        with torch.no_grad():
            layer(torch.randn(1,8,32),cache_params=cache,cache_position=0)
            conv=cache.conv_states[0].clone(); rs=cache.recurrent_states[0].clone()
            for q in [1,4,16]: layer(torch.randn(1,q,32),cache_params=cache,cache_position=8,return_kv_cache=False)
            self.assertTrue(torch.equal(conv,cache.conv_states[0])); self.assertTrue(torch.equal(rs,cache.recurrent_states[0]))
    def test_detach_conv(self):
        cache=SparseKVCache(); cache.conv_states[0]=torch.randn(1,3,4,requires_grad=True)
        self.assertFalse(cache.detach().conv_states[0].requires_grad)
    def test_history_labels(self):
        for w in [1,8,16,32]:
            proc=BuiltinActionProcessor(bar(w),False)
            x,l,a=proc.process(list(range(30,48)),True,True,None,{})
            x2,l2,a2=proc.process(list(range(30,48)),False,True,None,{})
            self.assertEqual(x,x2);self.assertEqual(a,a2); self.assertTrue(all(v==-100 for v in l)); self.assertTrue(any(v!=-100 for v in l2))
    def test_bar_labels(self):
        for w,n in itertools.product([1,8,16,32],[0,1,7,8,15,16,17,31,32,33]):
            ids,mask,lab=bar(w).build_io_from_ids(list(range(30,30+n)))
            self.assertEqual(len(ids),len(lab));self.assertEqual(len(ids),len(mask))
            # Under CE shift the complete BOS block predicts the first code block.
            if n:
                start=ids.index(100)
                self.assertEqual(ids[start:start+w],[100]*w)
                self.assertEqual(lab[start+1:start+w+1],list(range(30,30+min(w,n)))+[102]*max(0,w-n))
    def test_bar_overflow(self):
        with self.assertRaises(ValueError):bar(16,1).build_io_from_ids(list(range(33)))
        with self.assertRaises(ValueError):bar(1).build_io_from_ids([42]*99)
    def test_wrong_grid(self):
        with self.assertRaises(ValueError): self.m.mask_helper.build_context_layout(torch.tensor([[4.,1.,1.,4.]]),[(1,4,4)])
    def test_wrong_boundary(self):
        helper=self.m.mask_helper
        for a,g in [([4.,3.01,3.01,4.],[]),([4.,1.,1.,4.],[(1,2,4)])]:
            with self.assertRaises(ValueError):helper.build_context_layout(torch.tensor([a]),g,kv_len=2)
    def test_changed_prefix_rejected(self):
        with torch.no_grad():
            e,a,g=join(sample_segments(self.m)[:3]); _,c,_=self.m.forward_context(e,a,g)
            mask=torch.cat([a,torch.tensor([[4.]])],1);mask[0,0]=5
            with self.assertRaises(ValueError):self.m.forward_context(self.m.vlm.embed(torch.tensor([[22]])),mask,g,c)
    def test_padding_rejected(self):
        with self.assertRaises(ValueError):self.m.mask_helper.build_context_layout(torch.tensor([[0.,4.]]),[])
    def test_missing_state_rejected(self):
        with self.assertRaises(ValueError):self.m.encode_context_segment(torch.tensor([[8]]),torch.tensor([[2.]]))
    def test_temporal_fusion_rejected(self):
        with self.assertRaises(ValueError):self.m.encode_context_segment(torch.tensor([[7]]),torch.tensor([[1.]]),{'a':torch.randn(1,2,3,8,8)})
    def test_gradients_actual_ce(self):
        m=self.m.train(); segs=sample_segments(m); E,A,G=join(segs)
        # Reuse the same fixed encoded inputs to isolate hybrid decoder gradients.
        segs=[(e.detach(),a,g) for e,a,g in segs]; E,A,G=join(segs)
        labels=torch.full(A.shape,-100,dtype=torch.long);labels[:,1::3]=32
        h,_,_=m.forward_context(E,A,G); loss,_=m.ar_helper.cal_ce_loss(h,labels,m);loss.backward()
        grads={n:p.grad.clone() for n,p in m.vlm.named_parameters() if p.grad is not None};m.zero_grad(set_to_none=True)
        h2,_,_=run_chunks(m,segs);loss2,_=m.ar_helper.cal_ce_loss(h2,labels,m);loss2.backward()
        torch.testing.assert_close(loss,loss2,atol=2e-5,rtol=2e-5)
        for n,p in m.vlm.named_parameters():
            if n in grads:torch.testing.assert_close(p.grad,grads[n],atol=2e-4,rtol=2e-4,msg=n)
    def test_encoder_gradients(self):
        m=self.m.train(); E,A,G=join(sample_segments(m));h,_,_=m.forward_context(E,A,G)
        labels=torch.full(A.shape,-100,dtype=torch.long);labels[:,-1]=25
        loss,_=m.ar_helper.cal_ce_loss(h,labels,m);loss.backward()
        self.assertGreater(float(m.vision_tower.patch_embed.proj.weight.grad.abs().sum()),0)
        self.assertGreater(float(m.proprio_embedder.mlp[3].weight.grad.abs().sum()),0)
    def test_bar_future_target_isolation(self):
        for w in [1,8,16,32]:
            ids,mask,labels=bar(w).build_io_from_ids([32]*w+[33]*w)
            x=torch.tensor([ids]);a=torch.tensor([mask]);e=self.m.vlm.embed(x)
            with torch.no_grad():
                h,_,_=self.m.forward_context(e,a)
                start=ids.index(100);end=start+w
                x2=x.clone();x2[:,end:end+w]=45
                h2,_,_=self.m.forward_context(self.m.vlm.embed(x2),a)
                torch.testing.assert_close(h[:,:end],h2[:,:end],atol=1e-6,rtol=0)
    def test_multicamera_batch(self):
        ids=torch.tensor([[9,7,7,7,7,10,9,7,7,10,8]]*2)
        a=torch.tensor([[4.,1.,1.,1.,1.,4.,4.,1.,1.,4.,2.]]*2)
        px={'front':torch.randn(2,1,3,16,16),'wrist':torch.randn(2,1,3,8,16)}
        with torch.no_grad():
            e,g=self.m.encode_context_segment(ids,a,px,torch.randn(2,1,2))
            self.assertEqual(g,[[1,4,4],[1,2,4]])
            h,c,p=self.m.forward_context(e,a,g)
            self.assertEqual(tuple(h.shape),(2,11,32));self.assertEqual(tuple(p.shape),(3,2,11))
    def test_reject_legacy_cache(self):
        c=SparseKVCache()
        with self.assertRaises(ValueError):self.m.forward_context(self.m.vlm.embed(torch.tensor([[21]])),torch.tensor([[4.]]),(),c)
    def test_missing_image_rejected(self):
        with self.assertRaises(ValueError):self.m.encode_context_segment(torch.tensor([[7]]),torch.tensor([[1.]]))
    def test_wrong_state_count(self):
        with self.assertRaises(ValueError):self.m.encode_context_segment(torch.tensor([[8]]),torch.tensor([[2.]]),proprio=torch.randn(1,2,2))
    def test_dtype_consistency(self):
        with torch.no_grad(),torch.autocast('cpu',dtype=torch.bfloat16):
            segs=sample_segments(self.m)
            self.assertEqual({e.dtype for e,_,_ in segs},{self.m.vlm.input_proj.weight.dtype})
    def test_legacy_single_turn_prefill(self):
        ids=torch.tensor([[14,9,7,7,7,7,10,8,20]])
        a=torch.tensor([[4.,4.,1.,1.,1.,1.,4.,2.,4.]])
        px={'front':torch.randn(1,1,3,16,16)};state=torch.randn(1,1,2)
        with torch.no_grad():
            old,c,p=self.m.vlm_prefill(ids,a.clone(),px,proprio=state)
            e,g=self.m.encode_context_segment(ids,a,px,state)
            new,cn,pn=self.m.forward_context(e,a,g)
            torch.testing.assert_close(old,new,atol=1e-6,rtol=1e-6); self.assertTrue(torch.equal(p,pn))
    def test_native_api_multisession(self):
        with torch.no_grad():
            m=self.m; s1=sample_segments(m);s2=sample_segments(m); c1=c2=None; a1=[];a2=[];g1=[];g2=[];h1=[];h2=[]
            for x,y in zip(s1,s2):
                for k,(e,a,g) in enumerate((x,y)):
                    if k==0: a1.append(a);g1+=g;h,c1,_=m.forward_context(e,torch.cat(a1,1),g1,c1);h1.append(h)
                    else: a2.append(a);g2+=g;h,c2,_=m.forward_context(e,torch.cat(a2,1),g2,c2);h2.append(h)
            for ss,hs in [(s1,h1),(s2,h2)]:
                h,_,_=m.forward_context(*join(ss));torch.testing.assert_close(h,torch.cat(hs,1),atol=3e-5,rtol=3e-5)

if __name__=='__main__': unittest.main(verbosity=2)
