"""Real source policy/codec regression, not pretrained or RoboDojo benchmark."""
import unittest
from unittest.mock import patch
import torch
from native_policy_fixture import make_policy, turn
from g05.models.g05 import native_context as native
from g05.data.native_episodes import history_indices, completed_history, collate_episodes
from g05.utils.data.data_utils import collate_fn_pad_sequences

class NativePolicyTest(unittest.TestCase):
    def setUp(self):
        self.p=make_policy(); self.p.eval(); self.b=turn()
    def prefill(self): return self.p.prefill(self.b['samples'],self.b['pixel_values'])
    def generate(self,s): return self.p.generate_action(s,self.b['samples'],action_source='ar')
    def test_real_codec_policy(self):
        s=self.prefill(); r=self.generate(s)
        self.assertEqual(tuple(r['action'].shape),(1,8,5)); self.assertTrue(torch.isfinite(r['action']).all())
        self.assertEqual(r['decoded_action_tokens'][0].numel(),39); self.assertFalse(r['ar_absent_keys'][0])
        s.check_invariants(); self.assertEqual(s.input_ids.size(1),80)
    def test_exact_replay(self):
        s=self.prefill(); r=self.generate(s)
        sample={**self.b['samples'][0],'action':r['decoded_action_tokens'][0].tolist()}
        self.p.processor.batchify_action=False
        ids,lab,mask,emb,grids,_=native.encode(self.p,sample,self.b['pixel_values'])
        self.assertTrue(torch.equal(ids,s.input_ids)); self.assertTrue(torch.equal(mask,s.attention_mask))
        h,cache,pos=self.p.model.forward_context(emb,mask,grids)
        self.assertTrue(torch.equal(pos,s.position_ids)); torch.testing.assert_close(h[:,-1],s.last_hidden,atol=2e-5,rtol=2e-5)
    def test_two_turn_replay(self):
        s=self.prefill(); r=self.generate(s); a={**self.b['samples'][0],'action':r['decoded_action_tokens'][0].tolist()}
        b=turn('right'); old_ids=s.input_ids.clone(); old_hidden=s.last_hidden.clone()
        t=self.p.append_context(s,b['samples'],b['pixel_values'],executed_steps=3); r2=self.p.generate_action(t,b['samples'],action_source='ar')
        b2=native.receipt({**b['samples'][0],'action':r2['decoded_action_tokens'][0].tolist()},3)
        self.assertTrue(torch.equal(s.input_ids,old_ids)); self.assertTrue(torch.equal(s.last_hidden,old_hidden))
        self.p.processor.batchify_action=False
        ea=native.encode(self.p,a,self.b['pixel_values']); eb=native.encode(self.p,b2,b['pixel_values'])
        ids=torch.cat([ea[0],eb[0]],-1); mask=torch.cat([ea[2],eb[2]],-1)
        self.assertTrue(torch.equal(t.input_ids,ids)); self.assertTrue(torch.equal(t.attention_mask,mask))
        h,_,pos=self.p.model.forward_context(torch.cat([ea[3],eb[3]],1),mask,ea[4]+eb[4])
        self.assertTrue(torch.equal(pos,t.position_ids)); torch.testing.assert_close(h[:,-1],t.last_hidden,atol=3e-5,rtol=3e-5)
    def test_decode_failure_rollback(self):
        s=self.prefill(); ids=s.input_ids.clone(); old=s.kv_cache
        with patch.object(self.p.processor,'decode_ar',side_effect=RuntimeError('test decode failure')):
            with self.assertRaisesRegex(RuntimeError,'decode failure'): self.generate(s)
        self.assertIs(s.kv_cache,old); self.assertTrue(torch.equal(s.input_ids,ids))
    def test_budget_rollback(self):
        s=self.prefill(); old=s.kv_cache; self.p.config.native_context_max_tokens=20
        with self.assertRaisesRegex(RuntimeError,'budget'): self.generate(s)
        self.assertIs(s.kv_cache,old)
    def test_invalid_decode_rollback(self):
        s=self.prefill(); old=s.kv_cache
        fake=([torch.zeros(8,5)],[torch.zeros(39,dtype=torch.long)],[''],[{'left'}])
        with patch.object(self.p.processor,'decode_ar',return_value=fake):
            with self.assertRaisesRegex(RuntimeError,'decode failed'): self.generate(s)
        self.assertIs(s.kv_cache,old)
    def test_text_then_action(self):
        b=turn(cot=True); s=self.p.prefill(b['samples'],b['pixel_values']); n=s.input_ids.size(1)
        # Force only text logits for this protocol test; all forwards and codec real.
        old_decode=self.p.model.vlm.decode
        def decode(h):
            logits=old_decode(h); logits.fill_(-100); logits[...,13]=100; return logits
        with patch.object(self.p.model.vlm,'decode',side_effect=decode):
            self.p.generate_text(s,stop_token_ids=[13],max_new_tokens=2)
        self.assertEqual(s.input_ids.size(1),n+1); self.assertEqual(int(s.input_ids[0,-1]),13)
        self.p.generate_action(s,b['samples'],action_source='ar'); s.check_invariants()
    def test_text_failed_stop_transaction(self):
        s=self.prefill(); old=s.kv_cache
        with patch.object(self.p.model.ar_helper,'_sample',return_value=torch.tensor([14])):
            with self.assertRaisesRegex(RuntimeError,'stop token'): self.p.generate_text(s,stop_token_ids=[13],max_new_tokens=2)
        self.assertIs(s.kv_cache,old)
    def test_multi_episode_grad(self):
        self.p.train(); b2=turn('same'); b2['executed_steps']=8
        loss,_=self.p({'native_episodes':[[self.b,b2],[turn('right')]]}); self.assertTrue(torch.isfinite(loss)); loss.backward()
        for module in (self.p.model.vision_tower,self.p.model.proprio_embedder,self.p.model.vlm):
            self.assertTrue(any(x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum()>0 for x in module.parameters()))
        self.assertIsNone(self.p.model.ar_helper._last_ce_cache)
    def test_receipt_template(self):
        s={**self.b['samples'][0],'template':self.b['samples'][0]['template'].replace('_text_!>','_text_!_200>')}
        r=native.receipt(s,3); self.assertEqual(r['command'],s['command']); self.assertIn('<executionreceipt_text_!>',r['template'])
        with self.assertRaises(ValueError): native.receipt(s,-1)
    def test_two_stream_isolation(self):
        s=self.prefill(); h=s.last_hidden.clone(); b=turn('right'); t=self.p.append_context(None,b['samples'],b['pixel_values']); self.generate(t)
        self.assertTrue(torch.equal(h,s.last_hidden)); self.generate(s)
    def test_rule_field_offsets(self):
        ar=self.p.model.ar_helper
        for c in ar._bar_joint_token_constraints:
            marker=ar._bar_position_constraints[c['offset']-1]
            self.assertEqual(marker['name'],f"<{c['name']}>")
            self.assertIn('fixed_token',marker)
            for j in range(c['num_tokens']): self.assertNotIn('fixed_token',ar._bar_position_constraints[c['offset']+j])
        self.assertEqual(len(ar._bar_joint_token_constraints),1)
    def test_dual_rule_marker_offsets(self):
        p=make_policy(dual_rule=True).eval(); ar=p.model.ar_helper
        self.assertEqual(len(ar._bar_joint_token_constraints),2)
        for c in ar._bar_joint_token_constraints:
            self.assertEqual(ar._bar_position_constraints[c["offset"]-1]["name"], f"<{c['name'] }>")
        s=p.prefill(self.b["samples"],self.b["pixel_values"]); r=p.generate_action(s,self.b["samples"],action_source="ar")
        self.assertEqual(r["action"].shape,(1,8,6)); self.assertFalse(r["ar_absent_keys"][0])
    def test_duplicate_generation_rejected(self):
        s=self.prefill(); self.generate(s); old=s.kv_cache
        with self.assertRaisesRegex(ValueError,"EOV boundary"):self.generate(s)
        self.assertIs(s.kv_cache,old)
    def test_no_suffix_target_leak(self):
        s=self.prefill(); b={**self.b['samples'][0],'template':self.b['samples'][0]['template']+'<answer_text>'}
        with self.assertRaisesRegex(ValueError,'suffix'): self.p.generate_action(s,[b],action_source='ar')
    def test_codec_noncausal_probe(self):
        s=self.prefill(); r=self.generate(s); codes=r['decoded_action_tokens'][0].clone(); ar=self.p.model.ar_helper
        index=max(i for i,c in ar._bar_position_constraints.items() if i<codes.numel() and 'token_lower' in c and 'gripper' not in c['name'])
        c=ar._bar_position_constraints[index]; codes[index]=c['token_lower']+(int(codes[index])-c['token_lower']+1)%(c['token_upper']-c['token_lower'])
        r2=self.p.processor.decode_ar([codes],8,5,generated_ids_are_action_only=True)[0][0]
        error=(r2[:4]-r['action'][0,:4]).abs().max().item(); print('codec_prefix_perturbation',error)
        self.assertGreater(error,1e-8)

class EpisodeTest(unittest.TestCase):
    def test_indices(self):
        for start,end in [(0,33),(33,103),(103,200)]:
            for current in range(start,end):
                h=history_indices(start,end,current,8,4)
                self.assertTrue(all(start<=i and i+8<=current for i in h)); self.assertTrue(len(h)<=3)
    def test_collate(self):
        def row(i): return {'idx':i,'samples':{'command':str(i)},'action':torch.ones(8,2),'pixel_values':{'image':torch.ones(1,3,8,8)}}
        a,b=row(16),row(8); a['native_history']=[row(0),row(8)]
        out=collate_episodes([a,b],collate_fn_pad_sequences)
        self.assertEqual([len(x) for x in out['native_episodes']],[3,1]); self.assertEqual(out['native_episodes'][0][2]['executed_steps'],8); self.assertIn('samples',a)
    def test_bounds_and_padded_history(self):
        class D:
            obs_size=1;past_action_size=0;state_meta=[];image_meta=[];action_meta=[];model_fps=fps=30;action_size=8;_start_idx=8;_end_idx=48
            episode_data_index={'from':torch.tensor([0,24]),'to':torch.tensor([24,48])}
            def load_native_history(self,i):return {'idx':i,'action_is_pad':torch.tensor([i==24])}
        d=D(); self.assertEqual([r['idx'] for r in completed_history(d,40,4)],[32]); d.state_meta=[{'time_offset':1}]
        with self.assertRaisesRegex(ValueError,'future'):completed_history(d,40,4)

if __name__=='__main__': unittest.main()
