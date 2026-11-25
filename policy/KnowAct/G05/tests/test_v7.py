"""V7 regression of executed-only history, real codec and BAR supervision."""
import unittest
from unittest.mock import patch
from dataclasses import replace
import torch
from native_policy_fixture import make_policy, turn
from g05.models.g05 import native_context as n
from g05.data.native_episodes import completed_history, hold_pad, collate_episodes
from g05.utils.data.data_utils import collate_fn_pad_sequences

class ExecutedHistory(unittest.TestCase):
    def setUp(self):
        self.p=make_policy().eval();self.p.config.native_replan_steps=4;self.b=turn()
    def proposal(self):
        s=self.p.prefill(self.b['samples'],self.b['pixel_values']);self.p.generate_action(s,self.b['samples'],action_source='ar');return s
    def test_future_commands_do_not_enter_context(self):
        s=self.proposal();sample=s.pending_action; base=replace(s,pending_action=None,generated_ids=None);v=sample['action']['value'];v2=v.clone();v2[4:]+=0.9
        changed=replace(s,pending_action={**sample,'action':{**sample['action'],'value':v2}})
        a=n.finalize_execution(self.p,s,4);b=n.finalize_execution(self.p,changed,4)
        self.assertTrue(torch.equal(a.input_ids,b.input_ids));torch.testing.assert_close(a.last_hidden,b.last_hidden,rtol=0,atol=0)
        self.assertIsNotNone(s.pending_action)
    def test_full_replay_training_layout(self):
        for count in (1,3,4,7,8):
            s=self.proposal();sample=s.pending_action; base=replace(s,pending_action=None,generated_ids=None);now=turn('right');old={**sample,'action':{**sample['action'],'value':hold_pad(sample['action']['value'],count)}}
            old_enc=n.encode(self.p,old,self.b['pixel_values']);new_enc=n.encode(self.p,n.receipt(now['samples'][0],count,executed_only=True),now['pixel_values'],prefix=True)
            t=self.p.append_context(s,now['samples'],now['pixel_values'],executed_steps=count)
            ids=torch.cat([old_enc[0],new_enc[0]],1);mask=torch.cat([old_enc[2],new_enc[2]],1)
            self.assertTrue(torch.equal(ids,t.input_ids));self.assertTrue(torch.equal(mask,t.attention_mask))
            h,_,pos=self.p.model.forward_context(torch.cat([old_enc[3],new_enc[3]],1),mask,old_enc[4]+new_enc[4]);self.assertTrue(torch.equal(pos,t.position_ids));torch.testing.assert_close(h[:,-1],t.last_hidden,rtol=3e-5,atol=3e-5)
            self.assertIsNone(t.pending_action);self.assertIsNotNone(s.pending_action)
    def test_zero_execution(self):
        s=self.proposal();base=replace(s,pending_action=None,generated_ids=None);now=turn('same');a=self.p.append_context(s,now['samples'],now['pixel_values'],executed_steps=0)
        b=n.append_observation(self.p,base,[n.receipt(now['samples'][0],0,executed_only=True)],now['pixel_values']);self.assertTrue(torch.equal(a.input_ids,b.input_ids));torch.testing.assert_close(a.last_hidden,b.last_hidden)
    def test_invalid_receipt_rollback(self):
        s=self.proposal();cache=s.kv_cache
        for count in (None,-1,9,True,1.5):
            with self.assertRaises(ValueError):self.p.append_context(s,self.b['samples'],self.b['pixel_values'],executed_steps=count)
        self.assertIs(cache,s.kv_cache);self.assertIsNotNone(s.pending_action)
        with self.assertRaisesRegex(RuntimeError,'pending'):self.p.generate_text(s,max_new_tokens=1)
    def test_only_committed_cache_retained(self):
        s=self.p.prefill(self.b['samples'],self.b['pixel_values']);cache=s.kv_cache;ids=s.input_ids.clone()
        self.p.generate_action(s,self.b['samples'],action_source='ar')
        self.assertIs(cache,s.kv_cache);self.assertTrue(torch.equal(ids,s.input_ids));self.assertIsNotNone(s.pending_action)
        with self.assertRaisesRegex(RuntimeError,'pending'):self.p.generate_action(s,self.b['samples'],action_source='ar')
    def test_codec_failure_no_mutation(self):
        s=self.proposal();old=s.kv_cache
        with patch.object(self.p.processor,'preprocess',side_effect=RuntimeError('codec failed')):
            with self.assertRaisesRegex(RuntimeError,'codec failed'):self.p.append_context(s,self.b['samples'],self.b['pixel_values'],executed_steps=3)
        self.assertIs(s.kv_cache,old);self.assertIsNotNone(s.pending_action)
    def test_no_gt_in_proposal(self):
        s=self.proposal();sample=s.pending_action; base=replace(s,pending_action=None,generated_ids=None)
        self.assertFalse(torch.equal(sample['action']['value'],self.b['samples'][0]['action']['value']))
        self.assertEqual(sample['action']['value'].shape,(8,5))
    def test_mask_only_fixed_fields(self):
        e=n.encode(self.p,self.b['samples'][0],self.b['pixel_values']);ids,labels,mask=e[:3];out=n.mask_fixed_action_targets(self.p,ids,labels,mask)
        changed=(labels!=out);fixed=sum('fixed_token' in c for c in self.p.model.ar_helper._bar_position_constraints.values());self.assertEqual(int(changed.sum()),fixed);self.assertTrue((out[changed]==-100).all());self.assertTrue((out[~changed]==labels[~changed]).all())
    def test_mask_ce_backward(self):
        self.p.train();self.p.config.native_mask_fixed_action_targets=True
        loss,_=self.p({'native_episodes':[[self.b]]});loss.backward();self.assertTrue(torch.isfinite(loss));self.assertTrue(any(x.grad is not None and x.grad.abs().sum()>0 for x in self.p.model.parameters()))

class DataTest(unittest.TestCase):
    def test_hold_pad_no_future_dependency(self):
        a=torch.arange(40.).view(8,5);b=a.clone();b[3:]=999;self.assertTrue(torch.equal(hold_pad(a,3),hold_pad(b,3)));self.assertTrue(torch.equal(a[:3],hold_pad(a,3)[:3]));self.assertEqual(float(a[-1,-1]),39.)
    def test_history_indices_actual_strides_and_loss_mask(self):
        class D:
            obs_size=1;past_action_size=0;state_meta=[];image_meta=[];action_meta=[];model_fps=fps=30;action_size=8;_start_idx=0;_end_idx=40
            episode_data_index={'from':torch.tensor([0,20]),'to':torch.tensor([20,40])}
            def load_native_history(self,i):
                a=torch.arange(40.).reshape(8,5)+i;return {'idx':i,'action_is_pad':torch.tensor([False]*4+[True]*4),'samples':{'action':{'value':a}},'action':a}
        d=D();h=completed_history(d,30,4,4);self.assertEqual([x['idx'] for x in h],[22,26]);self.assertTrue(all(torch.equal(x['action'][4:],x['action'][3:4].expand(4,5)) for x in h));self.assertTrue(all(x['_native_executed_steps']==4 for x in h))
        cur=d.load_native_history(30);cur['native_history']=h;batch=collate_episodes([cur],collate_fn_pad_sequences);ep=batch['native_episodes'][0];self.assertEqual([t.get('supervise',True) for t in ep],[False,False,True]);self.assertEqual([t.get('executed_steps') for t in ep],[None,4,4]);self.assertEqual(float(ep[-1]['samples'][0]['action']['value'][-1,-1]),69.)
    def test_invalid_stride(self):
        class D:
            obs_size=1;past_action_size=0;state_meta=[];image_meta=[];action_meta=[];model_fps=fps=30;action_size=8;_start_idx=0;_end_idx=20;episode_data_index={'from':torch.tensor([0]),'to':torch.tensor([20])}
        for stride in (-1,9):
            with self.assertRaisesRegex(ValueError,'replan'):completed_history(D(),16,3,stride)
class LayoutTest(unittest.TestCase):
    def test_real_robodojo_grouped_padding(self):
        from pathlib import Path
        from types import SimpleNamespace
        import yaml
        from g05.models.g05.inferencer import action_padding_for_history
        from g05.data_processor.transforms.action_state_merger import GroupedPaddingMerger
        from g05.data_processor.transforms.action_filter import DummyActionFilter
        cfg=Path(__file__).resolve().parents[1]/'configs/data'
        meta=yaml.safe_load((cfg/'robodojo.yaml').read_text())['processors']['robodojo']['shape_meta']
        layout=yaml.safe_load((cfg/'parts_meta/dual_arm_grouped_0409.yaml').read_text())
        merger=GroupedPaddingMerger(max_action_shape_meta=layout['parts_meta'],merge_spec=layout['merge_spec'])
        merger.set_shape_meta(meta)
        p=SimpleNamespace(action_state_merger=merger,shape_meta=meta,action_filter=DummyActionFilter())
        mask=action_padding_for_history(p,20)
        self.assertEqual(mask.nonzero().flatten().tolist(),[6,7,8,16,17,18])
        with self.assertRaises(ValueError):action_padding_for_history(p,14)
        p.use_stepwise_action_norm=True
        with self.assertRaisesRegex(ValueError,'normalization'):action_padding_for_history(p,20)

    def test_applied_receipt_real_policy_adapter(self):
        from test_xpl_integration import adapter,obs
        m=adapter();m.policy.config.native_replan_steps=4;m.action_steps=4
        m._predict_chunks([obs(5),obs(9)],[5,9]);old=m._native_states[5]
        self.assertIsNotNone(old[0].pending_action)
        m._predict_chunks([obs(5,{'plan':0,'steps':3})],[5]);s=m._native_states[5][0]
        self.assertIsNotNone(s.pending_action);self.assertGreater(s.input_ids.shape[-1],old[0].input_ids.shape[-1])
        with self.assertRaisesRegex(RuntimeError,'receipt'):m._predict_chunks([obs(5,{'plan':0,'steps':3})],[5])
        m._predict_chunks([obs(5,{'plan':1,'steps':0})],[5]);self.assertIsNotNone(m._native_states[5][0].pending_action)
        self.assertEqual(m._native_states[9][1],0)

if __name__=='__main__':unittest.main()
