"""Actual XPL adapter+deployment code, actual G05Policy and codec; simulated IO boundary.
Requires original XPolicyLab installed on PYTHONPATH, or extracted unmodified adapter.
The small processor fixture is not the real robot normalization stack.
"""
import os,threading,unittest
from unittest.mock import patch
import torch
from native_policy_fixture import make_policy,turn
from g05.models.g05.inferencer import PolicyInferencer
from XPolicyLab.policy.KnowAct.model import Model
from XPolicyLab.policy.KnowAct import deploy

class IOFixture:
    pad_token_id=0; num_obs_steps=1
    shape_meta = {"action": [dict(key=side + "_" + part, raw_shape=size)
                             for side in ("left", "right")
                             for part, size in (("control", 2), ("gripper", 1))]}
    def preprocess(self,obs):
        b=turn(obs.get('instruction','same')); return {'samples':b['samples'][0],'pixel_values':{k:v[0] for k,v in b['pixel_values'].items()},'proprio':torch.zeros(1,2)}
    def postprocess(self,b):
        a=b['action'][0]; return {'action':{'left_control':a[:,:2],'right_control':a[:,2:4],'left_gripper':a[:,4:5],'right_gripper':a[:,4:5]}}

def adapter():
    m=Model.__new__(Model);m.demo_head=None;m.embodiment_type="robodojo";m.native_context=True;m.keep_history=True;m._native_lock=threading.RLock();m._native_states={};m._history_buffers={};m.batch_size=2;m.action_steps=8;m.action_type='joint';m.robot_action_dim_info={'arm_dim':[2,2],'ee_dim':[1,1]};m.processor=IOFixture();m.policy=make_policy();m.inferencer=PolicyInferencer(m.policy,m.processor,'cpu','ar');m._to_g05_raw_obs=lambda obs:obs;m._build_obs_dict=lambda raw,p:raw;m.reset();return m

def obs(k,receipt=None):
    o={'env_idx':k,'instruction':'left' if k==5 else 'right'}
    if receipt is not None:o['__knowact_execution__']=receipt
    return o

class TestIntegration(unittest.TestCase):
    def test_real_adapter_and_inferencer(self):
        m=adapter();m.update_obs_batch([obs(5),obs(9)]);out=m.get_action_batch([5,9]);self.assertEqual([len(x) for x in out],[8,8]);s=m._native_states[5][0];ids=s.input_ids.clone()
        m.update_obs_batch([obs(9,{'plan':0,'steps':3})]);out=m.get_action_batch([9]);self.assertEqual(len(out[0]),8);self.assertTrue(torch.equal(s.input_ids,ids));self.assertGreater(m._native_states[9][0].input_ids.size(1),s.input_ids.size(1))
    def test_receipt_replay_rejected(self):
        m=adapter();m._predict_chunks([obs(5)],[5]);o=obs(5,{'plan':0,'steps':3});m._predict_chunks([o],[5]);old=m._native_states[5]
        with self.assertRaisesRegex(RuntimeError,'receipt'):m._predict_chunks([o],[5])
        self.assertIs(m._native_states[5],old)
    def test_batch_atomic_failure(self):
        m=adapter();m._predict_chunks([obs(5),obs(9)],[5,9]);old=dict(m._native_states)
        with self.assertRaises(ValueError):m._predict_chunks([obs(5,{'plan':0,'steps':3}),obs(9,{'plan':0,'steps':100})],[5,9])
        self.assertEqual({k:id(v) for k,v in old.items()},{k:id(v) for k,v in m._native_states.items()})
        m.reset();self.assertFalse(m._native_states)
    def test_reset_ablation(self):
        m=adapter();m.keep_history=False;m._predict_chunks([obs(5)],[5]);n=m._native_states[5][0].input_ids.size(1);m._predict_chunks([obs(5,{'plan':0,'steps':3})],[5]);self.assertEqual(m._native_states[5][0].input_ids.size(1),n)
    def test_batch_deploy_dropouts(self):
        m=adapter(); history=[]
        class Env:
            counts={5:0,9:0};limits={5:2,9:7}
            def is_episode_end(self):return not self.get_running_env_idx_list()
            def get_running_env_idx_list(self):return [k for k in self.counts if self.counts[k]<self.limits[k]]
            def get_obs_batch(self,ks):return [obs(k) for k in ks]
            def take_action_batch(self,actions,ks):
                for k,a in zip(ks,actions):self.counts[k]+=1
        class Client:
            def call(self,func_name,obs=None):
                if func_name=='update_obs_batch':history.append(obs);return m.update_obs_batch(obs)
                if func_name=='get_action_batch':return m.get_action_batch(obs)
                return getattr(m,func_name)()
        env=Env()
        with patch.dict(os.environ,{'KNOWACT_NATIVE_CONTEXT':'true','KNOWACT_REPLAN_STEPS':'3'}):deploy.eval_one_episode_batch(env,Client())
        self.assertEqual(env.counts,env.limits);self.assertEqual(history[1][0]['env_idx'],9);self.assertEqual(history[1][0]['__knowact_execution__'],{'plan':0,'steps':3});self.assertEqual(history[2][0]['__knowact_execution__'],{'plan':1,'steps':3})

if __name__=='__main__': unittest.main()
