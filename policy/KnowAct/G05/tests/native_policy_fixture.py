"""Actual small G05Policy + ActionCodec math. Minimal optional HF/base imports only.
No pretrained weights, no real text tokenizer; the deterministic fixture maps words.
"""
import sys, types, re
import native_context_imports
import transformers
import torch
from torch import nn
from omegaconf import OmegaConf
from native_context_fixture import tiny
if not hasattr(transformers, 'PretrainedConfig'):
    class PretrainedConfig:
        def __init__(self, **kw): self.__dict__.update(kw)
    class PreTrainedModel(nn.Module):
        def __init__(self, config): super().__init__(); self.config=config
        def post_init(self): self.apply(self._init_weights)
    sys.modules['transformers'].PretrainedConfig=PretrainedConfig
    sys.modules['transformers'].PreTrainedModel=PreTrainedModel
try: import git
except ImportError:
    mod=types.ModuleType('git'); mod.Repo=lambda *a,**kw: None; sys.modules['git']=mod
from g05.models.g05.g05_policy import G05Policy
from g05.models.g05.io.input_preprocessor import InputPreprocessor, BuiltinTextProcessor, BuiltinProprioMLPProcessor, BuiltinActionProcessor, BuiltinQwen35ImageProcessor
from g05.tokenizer.interface.vq_base import VQActionTokenizer
from g05.tokenizer.interface.bar_builder import BARBuilder
from g05.utils.common.special_tokens import SpecialTokenManager
from g05.utils.training.accuracy_accumulator import TrainAccuracyAccumulator

class TextTokenizer:
    eos_token_id=2; pad_token_id=0
    def __call__(self,s,**kw): return {'input_ids':self.encode(s,**kw)}
    def encode(self,s,**kw):
        words=re.findall(r'<[^>]+>|\w+|[^\w\s]', s)
        fixed={'<EOV>':13,'<eos>':2,'<|eot|>':2,'left':14,'right':15,'answer':16,'same':17,'Action':11,'|':12}
        return [fixed.get(w,18+sum(w.encode())%13) for w in words]
    def decode(self,ids,**kw): return ' '.join(str(int(i)) for i in ids)
    def batch_decode(self,rows,**kw): return [self.decode(r) for r in rows]
    def get_vocab(self): return {'<eos>':2,'<EOV>':13}

def make_policy(seed=11, dual_rule=False):
    m=tiny(seed)
    # Actual codec backend, two groups and two RVQ levels; small FSQ to fit test vocab.
    cfg=OmegaConf.create(dict(vqvae_type='g05.tokenizer.models.actioncodec2_v2.wrapper.ActionCodecV2Wrapper',block_wise_autoregressive=True,block_size=16,use_extra_tokens=True,use_group_markers=True,group_order_shuffle=False,dropout_noop_parts=False,parts_meta={'left':2,'right':2,'gripper':1},rule_based_key_patterns=['gripper'],num_residuals=2,eval=True,model_arch=dict(max_component_dim=3,horizon=8,horizon_patch_size=2,conv_in_action_kernel=2,encoder_channels=16,latent_dim=16,c_mults=[1],strides=[[1,1]],transformer_depths=[1],num_heads=2,dim_heads=32,n_codebooks=2,use_fsq=True,fsq_levels=[4,4],codebook_size=16,codebook_dim=2,use_block_dct=True,block_dct_block_size=2)))
    if dual_rule:
        cfg.parts_meta["right_gripper"]=1
    codec=VQActionTokenizer(cfg,vocab_offset=40,hf_vocab_size=40,device='cpu'); codec.set_bar_ids(100,101,102)
    codec.action_tokenizer.model.eval()
    tok=TextTokenizer()
    pp=InputPreprocessor.__new__(InputPreprocessor)
    pp.tokenizer=tok; pp.action_tokenizer=codec; pp.batchify_action=True; pp.pi05_ft_mode=False; pp.pad_token_id=0; pp.model_cfg=m.cfg
    pp.control_tokens={'eoc':'EOC','eov':'EOV'}; pp.pred_eov=True; pp.eov_token_id=13
    pp.token_manager=SpecialTokenManager(tokens={'eos':'<|eot|>'})
    pp.bar_builder=BARBuilder(tok,block_wise=True,**codec.bar_config)
    im=BuiltinQwen35ImageProcessor.__new__(BuiltinQwen35ImageProcessor)
    im.patch_size=4; im.spatial_merge_size=2; im.image_token_index=7; im.boi_token_index=9; im.eoi_token_index=10
    pp.modality_processors={'text':BuiltinTextProcessor(),'proprio':BuiltinProprioMLPProcessor(8),'action':BuiltinActionProcessor(pp.bar_builder,False),'image':im}
    p=G05Policy.__new__(G05Policy); nn.Module.__init__(p); p.model=m; p.processor=pp; p.action_tokenizer=codec
    p.config=OmegaConf.create(dict(native_context=True,native_context_max_tokens=4096,horizon_steps=8,action_dim=6 if dual_rule else 5,pad_token_id=0,eos_token_id=2)); p.model_config=p.config
    p.model.cfg.eos_token_id=2
    p.continuous_action=False; p.discrete_action=True; p.predict_cot=False; p.language_loss_weight=1.; p._train_acc=TrainAccuracyAccumulator(); p._fwd_step=0
    p._sync_bar_runtime(codec.token_config); p._sync_bar_position_constraints(codec.token_config); p._sync_bar_joint_token_constraints(codec.token_config)
    ar=m.ar_helper; ar.set_token_index_ranges(codec.action_token_begin_idx,codec.action_token_end_idx); ar.do_sample=False
    return p

def turn(command='left', direction=1., cot=False, pixels=True):
    t=torch.linspace(-.5,.5,8)
    # Native flat codec layout [left(2), right(2), gripper(1)]
    a=torch.stack((t*direction,t*.5,t*-.4,t*direction,(t>0).float()),-1)
    sample={'template':'<image_image_!> Task: <command_text_!> <proprio_proprio_!> Action: <EOV><EOC><action_action>|<eos>' if not cot else '<image_image_!> Task: <command_text_!> <proprio_proprio_!><EOC><answer_text> Action: <EOV><action_action>|<eos>', 'image':(8,8),'command':command,'proprio':{'value':torch.tensor([[.2,.1]])},'action':{'value':a,'action_op_mask':torch.ones(5,dtype=torch.bool),'action_dim_is_pad':torch.zeros(5,dtype=torch.bool)},'answer':'answer'}
    return {'samples':[sample],'pixel_values':{'image':torch.ones(1,1,3,8,8)*.1},'action':a[None]}
