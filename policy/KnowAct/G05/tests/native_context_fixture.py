import native_context_imports
import torch
from omegaconf import OmegaConf
from g05.models.g05.g05_model_qwen35 import G05ModelQwen35
from g05.models.g05.qwen35.gated_deltanet import Qwen3_5GatedDeltaNet
from g05.models.g05.helpers.mask_helper import MaskHelperQwen35
from g05.models.kv_cache import SparseKVCache
from g05.models.g05.io.input_preprocessor import BuiltinActionProcessor, TOKEN_INDEX
from g05.tokenizer.interface.bar_builder import BARBuilder

torch.set_num_threads(1)

def config():
    mixture=dict(name='g05.models.g05.qwen35.mixture_qwen35.MixtureQwen35',hidden_size=32,intermediate_size=64,num_hidden_layers=4,num_attention_heads=2,num_key_value_heads=1,head_dim=16,rms_norm_eps=1e-6,max_position_embeddings=4096,attention_bias=False,input_type='embedding',output_type='lm_head',vocab_size=128,pad_token_id=0,adaptive_mode=None,time_hidden_size=0,use_final_norm=True,hidden_act='silu',rope_parameters=dict(rope_theta=10000000.,rope_type='default',mrope_section=[2,1,1],mrope_interleaved=True,partial_rotary_factor=.5),linear_attn_backend='torch',linear_conv_kernel_dim=4,linear_key_head_dim=8,linear_value_head_dim=8,linear_num_key_heads=2,linear_num_value_heads=2,layer_types=['linear_attention','linear_attention','linear_attention','full_attention'])
    expert=dict(mixture); expert.update(input_type='linear',output_type='linear',input_dim=2,output_dim=2,layer_types=['full_attention']*4)
    vision=dict(name='g05.models.g05.qwen35.vision.Qwen3_5VisionModel',depth=1,hidden_size=32,num_heads=2,patch_size=4,temporal_patch_size=2,spatial_merge_size=2,in_channels=3,intermediate_size=64,out_hidden_size=32,num_position_embeddings=64,temporal_freq=0,hidden_act='gelu_pytorch_tanh')
    return OmegaConf.create(dict(image_token_index=7,pad_token_id=0,attn_implementation='eager',checkpoint_vision=False,checkpoint_vlm=False,checkpoint_action_expert=False,proprio_encoder='mlp',proprio_dim=2,position_ids_type='pi0fast',action_position_offset=None,vlm=mixture,action_expert=expert,vision=vision,fm=dict(time_convention='pi_convention',num_inference_steps=2,horizon_steps=4,action_dim=2,fm_weight=1.,final_action_clip_value=None,flow_sig_min=.001,padding_action_weight=0.,zero_pad_action_target=False,joint_training=False,flow_sampling='beta'),ar=dict(ce_weight=1.,vocab_size=128,use_fused_ce=False,ce_z_loss_scale=0.,block_wise_autoregressive=True,bos_blk_id=100,eos_blk_id=101,block_size=16)))

def tiny(seed=11):
    torch.manual_seed(seed); m=G05ModelQwen35(config()); m.proprio_embedder.state_token_id=8
    # Nonzero state encoder so tests really exercise state inputs.
    torch.nn.init.normal_(m.proprio_embedder.mlp[3].weight,std=.02)
    return m

class MiniTokenizer:
    """Only a test token-ID wrapper, not an ActionCodec implementation."""
    def encode(self,text,add_special_tokens=False):
        if text == 'Action: ': return [11]
        if text == '|': return [12]
        if not text: return []
        return list(map(int,text.split()))

def bar(width=16,num_block=None):
    return BARBuilder(MiniTokenizer(),block_wise=True,block_size=width,bos_blk_id=100,eos_blk_id=101,pad_action_id=102,num_block=num_block)

def sample_segments(m):
    # two independently encoded observations with different image resolutions,
    # text-only and state-only events, then the original BARBuilder output.
    out=[]
    def add(ids,mask,images=None,proprio=None):
        ids=torch.tensor([ids]); mask=torch.tensor([mask],dtype=torch.float32)
        emb,grid=m.encode_context_segment(ids,mask,images,proprio)
        out.append((emb,mask,grid))
    add([14,15],[4,4]); add([9,7,7,7,7,10],[4,1,1,1,1,4],{'front':torch.randn(1,1,3,16,16)})
    add([8],[2],proprio=torch.randn(1,1,2)); add([20,21],[4,4]);
    ids,mask,labels=bar().build_io_from_ids(list(range(30,48)))
    add(ids,mask); add([9,7,7,10],[4,1,1,4],{'wrist':torch.randn(1,1,3,8,16)}); add([8,22],[2,4],proprio=torch.randn(1,1,2))
    return out
