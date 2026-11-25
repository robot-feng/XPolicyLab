"""Use the released joint AR/FM G05 config with the single-arm Panda observation schema."""

from pathlib import Path
import os

from XPolicyLab.utils.process_data import get_robot_action_dim_info
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf as OC
from g05.utils.config.config_resolvers import register_default_resolvers


def configure(root, codec):
    source = Path(__file__).resolve().parents[3] / "policy/G05/G05"
    os.chdir(source)
    register_default_resolvers()
    with initialize_config_dir(version_base=None, config_dir=str(source / "configs")):
        cfg = compose(config_name="train", overrides=["task=robodojo_arx_x5_joint"])
    OC.set_struct(cfg, False)
    cfg.run_dir = str(root / "run")
    cfg.output_dir = cfg.run_dir
    cfg.ckpt_path = str(root / "base/checkpoints/model_state_dict.pt")
    cfg.model.model_arch.hf_processor_path = str(root / "base/hf_processor")
    cfg.model.model_arch.pretrained_model_path = None
    cfg.tokenizer.vq_config.ckpt_dir = str(codec)
    cfg.model.tokenizer = cfg.tokenizer
    cfg.model.model_arch.fm.num_flow_samples = 2
    cfg.model.model_arch.ar.use_fused_ce = False
    cfg.model.model_arch.checkpoint_vision = False
    cfg.model.model_arch.checkpoint_action_decoder = False
    cfg.model.processor.num_input_cameras = cfg.model.processor.num_output_cameras = 2
    cfg.model.processor.use_stepwise_action_norm = False
    cfg.model.processor.norm_exception_mode = None
    # The official single-arm config calls the two cameras image and wrist_image.
    libero = OC.load(source / "configs/data/libero.yaml")
    dataset = libero.embodiment_datasets.libero
    processor = libero.processors.libero
    schema = dataset.shape_meta
    dims = get_robot_action_dim_info("panda")
    (arm,) = dims["arm_dim"]
    (grip,) = dims["ee_dim"]
    for modality, field in [("action", "action"), ("state", "observation.state")]:
        schema[modality] = [
            dict(key=k, lerobot_key=field, start_index=i, raw_shape=d, shape=d, time_offset=0)
            for k, i, d in [("right_arm", 0, arm), ("right_gripper", arm, grip)]
        ]
    for im in schema.images:
        im.raw_shape = [3, 256, 256]
        im["shape"] = [3, 256, 256]
    dataset.embodiment_type = "panda"
    dataset.dataset_groups = []
    processor.shape_meta = schema
    processor.action_state_transforms = None
    processor.action_filter = {
        "_target_": "g05.data_processor.transforms.action_filter.DummyActionFilter"
    }
    cfg.data.embodiment_datasets = {"panda": dataset}
    cfg.data.processors = {"panda": processor}
    cfg.model.processor.norm_exception_mode = {
        "action": {"right_gripper": "q01/q99"},
        "state": {"right_gripper": "q01/q99"},
    }
    cfg.logger.task = "panda"
    cfg.logger.experiment_name = "g05-robomme"
    cfg.logger.mode = "disabled"
    return cfg
