"""Official RoboMME pi05_baseline backend (joint control, two RGB cameras)."""

from pathlib import Path
import sys

def load_policy(checkpoint: Path, seed: int):
    source = Path(__file__).resolve().parent / "robomme_runtime" / "src"
    if not (source / "mme_vla_suite").is_dir():
        raise FileNotFoundError("Run bash policy/Pi_05/install_robomme.sh first")
    # Both upstream trees provide `openpi`; select the official RoboMME fork
    # before importing either. A fresh policy server is required when switching.
    if "openpi" in sys.modules:
        loaded = Path(sys.modules["openpi"].__file__).resolve()
        if not loaded.is_relative_to(source):
            raise RuntimeError("Native openpi is already loaded; start a fresh RoboMME policy server")
    sys.path.insert(0, str(source))
    from mme_vla_suite.policies import policy_config
    from mme_vla_suite.training import config

    return policy_config.create_trained_policy(config.get_config("pi05_baseline"), checkpoint, seed=seed)
