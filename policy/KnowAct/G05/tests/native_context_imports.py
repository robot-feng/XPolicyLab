"""CPU-only source test loader. No substitution of model/attention/codec formatting math.
In the official environment, use ordinary imports. Standalone mode avoids the
policy package's eager imports of optional training/deployment dependencies.
"""
import os, sys, types, importlib.util
from pathlib import Path
ROOT = Path(os.environ.get('G05_TEST_SOURCE', Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT/'src'))
if os.environ.get('G05_TEST_STANDALONE', '1') == '1':
    pkg=types.ModuleType('g05.models.g05'); pkg.__path__=[str(ROOT/'src/g05/models/g05')]; sys.modules[pkg.__name__]=pkg
    # Only eager attention is exercised. This registry is not the attention implementation.
    if importlib.util.find_spec('transformers') is None:
        pkg=types.ModuleType('transformers'); pkg.__path__=[]; sys.modules['transformers']=pkg
        mod=types.ModuleType('transformers.modeling_utils'); mod.ALL_ATTENTION_FUNCTIONS={}; sys.modules[mod.__name__]=mod
    # Trainer monitor is disabled; no optimizer/data/forward code is replaced.
    mod=types.ModuleType('g05.utils.training.train_utils'); mod.get_global_monitor=lambda: None; sys.modules[mod.__name__]=mod
