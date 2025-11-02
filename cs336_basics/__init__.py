import importlib.metadata
from . import nn_utils
from . import data
from . import tokenizer
from .train_bpe import BPE

__version__ = importlib.metadata.version("cs336_basics")
