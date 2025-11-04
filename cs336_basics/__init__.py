import importlib.metadata
from . import nn_utils
from . import data
from . import tokenizer
from .train_bpe import BPE
from .model import Linear, Embedding, RMSNorm, silu, RoPE, scaled_dot_product_attention, MultiHeadAttention, SwiGLU

__version__ = importlib.metadata.version("cs336_basics")
