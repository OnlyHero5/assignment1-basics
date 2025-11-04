import importlib.metadata
from . import nn_utils
from . import data
from . import tokenizer
from .train_bpe import BPE
<<<<<<< HEAD
from .model import Linear, Embedding, RMSNorm, silu, RoPE, scaled_dot_product_attention, MultiHeadAttention, SwiGLU
=======
from .model import Linear, Embedding, RMSNorm, silu, RoPE, scaled_dot_product_attention, MultiHeadAttention
>>>>>>> f62ad35b825384fc50d967c856b92bb75d556a86

__version__ = importlib.metadata.version("cs336_basics")
