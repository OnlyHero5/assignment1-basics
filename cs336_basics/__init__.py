import importlib.metadata
from . import nn_utils
from . import data
from . import tokenizer
from .train_bpe import BPE
from .model import Linear, Embedding, RMSNorm, silu, RoPE, scaled_dot_product_attention, MultiHeadAttention, SwiGLU, TransformerBlock, TransformrLM
from .optimizer import AdamW, get_lr_cosine_schedule
from .serialization import save_checkpoint, load_checkpoint

__version__ = importlib.metadata.version("cs336_basics")
