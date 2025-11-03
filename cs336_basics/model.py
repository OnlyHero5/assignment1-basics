"""
Transformer Language Model Implementation

本模块实现了完整的 Transformer 语言模型，包括：
- 基础层：Linear, Embedding, RMSNorm
- 激活函数：SiLU
- 位置编码：RoPE (Rotary Position Embedding)
- 注意力机制：Scaled Dot-Product Attention, Multi-Head Attention
- 前馈网络：SwiGLU
- 完整模型：TransformerBlock, TransformerLM

Author: PSX
Date: 2025-11-01
"""
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple
import math


# ============================================================
# 第一部分：基础层 - Linear
# ============================================================
class Linear(nn.Module):

    def __init__(self, d_in: int, d_out: int, bias: bool = True):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out

        # 初始化权重矩阵 y = x @ W.T + b
        self.weight = nn.Parameter(torch.empty(d_out, d_in))

        # Kaiming 初始化
        bound = math.sqrt(6.0 / d_in)
        nn.init.uniform_(self.weight, -bound, bound)

        # 创建偏置
        if bias:
            self.bias = nn.Parameter(torch.zeros(d_out))
        else:
            self.register_parameter("bias", None)
    
    def forward(self, x: Tensor) -> Tensor:
        out = torch.matmul(x, self.weight.t())
        if self.bias is not None:
            out = out + self.bias
        
        return out
    

# ============================================================
# 第二部分：Embedding 层
# ============================================================
class Embedding(nn.Module):

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
    
        # 创建嵌入矩阵
        self.weight = nn.Parameter(torch.empty(vocab_size, d_model))
        # pytorch默认std=1.0,这里采用bert工程实践经验std=0.02
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, tokens_id: Tensor) -> Tensor:
        """
        前向传播：查表操作
        
        参数:
            token_ids: token ID 张量 (batch_size, seq_len)
        
        返回:
            嵌入向量 (batch_size, seq_len, d_model)
        """
        return self.weight[tokens_id]



# ============================================================
# 第三部分：RMSNorm 层
# ============================================================
class RMSNorm(nn.Module):

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播
        
        参数:
            x: 输入张量 (..., d_model)
        
        返回:
            归一化后的张量 (..., d_model)
        """
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)

        x_normed = x / rms
        x_normed = x_normed * self.weight
        return x_normed


# 激活函数
def silu(x: Tensor) -> Tensor:
    """
    SiLU 激活函数
    """
    exp_minus_x = torch.exp(-x)
    sigmoid_x = 1.0 / (1.0 + exp_minus_x)
    return x * sigmoid_x



# ============================================================
# 第五部分：RoPE 位置编码 (Day 2 上午)
# ============================================================
class RoPE(nn.Module):
    """
    旋转位置编码 (Rotary Position Embedding)
    
    RoPE 通过旋转操作将位置信息编码到 Q 和 K 中。
    相比绝对位置编码，RoPE 具有更好的外推能力。
    
    核心思想:
        将 d_head 维向量看作 d_head/2 个复数
        对每个复数应用旋转变换: z' = z * e^(i*m*θ)
        其中 m 是位置，θ 是频率
    
    参数:
        d_head: 每个注意力头的维度 (必须是偶数)
        max_seq_len: 最大序列长度
        theta: 频率基数，默认 10000.0
    
    形状:
        输入: (batch, seq_len, num_heads, d_head)
        输出: (batch, seq_len, num_heads, d_head)
    
    参考:
        RoFormer: Enhanced Transformer with Rotary Position Embedding
        https://arxiv.org/abs/2104.09864
    
    示例:
        >>> rope = RoPE(d_head=64, max_seq_len=2048)
        >>> x = torch.randn(2, 10, 8, 64)  # (batch, seq, heads, d_head)
        >>> pos = torch.arange(10).unsqueeze(0).expand(2, -1)  # (batch, seq)
        >>> x_rotated = rope(x, pos)
    """
    def __init__(self, d_head: int, max_seq_len: int, theta: float = 10000.0):
        super().__init__()
        assert d_head % 2 == 0, "d_head must be even for RoPE"

        self.d_head = d_head
        self.max_seq_len = max_seq_len
        self.theta = theta

        # 预计算频率和旋转角度
        inv_freq = self._compute_inv_frequencies()
        self.register_buffer("inv_freq", inv_freq)  # 注册为模型参数，避免梯度计算

        cos, sin = self._precompute_cos_sin(inv_freq, max_seq_len)
        self.register_buffer("cos", cos)  # 注册为模型参数，避免梯度计算
        self.register_buffer("sin", sin)  # 注册为模型参数，避免梯度计算
    
    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
        """
        应用 RoPE
        
        参数:
            x: 输入，支持两种形状：
               - 3D: (..., seq_len, d_model)
               - 4D: (batch, seq_len, num_heads, d_head)
            token_positions: (..., seq_len)
        
        返回:
            旋转后的张量，形状与输入相同
        """

        # 首先判断输入的x是什么形状
        needs_unsqueeze = (x.ndim == 3)
        if needs_unsqueeze:
            x = x.unsqueeze(-2)
        
        # 获取预处理的cos/sin值
        cos, sin = self._get_cos_sin_for_positions(token_positions)
        # 拆分实部虚部
        x_real, x_imag = self._reshape_for_rotation(x)
        # 应用旋转
        x_rotated = self._apply_rotation(x_real, x_imag, cos, sin)

        if needs_unsqueeze:
            x_rotated = x_rotated.squeeze(-2)
        
        return x_rotated

    def _compute_inv_frequencies(self) -> Tensor:
        """
        计算逆频率向量
        
        公式: inv_freq[i] = 1 / (theta^(2i/d_head))
        其中 i = 0, 1, ..., d_head/2 - 1
        
        返回:
            inv_freq: 形状 (d_head // 2,)
        
        示例:
            >>> rope = RoPE(d_head=4, max_seq_len=10, theta=10000.0)
            >>> rope.inv_freq
            tensor([1.0000, 0.0100])  # [1/10000^0, 1/10000^(2/4)]
        """
        indices = torch.arange(0, self.d_head, 2, dtype=torch.float32)
        exponents = indices / self.d_head
        inv_freq = 1.0 / (self.theta ** exponents)
        return inv_freq
    
    def _precompute_cos_sin(self, inv_freq: Tensor, max_seq_len: int) -> Tuple[Tensor, Tensor]:
        """
        预计算所有位置的 cos 和 sin 值
        
        参数:
            inv_freq: 逆频率向量 (d_head // 2,)
            max_seq_len: 最大序列长度
        
        返回:
            cos: 余弦值 (max_seq_len, d_head // 2)
            sin: 正弦值 (max_seq_len, d_head // 2)
        
        计算过程:
            1. 生成位置索引: [0, 1, 2, ..., max_seq_len-1]
            2. 计算角度矩阵: positions ⊗ inv_freq (外积)
            3. 计算 cos 和 sin
        """
        # 生成位置索引
        positions = torch.arange(max_seq_len, dtype=torch.float32)  # (max_seq_len,)
        # 计算角度矩阵
        freqs = torch.outer(positions, inv_freq)

        cos = torch.cos(freqs)
        sin = torch.sin(freqs)

        return cos, sin
    
    def _reshape_for_rotation(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        将输入重排为复数形式 (实部, 虚部)
        
        参数:
            x: 输入张量 (..., d_head)
        
        返回:
            x_real: 实部 (..., d_head // 2)
            x_imag: 虚部 (..., d_head // 2)
        
        转换示例:
            输入: [x0, x1, x2, x3, x4, x5]
            实部: [x0, x2, x4]  # 偶数索引
            虚部: [x1, x3, x5]  # 奇数索引
        """
        x_reshaped = x.reshape(*x.shape[:-1], -1, 2)
        x_real = x_reshaped[..., 0]  # 实部
        x_imag = x_reshaped[..., 1]  # 虚部
        return x_real, x_imag
    
    def _apply_rotation(self, 
                        x_real: Tensor,
                        x_imag: Tensor,
                        cos: Tensor,
                        sin: Tensor) -> Tensor:
        """
        应用复数旋转变换
        
        复数乘法公式:
            (a + bi) * (cos + i*sin) = (a*cos - b*sin) + i*(a*sin + b*cos)
        
        参数:
            x_real: 实部 (..., d_head // 2)
            x_imag: 虚部 (..., d_head // 2)
            cos: 余弦值 (..., d_head // 2)
            sin: 正弦值 (..., d_head // 2)
        
        返回:
            旋转后的张量 (..., d_head)
        """
        x_rotated_real = x_real * cos - x_imag * sin
        x_rotated_imag = x_real * sin + x_imag * cos

        x_rotated = torch.stack([x_rotated_real, x_rotated_imag], dim=-1)
        x_rotated = x_rotated.reshape(*x_real.shape[:-1], -1)  # 展平
        return x_rotated
    
    def _get_cos_sin_for_positions(self, token_positions: Tensor) -> Tuple[Tensor, Tensor]:
        """
        根据位置索引获取对应的 cos 和 sin 值
        
        参数:
            token_positions: 位置索引 (batch, seq_len)
        
        返回:
            cos: 对应位置的余弦值 (batch, seq_len, 1, d_head // 2)
            sin: 对应位置的正弦值 (batch, seq_len, 1, d_head // 2)
        """
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]

        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)

        return cos, sin
        

    