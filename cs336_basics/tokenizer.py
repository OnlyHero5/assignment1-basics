# -*- coding: utf-8 -*-
# 文件：nn.utils.py
# 作者：PSX
# 描述：tokenzier类的作用是根据vocab和merges将文本转换为token，
#       并将token转换为id，以及将id转换为token。
#       其中，vocab是一个字典，key是token，value是id。
#       merges是一个列表，每个元素是一个元组，元组的第一个元素是token的第一个字节，
#       第二个元素是token的第二个字节。
#       special_tokens是一个列表，每个元素是一个特殊token。
#       整体思路是类接受输入，先byte级别拆分，然后根据merges进行合并，最后根据vocab进行映射。
# 日期：2025-10-26


from typing import List, Tuple, Dict, Optional
import re
import torch


class Tokenizer:

    def __init__(
            self,
            vocab: dict[int, bytes],
            merges: list[tuple[bytes, bytes]],
            special_tokens: list[str] | None = None
            ):
        
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        self.reverse_vocab = {v: k for k, v in vocab.items()}
        # merges的快速查找版本
        self.merge_ranks = {pair:i  for i, pair in enumerate(merges)}

    
    # decoder: id -> token -> text
    def decode(self, token_ids: List[int]) -> str:
        bytes_data = b""
        for token_id in token_ids:
            if token_id in self.vocab:
                token = self.vocab[token_id]
                bytes_data += token.decode("utf-8", errors="replace")
        text = bytes_data.decode("utf-8", errors="replace")
        return text

    