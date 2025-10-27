# -*- coding: utf-8 -*-
# 文件：tokenizer.py
# 作者：PSX
# 描述：tokenzier类的作用是根据vocab和merges将文本转换为token，
#       并将token转换为id，以及将id转换为token。
#       其中，vocab是一个字典，key是token，value是id。
#       merges是一个列表，每个元素是一个元组，元组的第一个元素是token的第一个字节，
#       第二个元素是token的第二个字节。
#       special_tokens是一个列表，每个元素是一个特殊token。
#       整体思路是类接受输入，先byte级别拆分，然后根据merges进行合并，最后根据vocab进行映射。
# 日期：2025-10-26 2025-10-27
#


from typing import List, Tuple, Dict, Iterable, Iterator
import re

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
                token_bytes = self.vocab[token_id]
                bytes_data += token_bytes

        text = bytes_data.decode("utf-8", errors="replace")

        return text

    # 将文本按特殊token分割（内部方法）
    def _split_by_special_tokens(self, text: str) -> list[tuple[str, bool]]:
        
        # 特殊情况处理
        if not self.special_tokens:
            return [(text, False)]
        if not text:
            return []
        if text in self.special_tokens:
            return [(text, True)]
        
        # 按特殊token的长度进行降序排序，长的special token优先匹配
        sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)

        # 构建正则表达式
        escaped_tokens = [re.escape(token) for token in sorted_special_tokens]

        # 使用或运算符进行拼接
        pattern = "|".join(escaped_tokens)

        # 使用正则表达式进行分割
        parts = re.split(f'({pattern})', text)
        
        #构造结果
        result = []
        for part in parts:
            if part:
                is_special = part in self.special_tokens
                result.append((part, is_special))
        
        return result
    

    # 将两个相邻的token合并为一个token对（内部方法）
    def _get_pairs(self, tokens: list[bytes]) -> set[tuple[bytes, bytes]]:
        pairs = set()

        # 特殊情况处理
        if len(tokens) == 1:
            return pairs
        if len(tokens) == 0:
            return pairs

        # 遍历tokens，将相邻的token合并为一个token对
        for i in range(len(tokens) - 1):
            pairs.add((tokens[i], tokens[i+1]))
        
        return pairs

    # 合并一个tokens列表里所有的指定token对，返回合并后的tokens（内部方法）
    def _merge_pair(self, tokens: list[bytes], pair: tuple[bytes, bytes]) -> list[bytes]:
        # 特殊情况处理
        if len(tokens) == 1:
           return tokens
        if len(tokens) == 0:
           return tokens
        if pair is None or len(pair) == 0:
            return tokens
        
        new_tokens = []
        i = 0
        
        while i < len(tokens):
            # 检查是否能合并
            if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == pair:
                # 合并
                new_tokens.append(tokens[i] + tokens[i+1])
                i += 2  # 跳过下一个token
            else:
                new_tokens.append(tokens[i])
                i += 1
        
        return new_tokens
    
    # 整合全部辅助方法，对token列表重复应用BPE合并规则，直到不再有合并操作
    def _apply_bpe_merges(self, tokens:list[bytes]) -> list[bytes]:
        
        while True:

            # 获取所有可能的相邻对
            possible_pairs = self._get_pairs(tokens)

            # 过滤出现在merge_ranks中的对
            possible_merges = []
            for pair in possible_pairs:
                if pair in self.merge_ranks.keys():
                    priority = self.merge_ranks[pair]
                    possible_merges.append((pair, priority))
            
            # 如果没有可合并的对，结束
            if not possible_merges:
                break

            # 选择优先级最高的对进行合并
            best_pair = min(possible_merges, key=lambda x: x[1])[0]
            tokens = self._merge_pair(tokens, best_pair)
        
        return tokens
    
    # 流式编码，逐个生成 token ID
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
       
        for item in iterable:
            if isinstance(item, str):
                segments = self._split_by_special_tokens(item)

                for segment, is_special in segments:
                    if is_special:
                        token_bytes = segment.encode("utf-8")
                        token_id = self.reverse_vocab[token_bytes]
                        yield token_id
                    else:
                        # 对普通文本进行BPE编码
                        tokens = [bytes([b]) for b in segment.encode("utf-8")]
                        tokens = self._apply_bpe_merges(tokens)

                        for token in tokens:
                            token_id = self.reverse_vocab[token]
                            yield token_id
    
    # 实现最终的encode
    def encode(self, text: str) -> list[int]:
        return list(self.encode_iterable([text]))
