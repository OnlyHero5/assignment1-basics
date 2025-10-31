"""
BPE (Byte-Pair Encoding) Training Implementation

This module implements the training algorithm for Byte-Pair Encoding tokenizers,
following the approach used in GPT-2.

Algorithm Overview:
    1. Pre-tokenize corpus using regex
    2. Initialize vocab with 256 bytes + special tokens
    3. Iteratively merge most frequent adjacent pairs
    4. Record merge history for later use

References:
    - GPT-2 Paper: https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
    - Original BPE Paper: https://arxiv.org/abs/1508.07909
    - SentencePiece: https://github.com/google/sentencepiece

Author: PSX
Date: 2025-10-28
"""
import regex 
import os
from collections import Counter

# GPT-2 预分词正则表达式
GPT2_SPLIT_PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


#实现 train_bpe 的主函数
def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    
    # 异常处理
    if special_tokens is None:
        special_tokens = []
    
    min_vocab_size = 256 + len(special_tokens)
    if vocab_size < min_vocab_size:
        raise ValueError(f"vocab_size must be at least {min_vocab_size}, got {vocab_size}")
    

    # ======================读取并预分词======================
    print(f"Step 1: Reading and pre-tokenizing corpus from {input_path}...")

    
    pattern = _get_pretokenized_pattern(special_tokens)
    # 读取文本
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    print(f" Corpus size: {len(text)} characters")

    # 预分词
    words = _pretokenize(text, pattern)
    print(f" Pre-tokenized into {len(words)} words")

    # ======================初始化词表和token序列======================
    print("\n Step 2: Initializing vocabulary ...")
    vocab = _initialize_vocab(special_tokens)
    token_sequences = _word_to_token_sequence(words, vocab)

    print(f" Initialized vocab size: {len(vocab)}")
    print(f" Tokenized sequences: {len(token_sequences)} words")

    total_tokens = sum(len(seq) for seq in token_sequences)
    print(f" Total tokens: {total_tokens}")

    # 初始化merges列表
    merges = []

    # ======================迭代合并======================
    num_base_tokens = len(vocab)
    num_merges = vocab_size - num_base_tokens

    print(f"\n Step 3: Merging tokens...")

    # 主循环
    for i in range(num_merges):
        # 1. 统计所有相邻pair的频率
        pair_frequency, pair_first_pos = _count_pair(token_sequences)

        if not pair_frequency:
            print(f"No more merges available at iteration {i+1}")
            break
        
        # 2. 合并频率最高的pair
        max_freq = max(pair_frequency.values())

        candidates = [pair for pair, freq in pair_frequency.items() if freq == max_freq]

        best_pair = min(candidates, key=lambda p: pair_first_pos[p])

        # 3. 创建新的token
        new_token_id = len(vocab)
        token1_bytes = vocab[best_pair[0]]
        token2_bytes = vocab[best_pair[1]]
        new_token_bytes = token1_bytes + token2_bytes

        # 4. 记录merge操作
        vocab[new_token_id] = new_token_bytes
        merges.append((token1_bytes, token2_bytes))

        _merge_pair(token_sequences, best_pair, new_token_id)

        # 5. 打印进度
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  Merge {i+1}/{num_merges}: "
                  f"{token1_bytes!r} + {token2_bytes!r} = {new_token_bytes!r} "
                  f"(freq: {max_freq})")

    #================= step 4: 打印最终结果 =====================
    print(f"\n Training completed!")
    print(f" Final vocab size: {len(vocab)}")
    print(f" Total merges: {len(merges)}")
    return vocab, merges



# 预分词的规则指定：包括输入的special_tokens和GPT2的规则
def _get_pretokenized_pattern(special_tokens: list[str]) -> regex.Pattern:

    # GPT-2 预分词正则表达式
    base_pattern = GPT2_SPLIT_PATTERN

    if special_tokens:
        # 转义特殊字符
        escaped_tokens = [regex.escape(token) for token in special_tokens]
        special_pattern = "|".join(escaped_tokens)
        finally_pattern = f"({special_pattern})|{base_pattern}"
    else: 
        finally_pattern = base_pattern
    
    # 编译正则表达式
    return regex.compile(finally_pattern)

# 根据正则表达式进行预分词
def _pretokenize(text: str, pattern: regex.Pattern) -> list[bytes]:

    words = []
    for match in pattern.finditer(text):
        word = match.group(0)

        word_bytes = word.encode("utf-8")
        words.append(word_bytes)
    return words

# 初始化词表
def _initialize_vocab(special_tokens: list[str]) -> dict[int, bytes]:

    vocab = {}
    # 添加256个byte
    for i in range(256):
        vocab[i] = bytes([i])
    
    # 添加特殊token
    current_index = 256
    for token in special_tokens:
        token_bytes = token.encode("utf-8")
        vocab[current_index] = token_bytes
        current_index += 1
    
    return vocab


# 把Word转成Token
def _word_to_token_sequence(words: list[bytes], 
                            vocab: dict[int, bytes]
                            ) -> list[list[int]]:
    bytes_to_id = {v: k for k, v in vocab.items()}
    token_sequences = []

    for word in words:
        if word in bytes_to_id:
            token_sequences.append([bytes_to_id[word]])
        else:
            # 如果word不在vocab中，进行byte级拆分
            tokens = []
            for byte_val in word:
                single_byte = bytes([byte_val])
                token_id = bytes_to_id[single_byte]
                tokens.append(token_id)

            token_sequences.append(tokens)

    return token_sequences


# 统计相邻出现的频率
def _count_pair(token_sequences: list[list[int]]
                ) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int], int]]:
        
    pair_counts = {}  # ⭐ 改用dict而不是Counter
    pair_first_pos = {}  # 记录每个pair第一次出现的位置
    global_position = 0
    
    for tokens in token_sequences:
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            
            if pair not in pair_counts:
                pair_first_pos[pair] = global_position
                pair_counts[pair] = 0
            
            pair_counts[pair] += 1
            global_position += 1
    
    return pair_counts, pair_first_pos
    

# 合并操作
def _merge_pair(
        token_sequences: list[list[int]],
        pair_to_merge: tuple[int, int],
        new_token_id: int,
        ) -> None:
    for tokens in token_sequences:
        i = 0
        while i < len(tokens) - 1:
            if (tokens[i], tokens[i+1]) == pair_to_merge:
                tokens[i] = new_token_id
                del tokens[i+1]
            else:
                i += 1
         
