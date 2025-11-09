from train_bpe import BPE

BPE.train_bpe(input_path="../data/TinyStoriesV2-GPT4-train.txt", vocab_size=10000, speical_tokens=["<|endoftext|>"])