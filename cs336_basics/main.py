from train_bpe import BPE

BPE.train_bpe(input_path="../data/", 
              vocab_size=32000, 
              special_tokens=["<|endoftext|>"])