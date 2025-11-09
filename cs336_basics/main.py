from train_bpe import BPE

BPE.train_bpe(input_path="../data/owt_train.txt", 
              vocab_size=32000, 
              special_tokens=["<|endoftext|>"])