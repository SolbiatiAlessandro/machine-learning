import torch
import sys
import os

# Assuming your notebook's working directory is set such that ../llm-tokenizer is reachable:
tokenizer_dir = os.path.abspath(os.path.join(os.getcwd(), '../llm_tokenizer'))
if tokenizer_dir not in sys.path:
    sys.path.insert(0, tokenizer_dir)

import BPETokenizer  # Now you should be able to import it

class DataLoader:
    def __init__(self, config):
        
        self.config = config

        with open('input.txt', 'r') as f:
            text = f.read()
        len(text)

        
        tokenizer = BPETokenizer.Tokenizer(text, encoding_vocab_size=2000, raw_tokens=False)
        tokenizer.load_from_file()
        encoded_dataset = tokenizer.encode(text, raw_tokens=False)
        print(f"max vocabulary size={max(encoded_dataset)}, compression ratio={len(encoded_dataset) / len(text)}")
        split = int(len(encoded_dataset) * 0.80)
        self.train_data =  torch.tensor(encoded_dataset[:split])
        self.val_data = torch.tensor(encoded_dataset[split+config.block_size:])
        print(f"train_data.shape={self.train_data.shape}, val_data.shape={self.val_data.shape}")
        
        self.train_data_ix = 0
        self.val_data_ix = 0
        self.batch_step = self.config.batch_size * self.config.block_size 
        
    def next_batch(self, mode="train", device='cpu'):
        """ mode=["train", "eval"] """
        if mode == "train":
            x, y = self._next_batch_train()
        else:
            x, y = self._next_batch_eval()
        return x.to(device), y.to(device)
    
    def _next_batch_train(self):
        
        data = self.train_data
        ix = self.train_data_ix 
        
        buf = data[ix:ix+self.batch_step + 1]     
        x = buf[:-1].view(self.config.batch_size, self.config.block_size)
        y = buf[1:].view(self.config.batch_size, self.config.block_size)
        
        self.train_data_ix += self.batch_step 
        if self.train_data_ix + self.batch_step + 1 > len(self.train_data):
            self.train_data_ix = 0
        
        return x, y
    
    def _next_batch_eval(self):
        
        data = self.val_data
        ix = self.val_data_ix 
        
        buf = data[ix:ix+self.batch_step + 1]     
        x = buf[:-1].view(self.config.batch_size, self.config.block_size)
        y = buf[1:].view(self.config.batch_size, self.config.block_size)
        
        self.val_data_ix += self.batch_step 
        if self.val_data_ix + self.batch_step + 1 > len(self.val_data):
            self.val_data_ix = 0
        
        return x, y
    

        
        

