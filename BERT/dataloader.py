import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from random import random
import math

# Assuming your notebook's working directory is set such that ../llm-tokenizer is reachable:
tokenizer_dir = os.path.abspath(os.path.join(os.getcwd(), '../llm-tokenizer'))
if tokenizer_dir not in sys.path:
    sys.path.insert(0, tokenizer_dir)

import BPETokenizer  # Now you should be able to import it



class DataLoader:
    """next token prediction DataLoader used for GPT like models"""
    def __init__(self, config, name="tinyshakespeare"):
        """
        name = tinyshakespeare
        input text = tinyshakespeare.txt
        tokenizer = tokenizer_tinyshakespeare.pickle
        """
        
        self.config = config
        data_filename = f"{name}.txt"
        with open(data_filename, 'r') as f:
            text = f.read()
        print(f"[DataLoader] {data_filename} size = {len(text)}")

        self.tokenizer = BPETokenizer.Tokenizer(text, encoding_vocab_size=2000, raw_tokens=False, name=name)
        self.tokenizer.load_from_file()
        print(f"[DataLoader] loaded tokenizer {self.tokenizer._filename()}")
        encoded_dataset = self.tokenizer.encode(text, raw_tokens=False)
        print(f"[DataLoader] max vocabulary size={max(encoded_dataset)}, compression ratio={len(encoded_dataset) / len(text)}")
        split = int(len(encoded_dataset) * 0.80)
        self.train_data =  torch.tensor(encoded_dataset[:split])
        self.val_data = torch.tensor(encoded_dataset[split+config.block_size:])
        print(f"[DataLoader] train_data.shape={self.train_data.shape}, val_data.shape={self.val_data.shape}")
        
        self.train_data_ix = 0
        self.val_data_ix = 0
        self.batch_step = self.config.batch_size * self.config.block_size 
        
    def next_batch(self, mode="train", device='cpu'):
        """ mode=["train", "eval"] """
        if mode == "train":
            x, y = self._next_batch_train()
        else:
            x, y = self._next_batch_eval()
        if device:
            return x.to(device), y.to(device)
        return x, y
    
    def _next_batch_train(self):
        
        data = self.train_data
        ix = int(random() * (len(data) - 2*self.batch_step))
        
        buf = data[ix:ix+self.batch_step + 1]     
        x = buf[:-1].view(self.config.batch_size, self.config.block_size)
        y = buf[1:].view(self.config.batch_size, self.config.block_size)
        
        self.train_data_ix += self.batch_step 
        if self.train_data_ix + self.batch_step + 1 > len(self.train_data):
            self.train_data_ix = 0
        
        return x, y
    
    def _next_batch_eval(self):
        
        data = self.val_data
        ix = int(random() * (len(data) - 2*self.batch_step))
        
        buf = data[ix:ix+self.batch_step + 1]     
        x = buf[:-1].view(self.config.batch_size, self.config.block_size)
        y = buf[1:].view(self.config.batch_size, self.config.block_size)
        
        self.val_data_ix += self.batch_step 
        if self.val_data_ix + self.batch_step + 1 > len(self.val_data):
            self.val_data_ix = 0
        
        return x, y
    
class BERTDataLoader(DataLoader):
    """data loader for BERT-like MLM + NSP loss"""
    def __init__(self, config, name="tinyshakespeare"):
        super().__init__(config, name=name)
        self.max_vocab_size = max(self.tokenizer.encoding_map.values())
        self.CLS =self.max_vocab_size + 1
        self.SEP = self.CLS + 1
        self.MASK = self.SEP + 1
        print(f"[BERTDataLoader] new tokens: {self.max_vocab_size}, {self.CLS}, {self.SEP}, {self.MASK}")
        
    def next_batch(self, device='cpu', test=False, mode="train"):
        """return x, y for Masked Language Model (MLM), MLM loss mask, y for Next Sentence Prediction"""
        _x, _ = super().next_batch(device=None, mode=mode)
        x, y_MLM, ix_MLM, y_NSP = [], [], [], []
        assert len(_x) % 2 == 0, "BERTDataLoader batch size should be % 2 == 0"
        for ix in range(int(len(_x) / 2)): 
            x0 = _x[ix].clone()
            if random() < 0.5:
                x1 = _x[ix+1].clone()
                y_NSP.append(torch.tensor(1))
            else:
                __x, _ = super().next_batch(device=None)
                x1 = __x[0].clone()
                y_NSP.append(torch.tensor(0))
            y_MLM.append(self._make_input(x0, x1))
            x0, ix0 = self._mask(x0)
            x1, ix1 = self._mask(x1)
            x.append(self._make_input(x0, x1))
            ix_MLM.append(torch.cat([
                torch.tensor([0]), 
                ix0,
                torch.tensor([0]), 
                ix1,
                torch.tensor([0])
                ]))
            
        if test:
            for i, xx in enumerate(x):
                if y_NSP[i] == 1:
                    a, b = y_MLM[i][1:1+BERTConfig.block_size], _x[i]
                    assert (a == b).all(), (a, b)
                    a, b = y_MLM[i][1+BERTConfig.block_size+1:1+2*BERTConfig.block_size+1], _x[i+1]
                    assert (a == b).all(), (a, b)
                    print(f"[BERTDataLoader] {i} TEST PASSED same sentence")
                    
        # turn indexes of MLM into a average loss mask
        ix_MLM = torch.stack(ix_MLM).to(device)
        ix_MLM = (ix_MLM / ix_MLM.sum()).nan_to_num(0.0)
            
        return (
            torch.stack(x).to(device), 
            torch.stack(y_MLM).to(device), 
            ix_MLM, 
            torch.stack(y_NSP).to(device)
        )
    
    def _make_input(self, x0, x1):
        return torch.cat([
            torch.tensor([self.CLS]), 
            x0,
            torch.tensor([self.SEP]), 
            x1,
            torch.tensor([self.SEP])
        ])
    
    def _mask(self, x):
        ix = []
        for i, v in enumerate(x):
            if random() < 0.15:
                ix.append(torch.tensor(1))
                r2 = random()
                if r2 < 0.80:
                    x[i] = self.MASK
                elif 0.80 <= r2 < 0.90:
                    x[i] = int(random() * (self.max_vocab_size - 100))
            else: ix.append(torch.tensor(0))
        return x, torch.tensor(ix)