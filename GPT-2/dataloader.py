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
        """
        name = tinyshakespeare
        input text = tinyshakespeare.txt
        tokenizer = tokenizer_tinyshakespeare.pickle
        """
        
        self.config = config
        dataset_name = config.dataset
        self.DATASET_FOLDER = "./datasets"
       
        self.tokenizer = BPETokenizer.Tokenizer("", encoding_vocab_size=2000, raw_tokens=False, name=config.tokenizer_name)
        self.tokenizer.load_from_file()
        print(f"[DataLoader.__init__] loaded tokenizer {self.tokenizer._filename()}")
        
        self.train_data =  self._load_dataset(f"{dataset_name}_train.txt")
        self.val_data = self._load_dataset(f"{dataset_name}_val.txt")
        print(f"[DataLoader.__init__] train_data.shape={self.train_data.shape}, val_data.shape={self.val_data.shape}")
        
        self.train_data_ix = 0
        self.val_data_ix = 0
        self.batch_step = self.config.batch_size * self.config.block_size 
        
    def _load_dataset(self, filename):
        if self.tokenizer is None: 
            raise Exception("[DataloaderException] Need to load dataset after loading tokenizer")
        filename = f"{self.DATASET_FOLDER}/{filename}"
        with open(filename, 'r') as f:
            text = f.read()
        print(f"[DataLoader._load_dataset] {filename}: size = {len(text)}")
        
        encoded_dataset = self.tokenizer.encode(text, raw_tokens=False)
        print(f"[DataLoader._load_dataset] {filename}: max vocabulary size={max(encoded_dataset)}, compression ratio={len(encoded_dataset) / len(text)}")
        
        return torch.tensor(encoded_dataset, device='cpu')

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
    

        
        

