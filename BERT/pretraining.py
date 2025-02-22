from dataclasses import dataclass, field
import uuid
import torch
import torch.nn as nn
import torch.nn.functional as F
from random import random
import math

from dataloader import BERTDataLoader
from model import BERT
from utils import device, get_free_gpu_memory, LossLogs, save_checkpoint, load_checkpoint

print(f"[pretraining.py] Available GPU memory: {get_free_gpu_memory()[0]:,} MB")

@dataclass
class BERTConfig:
    # model is going to be (BERT_batch_size, input_size, embedding_size or hidden_layer_size)
    BERT_batch_size = 64
    batch_size = BERT_batch_size * 2
    block_size = 40
    embedding_size = 256*2
    vocab_size = 2220 # new tokens: 2215, 2216, 2217, 2218 from dataloader
    input_size = 2 * block_size + 3
    hidden_layer_size = 1024
    attention_heads = 8
    attention_size = embedding_size // attention_heads
    train_epochs = 10000
    val_epochs = 30
    val_interval = 100
    transformer_blocks = 10
    model_name: str = field(default_factory=lambda: f"BERT_{uuid.uuid4()}")
    checkpoint_interval = 1000

                            
config = BERTConfig()                           
model = BERT(config)
model.to(device)

MLMloss = LossLogs("MLM", scale=0.1)
NSPloss = LossLogs("NSP")
data_loader = BERTDataLoader(config)
print(f"[pretraining.py] BERT ready for pretraining with {sum(p.numel() for p in model.parameters()):,} parameters")

print(f"[pretraining.py] Available GPU memory: {get_free_gpu_memory()[0]:,} MB")

optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
prev_train_epoch = 0

for train_epoch in range(config.train_epochs):
    
    train_epoch += prev_train_epoch
    
    optimizer.zero_grad()
    
    x, y_MLM, loss_mask_MLM, y_NSP = data_loader.next_batch(device=device)
    pred_NSP, logits_MSM, loss_NSP, loss_MLM = model(x, y_NSP=y_NSP, y_MLM=y_MLM, loss_mask_MLM=loss_mask_MLM)
    loss = loss_NSP + loss_MLM
    loss.backward()
    optimizer.step()
    MLMloss.log_train(train_epoch, loss_MLM.item())
    NSPloss.log_train(train_epoch, loss_NSP.item())
    
    if train_epoch % config.val_interval == 0:
        
        with torch.no_grad():
            model.eval()
            for val_epoch in range(config.val_epochs):
                x, y_MLM, loss_mask_MLM, y_NSP = data_loader.next_batch(device=device, mode="eval")
                pred_NSP, logits_MSM, loss_NSP, loss_MLM = model(x, y_NSP=y_NSP, y_MLM=y_MLM, loss_mask_MLM=loss_mask_MLM)
                MLMloss.log_val(train_epoch, val_epoch, loss_MLM.item())
                NSPloss.log_val(train_epoch, val_epoch, loss_NSP.item())
            model.train()
        
        
        log_string = f"[{train_epoch}/{config.train_epochs}]"
        log_string += f"[MLM losses: train={loss_MLM.item():.3f},val={MLMloss.get_val_loss(train_epoch):.3f}]"
        log_string += f"[NSP losses: train={loss_NSP.item():.3f},val={NSPloss.get_val_loss(train_epoch):.3f}]"
        print(log_string)
    
    if train_epoch % config.checkpoint_interval == 0:
        checkpoint_path = save_checkpoint(model, optimizer, train_epoch, MLMloss, NSPloss, config.model_name)
        
    
# 1 attention only [4500/10000][MLM losses: train=6.047,val=5.977][NSP losses: train=0.708,val=0.690]
# 1 attention + 1 MLP [4500/10000][MLM losses: train=6.124,val=5.418][NSP losses: train=0.687,val=0.695]
# 1 block @2M params [3000/10000][MLM losses: train=3.718,val=3.398][NSP losses: train=0.688,val=0.695]
# 4 blocks @4M params [3000/10000][MLM losses: train=2.516,val=2.724][NSP losses: train=0.645,val=0.667]
# 8 blocks @20M params [5500/10000][MLM losses: train=2.562,val=2.664][NSP losses: train=0.478,val=0.485]
