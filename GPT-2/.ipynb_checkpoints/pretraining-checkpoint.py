from GPT import GPT, GPTConfig
from dataloader import DataLoader
from utils import device, get_free_gpu_memory, LossLogs, save_checkpoint, load_checkpoint

config = GPTConfig()
config.batch_size = 32
config.block_size = 64
config.epochs = 10000
config.validation_frequency = 20
config.validation_epochs = 1

import wandb
import random
import dataclasses

# start a new wandb run to track this script
wandb.init(
    # set the wandb entity where your project will be logged (generally your team name)
    entity="lessandro",

    # set the wandb project where this run will be logged
    project="GPT2",

    # track hyperparameters and run metadata
    config=dataclasses.asdict(config)
)

import torch
import torch.nn.functional as F
import gc

print(f"[pretraining.py] Available GPU memory: {get_free_gpu_memory()[0]:,} MB")

device = 'cuda' if torch.cuda.is_available else 'cpu'

if torch.cuda.is_available():
    current_device = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(current_device)
    device_props = torch.cuda.get_device_properties(current_device)
    memory_summary = torch.cuda.memory_summary(device=current_device, abbreviated=True)
    
    print("Current device index:", current_device)
    print("Running on GPU:", device_name)
    print("GPU properties:")
    print("  - Compute Capability:", f"{device_props.major}.{device_props.minor}")
    print("  - Total Memory:", f"{device_props.total_memory / (1024**3):.2f} GB")
    print("  - Multiprocessor Count:", device_props.multi_processor_count)
    print("  - Max Threads per Multiprocessor:", device_props.max_threads_per_multi_processor)
else:
    print("CUDA is not available, running on CPU.")

data_loader = DataLoader(config)

model = GPT(config)
model.to(device)
print(sum(p.numel() for p in model.parameters()))

optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005)
# train_losses, val_losses = [], []
bb = config.batch_size * config.block_size

NTPloss = LossLogs("NTP", wandb=wandb)



for train_epoch in range(config.epochs):
    optimizer.zero_grad()
    X, y = data_loader.next_batch(device=device)
    logits = model(X)
    train_loss = F.cross_entropy(logits.view(bb, -1), y.view(bb))
    train_loss.backward()
    optimizer.step()
    
    #print(train_loss)
    NTPloss.log_train(train_epoch, train_loss.item())
    
    
    del X, y, logits
    gc.collect()
    torch.cuda.empty_cache()
    
    if train_epoch % config.validation_frequency == 0:
        model.eval()
        with torch.no_grad():
            epoch_val_losses = []
            for val_epoch in range(config.validation_epochs):
                X, y = data_loader.next_batch(mode="eval", device=device)
                logits = model(X)
                val_loss = F.cross_entropy(logits.view(bb, -1), y.view(bb))
                NTPloss.log_val(train_epoch, val_epoch, val_loss.item())
                
                del X, y, logits
                gc.collect()
                torch.cuda.empty_cache()
                
            
            model.train()
    
        
            print(f"train={train_loss.item():.3f},val={NTPloss.get_val_loss(train_epoch):.3f}")

# train=1.163,val=5.416
# [880/10000] train_loss=3.59, val_loss=4.00