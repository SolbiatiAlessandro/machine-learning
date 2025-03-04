import torch
import subprocess
from dataclasses import dataclass
from collections import defaultdict

from dataclasses import dataclass, field
import uuid



def get_free_gpu_memory():
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            encoding="utf-8"
        )
        free_memory = [int(x) for x in output.strip().split("\n")]
        return free_memory
    except Exception as e:
        print("Error querying nvidia-smi:", e)
        return None
    

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


class LossLogs:
    """
    usage
    
    MLMloss.log_train(train_epoch, loss_MLM.item())
    MLMloss.log_val(train_epoch, val_epoch, loss_MLM.item()) 
    
    """
    def __init__(self, name, wandb=None, scale=1):
        self.train_x, self.val_x = [], []
        self.train_loss, self.val_loss = [], []
        self.val_loss_bags = {}
        self.name = name
        self.scale = scale
        self.wandb = wandb
        self.infra = defaultdict(list)
    
    def log_train(self, ix, loss, infra_metrics=None):
        if self.wandb: 
            # print("[LossLogs.log_train] logging wandb")
            self.wandb.log({f"{self.name}_train_loss": loss})
        self.train_x.append(ix)
        self.train_loss.append(loss)
        if infra_metrics:
            if self.wandb:
                self.wandb.log(infra_metrics)
            for k, v in infra_metrics.items():
                self.infra[k].append(v)
        
    def log_val(self, train_ix, val_ix, loss):
        if train_ix in self.val_loss_bags.keys():
            self.val_loss_bags[train_ix].append(loss)
        else:
            self.val_loss_bags[train_ix] = [loss]
        
    def get_val_loss(self, train_ix):
        self.val_x.append(train_ix)
        val_loss = sum(self.val_loss_bags[train_ix]) / len(self.val_loss_bags[train_ix])
        self.val_loss.append(val_loss)
        if self.wandb: self.wandb.log({f"{self.name}_val_loss": val_loss})
        return val_loss
    
    def train_loss_series(self, aggr=100):
        ixs, xs, aggcum = [], [], 0
        for ix in range(len(self.train_x)):
            aggcum += self.train_loss[ix]
            if (ix + 1) % aggr == 0:
                xs.append((aggcum/aggr)*self.scale)
                aggcum = 0
                ixs.append(ix)
        return ixs, xs
    
    def val_loss_series(self):
        return self.val_x, [self.scale * x for x in self.val_loss]
    
    def state_dict(self):
        # Return a shallow copy of all instance attributes.
        # Exclude the wandb object from the state
        state = self.__dict__.copy()
        if 'wandb' in state:
            state['wandb'] = None
        return state

    def load_state_dict(self, state):
        # Update the instance attributes with those saved in the checkpoint.
        self.__dict__.update(state)
        
import os
import torch

def save_checkpoint(model, optimizer, train_epoch, loss_logger, model_name):
    # Ensure the checkpoints folder exists
    checkpoint_dir = "./checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create a filename that includes the epoch number
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{train_epoch}_{model_name}.pth")
    
    checkpoint = { 
        'train_epoch': train_epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss_state': loss_logger.state_dict(),
        'model_name': model_name
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")
    return checkpoint_path

def load_checkpoint(model, optimizer, checkpoint_path, loss_logger):
    checkpoint = torch.load(checkpoint_path)
    
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)

    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    train_epoch = checkpoint['train_epoch']
    if loss_logger:
        loss_logger.load_state_dict(checkpoint['loss_state'])
    model_name = checkpoint['model_name']
    print(f"Checkpoint loaded from {checkpoint_path}, resuming at epoch {train_epoch}")
    return train_epoch, model_name
