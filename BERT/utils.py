import torch
import subprocess
from dataclasses import dataclass

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
    def __init__(self, name, scale=1):
        self.train_x, self.val_x = [], []
        self.train_loss, self.val_loss = [], []
        self.val_loss_bags = {}
        self.name = name
        self.scale = scale
    
    def log_train(self, ix, loss):
        self.train_x.append(ix)
        self.train_loss.append(loss)
        
    def log_val(self, train_ix, val_ix, loss):
        if train_ix in self.val_loss_bags.keys():
            self.val_loss_bags[train_ix].append(loss)
        else:
            self.val_loss_bags[train_ix] = [loss]
        
    def get_val_loss(self, train_ix):
        self.val_x.append(train_ix)
        val_loss = sum(self.val_loss_bags[train_ix]) / len(self.val_loss_bags[train_ix])
        self.val_loss.append(val_loss)
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
        return self.__dict__.copy()

    def load_state_dict(self, state):
        # Update the instance attributes with those saved in the checkpoint.
        self.__dict__.update(state)

def save_checkpoint(model, optimizer, train_epoch, MLMloss, NSPloss, model_name):
    checkpoint_path = f"checkpoint_latest_{model_name}.pth"
    checkpoint = {
        'train_epoch': train_epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'MLMloss_state': MLMloss.state_dict(),
        'NSPloss_state': NSPloss.state_dict(),
        'model_name': model_name
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")
    return checkpoint_path

def load_checkpoint(model, optimizer, checkpoint_path, MLMloss, NSPloss):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    train_epoch = checkpoint['train_epoch']
    if MLMloss:
        MLMloss.load_state_dict(checkpoint['MLMloss_state'])
    if NSPloss:
        NSPloss.load_state_dict(checkpoint['NSPloss_state'])
    model_name = checkpoint['model_name']
    print(f"Checkpoint loaded from {checkpoint_path}, resuming at epoch {train_epoch}")
    return train_epoch, model_name
