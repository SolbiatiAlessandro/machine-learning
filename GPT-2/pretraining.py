import os
import sys
import torch
from torch.distributed import init_process_group, destroy_process_group

ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    print(f"torch.distributed: ddp_rank={ddp_rank}, ddp_local_rank={ddp_local_rank}, ddp_world_size={ddp_world_size}")
    master_process = ddp_rank == 0
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)


import subprocess
import sys


subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "datasets"])


from GPT import GPT, GPTConfig
from dataloader import DataLoader
from evaluation import evaluate_downstream_cbt_with_probs
from generate import sample_generations
from utils import device, get_free_gpu_memory, LossLogs, save_checkpoint, load_checkpoint

config = GPTConfig()
config.mini_batch_size = 64
config.total_batch_size = 64*64
config.block_size = 1024
config.epochs = 1000000
config.validation_frequency = 10
config.validation_epochs = 5
config.dataset = "wikitext"
config.tokenizer_name = "wikitext2_18k"
config.downstream_evals_iterations = 300
config.downstream_evals_frequency = 100

"""
100k tokens/s
6h training
6M per min
360M per h
2B tokens

> All models were trained for a total of 300 billion tokens.
To train all versions of GPT-3, we use Adam with β1 = 0.9, β2 = 0.95, and  = 10−8
, we clip the global norm of the
gradient at 1.0, and we use cosine decay for learning rate down to 10% of its value, 
> over 260 billion tokens = 86%
(after 260 billion tokens, training continues at 10% of the original learning rate). 
There is a linear LR warmup 
> over the first 375 million tokens = 1%
We also gradually increase the batch size linearly from a small value (32k tokens) to the full value over
the first 

> 4-12 billion tokens of training, depending on the model size
= 5%
"""

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
#config.vocab_size = data_loader.vocab_size
config.vocab_size = 17792

model = GPT(config)
model.to(device)

model = torch.compile(model)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

torch.set_float32_matmul_precision("high")

optimizer = model.configure_optimizer(weight_decay=0.1, learning_rate=3e-4, device=device)
# train_losses, val_losses = [], []


NTPloss = LossLogs("NTP", wandb=wandb)

config.epochs = 30000

import math

max_lr = 6e-4
min_lr = max_lr * 0.1
# using tokens instead of steps
warmup_step = 2e7 # 20M tokens
warmup_batch = 1e9 # 1B tokens
asymptotic_step = 1e9 # 1B tokens

def get_lr(step):
    # Linear warmup for the first 10% of steps
    if step < warmup_step:
        return max_lr * (step+1) / warmup_step
    
    # Cosine decay from max_lr to min_lr (10% of max_lr) over remaining steps
    # The decay continues until asymptotic_step (50% of total epochs)
    if step < asymptotic_step:
        decay_ratio = (step - warmup_step) / (asymptotic_step - warmup_step)
        # Cosine decay formula: min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(π * decay_ratio))
        return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * decay_ratio))
    
    # After asymptotic_step, maintain the min_lr (10% of max_lr)
    return min_lr

    
def get_batch_size(step, start_batch_size = 64):
    if step < warmup_batch:
        return start_batch_size + int((config.total_batch_size - start_batch_size) * (step+1) / warmup_batch)
    return config.total_batch_size


#training loop --------------


from time import time
train_epoch = 0
tokens = 1
optimizer.zero_grad()
while True:
    
    train_epoch += 1
    total_batch_iteration_time = 0
    batch_size = get_batch_size(tokens)
    epoch_train_loss = 0.0
    t0 = time()
    mini_batch_steps = int(batch_size / (config.mini_batch_size * ddp_world_size))
    for mini_batch_step in range(mini_batch_steps):
        
        
    
        bb = config.mini_batch_size * config.block_size
        X, y = data_loader.next_batch(device=device, batch_size=config.mini_batch_size)
        t01 = time()
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits = model(X)
            train_loss = F.cross_entropy(logits.view(bb, -1), y.view(bb))
            
        train_loss = train_loss / mini_batch_steps
        epoch_train_loss += train_loss.detach()
        train_loss.backward()  
        
        torch.cuda.synchronize()
        
    t1 = time()
    dt = (t1 - t0) * 1000 
    tokens += X.shape[0]*X.shape[1]*mini_batch_steps
    tps = 1000*X.shape[0]*X.shape[1]*mini_batch_steps/dt
        
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = get_lr(tokens)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    optimizer.step()
    optimizer.zero_grad()

    infra_metrics = {
        'infra/iteration_time(ms)': dt,
        'infra/tokens_per_second': tps,
        'training/tokens': tokens,
        'training/norm': norm,
        'training/batch_size': batch_size,
        'training/learning_rate': lr,
        'training/mini_batch_steps': mini_batch_steps,
    }
    NTPloss.log_train(train_epoch, epoch_train_loss.item(), infra_metrics=infra_metrics)


    loss_string = f"[{train_epoch}*{mini_batch_steps}/{config.epochs}] train_loss={epoch_train_loss.item():.3f}"
    infra_string = ", ".join([f"{k}={v:.6f}" for k,v in infra_metrics.items()])
    print(loss_string + infra_string)



    if train_epoch >= config.downstream_evals_frequency and (train_epoch % config.downstream_evals_frequency == 0 or train_epoch % (config.downstream_evals_frequency + 1) == 0):
        accuracy, _, skipped = evaluate_downstream_cbt_with_probs(
            model=model,
            tokenizer=data_loader.tokenizer,
            device=device,
            dataset_split="validation",
            verbose=False,
            max_context_length=config.block_size - 1,
            max_examples=config.downstream_evals_iterations  
        )
        wandb.log({
            'downstream/children_book_text_accuracy': accuracy,
            'downstream/children_book_text_skipped': skipped,
        })

        generated_evals = sample_generations(
            model, 
            data_loader.tokenizer, 
            config, 
            device=device,
            wandb_obj=wandb,
            iteration=train_epoch)


    if (train_epoch + 1) % config.validation_frequency == 0:
        model.eval()
        with torch.no_grad():
            epoch_val_losses = []
            for val_epoch in range(config.validation_epochs):
                X, y = data_loader.next_batch(mode="eval", device=device, batch_size=config.mini_batch_size)
                logits = model(X)
                val_loss = F.cross_entropy(logits.view(bb, -1), y.view(bb))
                NTPloss.log_val(train_epoch, val_epoch, val_loss.item())

                del X, y, logits
                gc.collect()
                torch.cuda.empty_cache()


                model.train()
                loss_string = f"[{train_epoch}/{config.epochs}] train_loss={epoch_train_loss.item():.3f},val_loss={NTPloss.get_val_loss(train_epoch):.3f}, "
                loss_string += f", norm={norm}"
                infra_string = ", ".join([f"{k}={v:.3f}" for k,v in infra_metrics.items()])
                print(loss_string + infra_string)
