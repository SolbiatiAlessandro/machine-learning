
import torch

# Check if CUDA is available
if torch.cuda.is_available():
    # Get the number of available CUDA devices
    num_devices = torch.cuda.device_count()
    print("Number of CUDA devices:", num_devices)

    # Get information about the current device
    current_device_id = torch.cuda.current_device()
    print("Current device ID:", current_device_id)

    # Get the name of the current device
    device_name = torch.cuda.get_device_name(current_device_id)
    print("Device name:", device_name)

    # Get detailed device properties
    device_props = torch.cuda.get_device_properties(current_device_id)
    print("Device properties:", device_props)

    # Get memory information
    memory_allocated = torch.cuda.memory_allocated(current_device_id)
    memory_reserved = torch.cuda.memory_reserved(current_device_id)
    print("Memory allocated:", memory_allocated)
    print("Memory reserved:", memory_reserved)
else:
    print("CUDA is not available.")

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

hyperparams = {
    'block_size': 16,
    'dmodel': 32,
    'hidden_layer': 2048,
    'batch_size': 1024*8,
    'epochs': 100000
}

# read it in to inspect it
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    
    
# other tokenizer can use 
# https://github.com/openai/tiktoken
# https://github.com/google/sentencepiece 

characters = set()
for c in text:
    characters.add(c)
vocab_size = len(characters)
int_to_characters = {i: c for i, c in enumerate(characters)}
characters_to_int = {c: i for i, c in enumerate(characters)}

# could also be done w torch.stack
data = [characters_to_int[c] for c in text]
n = int(0.9 * len(data))
train_data = torch.tensor(data[:n], device=device)
val_data = torch.tensor(data[n:], device=device)

torch.manual_seed(1337)
batch_size = hyperparams['batch_size']
block_size = hyperparams['block_size']

def get_batch(split , batch_size=batch_size):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(0, len(data) - block_size, (batch_size,), device=device)
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y


import torch.nn as nn
import torch.nn.functional as F
import math

block_size = hyperparams['block_size']
dmodel = hyperparams['dmodel']
hidden_layer = hyperparams['hidden_layer']

class PositionalEmbeddings(nn.Module):
    def __init__(self, block_size, dmodel, vocab_size, device=device):
        super().__init__()
        positional_embeddings = torch.zeros(block_size, dmodel, device=device)
        for pos in range(block_size):
            for i in range(dmodel // 2):
                denominator = 10000 ** (2 * i / dmodel)
                positional_embeddings[pos, 2*i]   = math.sin(pos / denominator)
                positional_embeddings[pos, 2*i+1] = math.cos(pos / denominator)
        self.register_buffer("positional_embeddings", positional_embeddings)
        self.dmodel = dmodel
        
        self.embeddings = nn.Embedding(vocab_size, dmodel, device=device)
        self.dropout = nn.Dropout(0.10)
        
    def forward(self, x):
        self.out = (self.embeddings(x) + self.positional_embeddings) * math.sqrt(self.dmodel)
        self.out = self.dropout(self.out)
        return self.out

class ResidualMLPLayer(nn.Module):
    def __init__(self, fan_in, hidden_layer_size,device=device):
        # output is going to be fan_in * 2
        super().__init__()
        self.l1 = nn.Linear(fan_in, hidden_layer_size, device=device)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_layer_size, fan_in, device=device)
        self.layer_norm = nn.LayerNorm(fan_in, device=device)
        self.dropout = nn.Dropout(p=0.10)
    
    def forward(self, x):
        mlp = self.l1(x)
        mlp = self.relu(mlp)
        mlp = self.l2(mlp)
        mlp = self.dropout(mlp)
        x = x + mlp
        x = self.layer_norm(x)
        self.out = x
        return self.out

    
class ScaledDotProductAttention(nn.Module):
    
    def __init__(self, fan_in, query_size = 64, value_size = 64, device=device):
        # fan_in = embedding_dim
        super().__init__()
        self.query_layer = nn.Linear(fan_in, query_size, device=device)
        self.key_layer = nn.Linear(fan_in, query_size, device=device)
        self.value_layer = nn.Linear(fan_in, value_size, device=device)
        self.query_size = query_size
        self.flatten = nn.Flatten()
        
    def forward(self, x):
        query = self.query_layer(x)
        key = self.key_layer(x)
        attention = (query @ key.transpose(2, 1)) / math.sqrt(self.query_size)
        mask = torch.tril(torch.ones_like(attention, dtype=torch.bool))
        masked_attention = attention.masked_fill(~mask, float('-inf'))
        masked_attention = F.softmax(masked_attention, dim=2)
        
        value = self.value_layer(x)
        x = (masked_attention @ value)
        self.out = x
        return self.out
    
class MultiHeadAttention(nn.Module):
    
    def __init__(self, fan_in, heads=4, query_size = 64, value_size = 64, device=device):
        super().__init__()
        self.value_size = value_size
        self.heads = heads
        self.attention_heads = nn.ModuleList([
            ScaledDotProductAttention(fan_in, query_size, value_size, device) for _ in range(heads)])
        self.linear = nn.Linear(value_size * heads, fan_in, device=device)
        self.norm = nn.LayerNorm(fan_in, device=device)
        self.dropout = nn.Dropout(p=0.10)
        
    def forward(self, x):
        out = [attention_head(x) for attention_head in self.attention_heads]
        out = torch.cat(out, dim=2)
        out = self.linear(out)
        out = self.dropout(out)
        out = out + x # add
        out = self.norm(out) # norm
        self.out = out
        return self.out    

class Transformer(nn.Module):
    def __init__(self, vocab_size, dmodel, device=device):
        super().__init__()
        self.sequential = nn.Sequential(
            PositionalEmbeddings(block_size, dmodel, vocab_size, device=device),
            MultiHeadAttention(dmodel),
            ResidualMLPLayer(dmodel, hidden_layer, device=device),
            MultiHeadAttention(dmodel),
            ResidualMLPLayer(dmodel, hidden_layer, device=device),
            nn.Linear(dmodel, vocab_size, device=device)
        )
        
    def forward(self, x, targets=None):
        logits = self.sequential(x) # B, T, C  
        if targets is None: return logits
        B, T, C = logits.shape
        logits = logits.view(B*T, C)
        target = targets.view(B*T)
        loss = F.cross_entropy(logits, target)
        return logits, loss
    
    def generate(self, idx, max_token_generated):
        #        idx = torch.zeros((1, 1), dtype=torch.long, device=device)
        for _ in range(max_token_generated):
            logits = model.forward(idx[:, -8:])
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=1)
            pred = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, pred), dim=1)
        return idx

model = Transformer(vocab_size, dmodel, device=device)
print(f"parameters = {sum(len(p) for p in model.parameters())}")
print(model)
        

print("\n\n-- Generation Before Training --")
idx = torch.ones((1, block_size), dtype=torch.long, device=device)
print("".join(int_to_characters[x.item()] for x in model.generate(
    idx, 
    1000
)[0]))

print("\n\n-- Training --")
epochs = 100000
# lossi, vlossi = [], []
lossi = []

optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

#xval, yval = get_batch('val', batch_size=len(val_data))

#######################
# FIRST HOUR
#######################

# 1+ sec per epoch
one_hour_epochs = 3000

for epoch in range(one_hour_epochs):
    # forward
    xb, yb = get_batch('train')
    logits, loss = model(xb, targets=yb)
    lossi.append(loss.item())
    
    # backward
    optimizer.zero_grad()
    loss.backward()
    
    # update
    optimizer.step()
    
    if epoch % 10 == 0:
        
        with torch.no_grad():
            _vloss = []
            for _ in range(1):
                xval, yval = get_batch('val')
                _, vloss = model(xval, yval)
                _vloss.append(vloss.item())
            vloss = sum(_vloss)/len(_vloss)
            vlossi.append(vloss)
            print(f"{epoch}: training loss = {loss.item():.4f}, validation loss = {vloss:.4f}")

# Create a checkpoint dictionary
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': lossi,
}

# Save the checkpoint to a file
torch.save(checkpoint, 'model_checkpoint_18_Feb_A_3000_epochs.pth')


print("\n\n-- Generation After Training --")
idx = torch.ones((1, block_size), dtype=torch.long, device=device)
print("".join(int_to_characters[x.item()] for x in model.generate(
    idx, 
    1000
)[0]))


#######################
# SECOND HOUR
#######################

for epoch in range(one_hour_epochs):
    # forward
    xb, yb = get_batch('train')
    logits, loss = model(xb, targets=yb)
    lossi.append(loss.item())
    
    # backward
    optimizer.zero_grad()
    loss.backward()
    
    # update
    optimizer.step()
    
    if epoch % 10 == 0:
        
        with torch.no_grad():
            _vloss = []
            for _ in range(1):
                xval, yval = get_batch('val')
                _, vloss = model(xval, yval)
                _vloss.append(vloss.item())
            vloss = sum(_vloss)/len(_vloss)
            vlossi.append(vloss)
            print(f"{epoch}: training loss = {loss.item():.4f}, validation loss = {vloss:.4f}")


# Create a checkpoint dictionary
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': lossi,
}

# Save the checkpoint to a file
torch.save(checkpoint, 'model_checkpoint_18_Feb_A_6000_epochs.pth')


print("\n\n-- Generation After Training --")
idx = torch.ones((1, block_size), dtype=torch.long, device=device)
print("".join(int_to_characters[x.item()] for x in model.generate(
    idx, 
    1000
)[0]))


#######################
# THIRD HOUR
#######################


for epoch in range(one_hour_epochs):
    # forward
    xb, yb = get_batch('train')
    logits, loss = model(xb, targets=yb)
    lossi.append(loss.item())
    
    # backward
    optimizer.zero_grad()
    loss.backward()
    
    # update
    optimizer.step()
    
    if epoch % 10 == 0:
        
        with torch.no_grad():
            _vloss = []
            for _ in range(1):
                xval, yval = get_batch('val')
                _, vloss = model(xval, yval)
                _vloss.append(vloss.item())
            vloss = sum(_vloss)/len(_vloss)
            vlossi.append(vloss)
            print(f"{epoch}: training loss = {loss.item():.4f}, validation loss = {vloss:.4f}")

# Create a checkpoint dictionary
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': lossi,
}

# Save the checkpoint to a file
torch.save(checkpoint, 'model_checkpoint_18_Feb_A_9000_epochs.pth')


print("\n\n-- Generation After Training --")
idx = torch.ones((1, block_size), dtype=torch.long, device=device)
print("".join(int_to_characters[x.item()] for x in model.generate(
    idx, 
    1000
)[0]))

