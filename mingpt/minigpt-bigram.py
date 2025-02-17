
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
    'block_size': 8,
    'embedding_dim': 10,
    'hidden_layer': 10*10*5,
    'batch_size': 128,
    'epochs': 6000
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

xval, yval = get_batch('val', batch_size=len(val_data))

import torch.nn as nn
import torch.nn.functional as F

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size, device=device)
        
    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx) # B, T, C  
        if targets is None: return logits
        B, T, C = logits.shape
        logits = logits.view(B*T, C)
        target = targets.view(B*T)
        loss = F.cross_entropy(logits, target)
        return logits, loss
    
    def generate(self, idx, max_token_generated):
        #        idx = torch.zeros((1, 1), dtype=torch.long, device=device)
        for _ in range(max_token_generated):
            logits = model.forward(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=1)
            pred = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, pred), dim=1)
        return idx
        
        
model = BigramLanguageModel(vocab_size)
for p in model.parameters():
    p.require_grad = True
    #print(len(p))
    
print("\n\n-- Generation Before Training --")
print("".join(int_to_characters[x.item()] for x in model.generate(
    torch.zeros((1, 1), dtype=torch.long, device=device), 
    100
)[0]))

print("\n\n-- Training --")
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = hyperparams['epochs']
lossi, vlossi = [], []

#xval, yval = get_batch('val', batch_size=len(val_data))

for epoch in range(epochs):
    # forward
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    lossi.append(loss.item())
    
    # backward
    optimizer.zero_grad()
    loss.backward()
    
    # update
    optimizer.step()
    
    if epoch % 200 == 0:
        
        with torch.no_grad():
            _, vloss = model(xval, yval)
            vlossi.append(vloss.item())
            print(f"{epoch}: training loss = {loss.item():.4f}, validation loss = {vloss.item():.4f}")


print("\n\n-- Generation After Training --")
print("".join(int_to_characters[x.item()] for x in model.generate(
    torch.zeros((1, 1), dtype=torch.long, device=device), 
    100
)[0]))