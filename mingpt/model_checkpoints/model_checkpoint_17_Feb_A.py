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
        
    def forward(self, x):
        self.out = (self.embeddings(x) + self.positional_embeddings) * math.sqrt(self.dmodel)
        return self.out

class ResidualMLPLayer(nn.Module):
    def __init__(self, fan_in, hidden_layer_size,device=device):
        # output is going to be fan_in * 2
        super().__init__()
        self.l1 = nn.Linear(fan_in, hidden_layer_size, device=device)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_layer_size, fan_in, device=device)
        self.layer_norm = nn.LayerNorm(fan_in, device=device)
    
    def forward(self, x):
        mlp = self.l1(x)
        mlp = self.relu(mlp)
        mlp = self.l2(mlp)
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
    
    def __init__(self, fan_in, heads=2, query_size = 64, value_size = 64, device=device):
        super().__init__()
        self.value_size = value_size
        self.heads = heads
        self.attention_heads = nn.ModuleList([
            ScaledDotProductAttention(fan_in, query_size, value_size, device) for _ in range(heads)])
        self.linear = nn.Linear(value_size * heads, fan_in, device=device)
        self.norm = nn.LayerNorm(fan_in, device=device)
        
    def forward(self, x):
        out = [attention_head(x) for attention_head in self.attention_heads]
        out = torch.cat(out, dim=2)
        out = self.linear(out) + x # add
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
        