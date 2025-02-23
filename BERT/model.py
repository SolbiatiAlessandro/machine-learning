import torch
import torch.nn as nn
import torch.nn.functional as F


class InputRepresentation(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embeddings = nn.Embedding(config.vocab_size, config.embedding_size)
        self.segment_embeddings = nn.Embedding(2, config.embedding_size)
        self.position_embeddings = nn.Embedding(config.input_size + 1, config.embedding_size)
        self.register_buffer('segment_index', 
                             torch.tensor(sum([
                                 [0] * (config.block_size + 1), 
                                 [0], 
                                 [1] * (config.block_size + 1)], [])))
        self.register_buffer('position_index', 
                             torch.arange(config.input_size))
    
    def forward(self, x):
        if self.training:
            token_embeddings = self.token_embeddings(x)
            segment_embeddings = self.segment_embeddings(self.segment_index)
            position_embeddings = self.position_embeddings(self.position_index)
        else:
            # predicting one sentence only
            token_embeddings = self.token_embeddings(x)
            segment_embeddings = self.segment_embeddings(torch.tensor([0] * x.shape[1], device=x.device))
            position_embeddings = self.position_embeddings(torch.arange(x.shape[1]).to(x.device))
        x = position_embeddings + token_embeddings + segment_embeddings
        return x
            
            
    
    
    
class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.key = nn.Linear(config.embedding_size, config.attention_size)
        self.query = nn.Linear(config.embedding_size, config.attention_size)
        self.value = nn.Linear(config.embedding_size, config.attention_size)
        
    def forward(self, x):
        key = self.key(x)
        query = self.query(x)
        attention = F.softmax(key @ query.transpose(1, 2), dim=2)
        value = self.value(x)
        return attention @ value
    
class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = nn.ModuleList([SelfAttention(config) for _ in range(config.attention_heads)])
        self.linear = nn.Linear(config.embedding_size, config.embedding_size)
        
    def forward(self, x):
        heads = []
        for head in self.heads:
            heads.append(head(x))
        x = torch.cat(heads, dim=2)
        x = self.linear(x)
        return x
    

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.mlp = nn.ModuleList([
            nn.LayerNorm(config.embedding_size),
            nn.Linear(config.embedding_size, config.hidden_layer_size),
            nn.ReLU(),
            nn.Linear(config.hidden_layer_size, config.embedding_size)
        ])
        self.attention_norm = nn.LayerNorm(config.embedding_size)
        
    def forward(self, x):
        x = x + self.attention(self.attention_norm(x))
        mlp = self.mlp[0](x)
        for layer in self.mlp[1:]:
            mlp = layer(mlp)
        return x + mlp
    
    
class BERT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_representation = InputRepresentation(config)
        self.transformer_blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.transformer_blocks)])
        
        self.MLM_output = nn.Linear(config.embedding_size, config.vocab_size)
        self.NSP_output = nn.Linear(config.embedding_size, 1)
        
    def forward(self, x, y_NSP=None, y_MLM=None, loss_mask_MLM=None):
        """return (pred_NSP, logits_MSM, loss_NSP, loss_MLM)"""
        x = self.input_representation(x)
        for transformer in self.transformer_blocks:
            x = transformer(x)
            
        if not self.training:
            return x
        
        pred_NSP = F.sigmoid(self.NSP_output(x[:, 0, :]))
        logits_MSM = self.MLM_output(x)
        
        loss_NSP, loss_MLM = None, None
        if y_NSP is not None:
            loss_NSP = F.binary_cross_entropy(pred_NSP.view(-1), y_NSP.to(torch.float))
            
        if y_MLM is not None and loss_mask_MLM is not None:
            loss_MLM = F.cross_entropy(
                logits_MSM.reshape(self.config.BERT_batch_size * self.config.input_size, -1), 
                y_MLM.reshape(-1),
                reduction='none')
            loss_MLM = (loss_MLM * loss_mask_MLM.view(-1)).sum()
        
        
        return pred_NSP, logits_MSM, loss_NSP, loss_MLM