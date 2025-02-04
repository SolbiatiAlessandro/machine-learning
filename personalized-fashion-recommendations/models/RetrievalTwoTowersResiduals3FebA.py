import torch
import torch.nn as nn
import torch.optim as optim

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x):
        identity = x  # Save the input for the skip connection
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out + identity)  # Add skip connection
        return out

class CustomerTower(nn.Module):
    def __init__(self, customer_input_dim, embedding_dim, num_blocks=2):
        super(CustomerTower, self).__init__()
        self.initial_fc = nn.Linear(customer_input_dim, 1024)
        self.initial_relu = nn.ReLU()
        self.residual_blocks = ResidualBlock(1024)
        self.final_fc = nn.Linear(1024, embedding_dim)

    def forward(self, x):
        x = self.initial_fc(x)
        x = self.initial_relu(x)
        x = self.residual_blocks(x)
        x = self.final_fc(x)
        return x

class ArticleTower(nn.Module):
    def __init__(self, customer_input_dim, embedding_dim, num_blocks=2):
        super(ArticleTower, self).__init__()
        self.initial_fc = nn.Linear(customer_input_dim, 2048)
        self.initial_relu = nn.ReLU()
        self.residual_blocks = ResidualBlock(2048)
        self.final_fc = nn.Linear(2048, embedding_dim)

    def forward(self, x):
        x = self.initial_fc(x)
        x = self.initial_relu(x)
        x = self.residual_blocks(x)
        x = self.final_fc(x)
        return x

class TwoTowerRecommender(nn.Module):
    def __init__(self, customer_input_dim=13, article_input_dim=500, embedding_dim=32):
        super(TwoTowerRecommender, self).__init__()
        # Customer tower: small MLP for 13 features.
        self.customer_tower = CustomerTower(customer_input_dim, embedding_dim)
        
        # Article tower: deeper MLP for 500 features.
        self.article_tower = ArticleTower(article_input_dim, embedding_dim)
    
    def forward(self, customer_features, article_features):
        customer_emb = self.customer_tower(customer_features)   # [batch, embedding_dim]
        article_emb = self.article_tower(article_features)        # [batch, embedding_dim]
        
        # Compute similarity, e.g., dot product.
        logits = (customer_emb * article_emb).sum(dim=1)
        probabilities = torch.sigmoid(logits)
        return probabilities 
