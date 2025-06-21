import torch
from torch import nn

class Classifier(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        logits = self.model(x)
        probs = torch.sigmoid(logits.squeeze()).squeeze(-1)
        return probs