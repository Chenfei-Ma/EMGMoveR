import torch.nn as nn
import torch

class TransformerClassifier(nn.Module):
    def __init__(self,input_length, input_dim, num_classes, num_heads=2, num_layers=2, dropout=0.2):
        super(TransformerClassifier, self).__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dropout=dropout),
            num_layers=num_layers
        )
        self.fc = nn.Linear(input_dim * input_length, num_classes)

    def forward(self, x):
        # x shape: (batchsize, channel, timesteps)
        x = x.permute(2, 0, 1)  # Transform to (timesteps, batch_size, channel) for transformer
        x = self.transformer(x)
        x = x.permute(1, 2, 0)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x
