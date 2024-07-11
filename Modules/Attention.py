import torch
import torch.nn as nn

class ChannelMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(ChannelMultiHeadAttention, self).__init__()
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=False)

    def forward(self, x):
        # (batchsize, channel, timesteps) -> (timesteps, batchsize, channel)
        x = x.permute(2, 0, 1)
        attn_output, _ = self.mha(x, x, x)
        # (timesteps, batchsize, channel) -> (batchsize, channel, timesteps)
        attn_output = attn_output.permute(1, 2, 0)
        return attn_output