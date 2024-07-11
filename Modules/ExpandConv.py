import torch.nn as nn
from torch.nn.utils import weight_norm
from Modules.Chomp import Chomp1d, Chomp2d


class UpsampleExpandConv1DBlock(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size, stride, dilation, padding, dropout=0.2):
        super(UpsampleExpandConv1DBlock, self).__init__()

        self.upsp1 = nn.Upsample(scale_factor=2, mode='linear')
        self.conv1 = weight_norm(nn.Conv1d(input_channel, output_channel, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU() #nn.LeakyReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.upsp1, self.conv1, self.chomp1, self.relu1, self.dropout1)
        self.upsample = nn.ConvTranspose1d(input_channel, output_channel, 3, 2, 1, 1) #if input_channel != output_channel else None
        self.relu = nn.ReLU() #nn.LeakyReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        if self.upsample is not None:
            self.upsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = self.upsample(x)  # x if self.upsample is None else
        return self.relu(out + res)
