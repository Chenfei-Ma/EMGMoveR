import torch.nn as nn
import torch
from Modules.Attention import ChannelMultiHeadAttention
from Modules.Transformer import TransformerClassifier
from Modules.IsometricConv import IsometricConv1DBlock

class TemporalCompressConvNet(nn.Module):
    def __init__(self, num_inputs, hidden_layers, kernel_size=7, dropout=0.2):
        super(TemporalCompressConvNet, self).__init__()
        layers = []
        num_levels = len(hidden_layers)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else hidden_layers[i-1]
            out_channels = hidden_layers[i]
            layers += [IsometricConv1DBlock(n_inputs=in_channels,
                                            n_outputs=out_channels,
                                            kernel_size=kernel_size,
                                            stride=1,
                                            dilation=dilation_size,
                                            padding=(kernel_size-1) * dilation_size,
                                            chomp_size=(kernel_size-1) * dilation_size,
                                            dropout=dropout),
                       nn.AvgPool1d(2),
                       nn.BatchNorm1d(out_channels)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class AttentionTCN(nn.Module):
    def __init__(self, input_length, input_channels, kernel_size, hidden_layers, dropout=0.2):
        super(AttentionTCN, self).__init__()

        self.conv1 = nn.Conv1d(input_channels, input_channels, kernel_size=11, stride=2, padding=5)
        self.att1 = ChannelMultiHeadAttention(input_channels, 3)
        self.conv2 = TemporalCompressConvNet(input_channels,hidden_layers, kernel_size=kernel_size, dropout=dropout)
        self.att2 = ChannelMultiHeadAttention(hidden_layers[-1], 4)
        # self.classifier = TransformerClassifier(input_length//2**(len(hidden_layers)+1), hidden_layers[-1], 6)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_layers[-1] * input_length//2**(len(hidden_layers)+1), 60),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(60, 6),
        )
        self.init_weights()

    def init_weights(self):
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                layer.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x):
        x = self.conv1(x)
        ax = self.att1(x)
        x = self.conv2(ax)
        ax = self.att2(x)
        classification = self.classifier(ax.reshape(ax.shape[0], -1))
        return classification

if __name__ == "__main__":
    input_channels = 126
    input_length = 512
    ATCN = AttentionTCN(input_length, input_channels, 11, [32,16,8], 0.5)
    emg_tensor = torch.randn([16, 126, 512])
    classification = ATCN(emg_tensor)
    print(classification.shape)
    print(classification)
