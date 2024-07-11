import torch.nn as nn
import torch
from Modules.IsometricConv import IsometricConv1DBlock
from Modules.ExpandConv import UpsampleExpandConv1DBlock

class EncoderP(nn.Module):
    def __init__(self, input_channel, hidden_layers, kernel_size, dropout):
        super(EncoderP, self).__init__()
        layers = []
        num_levels = len(hidden_layers)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_channel if i == 0 else hidden_layers[i-1]
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
        out = self.network(x)
        return out


class EncoderS(nn.Module):
    def __init__(self, input_channel, hidden_layers, kernel_size, dropout):
        super(EncoderS, self).__init__()

        layers = []
        num_levels = len(hidden_layers)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_channel if i == 0 else hidden_layers[i - 1]
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
        out = self.network(x)
        return out


class Decoder(nn.Module):
    def __init__(self, hidden_layers, layer, output_channel, kernel_size, dropout=0.2):
        super(Decoder, self).__init__()

        layers = []
        num_levels = len(hidden_layers)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = hidden_layers[i]
            out_channels = output_channel if i == num_levels-1 else hidden_layers[i+1]
            layers += [layer(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size, padding=(kernel_size-1) * dilation_size, dropout=dropout),
                       nn.BatchNorm1d(out_channels)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        out = self.network(x)
        return out

class DEM(nn.Module):

    def __init__(self, input_length, input_channels, hidden_layers, kernel_size, dropout):
        super(DEM, self).__init__()
        self.encoderp = EncoderP(input_channels, hidden_layers, kernel_size, dropout=dropout)
        self.encoders = EncoderS(input_channels, hidden_layers, kernel_size, dropout=dropout)
        self.decoder = Decoder([i * 2 for i in hidden_layers[::-1]], UpsampleExpandConv1DBlock, input_channels, kernel_size, dropout=dropout)
        self.clr = nn.Linear(hidden_layers[-1]*4, 6)
        self.mp = nn.AdaptiveAvgPool1d(4)

    def forward(self, x):
        p = self.encoderp(x)
        s = self.encoders(x)
        d = self.decoder(torch.concatenate((p, s), axis=1))
        c = self.clr(self.mp(p).flatten(1,2))
        return d, p, s, c


if __name__ == "__main__":
    input = torch.randn([16, 126, 512])
    model = DEM(input_length=512, input_channels=126, hidden_layers=[32,16,8], kernel_size=11, dropout=0.2)
    print(model)
    output_d, output_p, output_s, output_c = model(input)
    print(f'd {output_d.shape} p {output_p.shape} d {output_s.shape} c {output_c.shape} ')
