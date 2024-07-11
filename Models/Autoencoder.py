import torch.nn as nn
import torch.nn.functional as F
import torch
from Modules.Attention import ChannelMultiHeadAttention
from Modules.Transformer import TransformerClassifier
from Modules.CompressConv import ConvCompress1DBlock

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, hidden_layers, kernel_size=7, dropout=0):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(hidden_layers)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else hidden_layers[i-1]
            out_channels = hidden_layers[i]
            layers += [ConvCompress1DBlock(n_inputs=in_channels,
                                            n_outputs=out_channels,
                                            kernel_size=kernel_size,
                                            stride=1,
                                            dilation=dilation_size,
                                            padding=(kernel_size-1) * dilation_size,
                                            chomp_size=(kernel_size-1) * dilation_size,
                                            dropout=dropout),
                       nn.BatchNorm1d(out_channels)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
class VAE(nn.Module):
    def __init__(self, input_length, input_channels, latent_dim, kernel_size, hidden_layers, dropout):
        super(VAE, self).__init__()

        # Encoder
        self.conv1 = nn.Conv1d(input_channels, 512, kernel_size=7, stride=2, padding=1)
        self.att1 = ChannelMultiHeadAttention(input_channels, 8)
        self.conv2 = TemporalConvNet(input_length,hidden_layers, kernel_size=kernel_size, dropout=dropout)
        self.att2 = ChannelMultiHeadAttention(input_channels, 8)

        self.fc1 = nn.Linear(64 * ((((input_length-5)//2+1)-5)//2+1), 256)
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

        # Decoder
        self.fc2 = nn.Linear(latent_dim, 256)
        self.fc3 = nn.Linear(256, 64 * ((((input_length-5)//2+1)-5)//2+1))
        self.deconv1 = nn.ConvTranspose1d(64, 32, kernel_size=7, stride=2, output_padding=1)
        self.att3 = ChannelMultiHeadAttention(32)
        self.deconv2 = nn.ConvTranspose1d(32, input_channels, kernel_size=7, stride=2, padding=3, output_padding=1)

        #classifier
        self.classifier = TransformerClassifier(latent_dim, 6)

    def encode(self, x):
        h = F.relu(self.conv1(x))
        h = self.att1(h)
        h = F.relu(self.conv2(h))
        h = self.att2(h)
        h = F.relu(self.conv3(h))
        h = self.att3(h)
        h = h.view(h.size(0), -1)
        h = F.relu(self.fc1(h))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.fc2(z))
        h = F.relu(self.fc3(h))
        h = h.view(h.size(0), 64, -1)
        h = F.relu(self.deconv1(h))
        h = self.att3(h)
        return self.deconv2(h)#torch.sigmoid(self.deconv2(h))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        classification = self.classifier(z.unsqueeze(1))
        return reconstructed, mu, logvar, classification

if __name__ == "__main__":
    input_channels = 168
    input_length = 512
    latent_dim = 128
    vae = VAE(input_length, input_channels, latent_dim)
    emg_tensor = torch.randn([32, 168, 512])
    reconstructed, mu, logvar, classification = vae(emg_tensor)
    print(reconstructed.shape)
    print(mu.shape)
    print(logvar.shape)
    print(classification.shape)
    print(classification)
