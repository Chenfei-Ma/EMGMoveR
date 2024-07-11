import torch.nn as nn

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            y = x
        else:
            y = x[..., :-self.chomp_size].contiguous()
        return y

class Chomp2d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp2d, self).__init__()
        if len(chomp_size) < 2:
            self.chomp_size = (chomp_size, chomp_size)
        else:
            self.chomp_size = chomp_size
    def forward(self, x):
        if self.chomp_size == (0,0):
            y = x
        else:
            y = x[..., self.chomp_size[0]//2:-self.chomp_size[0]//2, :-self.chomp_size[1]].contiguous()
        return y

