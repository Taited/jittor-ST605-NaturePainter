import jittor.nn as nn
from .norm_utils import get_norm_layer


class SPADE(nn.Module):
    def __init__(self, spade_ks, norm_type, norm_nc, label_nc):
        super().__init__()
        self.first_norm = get_norm_layer(norm_type, norm_nc)
        ks = spade_ks
        nhidden = 128
        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def execute(self, x, segmap):
        normalized = self.first_norm(x)
        segmap = nn.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        out = normalized * (1 + gamma) + beta
        return out