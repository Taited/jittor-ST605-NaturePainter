import jittor as jt
import jittor.nn as nn
from ..norms.norm_utils import get_spectral_norm


class OASIS_Discriminator(nn.Module):
    def __init__(self, is_spectral, num_res_blocks, semantic_nc):
        super().__init__()
        self.num_res_blocks = num_res_blocks
        
        output_channel = semantic_nc + 1 # for N+1 loss
        self.channels = [3, 128, 128, 256, 256, 512, 512]
        self.body_up   = nn.ModuleList([])
        self.body_down = nn.ModuleList([])
        # encoder part
        for i in range(num_res_blocks):
            self.body_down.append(residual_block_D(
                self.channels[i], self.channels[i+1], 
                is_spectral, -1, first=(i==0)))
        # decoder part
        self.body_up.append(residual_block_D(
            self.channels[-1], self.channels[-2], 
            is_spectral, 1))
        for i in range(1, num_res_blocks-1):
            self.body_up.append(residual_block_D(
                2*self.channels[-1-i], self.channels[-2-i], 
                is_spectral, 1))
        self.body_up.append(residual_block_D(
            2*self.channels[1], 64, is_spectral, 1))
        self.layer_up_last = nn.Conv2d(64, output_channel, 1, 1, 0)

    def execute(self, input):
        x = input
        #encoder
        encoder_res = list()
        for i in range(len(self.body_down)):
            x = self.body_down[i](x)
            encoder_res.append(x)
        #decoder
        x = self.body_up[0](x)
        for i in range(1, len(self.body_down)):
            x = self.body_up[i](jt.concat((encoder_res[-i-1], x), dim=1))
        ans = self.layer_up_last(x)
        return ans


class residual_block_D(nn.Module):
    def __init__(self, fin, fout, is_spectral, up_or_down, first=False):
        super().__init__()
        # Attributes
        self.up_or_down = up_or_down
        self.first = first
        self.learned_shortcut = (fin != fout)
        fmiddle = fout
        norm_layer = get_spectral_norm(is_spectral)
        if first:
            self.conv1 = nn.Sequential(
                norm_layer(nn.Conv2d(fin, fmiddle, 3, 1, 1)))
        else:
            if self.up_or_down > 0:
                self.conv1 = nn.Sequential(
                    nn.LeakyReLU(0.2), 
                    nn.Upsample(scale_factor=2), 
                    norm_layer(nn.Conv2d(fin, fmiddle, 3, 1, 1)))
            else:
                self.conv1 = nn.Sequential(
                    nn.LeakyReLU(0.2), 
                    norm_layer(nn.Conv2d(fin, fmiddle, 3, 1, 1)))
        self.conv2 = nn.Sequential(
            nn.LeakyReLU(0.2), 
            norm_layer(nn.Conv2d(fmiddle, fout, 3, 1, 1)))
        if self.learned_shortcut:
            self.conv_s = norm_layer(nn.Conv2d(fin, fout, 1, 1, 0))
        if up_or_down > 0:
            self.sampling = nn.Upsample(scale_factor=2)
        elif up_or_down < 0:
            self.sampling = nn.AvgPool2d(kernel_size=2, stride=2)
        else:
            self.sampling = nn.Sequential()

    def execute(self, x):
        x_s = self.shortcut(x)
        dx = self.conv1(x)
        dx = self.conv2(dx)
        if self.up_or_down < 0:
            dx = self.sampling(dx)
        out = x_s + dx
        return out

    def shortcut(self, x):
        if self.first:
            if self.up_or_down < 0:
                x = self.sampling(x)
            if self.learned_shortcut:
                x = self.conv_s(x)
            x_s = x
        else:
            if self.up_or_down > 0:
                x = self.sampling(x)
            if self.learned_shortcut:
                x = self.conv_s(x)
            if self.up_or_down < 0:
                x = self.sampling(x)
            x_s = x
        return x_s
