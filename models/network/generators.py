from jittor import nn
import jittor as jt

from .SPADE_block import ResnetBlock_with_SPADE


class OASIS_Generator(nn.Module):
    def __init__(self, channels_G, semantic_nc, z_dim, 
                 norm_type, spade_ks, no_3dnoise,
                 crop_size, num_res_blocks, aspect_ratio):
        super().__init__()
        
        self.no_3dnoise = no_3dnoise
        self.num_res_blocks = num_res_blocks
        self.z_dim = z_dim
        
        ch = channels_G
        self.channels = [16*ch, 16*ch, 16*ch, 8*ch, 4*ch, 2*ch, 1*ch]
        self.init_W, self.init_H = self.compute_latent_vector_size(crop_size,
                                   num_res_blocks,
                                   aspect_ratio)
        
        self.conv_img = nn.Conv2d(self.channels[-1], 3, 3, padding=1)
        self.up = nn.Upsample(scale_factor=2)
        self.body = nn.ModuleList([])
        
        for i in range(len(self.channels)-1):
            self.body.append(ResnetBlock_with_SPADE(
                self.channels[i], self.channels[i+1], z_dim, 
                 norm_type, spade_ks,
                 semantic_nc, no_3dnoise))
        if not no_3dnoise:
            self.fc = nn.Conv2d(semantic_nc + z_dim, 16 * ch, 3, padding=1)
        else:
            self.fc = nn.Conv2d(semantic_nc, 16 * ch, 3, padding=1)
        
        self.last_activation = nn.Tanh()

    def compute_latent_vector_size(self, crop_size, 
                                   num_res_blocks,
                                   aspect_ratio):
        w = crop_size // (2**(num_res_blocks-1))
        h = round(w / aspect_ratio)
        return h, w

    def execute(self, seg, z=None):
        if not self.no_3dnoise:
            z = jt.randn(seg.size(0), self.z_dim, dtype='float32')
            z = z.view(z.size(0), self.z_dim, 1, 1)
            z = z.expand(z.size(0), self.z_dim, seg.size(2), seg.size(3))
            seg = jt.concat((z, seg), dim = 1)
        x = nn.interpolate(seg, size=(self.init_W, self.init_H))
        x = self.fc(x)
        for i in range(self.num_res_blocks):
            x = self.body[i](x, seg)
            if i < self.num_res_blocks-1:
                x = self.up(x)
        x = self.conv_img(nn.leaky_relu(x, 2e-1))
        x = self.last_activation(x)
        return x
