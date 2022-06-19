from jittor import nn

from ..norms.norm_utils import get_spectral_norm
from ..norms.SPADE import SPADE


class ResnetBlock_with_SPADE(nn.Module):
    def __init__(self, fin, fout, z_dim, 
                 norm_type, spade_ks,
                 semantic_nc, no_3dnoise, 
                 is_spectral_norm=False):
        super().__init__()
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)
        sp_norm = get_spectral_norm(is_spectral_norm)
        self.conv_0 = sp_norm(nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1))
        self.conv_1 = sp_norm(nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1))
        if self.learned_shortcut:
            self.conv_s = sp_norm(nn.Conv2d(fin, fout, kernel_size=1, bias=False))

        spade_conditional_input_dims = semantic_nc
        if not no_3dnoise:
            spade_conditional_input_dims += z_dim

        self.norm_0 = SPADE(spade_ks, norm_type,
                            fin, spade_conditional_input_dims)
        self.norm_1 = SPADE(spade_ks, norm_type,
                            fmiddle, spade_conditional_input_dims)
        if self.learned_shortcut:
            self.norm_s = SPADE(spade_ks, norm_type,
                                fin, spade_conditional_input_dims)
        self.activ = nn.LeakyReLU(0.2)

    def execute(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x
        dx = self.conv_0(self.activ(self.norm_0(x, seg)))
        dx = self.conv_1(self.activ(self.norm_1(dx, seg)))
        out = x_s + dx
        return out
