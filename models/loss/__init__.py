from .gan_loss import GANLoss
from .perceptual_loss import PerceptualLoss
from .pixel_loss import (MSELoss,
                         L1Loss)
from .oasis_loss import OasisLoss
from .label_mix_loss import LabelMixLoss

__all__ = ['GANLoss', 'PerceptualLoss', 
           'L1Loss', 'MSELoss',
           'OasisLoss', 'LabelMixLoss']