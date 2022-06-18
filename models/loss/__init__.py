from .gan_loss import GANLoss
from .perceptual_loss import PerceptualLoss
from .pixel_loss import (MSELoss,
                         L1Loss)

__all__ = ['GANLoss', 'PerceptualLoss', 
           'L1Loss', 'MSELoss']