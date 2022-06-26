import jittor as jt
import jittor.nn as nn


class LabelMixLoss():
    def __init__(self, loss_weight=1.0):
        self.labelmix_function = nn.MSELoss()
        self.loss_weight = loss_weight
        
    def __call__(self, mask, output_D_mixed, 
                 output_D_fake, output_D_real):
        mixed_D_output = mask * output_D_real + \
            (1 - mask) * output_D_fake
        loss = self.loss_weight * \
            self.labelmix_function(mixed_D_output, output_D_mixed)
        return loss