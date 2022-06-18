from jittor import nn

from ..network.vgg19 import vgg19


class PerceptualLoss(nn.Module):
    def __init__(self, layer_weights=None) -> None:
        self.model = vgg19(pretrained=True)
        self.loss_func = nn.L1Loss()
        if layer_weights is None:
            self.layer_weights = [1.0 / 32, 1.0 / 16, 
                                  1.0 / 8, 1.0 / 4, 1.0]
        else:
            self.layer_weights = layer_weights
        
    def execute(self, results: dict, 
                source_key: str, target_key: str):
        src_features = self.model(results[source_key])
        tgt_features = self.model(results[target_key])
        loss = 0.
        for i in range(len(src_features)):
            loss += self.layer_weights[i] * \
                self.loss_func(src_features[i], 
                               tgt_features[i])
        return loss