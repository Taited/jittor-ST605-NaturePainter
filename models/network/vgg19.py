import sys
sys.path.append('~/codes/jittor-ST605-NaturePainter')

import jittor as jt
from jittor import nn
from dummy_vgg19 import vgg19 as official_vgg19_func

__all__ = ['vgg19']


def vgg19(pretrained=True):
    model = VGG19(make_vgg19_layers())
    if pretrained:
        weights = jt.load("jittorhub://vgg19.pkl")
        
        slice_1_state_dict = {
            '0.weight': weights['features.0.weight'],
            '0.bias': weights['features.0.bias'],
            '2.weight': weights['features.2.weight'],
            '2.bias': weights['features.2.bias']
        }
        model.features[0].load_state_dict(slice_1_state_dict)
        
        slice_2_state_dict = {
            '1.weight': weights['features.5.weight'],
            '1.bias': weights['features.5.bias'],
            '3.weight': weights['features.7.weight'],
            '3.bias': weights['features.7.bias']
        }
        model.features[1].load_state_dict(slice_2_state_dict)
        
        slice_3_state_dict = {
            '1.weight': weights['features.10.weight'],
            '1.bias': weights['features.10.bias'],
            '3.weight': weights['features.12.weight'],
            '3.bias': weights['features.12.bias'],
            '5.weight': weights['features.14.weight'],
            '5.bias': weights['features.14.bias'],
            '7.weight': weights['features.16.weight'],
            '7.bias': weights['features.16.bias']
        }
        model.features[2].load_state_dict(slice_3_state_dict)
        
        slice_4_state_dict = {
            '1.weight': weights['features.19.weight'],
            '1.bias': weights['features.19.bias'],
            '3.weight': weights['features.21.weight'],
            '3.bias': weights['features.21.bias'],
            '5.weight': weights['features.23.weight'],
            '5.bias': weights['features.23.bias'],
            '7.weight': weights['features.25.weight'],
            '7.bias': weights['features.25.bias']
        }
        model.features[3].load_state_dict(slice_4_state_dict)
        
        slice_5_state_dict = {
            '1.weight': weights['features.28.weight'],
            '1.bias': weights['features.28.bias'],
            '3.weight': weights['features.30.weight'],
            '3.bias': weights['features.30.bias'],
            '5.weight': weights['features.32.weight'],
            '5.bias': weights['features.32.bias'],
            '7.weight': weights['features.34.weight'],
            '7.bias': weights['features.34.bias']
        }
        model.features[4].load_state_dict(slice_5_state_dict)
        
    return model
    

class VGG19(nn.Module):
    def __init__(self, features, init_weights=True):
        super(VGG19, self).__init__()
        self.features = features
    
    def execute(self, x):
        results = []
        for feature_slice in self.features:
            x = feature_slice(x)
            results.append(x)
        return results
    

def make_vgg19_layers():
    # relu_1
    slice_1 = [
        nn.Conv(3, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv(64, 64, kernel_size=3, padding=1),
        nn.ReLU()
    ]
    
    # relu_2
    slice_2 = [
        nn.Pool(kernel_size=2, stride=2, op="maximum"),
        nn.Conv(64, 128, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv(128, 128, kernel_size=3, padding=1),
        nn.ReLU(),
    ]
    
    # relu_3
    slice_3 = [
        nn.Pool(kernel_size=2, stride=2, op="maximum"),
        nn.Conv(128, 256, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv(256, 256, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv(256, 256, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv(256, 256, kernel_size=3, padding=1),
        nn.ReLU()
    ]
    
    # relu_4
    slice_4 = [
        nn.Pool(kernel_size=2, stride=2, op="maximum"),
        nn.Conv(256, 512, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv(512, 512, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv(512, 512, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv(512, 512, kernel_size=3, padding=1),
        nn.ReLU()
    ]
    
    # relu_5
    slice_5 = [
        nn.Pool(kernel_size=2, stride=2, op="maximum"),
        nn.Conv(512, 512, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv(512, 512, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv(512, 512, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv(512, 512, kernel_size=3, padding=1),
        nn.ReLU()
    ]
    
    net_sequence = [
        nn.Sequential(*slice_1),
        nn.Sequential(*slice_2),
        nn.Sequential(*slice_3),
        nn.Sequential(*slice_4),
        nn.Sequential(*slice_5)
    ]
    
    return net_sequence


if __name__ == '__main__':
    self_vgg19 = vgg19(pretrained=True)
    official_vgg19 = official_vgg19_func(pretrained=True)
    x = jt.ones((1, 3, 256, 256))
    self_y = self_vgg19(x)
    pooling = nn.Pool(kernel_size=2, stride=2, op="maximum")
    self_y5 = pooling(self_y[4])
    official_y = official_vgg19(x)
    pass