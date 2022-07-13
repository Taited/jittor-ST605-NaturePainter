import jittor.nn as nn
from .spectralnorm import spectral_norm


_norm_types_ = ['instance', 'batch', 'syncbatch']


def get_spectral_norm(is_spectral=False):
    if is_spectral:
        return spectral_norm
    else:
        return nn.Identity()

def get_norm_layer(norm_type, norm_channel: int):
    assert norm_type in _norm_types_
    if norm_type == 'instance':
        return nn.InstanceNorm2d(norm_channel, affine=True)
    elif norm_type == 'batch':
        return nn.BatchNorm2d(norm_channel, sync=False, affine=False)
    elif norm_type == 'syncbatch':
        return nn.BatchNorm2d(norm_channel, sync=True, affine=False)