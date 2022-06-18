import jittor.nn as nn


_norm_types_ = ['instance', 'batch', 'syncbatch']


def get_norm_layer(norm_type, norm_channel: int):
    assert norm_type in _norm_types_
    if norm_type == 'instance':
        return nn.InstanceNorm2d(norm_channel)
    elif norm_type == 'batch':
        return nn.BatchNorm2d(norm_channel, sync=False)
    elif norm_type == 'syncbatch':
        return nn.BatchNorm2d(norm_channel, sync=True)