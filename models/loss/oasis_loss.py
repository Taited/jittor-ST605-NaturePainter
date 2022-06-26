import jittor as jt
import jittor.nn as nn


class OasisLoss():
    def __init__(self, no_balancing_inloss=False, 
                 contain_dontcare_label=False):
        self.no_balancing_inloss = no_balancing_inloss
        self.contain_dontcare_label = contain_dontcare_label

    def __call__(self, input, label, for_real):
        #--- balancing classes ---
        weight_map = get_class_balancing(input, label, 
                                         self.no_balancing_inloss,
                                         self.contain_dontcare_label)
        #--- n+1 loss ---
        target = get_n1_target(input, label, for_real)
        loss = cross_entropy_loss(input, target, reduction='none')
        if for_real:
            loss = jt.mean(loss * weight_map[:, 0, :, :])
        else:
            loss = jt.mean(loss)
        return loss


def get_class_balancing(input, label, 
                        no_balancing_inloss,
                        contain_dontcare_label):
    if not no_balancing_inloss:
        class_occurence = jt.sum(label, dims=(0, 2, 3))
        if contain_dontcare_label:
            class_occurence[-1] = 0
        num_of_classes = (class_occurence > 0).sum()
        coefficients = (1.0 / class_occurence) * \
            get_num_elements(label) / \
                (num_of_classes * label.shape[1])
        integers, _ = jt.argmax(label, dim=1, keepdims=True)
        if contain_dontcare_label:
            coefficients[-1] = 0
        weight_map = coefficients[integers]
    else:
        weight_map = jt.ones_like(input[:, :, :, :])
    return weight_map


def get_num_elements(input):
    b, c, w, h = input.shape
    return b * c * w * h

def get_n1_target(input, label, target_is_real):
    targets = get_target_tensor(input, target_is_real)
    num_of_classes = label.shape[1]
    _, integers = jt.argmax(label, dim=1)
    targets = targets[:, 0, :, :] * num_of_classes
    integers += targets
    integers = jt.clamp(integers, min_v=num_of_classes-1) - num_of_classes + 1
    return integers


def get_target_tensor(input, target_is_real):
    if target_is_real:
        # return jt.FloatTensor(1).fill_(1.0).requires_grad_(False).expand_as(input)
        return jt.ones_like(input)
    else:
        # return jt.FloatTensor(1).fill_(0.0).requires_grad_(False).expand_as(input)
        return jt.zeros_like(input)
    

# TODO 
# This cross entropy should be implemented by the official team.
# Their docs support arguments of reduction, but not implemented
# in this jittor version. This may change in the future.
def cross_entropy_loss(output, target, weight=None, ignore_index=None,reduction='mean'):
    target_shape = target.shape
    if len(output.shape) == 4:
        c_dim = output.shape[1]
        output = output.transpose((0, 2, 3, 1))
        output = output.reshape((-1, c_dim))

    target = target.reshape((-1, ))
    target_weight = jt.ones(target.shape[0], dtype='float32')
    if weight is not None:
        target_weight = weight[target]
    if ignore_index is not None:
        target_weight = jt.ternary(
            target==ignore_index,
            jt.array(0).broadcast(target_weight),
            target_weight
        )
    
    target = target.broadcast(output, [1])
    target = target.index(1) == target
    
    output = output - output.max([1], keepdims=True)
    logsum = output.exp().sum(1).log()
    loss = (logsum - (output*target).sum(1)) * target_weight
    if reduction == 'sum':
        return loss.sum()
    elif reduction == 'mean':
        return loss.mean() / target_weight.mean()
    else:
        return loss.reshape(target_shape) 
