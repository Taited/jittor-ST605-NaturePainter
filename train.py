from main import parse_args, train
import jittor as jt


jt.flags.use_cuda = 1
jt.misc.set_global_seed(seed=0)

opt = parse_args()
train(opt)
