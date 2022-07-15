from main import parse_args
from models import OASIS_Generator

import jittor as jt
import jittor.transform as transform
from jittor.dataset import Dataset
from PIL import Image
import glob
from tqdm import tqdm
import os
import os.path as osp
import numpy as np
jt.flags.use_cuda = 1
jt.misc.set_global_seed(seed=0)


class DummyDataset(Dataset):
    def __init__(self, path, 
                 transforms=None,
                 semantic_nc=29, 
                 *args, **kwargs):
        Dataset.__init__(self, *args, **kwargs)
        self.semantic_nc = semantic_nc
        self.labels = sorted(glob.glob(path + "/*.png"))
        self.total_len = len(self.labels)
        self.label_transforms = transform.Compose(transforms['label'])
    
    def __getitem__(self, index):
        label_path = self.labels[index % self.total_len]
        photo_id = label_path.split('/')[-1][:-4]
        img_B = Image.open(label_path)
        img_B = Image.fromarray(np.array(img_B).astype("uint8")[:, :, np.newaxis].repeat(3,2))
        img_B = self.label_transforms(img_B)
        img_B = jt.float32(np.array(img_B))
        img_B = jt.permute(img_B, (2, 0, 1))
        label = self.seg_onehot(img_B)
        results = {
            'label': label,
            'image': jt.float32(np.empty([1])),
            'photo_id': photo_id
        }
        return results
    
    def __len__(self):
        return self.total_len
    
    def seg_onehot(self, array):
        _, w, h = array.shape
        label = jt.zeros((self.semantic_nc, w, h))
        label = label.scatter_(0, array[0:1, :, :], jt.ones(1))
        return label
    
opt = parse_args()


# Configure data transforms
height = int(opt.load_size / opt.aspect_ratio)
width = opt.load_size
label_transforms = [
    transform.Resize(size=(height, width), mode=Image.NEAREST)]
dataloader = DummyDataset(opt.input_path,
                          semantic_nc=opt.semantic_nc,
                          transforms={
                              "label": label_transforms},
                          batch_size=opt.test_batch_size,
                          shuffle=False,
                          num_workers=opt.num_workers)

gen = OASIS_Generator(
    opt.channels_G, opt.semantic_nc,
    opt.z_dim, opt.norm_type,
    opt.spade_ks, opt.crop_size,
    opt.num_res_blocks, opt.aspect_ratio, 
    opt.no_3dnoise)

state_dict = jt.load('ckpt.pkl')
gen.load_state_dict(state_dict['EMA_generator'])
gen.eval()


save_root = opt.output_path
if not osp.exists(save_root):
    os.makedirs(save_root)


with jt.no_grad():
    with tqdm(total=len(dataloader)) as pbar:
        for _, data_batch in enumerate(dataloader):
                img = gen(data_batch['label'])
                img = img.detach().numpy()
                img = 255 * (img + 1) / 2
                img = np.clip(img, 0, 255).astype(np.uint8)
                for i in range(img.shape[0]):
                    img_sample = img[i, :, :, :]
                    img_sample = np.transpose(img_sample, (1, 2, 0))
                    img_sample = Image.fromarray(img_sample)
                    img_sample = img_sample.resize((512, 384))
                    img_sample.save(
                        osp.join(save_root, f"{data_batch['photo_id'][i]}.jpg"))
                    pbar.update(1)