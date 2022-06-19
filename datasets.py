import glob
import os
import numpy as np

from jittor.dataset import Dataset
import jittor.transform as transform
import jittor as jt
from PIL import Image


class FlickrDataset(Dataset):
    def __init__(self, root, is_train_phase=True, 
                 dataset_mode="train", transforms=None,
                 semantic_nc=29,
                 *args, **kwargs):
        """
            root: file root of dataset 
            is_train_phase: decide whether load images only \
                or both images and segmentation labels
            dataset_mdoe: directory name of sub dataset
        """
        Dataset.__init__(self, *args, **kwargs)
        self.label_transforms = transform.Compose(transforms['label'])
        self.img_transforms = transform.Compose(transforms['img'])
        self.is_train_phase = is_train_phase
        self.semantic_nc = semantic_nc
        
        self.labels = sorted(glob.glob(
            os.path.join(root, dataset_mode, "labels") + "/*.png"))
        self.total_len = len(self.labels)
        print(f"load {self.total_len} images in {dataset_mode}")

        if self.is_train_phase:
            self.files = sorted(glob.glob(
                os.path.join(root, dataset_mode, "imgs") + "/*.jpg"))
            # check correspondency
            assert len(self.labels) == len(self.files)
            
    def __getitem__(self, index):
        label_path = self.labels[index % self.total_len]
        photo_id = label_path.split('/')[-1][:-4]
        img_B = Image.open(label_path)
        img_B = Image.fromarray(np.array(img_B).astype("uint8")[:, :, np.newaxis].repeat(3,2))

        if self.is_train_phase:
            img_A = Image.open(self.files[index % self.total_len])
            if np.random.random() < 0.5:
                img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], "RGB")
                img_B = Image.fromarray(np.array(img_B)[:, ::-1, :], "RGB")
            img_A = self.img_transforms(img_A)
        else:
            img_A = np.empty([1])
        img_B = self.label_transforms(img_B)
        img_B = jt.float32(np.array(img_B))
        img_B = jt.permute(img_B, (2, 0, 1))
        label = self.seg_onehot(img_B)
        results = {
            'label': label,
            'image': jt.float32(img_A),
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
