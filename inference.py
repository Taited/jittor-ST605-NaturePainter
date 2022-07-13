import jittor as jt
import jittor.transform as transform
from PIL import Image
import copy
from tqdm import tqdm
import os
import os.path as osp
import numpy as np

jt.flags.use_cuda = 1

from main import parse_args
from models import OASIS_Generator
from datasets import FlickrDataset


def create_instance(opt):
    # Configure data transforms
    height = int(opt.load_size / opt.aspect_ratio)
    width = opt.load_size
    label_transforms = [
        transform.Resize(size=(height, width), mode=Image.NEAREST)]
    dataloader = FlickrDataset(opt.data_path,
                               dataset_mode="testB",
                               is_train_phase=False,
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
    gen_ema = copy.deepcopy(gen)
    
    instance_dict = {
        'dataloader': dataloader,
        'gen': gen,
        'gen_ema': gen_ema
    }
    return instance_dict


def load_model(opt, gen, gen_ema):
    state_dict = jt.load(opt.test_ckpt)
    gen.load_state_dict(state_dict['generator'])
    gen_ema.load_state_dict(state_dict['EMA_generator'])
    gen.eval()
    gen_ema.eval()
    return gen, gen_ema
    

def inference(opt, gen, gen_ema, dataloader):
    length = int(len(dataloader) / opt.test_batch_size)
    with jt.no_grad():
        with tqdm(total=length) as pbar:
            for _, batch_data in enumerate(dataloader):
                fake_img = gen(batch_data['label'])
                fake_img_ema = gen_ema(batch_data['label'])
                results = {
                    'fake_img': fake_img,
                    'fake_img_ema': fake_img_ema,
                    'photo_id': batch_data['photo_id']
                }
                save_imgs(opt.test_output, results)
                pbar.update(1)


def save_imgs(output_path, results: dict):
    save_root = output_path
    if not osp.exists(save_root):
        os.makedirs(save_root)

    for var_name in results:
        if var_name == 'photo_id':
            continue
        # create sub folder
        save_res_root = osp.join(save_root, var_name)
        if not osp.exists(save_res_root):
            os.mkdir(save_res_root)
            
        img = results[var_name].detach().numpy()
        img = 255 * (img + 1) / 2
        img = np.clip(img, 0, 255).astype(np.uint8)
        for i in range(img.shape[0]):
            img_sample = img[i, :, :, :]
            img_sample = np.transpose(img_sample, (1, 2, 0))
            img_sample = Image.fromarray(img_sample)
            img_sample = img_sample.resize((512, 384))
            img_sample.save(
                osp.join(
                    save_res_root, f"{results['photo_id'][i]}.jpg"))


if __name__ == '__main__':
    opt = parse_args()
    instance = create_instance(opt)
    dataloader = instance['dataloader']
    gen = instance['gen']
    gen_ema = instance['gen_ema']
    gen, gen_ema = load_model(opt, gen, gen_ema)
    inference(opt, gen, gen_ema, dataloader)