import argparse
import jittor as jt
import jittor.transform as transform
from PIL import Image
import time

from models import OASIS_Generator, OASIS_Discriminator
from utils.trainer import Trainer
from utils.logger import Logger
from datasets import FlickrDataset


def parse_args():
    parser = argparse.ArgumentParser()
    
    # model setting
    parser.add_argument('--num_res_blocks', type=int, default=6, help='number of residual blocks in G and D')
    parser.add_argument('--channels_G', type=int, default=64, help='# of gen filters in first conv layer in generator')
    parser.add_argument('--norm_type', type=str, default='syncbatch', help='which norm to use in generator before SPADE')
    parser.add_argument('--spade_ks', type=int, default=3, help='kernel size of convs inside SPADE')
    parser.add_argument('--is_EMA', type=bool, default=True, help='if specified, do *not* compute exponential moving averages')
    parser.add_argument('--EMA_decay', type=float, default=0.999, help='decay in exponential moving averages')
    parser.add_argument('--no_3dnoise', action='store_true', default=False, help='if specified, do *not* concatenate noise to label maps')
    parser.add_argument('--z_dim', type=int, default=64, help="dimension of the latent z vector")
    parser.add_argument('--is_spectral', type=bool, default=True, help="whether use spectral normalization")

    # data setting
    parser.add_argument("--data_path", type=str, default="./dataset/flickr/")
    parser.add_argument("--load_size", type=int, default=512, help="size of image")
    parser.add_argument("--crop_size", type=int, default=512, help="size of image")
    parser.add_argument("--aspect_ratio", type=int, default=2, help="ratio of image height")
    parser.add_argument("--semantic_nc", type=int, default=29, help="num of labels")
    parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
    parser.add_argument("--num_workers", type=int, default=8, 
                        help="number of cpu threads to use during batch generation")
    
    # optim setting
    parser.add_argument("--lr_G", type=float, default=0.0004, help="adam: learning rate of generator")
    parser.add_argument("--lr_D", type=float, default=0.0004, help="adam: learning rate of generator")
    parser.add_argument("--b1", type=float, default=0, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    
    # train setting
    parser.add_argument("--start_iter", type=int, default=0, help="iter to start training from")
    parser.add_argument("--total_iter", type=int, default=100000, help="number of training iterations")
    parser.add_argument("--output_path", type=str, default="./training_results/flickr")
    parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
    parser.add_argument("--sample_interval", type=int, default=500, help="interval of sampling iterations")
    parser.add_argument("--checkpoint_interval", type=int, default=20, help="interval between model checkpoints")
    parser.add_argument("--resume_from", type=str, default=None, help="interval between model checkpoints")
    
    opt = parser.parse_args()
    return opt


def train(opt):
    # Configure data transforms
    height = int(opt.load_size / opt.aspect_ratio)
    width = opt.load_size
    label_transforms = [
        transform.Resize(size=(height, width), mode=Image.NEAREST)]
    img_transforms = [
        transform.Resize(size=(height, width), mode=Image.BICUBIC),
        transform.ToTensor(),
        transform.ImageNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
    
    # Init dataloader
    # TODO semantic nc
    train_dataloader = FlickrDataset(opt.data_path, 
                                     dataset_mode="train",
                                     transforms={
                                         "label": label_transforms,
                                         "img": img_transforms},
                                     batch_size=opt.batch_size,
                                     shuffle=True,
                                     num_workers=opt.num_workers)

    # Create networks
    generator = OASIS_Generator(opt.channels_G, opt.semantic_nc,
                                opt.z_dim, opt.norm_type,
                                opt.spade_ks, opt.crop_size,
                                opt.num_res_blocks, opt.aspect_ratio, 
                                opt.no_3dnoise)
    discriminator = OASIS_Discriminator(opt.is_spectral, 
                                        opt.num_res_blocks,
                                        opt.semantic_nc)
    
    # Optimizers
    optimizer_G = jt.optim.Adam(generator.parameters(), lr=opt.lr_G, betas=(opt.b1, opt.b2))
    optimizer_D = jt.optim.Adam(discriminator.parameters(), lr=opt.lr_D, betas=(opt.b1, opt.b2))

    # Trainer
    trainer = Trainer(generator, discriminator,
                      optimizer_G, optimizer_D,
                      {}, {}, workspace=opt.output_path,
                      is_inference=False, is_EMA=opt.is_EMA, 
                      EMA_decay=opt.EMA_decay)
    
    # Logger for visualizing loss and print time
    logger = Logger(opt.output_path, opt.total_iter)
    
    # Resume Training
    start_iter = opt.start_iter
    if opt.resume_from is not None:
        start_iter = trainer.load_checkpoint(opt.resume_from)
    
    # Begin Training    
    for iter_id in range(start_iter, opt.total_iter):
        logger.update_timer('before_time')
        batch_data = train_dataloader[iter_id]
        logger.update_timer('data_time')
        results, log_var = trainer.train_step(batch_data)
        logger.update_timer('after_time')
        
        if jt.rank != 0:
            continue
        
        # print log and add to tensorboard
        if iter_id % opt.log_interval == 0:
            logger.print_log(iter_id, log_var)  # within tensorboard update
        
        # save images
        if iter_id % opt.img_interval == 0:
            results
            pass
    
        if iter_id % opt.val_interval == 0:
            # eval(epoch, writer)
            trainer.valid_step()
            # Save model checkpoints
            trainer.save_checkpoint(iter_id)


if __name__ == '__main__':
    opt = parse_args()
    train(opt)