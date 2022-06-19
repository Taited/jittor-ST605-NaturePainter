import copy
import jittor as jt
import jittor.nn as nn
import os
import os.path as osp


def start_grad(model):
    for param in model.parameters():
        if 'running_mean' in param.name() or 'running_var' in param.name(): continue
        param.start_grad()

def stop_grad(model):
    for param in model.parameters():
        param.stop_grad()
        
        
class Trainer:
    def __init__(self, generator: nn.Module, discriminator: nn.Module, 
                 optimizer_G: nn.Optimizer, optimizer_D: nn.Optimizer, 
                 disc_loss_dict, gen_loss_dict, workspace, 
                 is_inference=False, is_EMA=True, EMA_decay=0.9999):
        self.generator = generator
        self.is_EMA = is_EMA
        
        if not is_inference:
            self.discriminator = discriminator
            self.optim_G = optimizer_G
            self.optim_D = optimizer_D
            self.disc_loss_dict = disc_loss_dict
            self.gen_loss_dict = gen_loss_dict
        
        if self.is_EMA:
            self.EMA_gen = copy.deepcopy(self.generator)
            self.EMA_decay = EMA_decay
        
        # Init workspaces
        self.workspace = workspace
        self.ckpt_space = osp.join(workspace, 'ckpt')
        self.__prepare_workspace__()
        
    def __call__(self, data):
        EMA_results = None
        if self.is_EMA:
            EMA_results = self.EMA_gen(data)
        return self.generator(data), EMA_results
    
    def train_step(self, data):
        real_A, real_B = data['label'], data['image']
        fake_B, fake_B_EMA = self(real_A)
        results = {
            'real_A': real_A,
            'real_B': real_B,
            'fake_B': fake_B
        }
        
        # ---------------------
        #  Train Discriminator
        # ---------------------
        start_grad(self.discriminator)
        loss_D, log_D_loss = self._get_disc_loss(results)
        self.optim_D.step(loss_D)

        # ------------------
        #  Train Generators
        # ------------------
        stop_grad(self.discriminator)
        loss_G, log_G_loss = self._get_gen_loss(results)
        self.optim_G.step(loss_G)
        
        jt.sync_all(True)
        
        return results, log_G_loss
        
    def _get_disc_loss(self, results):
        fake_AB = jt.contrib.concat((results['real_A'], 
                                     results['fake_B']), 1) 
        pred_fake = self.discriminator(fake_AB.detach())
        loss_D_fake = self.criterion_GAN(pred_fake, False)
        real_AB = jt.contrib.concat((results['real_A'], 
                                     results['real_B']), 1)
        pred_real = self.discriminator(real_AB)
        loss_D_real = self.criterion_GAN(pred_real, True)
        loss_D = (loss_D_fake + loss_D_real) * 0.5
        log_var = {
            'loss_D_real': loss_D_real,
            'loss_D_fake': loss_D_fake,
            'loss_D': loss_D
        }
        return loss_D
    
    def _get_gen_loss(self, results):
        pass
    
    @jt.single_process_scope()
    def valid_step(self, epoch, writer):
        cnt = 1
        os.makedirs(f"{opt.output_path}/images/test_fake_imgs/epoch_{epoch}", exist_ok=True)
        for i, (_, real_A, photo_id) in enumerate(val_dataloader):
            fake_B = generator(real_A)
            
            if i == 0:
                # visual image result
                img_sample = np.concatenate([real_A.data, fake_B.data], -2)
                img = save_image(img_sample, f"{opt.output_path}/images/epoch_{epoch}_sample.png", nrow=5)
                writer.add_image('val/image', img.transpose(2,0,1), epoch)

            fake_B = ((fake_B + 1) / 2 * 255).numpy().astype('uint8')
            for idx in range(fake_B.shape[0]):
                cv2.imwrite(f"{opt.output_path}/images/test_fake_imgs/epoch_{epoch}/{photo_id[idx]}.jpg", fake_B[idx].transpose(1,2,0)[:,:,::-1])
                cnt += 1
    
    def save_checkpoint(self, epoch):
        file_name = f'checkpoint_{epoch}.pkl'
        save_path = osp.join(self.ckpt_space, file_name)
        data = {
            'epoch': epoch,
            'optim_D': self.optim_D,
            'optim_G': self.optim_G,
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict()
        }
        jt.save(data, save_path)
    
    def load_checkpoint(self, ckpt_path):
        state_dict = jt.load(ckpt_path)
        self.optim_D = state_dict['optim_D']
        self.optim_G = state_dict['optim_G']
        self.generator = state_dict['generator']
        self.discriminator = state_dict['discriminator']
        return state_dict['epoch']
    
    def __prepare_workspace__(self):
        if not osp.exists(self.workspace):
            os.makedirs(self.workspace, exist_ok=True)
            
        if not osp.exists(self.ckpt_space):
            os.mkdir(self.ckpt_space)
        