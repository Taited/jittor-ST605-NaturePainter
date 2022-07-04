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
                 gen_loss_dict, disc_loss_dict, workspace, 
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
            stop_grad(self.EMA_gen)
            self.EMA_decay = EMA_decay
        
        # Init workspaces
        self.workspace = workspace
        self.ckpt_space = osp.join(workspace, 'ckpt')
        self.__prepare_workspace__()
        
    def __call__(self, data):
        EMA_results = None
        with jt.no_grad():
            if self.is_EMA:
                EMA_results = self.EMA_gen(data)
        return self.generator(data), EMA_results
    
    def train_step(self, data):
        real_A, real_B = data['label'], data['image']
        fake_B, fake_B_EMA = self(real_A)
        results = {
            'real_A': real_A,
            'real_B': real_B,
            'fake_B': fake_B,
            'fake_B_EMA': fake_B_EMA
        }
        log_var = {}
        
        # ---------------------
        #  Train Discriminator
        # ---------------------
        # start_grad(self.discriminator)
        loss_D, log_D_loss = self._get_disc_loss(results)
        log_var.update(log_D_loss)
        # loss_D.sync()
        self.optim_D.step(loss_D)

        # ------------------
        #  Train Generators
        # ------------------
        # stop_grad(self.discriminator)
        loss_G, log_G_loss = self._get_gen_loss(results)
        log_var.update(log_G_loss)
        # loss_G.sync()
        self.optim_G.step(loss_G)
        
        jt.sync_all(True)
        # update ema
        if self.is_EMA:
            self.__update_ema()

        return results, log_var
    
    def valid_step(self, data):
        with jt.no_grad():
            real_A, real_B = data['label'], data['image']
            fake_B, fake_B_EMA = self(real_A)
            results = {
                'real_A': real_A,
                'real_B': real_B,
                'fake_B': fake_B,
                'fake_B_EMA': fake_B_EMA
            }
            return results

    def _get_disc_loss(self, results):
        log_var = {}
        pred_fake = self.discriminator(results['fake_B'].detach())
        pred_real = self.discriminator(results['real_B'])
        
        if 'oasis_loss' in self.disc_loss_dict:
            loss_func = self.disc_loss_dict['oasis_loss']
            loss_D_fake = loss_func(pred_fake, results['real_A'], for_real=False)
            loss_D_real = loss_func(pred_real, results['real_A'], for_real=True)
            log_var['loss_D_fake'] = loss_D_fake
            log_var['loss_D_real'] = loss_D_real
            
        if 'label_mix_loss' in self.disc_loss_dict:
            loss_func = self.disc_loss_dict['label_mix_loss']
            mixed_inp, mask = self._generate_labelmix(
                label=results['real_A'], 
                fake_image=results['fake_B'].detach(),
                real_image=results['real_B'])
            output_D_mixed = self.discriminator(mixed_inp)
            log_var['loss_label_mix'] = loss_func(
                mask, output_D_mixed, pred_fake, pred_real)

        # TODO
        # The sum of loss should be automatically
        loss_D = (loss_D_fake + loss_D_real) * 0.5
        log_var['loss_D'] = loss_D + log_var['loss_label_mix']
        return loss_D, log_var
    
    def _get_gen_loss(self, results):
        log_var = {}
        pred_fake = self.discriminator(results['fake_B'])
        
        if 'oasis_loss' in self.gen_loss_dict:
            loss_func = self.gen_loss_dict['oasis_loss']
            loss_G_fake = loss_func(pred_fake, results['real_A'], for_real=True)
            log_var['loss_G_fake'] = loss_G_fake
        
        if 'reconstruction_loss' in self.gen_loss_dict:
            loss_func = self.gen_loss_dict['reconstruction_loss']
            loss_reconstruction = loss_func(results)
            log_var['loss_reconstruction'] = loss_reconstruction
        
        loss_G = loss_G_fake + loss_reconstruction
        log_var['loss_G'] = loss_G
        return loss_G, log_var
    
    def _generate_labelmix(self, label, fake_image, real_image):
        target_map, _ = jt.argmax(label, dim=1, keepdims=True)
        all_classes = jt.unique(target_map)
        for c in all_classes:
            target_map[target_map == c] = jt.randint(0,2,(1,))
        target_map = target_map.float()
        mixed_image = target_map * real_image + \
            (1 - target_map) * fake_image
        return mixed_image, target_map
    
    def __update_ema(self):
        with jt.no_grad():
            ema_state_dict = self.EMA_gen.state_dict()
            gen_state_dict = self.generator.state_dict()
            for key in ema_state_dict:
                ema_state_dict[key] = copy.deepcopy( 
                    ema_state_dict[key].data * self.EMA_decay +
                    gen_state_dict[key].data * (1 - self.EMA_decay))

    def save_checkpoint(self, epoch):
        file_name = f'checkpoint_{epoch}.pkl'
        save_path = osp.join(self.ckpt_space, file_name)
        data = {
            'epoch': epoch,
            'optim_D': self.optim_D.state_dict(),
            'optim_G': self.optim_G.state_dict(),
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict()
        }
        if self.is_EMA:
            data['EMA_generator'] = self.EMA_gen.state_dict()
        jt.save(data, save_path)
    
    def load_checkpoint(self, ckpt_path):
        state_dict = jt.load(ckpt_path)
        self.optim_D.load_state_dict(state_dict['optim_D'])
        self.optim_G.load_state_dict(state_dict['optim_G'])
        self.generator.load_state_dict(state_dict['generator'])
        self.discriminator.load_state_dict(state_dict['discriminator'])
        if self.is_EMA:
            self.EMA_gen.load_state_dict(state_dict['EMA_generator'])
        print(f"Successfully loaded from checkpoint: {ckpt_path}")
        return state_dict['epoch']
    
    def __prepare_workspace__(self):
        if not osp.exists(self.workspace):
            os.makedirs(self.workspace, exist_ok=True)
            
        if not osp.exists(self.ckpt_space):
            os.mkdir(self.ckpt_space)
        