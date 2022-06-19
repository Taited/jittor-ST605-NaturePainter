import os
import os.path as osp
import time
from tensorboardX import SummaryWriter


class Logger:
    def __init__(self, workspace, total_iter) -> None:
        self.total_iter = total_iter
        self.workspace = workspace
        self.tf_space = osp.join(workspace, 'tf_logs')
        self.__prepare_workspace__()
        
        # tf log
        self.writter = SummaryWriter(self.tf_space)
        self.timmer = {
            'before_time': 0.0,
            'data_time': 0.0,
            'after_time': 0.0
        }
        
    def update_timer(self, update_mode):
        self.timmer[update_mode] = time.tiem()
    
    def print_log(self, cur_iter, log_var: dict):
        time_log = self.__calc_time__(cur_iter)
        loss_log = self.__update_loss__(cur_iter, log_var)
        print(f'iter: {cur_iter}/{self.total_iter} ' + \
            time_log + loss_log)
    
    def __update_loss__(self, cur_iter, log_var: dict):
        print_str = ''
        for var_name in log_var:
            loss_value = log_var[var_name].detach().cpu().numpy()
            self.writter(f'train/{var_name}', loss_value, cur_iter)
            print_str += f'{var_name}: {loss_value} '
        return print_str
    
    def __calc_time__(self, cur_iter):
        data_time = self.timmer['data_time'] - self.timmer['before_time']
        iter_time = self.timmer['after_time'] - self.timmer['before_time']
        res_time = (self.total_iter - cur_iter) * iter_time
        print_str = f'data time: {data_time}, \
            iter time: {iter_time}, \
                ETA: {self.__seconds_to_dhm__(res_time)} '
        return print_str
    
    def __seconds_to_dhm__(self, seconds):
        def _days(day):
            return "{} d ".format(day) if day > 1 else ""
        def _hours(hour):  
            return "{} h ".format(hour) if hour > 1 else ""
        def _minutes(minute):
            return "{} m".format(minute) if minute > 1 else ""     
        days = seconds // (3600 * 24)
        hours = (seconds // 3600) % 24
        minutes = (seconds // 60) % 60
        seconds = seconds % 60
        if days > 0 :
            return _days(days)+_hours(hours)+_minutes(minutes)
        if hours > 0 :
            return _hours(hours)+_minutes(minutes)
        if minutes > 0 :
            return _minutes(minutes)

    def __prepare_workspace__(self):
        if not osp.exists(self.workspace):
            os.makedirs(self.workspace)
        
        if not osp.exists(self.tf_space):
            os.makedirs(self.tf_space)