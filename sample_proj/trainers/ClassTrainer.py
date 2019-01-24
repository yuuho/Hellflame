import socket
from pathlib import Path
from importlib import import_module
import argparse

import torch
from tensorboardX import SummaryWriter

# Trainerクラスの雛形
class Trainer(object):

    def __init__(self,config):
        super().__init__()


        self.config = argparse.Namespace()
        self.config.__dict__.update(config)

        self.load_modules()


    def load_modules(self):
        self.modules = argparse.Namespace()
        self.modules.Model = getattr(import_module('models.'+self.config.model['name']),'Model')
        self.modules.Loss = getattr(import_module('losses.'+self.config.loss['name']),'Function')
        self.modules.Optimizer = getattr(import_module('optimizers.'+self.config.optimizer['name']),'Optimizer')
        self.modules.Dataset = getattr(import_module('datasets.'+self.config.dataset['name']),'Dataset')

    # 学習用コード
    def train(self):
        # define in each class
        return True
