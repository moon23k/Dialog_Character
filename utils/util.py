import time
import yaml
import random
import torch
import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn
from models.transformer.vanilla_module import Vanilla_Transformer
from models.transformer.light_module import Light_Transformer





def set_seed(SEED=42):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    cudnn.benchmark = False
    cudnn.deterministic = True



def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs



class Config(object):
    def __init__(self, args):    
        with open('configs/model.yaml', 'r') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
            params = params[args.model]

            for p in params.items():
                setattr(self, p[0], p[1])

        self.model = args.model
        self.task = args.task
        self.scheduler = args.scheduler
        self.clip = 1
        self.pad_idx = 1
        self.n_epochs = 1
        self.batch_size = 128
        self.best_valid_loss = float('inf')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if self.scheduler == 'constant':
            self.learning_rate = 1e-4

        elif self.scheduler in ['noam', 'cosine_annealing_warm']:
            self.learning_rate = 1e-9

        elif self.scheduler in ['exponential', 'step']:
            self.learning_rate = 1e-2


    def print_attr(self):
        for attribute, value in self.__dict__.items():
            print(f"* {attribute}: {value}")




def init_xavier(model):
    if hasattr(model, 'weight') and model.weight.dim() > 1:
        nn.init.xavier_uniform_(model.weight.data)



def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def load_model(config):
    if config.model == 'vanilla':
        model = Vanilla_Transformer(config)
    elif config.model == 'light':
        model = Light_Transformer(config)

    model.apply(init_xavier)
    model.to(config.device)
    print(f'{config.model} Transformer model has loaded.\nThe model has {count_params(model):,} parameters\n')
    return model