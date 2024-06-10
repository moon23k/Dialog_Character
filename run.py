import os, yaml, argparse, torch
from tokenizers import Tokenizer
from tokenizers.processors import TemplateProcessing
from module import (
    load_dataloader, load_model,
    Trainer, Tester, SeqGenerator
)




def set_seed(SEED=42):
    import random
    import numpy as np
    import torch.backends.cudnn as cudnn

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    cudnn.benchmark = False
    cudnn.deterministic = True




class Config(object):
    def __init__(self, args):    
        
        with open('config.yaml', 'r') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
            for group in params.keys():
                for key, val in params[group].items():
                    setattr(self, key, val)

        self.mode = args.mode
        self.hist_model = args.hist_model
        self.fusion_part = args.fusion_part
        self.search_method = args.search

        self.enc_fuse = 'enc' in self.fusion_part
        self.dec_fuse = 'dec' in self.fusion_part
        self.mname = f'{self.hist_model}_hist_{self.fusion_part}_fusion'
        self.ckpt = f'ckpt/{mname}_model.pt'

        use_cuda = torch.cuda.is_available()
        device_condition = use_cuda and self.mode != 'inference'
        self.device_type = 'cuda' if device_condition else 'cpu'
        self.device = torch.device(self.device_type)



    def print_attr(self):
        for attribute, value in self.__dict__.items():
            print(f"* {attribute}: {value}")




def load_tokenizer(config):
    assert os.path.exists(config.tokenizer_path)

    tokenizer = Tokenizer.from_file(config.tokenizer_path)    
    tokenizer.post_processor = TemplateProcessing(
        single=f"{config.bos_token} $A {config.eos_token}",
        special_tokens=[(config.bos_token, config.bos_id), 
                        (config.eos_token, config.eos_id)]
        )
    
    return tokenizer




def main(args):
    set_seed()
    config = Config(args)
    model = load_model(config)
    tokenizer = load_tokenizer(config)


    if config.mode == 'train':
        train_dataloader = load_dataloader(config, tokenizer, 'train')
        valid_dataloader = load_dataloader(config, tokenizer, 'valid')
        trainer = Trainer(config, model, train_dataloader, valid_dataloader)
        trainer.train()
    
    elif config.mode == 'test':
        tokenizer = load_tokenizer(config)
        test_dataloader = load_dataloader(config, tokenizer, 'test')
        tester = Tester(config, model, tokenizer, test_dataloader)
        tester.test()
    
    elif config.mode == 'inference':
        tokenizer = load_tokenizer(config)
        generator = SeqGenerator(config, model, tokenizer)
        generator.inference()
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', required=True)
    parser.add_argument('-hist_model', required=True)
    parser.add_argument('-fusion_part', required=True)
    parser.add_argument('-search', default='greedy', required=False)
    
    args = parser.parse_args()
    assert args.mode.lower() in ['train', 'test', 'inference']
    assert args.hist_model.lower() in ['std', 'evo']
    assert args.fusion_part.lower() in ['enc', 'dec', 'enc_dec']
    assert args.search.lower() in ['greedy', 'beam']

    if args.mode != 'train':
        mname = f'{args.hist_model}_hist_{args.fusion_part}_fusion'
        assert os.path.exists(f'ckpt/{mname}_model.pt')

    main(args)