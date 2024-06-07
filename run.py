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
        self.model_base = args.model_base
        self.model_type = args.model_type
        self.hist_apply = args.model_type
        self.search_method = args.search

        self.set_mname(args)
        self.ckpt = f'ckpt/{self.mname}_model.pt'

        use_cuda = torch.cuda.is_available()
        device_condition = use_cuda and self.mode != 'inference'
        self.device_type = 'cuda' if device_condition else 'cpu'
        self.device = torch.device(self.device_type)


    def set_mname(args):
        model_base_dict = {'standard': 'Std', 'evolved': 'Evo'}
        model_type_dict = {'parallel': 'Par', 'sequential': 'Seq'}
        hist_apply_dict = {'encoder': 'Enc', 'decoder': 'Dec', 'enc_dec': 'EncDec'}

        mname = f"{model_base_dict[args.model_base]}_\
                  {model_type_dict[args.model_type]}_\
                  {hist_apply_dict[args.hist_apply]}"
        setattr(self, 'mname', mname)



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
        test_dataloader = load_dataloader(config, 'test')
        tester = Tester(config, model, tokenizer, test_dataloader)
        tester.test()
    
    elif config.mode == 'inference':
        tokenizer = load_tokenizer(config)
        generator = SeqGenerator(config, model, tokenizer)
        generator.inference()
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', required=True)
    parser.add_argument('-model_base', required=True)
    parser.add_argument('-model_type', required=True)
    parser.add_argument('-hist_apply', required=True)
    parser.add_argument('-search', default='greedy', required=False)
    
    args = parser.parse_args()
    assert args.mode.lower() in ['train', 'test', 'inference']
    assert args.model_base.lower() in ['standard', 'evolved']
    assert args.model_type.lower() in ['parallel', 'sequential']
    assert args.hist_apply.lower() in ['encoder', 'decoder', 'enc_dec']
    assert args.search.lower() in ['greedy', 'beam']

    main(args)