import os, json, argparse, torch

from module import (
    load_dataloader,
    load_generator,
    load_discriminator,
    PreTrainer,
    Trainer,
    Tester,
    Generator
)

from transformers import (
    set_seed, 
    BlenderbotSmallTokenizer
)



class Config(object):
    def __init__(self, args):    

        self.mode = args.mode
        self.character = args.char        
        self.mname = "facebook/blenderbot_small-90M"

        self.clip = 1
        self.lr = 5e-5
        self.max_len = 128
        self.n_epochs = 10
        self.batch_size = 32
        self.iters_to_accumulate = 4

        self.early_stop = True
        self.patience = 3
        
        use_cuda = torch.cuda.is_available()
        self.device_type = 'cuda' if use_cuda else 'cpu'

        if self.mode == 'inference':
            self.device = torch.device('cpu')
        else:
            self.device = torch.device(self.device_type)

        self.g_ckpt = 'ckpt/generator.pt'
        self.d_ckpt = 'ckpt/discriminator.pt'



    def print_attr(self):
        for attribute, value in self.__dict__.items():
            print(f"* {attribute}: {value}")




def inference(g_model, g_tokenizer):
    g_model.eval()
    print(f'--- Inference Process Started! ---')
    print('[ Type "quit" on user input to stop the Process ]')
    
    while True:
        input_seq = input('\nUser Input Sequence >> ').lower()

        #End Condition
        if input_seq == 'quit':
            print('\n--- Inference Process has terminated! ---')
            break        

        #convert user input_seq into model input_ids
        input_ids = g_tokenizer(input_seq, return_tensors='pt')['input_ids']
        output_ids = g_model.generate(input_ids, max_new_tokens=128, use_cache=True)
        output_seq = g_tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

        #Search Output Sequence
        print(f"Model Out Sequence >> {output_seq}")



def main(args):
    set_seed(42)
    config = Config(args)    
    tokenizer = BlenderbotSmallTokenizer.from_pretrained(
        config.mname, 
        model_max_length=config.max_len
    )
    setattr(config, 'pad_id', tokenizer.pad_token_id)

    g_model = load_generator(config)
    d_model = load_discriminator(config)

    if config.mode == 'pretrain':
        train_dataloader = load_dataloader(config, tokenizer, 'valid')
        valid_dataloader = load_dataloader(config, tokenizer, 'valid')
        pretrainer = PreTrainer(config, g_model, d_model, train_dataloader, valid_dataloader)
        pretrainer.train()

    elif config.mode == 'train':
        train_dataloader = load_dataloader(config, tokenizer, 'valid')
        valid_dataloader = load_dataloader(config, tokenizer, 'valid')
        trainer = Trainer(config, g_model, d_model, train_dataloader, valid_dataloader)
        trainer.train()
    
    elif config.mode == 'test':
        test_dataloader = load_dataloader(config, tokenizer, 'test')
        tester = Tester(config, g_model, d_model, tokenizer, test_dataloader)    
        tester.test()

    elif config.mode == 'inference':
        inference(g_model, tokenizer)
    



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', required=True)
    parser.add_argument('-character', default=None, required=False)

    
    args = parser.parse_args()
    assert args.mode.lower() in ['pretrain', 'train', 'test', 'inference']
    if args.mode == 'pretrain':
        assert args.character.lower() in ['ted', 'barney', 'marshall', 'lily', 'robin']


    if args.mode == 'train':
        assert os.path.exists('ckpt/generator_base.pt')
        assert os.path.exists('ckpt/discriminator_base.pt')
    
    elif args.mode in ['test', 'inference']:
        assert os.path.exists('ckpt/discriminator.pt')
        assert os.path.exists('ckpt/generator.pt')

    main(args)