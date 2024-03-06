import os, torch
from model import Generator, Discriminator
from transformers import (
    BlenderbotSmallConfig,
    BlenderbotSmallForConditionalGeneration
)




def print_model_desc(model):
    def count_params(model):
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return params

    def check_size(model):
        param_size, buffer_size = 0, 0

        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_all_mb = (param_size + buffer_size) / 1024**2
        return size_all_mb

    print(f"--- Model Params: {count_params(model):,}")
    print(f"--- Model  Size : {check_size(model):.3f} MB\n")




def load_generator(config):
    if config.mode == 'pretrain':
        generator = BlenderbotSmallForConditionalGeneration.from_pretrained(config.g_mname)
        print(f"Generator for {config.mode.upper()} has loaded")
        print_model_desc(generator)
        return generator.to(config.device)

    generator_config = BlenderbotSmallConfig.from_pretrained(config.g_mname)
    generator = BlenderbotSmallForConditionalGeneration(generator_config)
    print(f"Generator for {config.mode.upper()} has loaded")

    if config.mode == 'train':
        ckpt = config.g_base_ckpt
    else:
        ckpt = config.g_ckpt
    
    assert os.path.exists(ckpt)
    generator_state = torch.load(ckpt, map_location=config.device)['model_state_dict']
    generator.load_state_dict(generator_state)

    print(f"Model States has loaded from {ckpt}")
    print_model_desc(generator)

    return generator.to(config.device)

