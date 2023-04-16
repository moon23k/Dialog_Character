import os, torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
from transformers import (BertModel, BertConfig,
                          BlenderbotSmallConfig,
                          BlenderbotSmallForConditionalGeneration)



class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()

        self.device = config.device

        if config.mode == 'pretrain':
            self.encoder = BertModel.from_pretrained(config.d_mname).to(self.device)
        else:
            bert_config = BertConfig.from_pretrained(config.d_mname)
            self.encoder = BertModel(bert_config).to(self.device)

        self.classifier = nn.Linear(self.encoder.config.hidden_size, 1)
        self.dropout = nn.Dropout(self.encoder.config.hidden_dropout_prob)
        
        self.device = config.device
        self.pad_id = self.encoder.config.pad_token_id
        self.criterion = nn.BCEWithLogitsLoss().to(self.device)
        self.outputs = namedtuple('Discriminator_Outputs', ('logit', 'loss'))

        
    def forward(self, input_ids, attention_mask, labels):
        out = self.encoder(input_ids, attention_mask).last_hidden_state
        out = self.classifier(out[:, 0])
        out = self.dropout(out).squeeze()

        loss = self.criterion(out, labels)
        return self.outputs(out, loss)



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

    ckpt = config.g_ckpt
    assert os.path.exists(ckpt)
    generator_state = torch.torch.load(ckpt, map_location=config.device)['model_state_dict']
    generator.load_state_dict(generator_state)

    print(f"Model States has loaded from {ckpt}")
    print_model_desc(generator)

    return generator.to(config.device)




def load_discriminator(config):
    discriminator = Discriminator(config)
    print(f"Discriminator for {config.mode.upper()} has loaded")

    if config.mode == 'train':
        assert os.path.exists(config.d_base_ckpt)
        model_state = torch.load(config.d_base_ckpt, map_location=config.device)['model_state_dict']        
        discriminator.load_state_dict(model_state)
        print(f"Model States has loaded from {config.d_ckpt}")        

    elif config.mode in ['test', 'inference']:
        assert os.path.exists(config.d_ckpt)
        model_state = torch.load(config.d_ckpt, map_location=config.device)['model_state_dict']        
        discriminator.load_state_dict(model_state)
        print(f"Model States has loaded from {config.d_ckpt}")

    print_model_desc(discriminator)
    return discriminator.to(config.device)