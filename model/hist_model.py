import torch
import torch.nn as nn
from collections import namedtuple
from .components import clones, Embeddings
from .encoders import (
    SublayerConnection, 
    PositionwiseFeedForward,
    StandardEncoder, EvolvedEncoder
)






class DecoderLayer(nn.Module):
    def __init__(self, config, fusion_flag=False):
        super(DecoderLayer, self).__init__()

        attn_params = {
            'embed_dim': config.hidden_dim,
            'num_heads': config.n_heads,
            'batch_first': True
        }
        self.fusion_flag = fusion_flag

        self.self_attn = nn.MultiheadAttention(**attn_params)
        self.cross_attn = nn.MultiheadAttention(**attn_params)
        self.pff = PositionwiseFeedForward(config)

        if fusion_flag:
            self.sublayer = clones(SublayerConnection(config), 4)
            self.hist_attn = nn.MultiheadAttention(**attn_params)
        else:
            self.sublayer = clones(SublayerConnection(config), 3)


    def forward(self, x, e_mem, e_mask, d_mask, h_mem=None, h_mask=None):
        
        x = self.sublayer[0](
            x, 
            lambda x: self.self_attn(
                x, x, x, 
                attn_mask=d_mask,
                need_weights=False
            )[0]
        )

        x = self.sublayer[1](
            x, 
            lambda x: self.cross_attn(
                x, e_mem, e_mem, 
                key_padding_mask=e_mask,
                need_weights=False
            )[0]
        )

        if self.fusion_flag:
            x = self.sublayer[2](
                x, 
                lambda x: self.hist_attn(
                    x, h_mem, h_mem, 
                    key_padding_mask=h_mask,
                    need_weights=False
                )[0]
            )

            return self.sublayer[3](x, lambda x: self.pff(x))


        return self.sublayer[2](x, lambda x: self.pff(x))            





class Decoder(nn.Module):
    def __init__(self, config, fusion_flag):
        super(Decoder, self).__init__()

        self.embeddings = Embeddings(config)
        self.layers = clones(DecoderLayer(config, fusion_flag), config.n_layers)
        self.norm = nn.LayerNorm(config.hidden_dim)
        


    def forward(self, x, e_mem, e_mask, d_mask, h_mem=None, h_mask=None):
        x = self.embeddings(x)
        for layer in self.layers:
            x = layer(x, e_mem, e_mask, d_mask, h_mem, h_mask)
        return self.norm(x)





class HistModel(nn.Module):
    def __init__(self, config):
        super(HistModel, self).__init__()

        #Attr Setup
        self.device = config.device
        self.pad_id = config.pad_id
        self.vocab_size = config.vocab_size
        self.enc_fuse = config.enc_fuse
        self.dec_fuse = config.dec_fuse

        
        #Module Setup
        if config.hist_model == 'std':
            self.hist_encoder = StandardEncoder(config)
        else:
            self.hist_encoder = EvolvedEncoder(config)

        self.encoder = StandardEncoder(config, self.enc_fuse)
        self.decoder = Decoder(config, self.dec_fuse)
        self.generator = nn.Linear(config.hidden_dim, self.vocab_size)


        #Output Setup
        self.criterion = nn.CrossEntropyLoss()
        self.out = namedtuple('Out', 'logit loss')


    @staticmethod    
    def shift_y(x):
        return x[:, :-1], x[:, 1:]


    def pad_mask(self, x):
        return x == self.pad_id


    def causal_mask(self, y):
        sz = y.size(1)
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1).to(self.device)


    def forward(self, hist, x, y):
        #Prerequisites
        y, label = self.shift_y(y)
        h_mask = self.pad_mask(hist)
        e_mask = self.pad_mask(x)
        d_mask = self.causal_mask(y)

        #Actual Process
        h_mem = self.hist_encoder(hist, h_mask)

        if self.enc_fuse:
            e_mem = self.encoder(x, e_mask, h_mem, h_mask)
        else:
            e_mem = self.encoder(x, e_mask)

        if self.dec_fuse:
            d_out = self.decoder(y, e_mem, e_mask, d_mask, h_mem, h_mask)
        else:
            d_out = self.decoder(y, e_mem, e_mask, d_mask)

        logit = self.generator(d_out)
        
        #Getting Outputs
        self.out.logit = logit
        self.out.loss = self.criterion(
            logit.contiguous().view(-1, self.vocab_size), 
            label.contiguous().view(-1)
        )

        return self.out