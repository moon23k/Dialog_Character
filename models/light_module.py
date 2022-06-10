import copy
import torch
import torch.nn as nn
from .layer import EncoderLayer, DecoderLayer, get_clones
from .embedding import TransformerEmbedding
from utils.train import create_src_mask, create_trg_mask




class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()

        self.n_layers = config.n_layers
        self.layer = EncoderLayer(config)


    def forward(self, src, src_mask):
        for _ in range(self.n_layers):
            src = self.layer(src, src_mask)

        return src




class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()

        self.n_layers = config.n_layers
        self.layers = DecoderLayer(config)


    def forward(self, memory, trg, src_mask, trg_mask):
        for _ in range(self.n_layers):
            trg, attn = self.layer(memory, trg, src_mask, trg_mask)
        
        return trg, attn




class Light_Transformer(nn.Module):
    def __init__(self, config):
        super(Light_Transformer, self).__init__()

        self.embedding = TransformerEmbedding(config)
        self.emb_fc = nn.Linear(config.emb_dim, config.hidden_dim)
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        
        self.fc_out = nn.Linear(config.hidden_dim, config.output_dim)
        self.device = config.device


    def forward(self, src, trg):
        src_mask = create_src_mask(src)
        trg_mask = create_trg_mask(trg)

        src, trg = self.embedding(src), self.embedding(trg) 

        enc_out = self.encoder(src, src_mask)
        dec_out, _ = self.decoder(enc_out, trg, src_mask, trg_mask)

        out = self.fc_out(dec_out)

        return out
        