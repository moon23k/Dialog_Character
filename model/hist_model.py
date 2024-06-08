import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn import functional as F
from .components import clones, Embeddings
from .encoders import StandardEncoder, EvolvedEncoder





class DecoderLayer(LayerBase):
    def __init__(self, config):
        super(DecoderLayer, self).__init__(config)

        self.self_attn = nn.MultiheadAttention(**self.attn_params)
        self.cross_attn = nn.MultiheadAttention(**self.attn_params)
        self.pff = PositionwiseFeedForward(config)

        if self.dec_fuse:
            self.sublayer = clones(SublayerConnection(config), 4)
        else:
            self.sublayer = clones(SublayerConnection(config), 3)


    def forward(self, x, memory, p_proj, e_mask=None, d_mask=None):
        
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
                x, memory, memory, 
                key_padding_mask=e_mask,
                need_weights=False
            )[0]
        )

        if self.dec_fuse:
            x = self.sublayer[2](
                x, 
                lambda x: self.ple_attn(
                    x, p_proj, p_proj, 
                    key_padding_mask=e_mask,
                    need_weights=False
                )[0]
            )

            return self.sublayer[3](x, lambda x: self.pff(x))


        return self.sublayer[2](x, lambda x: self.pff(x))            





class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()

        self.emb_mapping = nn.Sequential(
            nn.Linear(config.emb_dim, config.hidden_dim),
            nn.Dropout(config.dropout_ratio)
        )
        
        self.layers = clones(DecoderLayer(config), config.n_layers)
        self.norm = nn.LayerNorm(config.hidden_dim)
        


    def forward(self, x, memory, ple_out=None, e_mask=None, d_mask=None):
        
        x = self.emb_mapping(x)
        
        for layer in self.layers:
            x = layer(x, memory, ple_out, e_mask, d_mask)

        return self.norm(x)





class HistModel(ModelBase):
    def __init__(self, config):
        super(HistModel, self).__init__(config)

        #Attr Setup
        self.device = config.device
        self.pad_id = config.pad_id
        self.vocab_size = config.vocab_size

        
        #Module Setup
        if config.hist_model == 'std':
            self.hist_encoder = StandardEncoder(config)
        else:
            self.hist_encoder = EvolvedEncoder(config)
        
        self.encoder = StandardEncoder(config)
        self.decoder = Decoder(config)
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
        y, label = self.shift_y(y)

        e_mask = self.pad_mask(x)
        d_mask = self.causal_mask(y)

        memory = self.encoder(x, e_mask)
        dec_out = self.decoder(y, memory, e_mask, d_mask)

        logit = self.generator(dec_out)
        
        #Getting Outputs
        self.out.logit = logit
        self.out.loss = self.criterion(
            logit.contiguous().view(-1, self.vocab_size), 
            label.contiguous().view(-1)
        )

        return self.out