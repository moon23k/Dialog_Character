import copy, math, torch
import torch.nn as nn
from .components import clones, Embeddings, ModelBase




class Encoder(nn.Module):
    def __init__(self, config, ple):
        super(Encoder, self).__init__()

        self.embeddings = Embeddings(config)

    def forward(self, x, e_mask=None):
        x = self.ple(input_ids=x, attention_mask=e_mask).last_hidden_state
        return self.enc_mapping(x)





class StandardTransformer(ModelBase):
    def __init__(self, config):
        super(StandardTransformer, self).__init__(config)

        self.hist_encoder = Encoder(config)
        self.encoder = Encoder(config)
        



    def forward(self, hist, x, y):
        y, label = self.shift_y(y)

        e_mask = self.pad_mask(x)
        causal_mask = self.causal_mask(y)
        

        memory = self.encoder(input_ids, attention_mask)
        dec_out = self.decoder(y, memory, e_mask, causal_mask)
        logit = self.generator(dec_out)

        self.out.logit = logit
        self.out.loss = self.criterion(
            logit.contiguous().view(-1, self.vocab_size), 
            label.contiguous().view(-1)
        )

        return self.out
