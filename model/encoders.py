import torch
import torch.nn as nn
from torch.nn import functional as F
from .components import (
    clones, Embeddings, 
    GatedConvolution, SeparableConv1D,
    SublayerConnection, PositionwiseFeedForward
)






class EncoderCell(nn.Module):
    def __init__(self, config):
        super(EncoderCell, self).__init__()

        self.pad_id = config.pad_id
        self.glu = GatedConvolution(config.hidden_dim)
        
        self.attention = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=config.n_heads,
            batch_first=True
        )

        self.mid_layer_norm = nn.LayerNorm(config.pff_dim)
        self.layer_norms = clones(nn.LayerNorm(config.hidden_dim), 4)

        self.left_net = nn.Sequential(
            nn.Linear(config.hidden_dim, config.pff_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout_ratio)
        )

        self.right_net = nn.Sequential(
            nn.Conv1d(in_channels=config.hidden_dim, 
                      out_channels=config.hidden_dim//2, 
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(config.dropout_ratio)
        )

        self.sep_conv = SeparableConv1D(
            config.pff_dim, config.hidden_dim // 2, 9
        )

        self.pff = nn.Sequential(
            nn.Linear(config.hidden_dim, config.pff_dim),
            nn.SiLU(),
            nn.Linear(config.pff_dim, config.hidden_dim)
        )


    def forward(self, x, e_mask):
        ### Block_01
        B01_out = self.glu(self.layer_norms[0](x)) #Dim:512


        ### Block_02
        B02_normed = self.layer_norms[1](B01_out)        

        left_out = self.left_net(B02_normed)
        right_out = self.right_net(B02_normed.transpose(1, 2)).transpose(1, 2)

        right_out = F.pad(
            input=right_out, 
            pad=(0, left_out.size(-1) - right_out.size(-1), 0,0,0,0), 
            mode='constant', value=self.pad_id
        ) #Dim:2048          

        B02_out = left_out + right_out


        ### Block_03
        B03_out = self.mid_layer_norm(B02_out)
        
        B03_out = self.sep_conv(
            B03_out.transpose(1, 2)
        ).transpose(1, 2) #Dim:256
        
        B03_out = F.pad(
            input=B03_out,
            pad=(0, B01_out.size(-1) - B03_out.size(-1), 0, 0, 0, 0),
            mode='constant', value=self.pad_id
        )
        
        B03_out += B01_out #Dim:512


        ### Block_04
        B04_out = self.layer_norms[2](B03_out)
        
        attention_out = self.attention(
            B04_out, B04_out, B04_out,
            key_padding_mask = e_mask,
            need_weights=False
        )[0]
        
        B04_out += attention_out #Dim:512


        ### Block_05 & 06
        out = self.layer_norms[3](B04_out)
        out = self.pff(out) + B04_out #Dim:512

        return out 




class StandardEncoderLayer(nn.Module):
    def __init__(self, config, fusion_flag=False):
        super(StandardEncoderLayer, self).__init__()

        attn_params = {
            'embed_dim': config.hidden_dim,
            'num_heads': config.n_heads,
            'batch_first': True
        }
        self.fusion_flag = fusion_flag

        self.self_attn = nn.MultiheadAttention(**attn_params)
        self.pff = PositionwiseFeedForward(config)
        
        if fusion_flag:
            self.sublayer = clones(SublayerConnection(config), 3)
            self.hist_attn = nn.MultiheadAttention(**attn_params)
        else:
            self.sublayer = clones(SublayerConnection(config), 2)


    def forward(self, x, e_mask, h_mem=None, h_mask=None):

        x = self.sublayer[0](
            x, 
            lambda x: self.self_attn(
                x, x, x, 
                key_padding_mask=e_mask, 
                need_weights=False
            )[0]
        )

        if self.fusion_flag:            
            x = self.sublayer[1](
                x, 
                lambda x: self.hist_attn(
                    x, h_mem, h_mem, 
                    key_padding_mask=h_mask,
                    need_weights=False
                )[0]
            )
            return self.sublayer[2](x, self.pff)

        return self.sublayer[1](x, self.pff)




class EvolvedEncoder(nn.Module):
    def __init__(self, config):
        super(EvolvedEncoder, self).__init__()
        self.embeddings = Embeddings(config)
        self.cells = clones(EncoderCell(config), config.n_layers//2)


    def forward(self, x, e_mask):
        x = self.embeddings(x)
        for cell in self.cells:
            x = cell(x, e_mask)
        return x




class StandardEncoder(nn.Module):
    def __init__(self, config, fusion_flag=False):
        super(StandardEncoder, self).__init__()
        self.embeddings = Embeddings(config)
        self.layers = clones(StandardEncoderLayer(config, fusion_flag), config.n_layers)
        self.norm = nn.LayerNorm(config.hidden_dim)


    def forward(self, x, e_mask, h_mem=None, h_mask=None):
        x = self.embeddings(x)
        for layer in self.layers:
            x = layer(x, e_mask, h_mem, h_mask)
        return self.norm(x)