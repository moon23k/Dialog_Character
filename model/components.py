import torch, copy
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple





def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

    


class PositionwiseFeedForward(nn.Module):
    def __init__(self, config):
        super(PositionwiseFeedForward, self).__init__()
        
        self.w_1 = nn.Linear(config.hidden_dim, config.pff_dim)
        self.w_2 = nn.Linear(config.pff_dim, config.hidden_dim)
        
        self.norm = nn.LayerNorm(config.hidden_dim)
        self.dropout1 = nn.Dropout(config.dropout_ratio)
        self.dropout2 = nn.Dropout(config.dropout_ratio)


    def forward(self, x):
        x = self.norm(x)
        x = self.w_2(self.dropout1(F.gelu(self.w_1(x))))
        return self.dropout2(x)




class SublayerConnection(nn.Module):
    def __init__(self, config):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout_ratio)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))



class Embeddings(nn.Module):
    def __init__(self, config, hidden_dim):
        super(Embeddings, self).__init__()

        self.tok_emb = nn.Embedding(config.vocab_size, config.emb_dim)
        self.scale = math.sqrt(config.emb_dim)

        self.pos_emb = PositionalEncoding(config)
        self.pos_dropout = nn.Dropout(config.dropout_ratio)

        self.use_fc_layer = (config.emb_dim != hidden_dim)
        if self.use_fc_layer:
            self.fc = nn.Linear(config.emb_dim, hidden_dim)
            self.fc_dropout = nn.Dropout(config.dropout_ratio)


    def forward(self, x):
        out = self.tok_emb(x) * self.scale
        out = self.pos_dropout(self.pos_emb(out))

        if not self.use_fc_layer:
            return out
        return self.fc_dropout(self.fc(out))



#GLU
class GatedConvolution(nn.Module):
    def __init__(self, hidden_dim, kernel_size=3, padding=1):
        super(GatedConvolution,self).__init__()
        
        self.conv = nn.Conv1d(
            in_channels=hidden_dim, 
            out_channels=hidden_dim * 2,
            kernel_size=kernel_size, 
            padding=padding, bias=True
        )

        init.xavier_uniform_(self.conv.weight, gain=1)

    def forward(self,x):
        convoluted = self.conv(x.transpose(1,2)).transpose(1,2)
        out, gate = convoluted.split(int(convoluted.size(-1) / 2), -1)
        out = out * torch.sigmoid(gate)
        return out




class SeparableConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(SeparableConv1D, self).__init__()

        self.depth_wise = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            padding="same",
            groups=in_channels
        )
        
        self.point_wise = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1
        )

    def forward(self, x):
        out = self.depth_wise(x)
        out = self.point_wise(out)
        return out