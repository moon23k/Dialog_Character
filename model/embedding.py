import math
import torch
import torch.nn as nn




class TransformerEmbedding(nn.Module):
	def __init__(self, config):
		super(TransformerEmbedding, self).__init__()

		self.tok_emb = TokenEmbedding(config)
		self.pos_enc = PosEncoding(config)
		self.dropout = nn.Dropout(config.dropout_ratio)

	def forward(self, x):
		tok_emb = self.tok_emb(x)
		pos_enc = self.pos_enc(x)
		out = self.dropout(tok_emb + pos_enc)
		
		return out




class TokenEmbedding(nn.Module):
	def __init__(self, config):
		super(TokenEmbedding, self).__init__()

		self.embedding = nn.Embedding(config.input_dim, config.emb_dim)
		self.scale = torch.sqrt(torch.FloatTensor([config.emb_dim])).to(config.device)


	def forward(self, x):
		out = self.embedding(x)
		out = out * self.scale

		return out




class PosEncoding(nn.Module):
	def __init__(self, config, max_len=500):
		super(PosEncoding, self).__init__()
		
		pe = torch.zeros(max_len, config.emb_dim)
		position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, config.emb_dim, 2) * (-math.log(10000.0) / config.emb_dim))
		
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)

		self.pe = pe.to(config.device)

	
	def forward(self, x):
		out = self.pe[:x.size(1), :].detach()
		return out