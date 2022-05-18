import math
import torch
import torch.nn as nn
import torch.nn.functional as F




class MultiHeadAttn(nn.Module):
	def __init__(self, config):
		super(MultiHeadAttn, self).__init__()

		assert config.hidden_dim % config.n_heads == 0

		self.hidden_dim = config.hidden_dim
		self.n_heads = config.n_heads
		self.head_dim = config.hidden_dim // config.n_heads

		self.fc_q = nn.Linear(config.hidden_dim, config.hidden_dim)
		self.fc_k = nn.Linear(config.hidden_dim, config.hidden_dim)
		self.fc_v = nn.Linear(config.hidden_dim, config.hidden_dim)
		
		self.fc_out = nn.Linear(config.hidden_dim, config.hidden_dim)
		


	def forward(self, query, key, value, mask=None):
		Q, K, V = self.fc_q(query), self.fc_k(key), self.fc_v(value)
		Q, K, V = self.split(Q), self.split(K), self.split(V)

		out = self.calc_attn(Q, K, V, mask)
		out = self.concat(out)
		out = self.fc_out(out)

		return out


	def calc_attn(self, query, key, value, mask=None):
		d_k = key.size(-1)
		attn_score = torch.matmul(query, key.permute(0, 1, 3, 2))
		attn_score = attn_score / math.sqrt(d_k)

		if mask is not None:
			attn_score = attn_score.masked_fill(mask==0, -1e9)

		attn_prob = F.softmax(attn_score, dim=-1)
		attn = torch.matmul(attn_prob, value)
		
		return attn


	def split(self, x):
		batch_size = x.size(0)
		out = x.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
		
		return out


	def concat(self, x):
		batch_size = x.size(0)
		out = x.permute(0, 2, 1, 3).contiguous()
		out = out.view(batch_size, -1, self.hidden_dim)	
		
		return out




class PositionwiseFFN(nn.Module):
	def __init__(self, config):
		super(PositionwiseFFN, self).__init__()

		self.fc_1 = nn.Linear(config.hidden_dim, config.pff_dim)
		self.fc_2 = nn.Linear(config.pff_dim, config.hidden_dim)
		self.dropout = nn.Dropout(config.dropout_ratio)


	def forward(self, x):
		out = self.fc_1(x)
		out = self.dropout(F.relu(out))
		out = self.fc_2(out)

		return out




class ResidualConn(nn.Module):
	def __init__(self, config):
		super(ResidualConn, self).__init__()

		self.layer_norm = nn.LayerNorm(config.hidden_dim)
		self.dropout = nn.Dropout(config.dropout_ratio)


	def forward(self, x, sub_layer):
		out = x + sub_layer(x)
		out = self.layer_norm(out)
		out = self.dropout(out)
		
		return out