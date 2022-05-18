import copy
import torch
import torch.nn as nn
from .block import MultiHeadAttn, PositionwiseFFN, ResidualConn




class EncoderLayer(nn.Module):
	def __init__(self, config):
		super(EncoderLayer, self).__init__()

		self.m_attn = MultiHeadAttn(config)
		self.pff = PositionwiseFFN(config)
		self.residual_conn = get_clones(ResidualConn(config), 2)

	def forward(self, src, src_mask):
		out = self.residual_conn[0](src, lambda x: self.m_attn(src, src, src, src_mask))
		out = self.residual_conn[1](out, self.pff)
		return out




class DecoderLayer(nn.Module):
	def __init__(self, config):
		super(DecoderLayer, self).__init__()
		
		self.m_attn = MultiHeadAttn(config)
		self.pff = PositionwiseFFN(config)
		self.residual_conn = get_clones(ResidualConn(config), 3)


	def forward(self, memory, trg, src_mask, trg_mask):
		out = self.residual_conn[0](trg, lambda x: self.m_attn(trg, trg, trg, trg_mask))
		attn = self.residual_conn[1](out, lambda x: self.m_attn(out, memory, memory, src_mask))
		out = self.residual_conn[2](attn, self.pff)

		return out, attn


def get_clones(module, N):
	return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])