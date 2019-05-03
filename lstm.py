from __future__ import print_function

import torch
from torch import nn
from torch.autograd import Variable
import numpy as np



class LSTM(torch.nn.Module):
	def __init__(self, input_size, hidden_size, batch_size, output_size=1, vocab_size, num_layers=1, \
		bias=True, batch_first=True, dropout=0, bidirectional=False):
		super(lstm, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.batch_size = batch_size
		self.output_size = output_size
		self.vocab_size = vocab_size
		self.num_layers = num_layers
		self.bias = bias
		self.batch_first = batch_first
		self.dropout = dropout
		self.bidirectional = bidirectional
		self.num_directions = 1
		if bidirectional:
			self.num_directions = 2
		self.lstm = torch.nn.LSTM(self.input_size, self.hidden_size, self.num_layers, \
			self.bias, self.batch_first, self.dropout, self.bidirectional)
		self.embedding = nn.Embedding(self.vocab_size, input_size)
		self.output = nn.Linear(self.hidden_size*self.num_directions, self.output_size)
		self.h_0 = Variable(torch.randn(self.num_layers*self.num_directions, self.batch_size, self.hidden_size))
		self.c_0 = Variable(torch.randn(self.num_layers*self.num_directions, self.batch_size, self.hidden_size))
	
	def forward(self, x):
		x = self.embedding(x)
		x, (h_n, c_n) = self.lstm(x, (self.h_0, self.c_0))
		x = self.output(x)
		return x

