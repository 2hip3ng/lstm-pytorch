from __future__ import print_function

import torch
from torch import nn
from torch.autograd import Variable
import numpy as np



class LSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, vocab_size, output_size=1, num_layers=1, \
        bias=True, batch_first=True, dropout=0, bidirectional=False):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
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
    def forward(self, x, hidden=None, no_cuda=False):
        batch_size = x.size(0) # batch_size
        x = self.embedding(x)
        h_0 = Variable(torch.randn(self.num_layers*self.num_directions, batch_size, self.hidden_size))
        c_0 = Variable(torch.randn(self.num_layers*self.num_directions, batch_size, self.hidden_size))
        if torch.cuda.is_available() and  not no_cuda:
            h_0 = h_0.cuda()
            c_0 = c_0.cuda()
        x, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        x = self.output(x)
        # 取最后一个time step的输出
        x = x[:,-1,:]
        return x

