from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import create_dic_fuc
from LinearFunction import LinearFunction

class FCDecomp(nn.Module):
    def __init__(self, coefs, dictionary, bias_val, input_features, output_features, bias=True):
        super(FCDecomp, self).__init__()
        self.dictionary = nn.Parameter(dictionary, requires_grad=False)
        self.coefs = nn.Parameter(coefs, requires_grad=True)
        if bias:
            self.bias = nn.Parameter(bias_val, requires_grad=True)
        else:
            self.register_parameter('bias', None)

        # Not a very smart way to initialize weights
        #self.weight.data.uniform_(-0.1, 0.1)
        #if bias is not None:
            #self.bias.data.uniform_(-0.1, 0.1)

    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        self.weight = torch.mm(self.dictionary, self.coefs)
        #return torch.add(torch.mm(input, self.weight.t()), self.bias)
        return LinearFunction.apply(input, self.weight, self.bias)
        #return LinearDecomp.apply(input, self.coefs, self.dictionary, self.bias)