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
from FCDecomp import FCDecomp
from LinearFunction import LinearFunction

class ExpandFCDec(nn.Module):
    def __init__(self, coefs, bias_val, input_features, output_features, bias=True, is_coef_grad=False, is_bias_grad=False):
        super(ExpandFCDec, self).__init__()
        self.is_coef_grad = is_coef_grad
        self.is_bias_grad = is_bias_grad
        print(self.is_coef_grad)
        self.coefs = nn.Parameter(coefs, requires_grad=self.is_coef_grad)
        if bias:
            self.bias = nn.Parameter(bias_val, requires_grad=self.is_bias_grad)
        else:
            self.register_parameter('bias', None)
        # Not a very smart way to initialize weights
        #self.weight.data.uniform_(-0.1, 0.1)
        #if bias is not None:
            #self.bias.data.uniform_(-0.1, 0.1)

    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        return LinearFunction.apply(input, self.coefs)+self.bias
