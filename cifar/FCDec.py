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

class FCDec(nn.Module):
    def __init__(self, dictionary, input_features, output_features, is_dic_grad=False):
        super(FCDec, self).__init__()
        self.is_dic_grad = is_dic_grad
        print(self.is_dic_grad)
        self.dictionary = nn.Parameter(dictionary, requires_grad=self.is_dic_grad)

        # Not a very smart way to initialize
        #self.weight.data.uniform_(-0.1, 0.1)
        #if bias is not None:
            #self.bias.data.uniform_(-0.1, 0.1)

    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        return LinearFunction.apply(input, self.dictionary)
