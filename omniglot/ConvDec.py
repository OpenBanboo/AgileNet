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

class ConvDec(nn.Module):
    def __init__(self, weight, input_channels, output_channels, kernel_size, bias=False, is_weight_grad=False, padding=0):
        super(ConvDec, self).__init__()
        self.output_channels = output_channels
        self.input_channels = input_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.is_weight_grad = is_weight_grad
        print("Is training dictionary: ", self.is_weight_grad)
        self.weight = nn.Parameter(weight, requires_grad=self.is_weight_grad)

        # Not a very smart way to initialize weights
        #self.weight.data.uniform_(-0.1, 0.1)
        #if bias is not None:
            #self.bias.data.uniform_(-0.1, 0.1)

    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        return F.conv2d(input, self.weight, stride=1, padding=self.padding, dilation=1, groups=1)
