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

class ConvDecomp2d(nn.Module):
    def __init__(self, coefs, dictionary, bias_val, input_channels, output_channels, kernel_size, bias=True, is_dic_grad=False, is_coef_grad=False, is_bias_grad=False, padding=1):
        super(ConvDecomp2d, self).__init__()
        self.output_channels = output_channels
        self.input_channels = input_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.is_dic_grad = is_dic_grad
        self.is_coef_grad = is_coef_grad
        self.is_bias_grad = is_bias_grad
        print("Is training dictionary: ", self.is_dic_grad)
        print("Is training coefficient: ", self.is_coef_grad)
        print("Is training bias: ", self.is_bias_grad)
        self.dictionary = nn.Parameter(dictionary, requires_grad=self.is_dic_grad)
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
        weight_tmp = torch.mm(self.dictionary, self.coefs).cuda().t().contiguous()
        weight_tmp = weight_tmp.view(self.output_channels*self.input_channels, self.kernel_size, self.kernel_size)
        self.weight = weight_tmp.view(self.output_channels, self.input_channels, self.kernel_size, self.kernel_size)
        return F.conv2d(input, self.weight, bias=self.bias, stride=1, padding=self.padding, dilation=1, groups=1)
