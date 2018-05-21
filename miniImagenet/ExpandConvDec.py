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
import time

class ExpandConvDec(nn.Module):
    def __init__(self, coefs, bias_val, input_channels, output_channels, bias=True, is_coef_grad=False, is_bias_grad=False):
        super(ExpandConvDec, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.is_coef_grad = is_coef_grad
        self.is_bias_grad = is_bias_grad
        print(self.is_coef_grad)
        print(self.is_bias_grad)
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
        out = Variable(torch.Tensor(input.size()[0], self.output_channels,input.size()[2],input.size()[2]).zero_()).cuda()
        #print(out_tmp[:, 63, :, :].size())
        #out = out_tmp.permute(1,0,2,3)
        #out = []out.narrow(1,4,1)
        #for i in range(self.output_channels):
        #    out.append(input.narrow(1,0,1)*self.coefs[i][0].expand_as(input.narrow(1,0,1)))
        #    out.append(self.bias[i].expand_as(out[i]))
        #print(out.size())
        #print(out[0].size())
        '''
        for i in range(self.output_channels):
            for j in range(0,input.size()[1]):
                #out[:, i, :, :] = out[:, i:i+1, :, :] + (input[:,j:j+1,:,:]*self.coefs[i][j].expand_as(input[:,j:j+1,:,:])).cuda()#).view(-1,32,32) #out.narrow(1,i,1)
                #out[:, i, :, :] = out[:, i:i+1, :, :] + (input.narrow(1,j,1)*self.coefs[i][j]).cuda()#).view(-1,32,32) #out.narrow(1,i,1)
                out[:, i, :, :] = out[:, i:i+1, :, :] + (input.narrow(1,j,1)*self.coefs[i][j]).cuda()#).view(-1,32,32) #out.narrow(1,i,1)
            out[:, i, :, :] = out[:, i:i+1, :, :] + self.bias[i].cuda() ##out.narrow(1,i,1)
        #t2 = time.time()
        #print(t2-t1)
        '''
        size_input_permute = input.permute(1, 0, 2, 3).size()
        out = torch.mm(self.coefs, input.permute(1, 0, 2, 3).contiguous().view(-1, size_input_permute[1]*size_input_permute[2]*size_input_permute[3]))
        #out += self.bias.view(-1, 1)
        out = out.view(-1, size_input_permute[1], size_input_permute[2], size_input_permute[3]).permute(1, 0, 2, 3).contiguous()
        return out
