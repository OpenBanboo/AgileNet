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

# Inherit from Function (Example)
class LinearDecomp(Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    def forward(ctx, input, coefs, dictionary, bias=None):
        #ctx.save_for_backward(input, weight, bias)
        weight = torch.mm(dictionary, coefs).cuda() # reconstruct the weight
        ctx.save_for_backward(input, weight, dictionary, coefs, bias)
        # output = input.mm(weight.t())
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, weight, coefs, dictionary, bias = ctx.saved_variables
        grad_input = grad_input = grad_coefs = grad_bias = None
        # grad_dictionary does not need to be autograd

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        grad_weight = grad_output.t().mm(input) # do not output

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)

        # if ctx.needs_input_grad[1]:
        grad_weight = grad_output.t().mm(input) # do not output grad_weight

        if ctx.needs_input_grad[2]:
            grad_coefs = dictionary.t().mm(grad_weight)

        if ctx.needs_input_grad[3]:
            grad_dictionary = grad_weight.t().mm(grad_coefs.t())

        if bias is not None and ctx.needs_input_grad[4]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_coefs, grad_dictionary, grad_bias, grad_weight