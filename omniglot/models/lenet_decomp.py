'''LeNet in PyTorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.FCDecomp import FCDecomp
from layers.ConvDecomp2d import ConvDecomp2d
from layers import create_dic_fuc

class LeNet_Decomp(nn.Module):
    def __init__(self, path_pretrained_model="mymodel.pth"):
        self.path_pretrained_model = path_pretrained_model
        super(LeNet_Decomp, self).__init__()
        try:
            params = torch.load(self.path_pretrained_model)
            print("Loaded pretrained model.")
        except:
            raise Exception("No pretrained model saved.")
        
        # Conv Layer 1
        self.W_conv1 = params.items()[0]
        self.B_conv1 = params.items()[1][1].cuda()
        self.W_conv1 = self.W_conv1[1].view(18, 25)
        self.W_conv1 = self.W_conv1.t()
        self.D_conv1, self.X_a_conv1 = create_dic_fuc.create_dic(A=self.W_conv1, M=25, N=18, Lmax=17, Epsilon=0.7, mode=1)
        self.D_conv1 = self.D_conv1.cuda()
        self.X_a_conv1 = self.X_a_conv1.cuda()

        # Conv Layer 2
        self.W_conv2 = params.items()[2]
        self.B_conv2 = params.items()[3][1].cuda()
        self.W_conv2 = self.W_conv2[1].view(96, 25)
        self.W_conv2 = self.W_conv2.t()
        self.D_conv2, self.X_a_conv2 = create_dic_fuc.create_dic(A=self.W_conv2, M=25, N=96, Lmax=24, Epsilon=0.2, mode=1)
        self.D_conv2 = self.D_conv2.cuda()
        self.X_a_conv2 = self.X_a_conv2.cuda()

        # Layer FC1
        self.W_fc1 = params.items()[4][1]
        self.B_fc1 = params.items()[5][1].cuda()
        self.D_fc1, self.X_a_fc1 = create_dic_fuc.create_dic(A=self.W_fc1, M=120, N=400, Lmax=119, Epsilon=0.2, mode=1)
        self.D_fc1 = self.D_fc1.cuda()
        self.X_a_fc1 = self.X_a_fc1.cuda()
        
        # Layer FC2
        self.W_fc2 = params.items()[6][1] # Feching the last fully connect layer of the orinal model
        self.B_fc2 = params.items()[7][1].cuda()
        self.D_fc2, self.X_a_fc2 = create_dic_fuc.create_dic(A=self.W_fc2, M=84, N=120, Lmax=83, Epsilon=0.4, mode=1)
        self.D_fc2 = self.D_fc2.cuda()
        self.X_a_fc2 = self.X_a_fc2.cuda()
        
        # Layer FC3
        self.W_fc3 = params.items()[8][1] # Feching the last fully connect layer of the orinal model
        self.B_fc3 = params.items()[9][1].cuda()
        self.D_fc3, self.X_a_fc3 = create_dic_fuc.create_dic(A=self.W_fc3, M=10, N=84, Lmax=9, Epsilon=0.2, mode=1)
        self.D_fc3 = self.D_fc3.cuda()
        self.X_a_fc3 = self.X_a_fc3.cuda()
        
        #self.conv1 = nn.Conv2d(3, 6, 5)
        #self.conv2 = nn.Conv2d(6, 16, 5)
        #self.fc1   = nn.Linear(16*5*5, 120)
        #self.fc2   = nn.Linear(120, 84)
        #self.fc3   = nn.Linear(84, 10)

        self.conv1 = ConvDecomp2d(coefs=self.X_a_conv1, dictionary=self.D_conv1, bias_val=self.B_conv1, input_channels=3, output_channels=6, kernel_size=5, bias=True)
        self.conv2 = ConvDecomp2d(coefs=self.X_a_conv2, dictionary=self.D_conv2, bias_val=self.B_conv2, input_channels=6, output_channels=16, kernel_size=5, bias=True)
        self.fc1 = FCDecomp(coefs=self.X_a_fc1, dictionary=self.D_fc1, bias_val=self.B_fc1, input_features=400, output_features=120)
        self.fc2 = FCDecomp(coefs=self.X_a_fc2, dictionary=self.D_fc2, bias_val=self.B_fc2, input_features=120, output_features=84)
        self.fc3 = FCDecomp(coefs=self.X_a_fc3, dictionary=self.D_fc3, bias_val=self.B_fc3, input_features=84, output_features=10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out
