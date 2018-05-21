import argparse
import os
import shutil
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
cudnn.benchmark = True
torch.cuda.set_device(0)

def Adaptive_idx(A,D):
    '''
    Sort the index of Error columns
    @ MatrixXd E =  D*GD.inverse()*D.transpose()*A - A;
    @ vError_local[j] = errorID(E.col(j))
    @ Sort E_norm idx
    @ sort(vError.begin(), vError.end(), errorID::compare);
    '''
    GD = torch.mm(D.t(),D)
    E = torch.mm( torch.mm( torch.mm(D, GD.inverse()), D.t() ), A ) - A
    # Finding the norm of columns of matrix E (error)
    E_norm = torch.Tensor(E.shape[1])
    for i in range(0, A.shape[1]):
        #if (torch.norm(A[:, i], 2) > 1E-3):
        E_norm[i] = torch.norm(E[:, i], 2)
    _, index = torch.sort(E_norm, dim=0, descending=True)
    return index

def create_dic(A, M=50, N=10, Lmin=1, Lstep=1, Lmax=49, Epsilon=0.1, mode=0):
    '''
    Main function of create dictionary D (random)
    # Matrix A should be written in a general one line per row matrix format.
    :return: D
    # The output D is written in  "desired D directory" in a general one line per row matrix format.
    '''
    # Set random seed
    torch.manual_seed(0)
    # Set random seed (if using NVIDIA GPU) - ref: https://discuss.pytorch.org/t/are-gpu-and-cpu-random-seeds-independent/142
    # torch.cuda.manual_seed_all(0)

    # Time interval
    # timeval t1, t2;

    # Initialize matrix
    # A = torch.zeros(M, N)
    # D = torch.zeros(M, Lmin)
    # A = torch.randn(M, N)
    D = torch.randn(M, Lmin).cuda()
    #print(D)
    #print(A[:, 0])
    # D[:,0] = A[:,0].view(M, Lmin)
    '''
    # Load Cifar10 data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=512, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=512, shuffle=False, num_workers=2)
    dataiter = iter(trainloader)


    img, _ = dataiter.next()
    A = img[0,0,:,:].view(-1).view(1,N)
    print A.shape
    for i in range(M-1):
        A = torch.cat((A,img[i+1,0,:,:].view(-1).view(1, N)),dim=0)
    print A.shape
    '''

    # Normalize columns of matrix A
    for i in range(0, N):
        if (torch.norm(A[:, i], 2) > 1E-5):
            A[:, i] /= torch.norm(A[:, i], 2)

    # Create dictionary D randomly in lstep
    global_error_D = 100.00
    l = Lmin
    lold = 0
    perm_tensor = torch.randperm(N)
    error_D = 0.00

    # Adjust the length of D to meet the error requirement
    while ((global_error_D > (Epsilon * Epsilon)) and (l < Lmax + 1)):

        # Sort the index of Error columns if chosing Adaptive mode
        if mode == 1:
            idx = Adaptive_idx(A, D)

        # Resize D
        tempD = torch.zeros(M, l).cuda()
        for i in range(D.shape[1]):
        	tempD[:,i] = D[:,i]
        D = tempD
        i = lold

        for i in range(lold, l):
            if mode == 0:
                D[:, i] = A[:, perm_tensor[i]] #/ torch.norm(A[:, perm_tensor[i]], 2)
            elif mode == 1:
                D[:, i] = A[:, idx[0]]
            else:
                raise "Type 0/1 for --mode"

        X_a_tmp = torch.randn(l, N).cuda()
        X_a = torch.randn(l, N).cuda()

        X_a_tmp, _ = torch.gels(A, D)
        X_a = X_a_tmp[0:l,:]

        # Add a filter
        threshold = 0.0
        for i in range(X_a.shape[0]):
            for j in range(X_a.shape[1]):
                if abs(X_a[i][j]) < threshold:
                    X_a[i][j] = 0.0

        for i in range(0, N):

            tmp = torch.mm(D, X_a)[:,i] - A[:,i]
            error_D = error_D + torch.sum(tmp * tmp)

        error_D = error_D / N
        if error_D < global_error_D:
            global_error_D = error_D

        lold = l
        l = l + Lstep

    print("Oringinal size is: ", N)
    print("Dictionary size is: ",l)
    print("Final error is: ", global_error_D)

    return D, X_a
