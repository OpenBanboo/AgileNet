'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar
from torch.autograd import Variable

current_path = os.path.dirname(os.path.realpath(__file__))
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--epochs', type=int, default=1, metavar='N',
                    help='number of epochs to train (default: 1)')
parser.add_argument('--epochs_new', type=int, default=1, metavar='N',
                    help='number of epochs to train (default: 1)')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--save', dest='is_save', type=bool, default=False,
                    help='save the model and weights')
parser.add_argument('--dic_train', dest='is_dic_train', type=bool, default=False,
                    help='Use diction training method')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.7)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
args.cuda = not args.no_cuda and torch.cuda.is_available()

if args.cuda:
    torch.cuda.manual_seed(args.seed)
is_save = args.is_save
# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# Model
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7')
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
else:
    print('==> Building model..')
    # net = VGG('VGG19')
    # net = ResNet18()
    # net = PreActResNet18()
    net = LeNet()
    # net = GoogLeNet()
    # net = DenseNet121()
    # net = ResNeXt29_2x64d()
    # net = MobileNet()
    # net = DPN92()
    # net = ShuffleNetG2()
    # net = SENet18()

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model_train.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
optimizer = optim.SGD(list(filter(lambda p: p.requires_grad, net.parameters())), lr=args.lr, momentum=args.momentum, weight_decay=5e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
# Training
def train(model_train=net, epoch=1):
    #optimizer = optim.SGD(model_train.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    #optimizer = optim.SGD(list(filter(lambda p: p.requires_grad, model_train.parameters())), lr=args.lr, momentum=args.momentum)

    print('\nEpoch: %d' % epoch)
    model_train.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = model_train(inputs)
        loss = criterion(outputs, targets)
        #loss = F.nll_loss(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(inputs), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.data[0]))

        #progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #    % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(model_test=net, epoch=1):
    global best_acc
    model_test.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = model_test(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        #progress_bar(batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #    % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        #print('Saving..')
        #state = {
        #    'net': model_test.module if use_cuda else model_test,
        #    'acc': acc,
        #    'epoch': epoch,
        #}
        #if not os.path.isdir('checkpoint'):
        #    os.mkdir('checkpoint')
        #torch.save(state, './checkpoint/ckpt.t7')
        best_acc = acc
    print("The accuracy is (%): ", acc)

#for epoch in range(start_epoch, start_epoch+200):
for epoch in range(start_epoch, args.epochs):
    train(net, epoch)
    test(net, epoch)


for epoch in range(1, args.epochs + 1):
    train(model_train=net, epoch=epoch)
    test(model_test=net, epoch=epoch)

print("\nTesting original model: ")
test(model_test=net)

# Save the trained model if assigned
print(str(is_save))
if is_save:
    print("saving original weights...")
    torch.save(net.state_dict(), "mymodel_10classes_30epochs.pth")
    print(net)

print("******************************************************************************************************************")
print("******************************************************************************************************************")
print("******************************************************************************************************************")
print("******************************************************************************************************************")
print("******************************************************************************************************************")
print("******************************************************************************************************************")


# Initialize the new model
if args.is_dic_train:
    new_net = LeNet_Decomp(path_pretrained_model=current_path+"/mymodel.pth")
    if args.cuda:
        new_net.cuda()
    # First test the new net with factorization
    print("\Testing new model: ")
    test(model_test=new_net)
    optimizer = optim.SGD(list(filter(lambda p: p.requires_grad, new_net.parameters())), lr=args.lr, momentum=0.6, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    print("\Training new model: ")
    for epoch in range(1, args.epochs_new + 1):
        train(model_train=new_net, epoch=epoch)
        test(model_test=new_net, epoch=epoch)

'''
    #for epoch in range(start_epoch, start_epoch+200):
    for epoch in range(start_epoch, args.epochs_new + 1):
        train(new_net, epoch)
        print("new model accuracy: ")
        test(new_net, epoch)
    print("New model: ", new_net)
'''