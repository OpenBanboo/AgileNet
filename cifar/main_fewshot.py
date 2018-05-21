from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
import dataloader

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler

from torch.autograd import Variable
import create_dic_fuc
from FCDecomp import FCDecomp
from ConvDec import ConvDec
from ConvDecomp2d import ConvDecomp2d
from ExpandConvDec import ExpandConvDec
from FCDec import FCDec
from ExpandFCDec import ExpandFCDec
from BatchNorm2dDecomp import _BatchNorm, BatchNorm2d

# Training settings
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save', dest='is_save', type=bool, default=False,
                    help='save the model and weights')
parser.add_argument('--get_rid_of', dest='get_rid_of', type=int, default=9,
                    help='The number to get rid of.')
parser.add_argument('--ori_train', dest='is_ori_train', type=bool, default=False,
                    help='Execute original training with OCO_Net (Please input True/False, default: False)')
parser.add_argument('--dic_train', dest='is_dic_train', type=bool, default=False,
                    help='Use diction training with DD_Net (Please input True/False, default: False)')
parser.add_argument('--fewshot_train', dest='is_fewshot_train', type=bool, default=False,
                    help='Execute fewshot training with DDF_Net (Please input True/False, default: False)')
parser.add_argument('--save_ori', dest='is_save_ori', type=bool, default=False,
                    help='Save the trained original trained model? (Please input True/False, default: False)')
parser.add_argument('--save_decomp', dest='is_save_decomp', type=bool, default=False,
                    help='Save the trained decomposed trained model? (Please input True/False, default: False)')
parser.add_argument('--epochs', type=int, default=1, metavar='N',
                    help='number of epochs to train original net (default: 10)')
parser.add_argument('--epochs_decomp', type=int, default=1, metavar='N',
                    help='number of epochs to train dictionary decomposed net (default: 1)')
parser.add_argument('--epochs_fewshot', type=int, default=1, metavar='N',
                    help='number of epochs to fewshot train dictionary decomposed net (default: 1)')
parser.add_argument('--num_fewshot_batch', type=int, default=5, metavar='N',
                    help='number of fewshot batches (default: 5)')
args = parser.parse_args()
is_save = args.is_save
is_ori_train = args.is_ori_train
is_dic_train = args.is_dic_train
is_fewshot_train = args.is_fewshot_train
is_save_ori = args.is_save_ori
is_save_decomp = args.is_save_decomp
num_fewshot_batch = args.num_fewshot_batch
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}


'''
=========================================================================================
Network Model Definition
OCO_Net - Original Customized Output Net
DD_Net - Dictionary Decomposed-weight Net
DDF_Net - DIctionary Decomposed-weight Fewshot Net
=========================================================================================
'''


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

# Original Customized Output Model
class VGG_OCO_Net(nn.Module):
    def __init__(self, vgg_name, num_outputs=9):
        super(VGG_OCO_Net, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, num_outputs)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


# Dictionary Decomp Net (which is the Decomp_Net in)
class VGG_DD_Net(nn.Module):
    def __init__(self, vgg_name='', path_pretrained_model="mymodel.pth", number_outputs=9, epsilon=0.7):
        super(VGG_DD_Net, self).__init__()
        # Load the pretrained model
        # Load the saved weights
        self.path_pretrained_model = path_pretrained_model
        self.number_outputs = number_outputs
        self.epsilon = epsilon
        try:
            params = torch.load(self.path_pretrained_model)
            print("Loaded pretrained model.")
        except:
            raise("No pretrained model saved.")

        D_conv = []
        X_a_conv = []
        Biases = []
        for i in range(0,len(params.items())-2,6):
            W = params.items()[i]
            shape_W = W[1].size()
            W = W[1].view(shape_W[0], shape_W[1]*shape_W[2]*shape_W[3])
            W = W.t()
            W = W.cuda()
            D_conv_temp, X_a_conv_temp = create_dic_fuc.create_dic(A=W, M=shape_W[3]*shape_W[2]*shape_W[1], N=shape_W[0],
                Lmin=1, Lmax=shape_W[0]-1, Epsilon=self.epsilon, mode=1)
            D_conv.append(D_conv_temp)
            X_a_conv.append(X_a_conv_temp)
            B = params.items()[i+1][1]
            Biases.append(B)

        layers = []
        in_channels = 3
        i = 0
        for x in cfg[vgg_name]:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [ConvDec(weight=D_conv[i].t().contiguous().view(D_conv[i].t().size()[0],in_channels,3,3).cuda(), input_channels=in_channels,
                           output_channels=D_conv[i].t().size()[0], kernel_size=3, bias=False, is_weight_grad=True , padding=1),
                           ExpandConvDec(coefs=X_a_conv[i].t().cuda(), bias_val=Biases[i].cuda(), input_channels=D_conv[i].t().size()[0], output_channels=x, bias=True,
                           is_coef_grad=True, is_bias_grad=True),
                           BatchNorm2d(num_features=x, weight=params.items()[i*6+2][1].cuda(), bias=params.items()[i*6+3][1].cuda(),
                           running_mean=params.items()[i*6+4][1].cuda(), running_var=params.items()[i*6+5][1].cuda()),
                           nn.ReLU(inplace=True)]
                in_channels = x
                i = i + 1
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]

        self.features = nn.Sequential(*layers)

        print("Initializing conv layers done!")


        W_fc = params.items()[-2][1].cuda()
        D_conv_temp, X_a_conv_temp = create_dic_fuc.create_dic(A=W_fc, M=self.number_outputs, N=512, Lmax=511, Epsilon=0.1, mode=1)
        Biases_fc = params.items()[-1][1].cuda()

        layers = []
        layers += [FCDec(dictionary=X_a_conv_temp.cuda(), input_features=512, output_features=D_conv_temp.size()[0], is_dic_grad=True),
        ExpandFCDec(coefs=D_conv_temp.cuda(), bias_val=Biases_fc.cuda(), input_features=D_conv_temp.size()[0], output_features=self.number_outputs, bias=True,
        is_coef_grad=True, is_bias_grad=True)]


        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


# Dictionary Decomp Net (which is the Decomp_Net in)
class VGG_DDF_Net(nn.Module):
    def __init__(self, vgg_name='',path_pretrained_model="decompmodel_9_classes_98p.pth", number_outputs=10):
        super(VGG_DDF_Net, self).__init__()
        # Load the pretrained model
        # Load the saved weights
        self.path_pretrained_model = path_pretrained_model
        self.number_outputs = number_outputs
        try:
            params = torch.load(self.path_pretrained_model)
            print("Loaded pretrained model.")
        except:
            raise("No pretrained model saved.")


        # Build the layers
        layers = []
        in_channels = 3
        i = 0
        for x in cfg[vgg_name]:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                #ConvDec(weight=D_conv[i].t().contiguous().view(D_conv[i].t().size()[0],in_channels,3,3).cuda()
                layers += [ConvDec(weight=params.items()[i][1].cuda(),
                		   input_channels=in_channels, output_channels=params.items()[i][1].size()[0], kernel_size=3, bias=False, is_weight_grad=False , padding=1),
                           ExpandConvDec(coefs=params.items()[i+1][1].cuda(), bias_val=params.items()[i+2][1].cuda(), input_channels=params.items()[i][1].size()[0], output_channels=x,
                           bias=True, is_coef_grad=False, is_bias_grad=False),
                           BatchNorm2d(num_features=x, weight=params.items()[i+3][1].cuda(), bias=params.items()[i+4][1].cuda(),
                           running_mean=params.items()[i+5][1].cuda(), running_var=params.items()[i+6][1].cuda()),
                           nn.ReLU(inplace=True)]
                in_channels = x
                i = i + 7
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]

        self.features = nn.Sequential(*layers)

        D_fc = torch.ones(self.number_outputs, params.items()[-2][1].size()[-1])
        D_fc[self.number_outputs-1,:] = params.items()[-2][1][self.number_outputs-2,:]
        D_fc[0:self.number_outputs-1,:] = params.items()[-2][1]

        B_fc = torch.ones(self.number_outputs)
        B_fc[self.number_outputs-1] = params.items()[-1][1][self.number_outputs-2]
        B_fc[0:self.number_outputs-1] = params.items()[-1][1]

        layers = []
        layers += [FCDec(dictionary=params.items()[-3][1].cuda(), input_features=512, output_features=params.items()[-3][1].size()[0], is_dic_grad=False),
        ExpandFCDec(coefs=D_fc.cuda(), bias_val=B_fc.cuda(), input_features=params.items()[-3][1].size()[0], output_features=self.number_outputs,
        bias=True, is_coef_grad=True, is_bias_grad=True)]

        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

'''
=========================================================================================
Training and Test Definition
=========================================================================================
'''
criterion = nn.CrossEntropyLoss()
def train(model_train=None, data_loader=None, learning_rate=0.01 ,epoch=1, num_batch=468, is_fewshot_dic_train=False):
    '''
    @model_train: model/net (pass in object of OCO_Net/DD_Net/DDF_Net)
    @data_loader: dataloader (pass in object of exp: train_loader/train_loader_except9/train_loader_only9)
    @epoch: number of training epochs
    @num_batch: 884 is the default value of maximum batch number if batch_size=64
    '''
    optimizer = optim.SGD(list(filter(lambda p: p.requires_grad, model_train.parameters())), lr=learning_rate, momentum=args.momentum, weight_decay=5e-4)
    model_train.train()
    for batch_idx, (data, target) in enumerate(data_loader):
        #target[torch.eq(target,target.clone().fill_(9))]=6
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model_train(data)
        loss = criterion(output, target)
        loss.backward()

        # To enable one row leanring: is_fewshot_dic_train=True
        if is_fewshot_dic_train:
            (model_train.classifier[1]).coefs.grad.data[0:9].zero_()
            (model_train.classifier[1]).bias.grad.data[0:9].zero_()

        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(data_loader.dataset),
                100. * batch_idx / len(data_loader), loss.data[0]))
        if batch_idx==num_batch:
            return

def test(model_test=None, data_loader=None):
    '''
    @model_train: model/net (pass in object of OCO_Net/DD_Net/DDF_Net)
    @data_loader: dataloader (pass in object of exp: test_loader/test_loader_except9/test_loader_only9)
    '''
    model_test.eval()
    test_loss = 0
    correct = 0
    for data, target in data_loader:
        #target[torch.eq(target,target.clone().fill_(9))]=6
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model_test(data)
        #print(output)
        test_loss += criterion(output, target).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(data_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(data_loader.dataset),
        100. * correct / len(data_loader.dataset)))


'''
=========================================================================================
Loading Dataset
./dataloader/cifar.py - All 10 classes dataset from CIFAR10 (50000 training + 10000 test)
./dataloader/cifar_subset.py - All classe but not number 9 from MNIST (45000 training)
./dataloader/cifar_subset_only9.py - Number 9 class only
=========================================================================================
'''

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


trainset = dataloader.cifar.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = dataloader.cifar.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

trainset_exclude9 = dataloader.cifar_subset.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
train_loader_exclude9 = torch.utils.data.DataLoader(trainset_exclude9, batch_size=128, shuffle=True, num_workers=2)

testset_exclude9 = dataloader.cifar_subset.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
test_loader_exclude9 = torch.utils.data.DataLoader(testset_exclude9, batch_size=100, shuffle=False, num_workers=2)

trainset_only9 = dataloader.cifar_subset_only9.CIFAR10(root='./data', train=True, download=True, transform=transform_train, keep_only=9)
train_loader_only9 = torch.utils.data.DataLoader(trainset_exclude9, batch_size=128, shuffle=True, num_workers=2)

testset_only9 = dataloader.cifar_subset_only9.CIFAR10(root='./data', train=False, download=True, transform=transform_test, keep_only=9)
test_loader_only9 = torch.utils.data.DataLoader(testset_only9, batch_size=100, shuffle=False, num_workers=2)

'''
=========================================================================================
Phase One: Training original model with 9 classes
=========================================================================================
'''
is_ori_train=True
is_save_ori=True
if is_ori_train is True:
    print("================Original Training===================")
    # Initializing the model
    model = VGG_OCO_Net('VGG16', num_outputs=9)
    if args.cuda:
        model.cuda()
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    for epoch in range(1, args.epochs + 1):
        train(model_train=model, data_loader=train_loader_exclude9, epoch=epoch)
        test(model_test=model, data_loader=test_loader_exclude9)

    print("\nTesting original model: ")
    test(model_test=model, data_loader=test_loader_exclude9)

    if is_save_ori:
        print("saving original weights...")
        torch.save(model.state_dict(), "./weights/originalmodel_9_classes_vgg16.pth")
        print("Saving weights for 9 class/outputs")
'''
=========================================================================================
Phase Two: Decompose the original model weights
=========================================================================================
'''

is_dic_train=True
is_save_decomp=True
if is_dic_train:
    print("================Dictionary Learning Experiment===================")
    # Initialize the new model
    new_model = VGG_DD_Net(vgg_name='VGG16',
        path_pretrained_model="./weights/originalmodel_9_classes_vgg16.pth",
        number_outputs=9,
        epsilon=0.9)
    if args.cuda:
        new_model.cuda()
        new_model = torch.nn.DataParallel(new_model, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    print("\Testing new model before training: ")
    test(model_test=new_model, data_loader=test_loader_exclude9)
    print("\Training new model: ")
    for epoch in range(1, args.epochs_decomp + 1):
        train(model_train=new_model, data_loader=train_loader_exclude9, epoch=epoch, learning_rate=0.01)
        test(model_test=new_model, data_loader=test_loader_exclude9)
    if is_save_decomp:
        print("saving decomposed weights...")
        torch.save(new_model.state_dict(), "./weights/decompmodel_9_classes_vgg16_decompH_lr0.02_eps0.9_0.1.pth")
        print("Saving decomposed weights for 9 class/outputs done.")

'''
=========================================================================================
Phase Three: Fewshot learning with dictionary
=========================================================================================
'''

num_fewshot_batch=1
is_fewshot_train=True
if is_fewshot_train:
    print("================Fewshot Experiment====================")
    few_shot_model = VGG_DDF_Net(vgg_name='VGG16',path_pretrained_model="./weights/decompmodel_9_classes_vgg16_decompH_lr0.02_eps0.9_0.1.pth", number_outputs=10)
    print("***Testing fewshot model before training: ")
    print("On Original dataset (10 classes)")
    test(model_test=few_shot_model, data_loader=test_loader)
    print("On old dataset (9 classes)")
    test(model_test=few_shot_model, data_loader=test_loader_exclude9)
    print("On New dataset (1 class)")
    test(model_test=few_shot_model, data_loader=test_loader_only9)

    print("================Start Fewshot Training=================")
    print("Training fewshot model: ")
    for epoch in range(1, args.epochs_fewshot + 1):
        train(model_train=few_shot_model, data_loader=train_loader, epoch=epoch, num_batch=num_fewshot_batch, is_fewshot_dic_train=True, learning_rate=0.1)
        test(model_test=few_shot_model, data_loader=test_loader)
        test(model_test=few_shot_model, data_loader=test_loader_exclude9)
        test(model_test=few_shot_model, data_loader=test_loader_only9)
        print("---------------------------------------------------")

    print("***Testing fewshot model after training: ")
    print("On Original dataset (10 classes)")
    test(model_test=few_shot_model, data_loader=test_loader)
    print("On old dataset (9 classes)")
    test(model_test=few_shot_model, data_loader=test_loader_exclude9)
    print("On New dataset (1 class)")
    test(model_test=few_shot_model, data_loader=test_loader_only9)
