from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import torch.optim as optim
from torchvision import datasets, transforms
import dataloader

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler

from torch.autograd import Variable
import create_dic_fuc
from FCDecomp import FCDecomp
from ConvDecomp2d import ConvDecomp2d

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
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

# Original Customized Output Model
class OCO_Net(nn.Module):
    def __init__(self, num_outputs=9):
		super(OCO_Net, self).__init__()
		self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
		self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
		self.conv2_drop = nn.Dropout2d()
		self.fc1 = nn.Linear(320, 50)
		self.fc2 = nn.Linear(50, num_outputs)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

# Original Customized Output Model
class OCO_Net_Conventional(nn.Module):
    def __init__(self, num_outputs=10, path_pretrained_model="./weights/originalmodel_9_classes_98p.pth"):
        super(OCO_Net_Conventional, self).__init__()
        # Load the pretrained model
        # Load the saved weights
        self.path_pretrained_model = path_pretrained_model
        self.number_outputs = num_outputs
        try:
            params = torch.load(self.path_pretrained_model)
            print("Loaded pretrained model.")
        except:
            raise("No pretrained model saved.")
        # Conv layer 1
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        #self.conv1.weight = params.items()[0][1]
        #self.conv1.bias = params.items()[1][1]
        # Conv layer 2
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        #self.conv2.weight = params.items()[2][1]
        #self.conv2.bias = params.items()[3][1]

        self.conv2_drop = nn.Dropout2d()
        # FC layer 1
        self.fc1 = nn.Linear(320, 50)
        #self.fc1.weight = params.items()[4][1]
        #self.fc1.bias = params.items()[5][1]
        # FC layer 1
        self.fc2 = nn.Linear(50, num_outputs)
        #self.fc2.weight = 

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

# Dictionary Decomp Net (which is the Decomp_Net in)
class DD_Net(nn.Module):
    def __init__(self, path_pretrained_model="mymodel.pth", number_outputs=9):
        super(DD_Net, self).__init__()
        # Load the pretrained model
        # Load the saved weights
        self.path_pretrained_model = path_pretrained_model
        self.number_outputs = number_outputs
        try:
            params = torch.load(self.path_pretrained_model)
            print("Loaded pretrained model.")
        except:
            raise("No pretrained model saved.")

        # Conv Layer 1
        self.W_conv1 = params.items()[0]
        self.B_conv1 = params.items()[1][1].cuda()
        self.W_conv1 = self.W_conv1[1].view(10, 25)
        self.W_conv1 = self.W_conv1.t()
        self.D_conv1, self.X_a_conv1 = create_dic_fuc.create_dic(A=self.W_conv1, M=25, N=10, Lmax=9, Epsilon=0.1, mode=1)
        self.D_conv1 = self.D_conv1.cuda()
        self.X_a_conv1 = self.X_a_conv1.cuda()

        # Conv Layer 2
        self.W_conv2 = params.items()[2]
        self.B_conv2 = params.items()[3][1].cuda()
        self.W_conv2 = self.W_conv2[1].view(200, 25)
        self.W_conv2 = self.W_conv2.t()
        self.D_conv2, self.X_a_conv2 = create_dic_fuc.create_dic(A=self.W_conv2, M=25, N=200, Lmax=199, Epsilon=0.1, mode=1)
        self.D_conv2 = self.D_conv2.cuda()
        self.X_a_conv2 = self.X_a_conv2.cuda()

        # Layer FC1
        self.W_fc1 = params.items()[4][1]
        self.B_fc1 = params.items()[5][1].cuda()
        self.D_fc1, self.X_a_fc1 = create_dic_fuc.create_dic(A=self.W_fc1, M=50, N=320, Lmax=319, Epsilon=0.1, mode=1)
        self.D_fc1 = self.D_fc1.cuda()
        self.X_a_fc1 = self.X_a_fc1.cuda()

        # Layer FC2
        self.W_fc2 = params.items()[6][1] # Feching the last fully connect layer of the orinal model
        self.B_fc2 = params.items()[7][1].cuda()
        self.D_fc2, self.X_a_fc2 = create_dic_fuc.create_dic(A=self.W_fc2, M=self.number_outputs, N=50, Lmax=49, Epsilon=0.1, mode=1)
        self.D_fc2 = self.D_fc2.cuda()
        self.X_a_fc2 = self.X_a_fc2.cuda()

        # Build the layers
        self.conv1 = ConvDecomp2d(coefs=self.X_a_conv1, dictionary=self.D_conv1, bias_val=self.B_conv1, input_channels=1, output_channels=10, kernel_size=5, bias=True, is_coef_grad=True, is_bias_grad=True)
        self.conv2 = ConvDecomp2d(coefs=self.X_a_conv2, dictionary=self.D_conv2, bias_val=self.B_conv2, input_channels=10, output_channels=20, kernel_size=5, bias=True, is_coef_grad=True, is_bias_grad=True)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = FCDecomp(coefs=self.X_a_fc1, dictionary=self.D_fc1, bias_val=self.B_fc1, input_features=320, output_features=50, is_coef_grad=True, is_bias_grad=True)
        self.fc2 = FCDecomp(coefs=self.X_a_fc2, dictionary=self.D_fc2, bias_val=self.B_fc2, input_features=50, output_features=self.number_outputs, is_coef_grad=True, is_bias_grad=True)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


# Dictionary Decomp Net (which is the Decomp_Net in)
class DDF_Net(nn.Module):
    def __init__(self, path_pretrained_model="decompmodel_9_classes_98p.pth", number_outputs=10):
        super(DDF_Net, self).__init__()
        # Load the pretrained model
        # Load the saved weights
        self.path_pretrained_model = path_pretrained_model
        self.number_outputs = number_outputs
        try:
            params = torch.load(self.path_pretrained_model)
            print("Loaded pretrained model.")
        except:
            raise("No pretrained model saved.")

        self.D_conv1 = params.items()[0][1].cuda()
        self.X_a_conv1 = params.items()[1][1].cuda()
        self.B_conv1 = params.items()[2][1].cuda()

        # Conv Layer 2
        self.D_conv2 = params.items()[3][1].cuda()
        self.X_a_conv2 = params.items()[4][1].cuda()
        self.B_conv2 = params.items()[5][1].cuda()


        # Layer FC1
        self.D_fc1 = params.items()[6][1].cuda()
        self.X_a_fc1 = params.items()[7][1].cuda()
        self.B_fc1 = params.items()[8][1].cuda()


        # Layer FC2
        self.D_fc2 = torch.ones(10, 9)
        self.D_fc2[9,:] = params.items()[9][1][8,:]
        self.D_fc2[0:9,:] = params.items()[9][1]
        self.D_fc2 = self.D_fc2.cuda()
        #self.X_a_fc2 = torch.rand(10, 50)
        #self.X_a_fc2[0:9,:] = params.items()[10][1]
        self.X_a_fc2 = params.items()[10][1].cuda()
        #print(self.X_a_fc2)
        self.B_fc2 = torch.ones(10)
        self.B_fc2[9] = params.items()[11][1][8]
        self.B_fc2[0:9] = params.items()[11][1]
        self.B_fc2 = self.B_fc2.cuda()

        #self.D_fc2 = params.items()[9][1].cuda()
        #self.X_a_fc2 = params.items()[10][1].cuda()
        #self.B_fc2 = params.items()[11][1].cuda()

        # Build the layers
        self.conv1 = ConvDecomp2d(coefs=self.X_a_conv1, dictionary=self.D_conv1, bias_val=self.B_conv1, input_channels=1, output_channels=10, kernel_size=5, bias=True,
        is_dic_grad=False, is_coef_grad=False, is_bias_grad=False)
        self.conv2 = ConvDecomp2d(coefs=self.X_a_conv2, dictionary=self.D_conv2, bias_val=self.B_conv2, input_channels=10, output_channels=20, kernel_size=5, bias=True,
        is_dic_grad=False, is_coef_grad=False, is_bias_grad=False)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = FCDecomp(coefs=self.X_a_fc1, dictionary=self.D_fc1, bias_val=self.B_fc1, input_features=320, output_features=50,
        is_dic_grad=False, is_coef_grad=False, is_bias_grad=False)
        self.fc2 = FCDecomp(coefs=self.X_a_fc2, dictionary=self.D_fc2, bias_val=self.B_fc2, input_features=50, output_features=self.number_outputs,
        is_dic_grad=True, is_coef_grad=False, is_bias_grad=True)
        #self.fc2.dictionary.grad.data[0:9].zero_()
        #self.fc2.bias.grad.data[0:9].zero_()

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

'''
=========================================================================================
Training and Test Definition
=========================================================================================
'''
def train(model_train=None, data_loader=None, epoch=1, num_batch=844, is_fewshot_dic_train=False, is_print=False):
    '''
    @model_train: model/net (pass in object of OCO_Net/DD_Net/DDF_Net)
    @data_loader: dataloader (pass in object of exp: train_loader/train_loader_except9/train_loader_only9)
    @epoch: number of training epochs
    @num_batch: 884 is the default value of maximum batch number if batch_size=64
    '''
    optimizer = optim.SGD(list(filter(lambda p: p.requires_grad, model_train.parameters())), lr=args.lr, momentum=args.momentum)
    model_train.train()
    #scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5,
    #verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
    for batch_idx, (data, target) in enumerate(data_loader):
        #target[torch.eq(target,target.clone().fill_(9))]=6
        #print(batch_idx)
        if is_print:
            print(target)
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model_train(data)
        loss = F.nll_loss(output, target)
        loss.backward()

        if is_fewshot_dic_train:
            model_train.fc2.dictionary.grad.data[0:9].zero_()
            model_train.fc2.bias.grad.data[0:9].zero_()

        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(data_loader.dataset),
                100. * batch_idx / len(data_loader), loss.data[0]))
        if batch_idx==num_batch:
            return
    #scheduler.step(val_loss)

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
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(data_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(data_loader.dataset),
        100. * correct / len(data_loader.dataset)))


'''
=========================================================================================
Loading Dataset
./dataloader/mnist.py - All 10 classes dataset from MNIST (60000 training + 10000 test)
./dataloader/mnist_subset.py - All classe but not number 9 from MNIST (54000 training)
./dataloader/mnist_subset_only9.py - Number 9 class only
=========================================================================================
'''
# Loading dataset - All
train_loader_original = DataLoader(
dataloader.mnist.MNIST('./data/original', train=True, download=True,
               transform=transforms.Compose([
                   transforms.ToTensor(),
                   transforms.Normalize((0.1307,), (0.3081,))
               ])), batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader_original = DataLoader(
dataloader.mnist.MNIST('./data/original', train=False,
                transform=transforms.Compose([
                   transforms.ToTensor(),
                   transforms.Normalize((0.1307,), (0.3081,))
               ])), batch_size=args.test_batch_size, shuffle=True, **kwargs)
print('==> Preparing Original MNIST data done..')
print("Original Batch size: ", train_loader_original.batch_size)
print("Original MNIST Dataset: ", train_loader_original.dataset)
print("Original Training Data: ", train_loader_original.dataset.train_data.shape)
print("Original Training Label: ", train_loader_original.dataset.train_labels.shape)
print("Original Test Data: ", test_loader_original.dataset.test_data.shape)
print("Original Test Label: ", test_loader_original.dataset.test_labels.shape)
print("===================================")

# Loading dataset - All but not 9
train_loader_except9 = DataLoader(
dataloader.mnist_subset.MNIST('./data/except9', train=True, download=True, get_rid_of=args.get_rid_of,
               transform=transforms.Compose([
                   transforms.ToTensor(),
                   transforms.Normalize((0.1307,), (0.3081,))
               ])), batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader_except9 = DataLoader(
dataloader.mnist_subset.MNIST('./data/except9', train=False, get_rid_of=args.get_rid_of,
                transform=transforms.Compose([
                   transforms.ToTensor(),
                   transforms.Normalize((0.1307,), (0.3081,))
               ])), batch_size=args.test_batch_size, shuffle=True, **kwargs)
print('==> Preparing data (Exclude 9) done..')
print("Batch size (Except9): ", train_loader_except9.batch_size)
print("MNIST Dataset (Except9): ", train_loader_except9.dataset)
print("Training Data (Except9): ", train_loader_except9.dataset.train_data.shape)
print("Training Label (Except9): ", train_loader_except9.dataset.train_labels.shape)
print("Test Data (Except9): ", test_loader_except9.dataset.test_data.shape)
print("Test Label (Except9): ", test_loader_except9.dataset.test_labels.shape)
print("===================================")

# Loading dataset - only 9
train_loader_only9 = DataLoader(
dataloader.mnist_subset_only9.MNIST('./data/only9', train=True, download=True, onlykeep=9,
               transform=transforms.Compose([
                   transforms.ToTensor(),
                   transforms.Normalize((0.1307,), (0.3081,))
               ])), batch_size=1, shuffle=True, **kwargs)
test_loader_only9 = DataLoader(
dataloader.mnist_subset_only9.MNIST('./data/only9', train=False, onlykeep=9,
                transform=transforms.Compose([
                   transforms.ToTensor(),
                   transforms.Normalize((0.1307,), (0.3081,))
               ])), batch_size=args.test_batch_size, shuffle=True, **kwargs)
print('==> Preparing data (Only 9) done..')
print("Batch size (Only9): ", train_loader_only9.batch_size)
print("MNIST Dataset (only 9): ", train_loader_only9.dataset)
print("Training Data (only 9): ", train_loader_only9.dataset.train_data.shape)
print("Training Label (only 9): ", train_loader_only9.dataset.train_labels.shape)
print("Test Data (only 9): ", test_loader_only9.dataset.test_data.shape)
print("Test Label (only 9): ", test_loader_only9.dataset.test_labels.shape)
print("===================================")

'''
=========================================================================================
Phase One: Training original model with 9 classes
=========================================================================================
'''
is_ori_train=False
if is_ori_train is True:
    print("================Original Training===================")
    # Initializing the model
    model = OCO_Net(num_outputs=9)
    if args.cuda:
        model.cuda()

    for epoch in range(1, args.epochs + 1):
        train(model_train=model, data_loader=train_loader_except9, epoch=1)
        test(model_test=model, data_loader=test_loader_except9)

    print("\nTesting original model: ")
    test(model_test=model, data_loader=test_loader_except9)

    if is_save_ori:
        print("saving original weights...")
        torch.save(model.state_dict(), "./weights/originalmodel_9_classes_98p.pth")
        print("Saving weights for 9 class/outputs")


'''
=========================================================================================
Phase Two: Decompose the original model weights
=========================================================================================
'''
is_dic_train=False
if is_dic_train:
    print("================Dictionary Learning Experiment===================")
    # Initialize the new model
    new_model = DD_Net(path_pretrained_model="./weights/originalmodel_9_classes_98p.pth")
    if args.cuda:
        new_model.cuda()

    print("\Testing new model before training: ")
    test(model_test=new_model,data_loader=test_loader_except9)
    print("\Training new model: ")
    for epoch in range(1, args.epochs_decomp + 1):
        train(model_train=new_model, data_loader=train_loader_except9, epoch=epoch)
        test(model_test=new_model, data_loader=test_loader_except9)
    if is_save_decomp:
        print("saving decomposed weights...")
        torch.save(new_model.state_dict(), "./weights/decompmodel_9_classes_98p.pth")
        print("Saving decomposed weights for 9 class/outputs done.")

''' optimizer=optimizer_fewshot,
=========================================================================================
Phase Three: Fewshot learning with dictionary
=========================================================================================
'''
num_fewshot_batch=1
if is_fewshot_train:
    print("================Fewshot Experiment====================")
    #params = torch.load("./weights/decompmodel_9_classes_98p.pth")
    few_shot_model = DDF_Net(path_pretrained_model="./weights/decompmodel_9_classes_98p.pth")
    print("Testing fewshot model before training: ")
    print("On Original dataset (10 classes)")
    test(model_test=few_shot_model, data_loader=test_loader_original)
    print("On old dataset (9 classes)")
    test(model_test=few_shot_model, data_loader=test_loader_except9)
    print("On New dataset (1 class)")
    test(model_test=few_shot_model, data_loader=test_loader_only9)

    print("================Start Fewshot Training=================")
    print("Training fewshot model: ")
    #optimizer_fewshot = optim.SGD(list(filter(lambda p: p.requires_grad, few_shot_model.parameters())), lr=args.lr, momentum=args.momentum)
    #scheduler_fewshot = optim.lr_scheduler.StepLR(optimizer_fewshot, step_size=3, gamma=0.1)
    for epoch in range(1, args.epochs_fewshot + 1):
        train(model_train=few_shot_model, data_loader=train_loader_original, epoch=epoch, num_batch=num_fewshot_batch, is_fewshot_dic_train=True)
        #scheduler_fewshot.step()
        test(model_test=few_shot_model, data_loader=test_loader_original)

    print("Testing fewshot model after training: ")
    print("On Original dataset (10 classes)")
    test(model_test=few_shot_model, data_loader=test_loader_original)
    print("On old dataset (9 classes)")
    test(model_test=few_shot_model, data_loader=test_loader_except9)
    print("On New dataset (1 class)")
    test(model_test=few_shot_model, data_loader=test_loader_only9)

'''
=========================================================================================
Phase Four: Conventional Few-shot
=========================================================================================
'''
is_conventional_train=True
is_save_conventional = True
if is_conventional_train:
    #new_model = DD_Net(path_pretrained_model="./weights/originalmodel_9_classes_98p.pth")

    print("================Original Training===================")
    # Initializing the model
    conventional_model = OCO_Net(num_outputs=10)
    #conventional_model2 = OCO_Net_Conventional(num_outputs=10, path_pretrained_model="./weights/originalmodel_9_classes_98p.pth")
    if args.cuda:
        conventional_model.cuda()
        #conventional_model2.cuda()

    for epoch in range(1, 10):

        train(model_train=conventional_model, data_loader=train_loader_except9, epoch=epoch)
        #train(model_train=conventional_model, data_loader=train_loader_only9, epoch=epoch, num_batch=num_fewshot_batch)
        #train(model_train=conventional_model, data_loader=train_loader_only9, epoch=epoch, num_batch=num_fewshot_batch)
        test(model_test=conventional_model, data_loader=test_loader_except9)
        test(model_test=conventional_model, data_loader=test_loader_original)
        test(model_test=conventional_model, data_loader=test_loader_only9)

    print("###########################################################################")
    if is_save_conventional:
        print("saving original weights...")
        torch.save(conventional_model.state_dict(), "./weights/conventionalmodel_9_classes.pth")
        print("Saving weights for 9 class/10outputs")

    for epoch in range(1, args.epochs + 1):
        print("EPOCHS: ", epoch)
        train(model_train=conventional_model, data_loader=train_loader_only9, epoch=epoch, num_batch=num_fewshot_batch, is_print=True)
        test(model_test=conventional_model, data_loader=test_loader_except9)
        test(model_test=conventional_model, data_loader=test_loader_original)
        test(model_test=conventional_model, data_loader=test_loader_only9)