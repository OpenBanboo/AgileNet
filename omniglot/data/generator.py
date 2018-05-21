from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import torch
import numpy as np
from torchvision import datasets, transforms
import random
import sys
from torch.autograd import Variable
from . import omniglot
#from . import mini_imagenet

def zca_whiten(X):
    """
    Applies ZCA whitening to the data (X)
    http://xcorr.net/2011/05/27/whiten-a-matrix-matlab-code/

    X: numpy 2d array
        input data, rows are data points, columns are features

    Returns: ZCA whitened 2d array
    """

    X = X[0]

    assert(X.ndim == 2)
    EPS = 10e-5

    #   covariance matrix
    cov = np.dot(X.T, X)
    #   d = (lambda1, lambda2, ..., lambdaN)
    d, E = np.linalg.eigh(cov)
    #   D = diag(d) ^ (-1/2)
    D = np.diag(1. / np.sqrt(d + EPS))
    #   W_zca = E * D * E.T
    W = np.dot(np.dot(E, D), E.T)

    X_white = np.dot(X, W)

    shape = np.shape(X_white)

    X_white = X_white.reshape(1,shape[0],shape[1])

    return X_white


class Generator(data.Dataset):
    def __init__(self, root, args, partition='train', dataset='omniglot',transform=None, shot=1, way=5 , fewshot=False, with_train=False, data_aug=False, both=False, whole=False):
        self.root = root
        self.partition = partition  # training set or test set
        self.args = args
        self.transform = transform
        self.shot = shot
        self.whole = whole
        self.way = way
        self.both = both
        self.fewshot = fewshot
        self.with_train = with_train
        self.data_aug = data_aug
        assert (dataset == 'omniglot' or
                dataset == 'mini_imagenet'), 'Incorrect dataset partition'
        self.dataset = dataset

        if self.dataset == 'omniglot':
            self.input_channels = 1
            self.size = (28, 28)
            # self.rnd_seed = random.seed(4) # best for 5 way
            # self.rnd_seed = random.seed(50) # best for 20 way
            self.rnd_seed = random.seed(4)
        else:
            self.input_channels = 3
            self.size = (84, 84)
            self.rnd_seed = random.seed(17)

        if dataset == 'omniglot':
            self.loader = omniglot.Omniglot(self.root, dataset=dataset)
            self.data = self.loader.load_dataset(self.partition, self.size)

            data_array = np.asarray(self.data.values())
            #print(np.shape(data_array))
            self.data_arr = data_array.reshape(-1,*data_array.shape[2:])
            #print(np.shape(self.data_arr))
            self.aug_data = []
            few_classes = random.sample(range(423), self.way)
            #self.rnd_seed = random.seed(6)
            rnd = random.sample(range(20),self.shot)

            if self.fewshot == False:
                for i in range(len(self.data_arr)):
                        self.aug_data.append(self.data_arr[i])
                        #self.aug_data.append(np.flip(self.data_arr[i], axis=2).copy())
                        #self.aug_data.append(np.flip(self.data_arr[i], axis=1).copy())
                        self.aug_data.append(np.rot90(self.data_arr[i], k=1, axes=(1, 2)).copy())
                        self.aug_data.append(np.rot90(self.data_arr[i], k=2, axes=(1, 2)).copy())
                        self.aug_data.append(np.rot90(self.data_arr[i], k=3, axes=(1, 2)).copy())
            elif self.both == True:

                self.few_data = []
                self.few_label = []
                #idx_perm = random.randint(1,101)
                #print("Ramdon index: ", idx_perm)

                for i in range(self.shot):
                    for s in range(self.way):
                        self.few_data.append(self.data_arr[rnd[i]+20*few_classes[s]])
                        self.few_label.append(s+1200)
                        if self.data_aug:
                            self.few_data.append(np.rot90(self.data_arr[rnd[i]+20*few_classes[s]], k=1, axes=(1, 2)).copy())
                            self.few_data.append(np.rot90(self.data_arr[rnd[i]+20*few_classes[s]], k=2, axes=(1, 2)).copy())
                            self.few_data.append(np.rot90(self.data_arr[rnd[i]+20*few_classes[s]], k=3, axes=(1, 2)).copy())
                            self.few_label.append(s+1200)
                            self.few_label.append(s+1200)
                            self.few_label.append(s+1200)


                self.data = self.loader.load_dataset('train', self.size)

                data_array = np.asarray(self.data.values())
                #print(np.shape(data_array))
                self.data_arr = data_array.reshape(-1,*data_array.shape[2:])


                for i in range(self.shot):
                    for s in range(1200):
                        self.few_data.append(self.data_arr[rnd[i]+20*s])
                        self.few_label.append(s)
                        if self.data_aug:
                            self.few_data.append(np.rot90(self.data_arr[rnd[i]+20*s], k=1, axes=(1, 2)).copy())
                            self.few_data.append(np.rot90(self.data_arr[rnd[i]+20*s], k=2, axes=(1, 2)).copy())
                            self.few_data.append(np.rot90(self.data_arr[rnd[i]+20*s], k=3, axes=(1, 2)).copy())
                            self.few_label.append(s)
                            self.few_label.append(s)
                            self.few_label.append(s)
            elif self.whole:
                self.few_data = []
                self.few_label = []

                self.loader = omniglot.Omniglot(self.root, dataset=dataset)
                self.data = self.loader.load_dataset('test', self.size)

                data_array = np.asarray(self.data.values())
                #print(np.shape(data_array))
                self.data_arr = data_array.reshape(-1,*data_array.shape[2:])   
                for i in range(20):
                    for s in range(self.way):
                        self.few_data.append(self.data_arr[i+20*few_classes[s]])
                        self.few_label.append(s+1200)
                        if self.data_aug:
                            self.few_data.append(np.rot90(self.data_arr[i+20*few_classes[s]], k=1, axes=(1, 2)).copy())
                            self.few_data.append(np.rot90(self.data_arr[i+20*few_classes[s]], k=2, axes=(1, 2)).copy())
                            self.few_data.append(np.rot90(self.data_arr[i+20*few_classes[s]], k=3, axes=(1, 2)).copy())
                            self.few_label.append(s+1200)
                            self.few_label.append(s+1200)
                            self.few_label.append(s+1200)   

                self.loader = omniglot.Omniglot(self.root, dataset=dataset)
                self.data = self.loader.load_dataset('train', self.size)

                data_array = np.asarray(self.data.values())
                #print(np.shape(data_array))
                self.data_arr = data_array.reshape(-1,*data_array.shape[2:])   
                for i in range(20):
                    for s in range(1200):
                        self.few_data.append(self.data_arr[i+20*s])
                        self.few_label.append(s)
                        if self.data_aug:
                            self.few_data.append(np.rot90(self.data_arr[i+20*s], k=1, axes=(1, 2)).copy())
                            self.few_data.append(np.rot90(self.data_arr[i+20*s], k=2, axes=(1, 2)).copy())
                            self.few_data.append(np.rot90(self.data_arr[i+20*s], k=3, axes=(1, 2)).copy())
                            self.few_label.append(s)
                            self.few_label.append(s)
                            self.few_label.append(s)      
            else:
                self.few_data = []
                self.few_label = []
                #idx_perm = random.randint(1,101)
                #print("Ramdon index: ", idx_perm)
                #few_classes = random.sample(range(423), self.way)


                rnd = random.sample(range(20),self.shot)
                for i in range(self.shot):
                    for s in range(self.way):
                        self.few_data.append(self.data_arr[rnd[i]+20*few_classes[s]])
                        self.few_label.append(s)
                        if self.data_aug:
                            #self.few_data.append(zca_whiten(self.data_arr[rnd[i]+20*few_classes[s]]).copy())
                            #self.few_data.append(np.flip(self.data_arr[rnd[i]+20*few_classes[s]], axis=1).copy())
                            self.few_data.append(np.rot90(self.data_arr[rnd[i]+20*few_classes[s]], k=1, axes=(1, 2)).copy())
                            self.few_data.append(np.rot90(self.data_arr[rnd[i]+20*few_classes[s]], k=2, axes=(1, 2)).copy())
                            self.few_data.append(np.rot90(self.data_arr[rnd[i]+20*few_classes[s]], k=3, axes=(1, 2)).copy())
                            #self.few_label.append(s)
                            self.few_label.append(s)
                            self.few_label.append(s)
                            self.few_label.append(s)
                            #self.few_label.append(s)
                            #self.few_label.append(s)


        elif dataset == 'mini_imagenet':
            self.loader = mini_imagenet.MiniImagenet(self.root)
            self.data, self.label_encoder = self.loader.load_dataset(self.partition, self.size)
            #print(np.asarray(self.data.values()).shape)
            data_array = np.asarray(self.data.values())
            #print(np.shape(data_array))
            self.data_arr = data_array.reshape(-1,*data_array.shape[2:])
            #print(np.shape(self.data_arr))
            self.few_data = []
            self.few_label = []

            few_classes = random.sample(range(20), self.way)
            #few_classes = [14, 16, 17, 18, 19]
            #few_classes = [1,2,3,4,5]
            self.rnd_seed = random.seed(3)
            rnd = random.sample(range(600),self.shot)
            #if self.fewshot and self.data_aug:
            #print("Classes chosen:", few_classes)
            #print("Rrandom seeds:", rnd)

            for i in range(self.shot):
                for s in range(self.way):
                    self.few_data.append(self.data_arr[rnd[i]+600*few_classes[s]])
                    if self.data_aug:
                        self.few_data.append(np.flip(self.data_arr[rnd[i]+600*few_classes[s]], axis=2).copy())
                        self.few_data.append(np.flip(self.data_arr[rnd[i]+600*few_classes[s]], axis=1).copy())
                        self.few_data.append(np.rot90(self.data_arr[rnd[i]+600*few_classes[s]], k=1, axes=(1, 2)).copy())
                        self.few_data.append(np.rot90(self.data_arr[rnd[i]+600*few_classes[s]], k=2, axes=(1, 2)).copy())
                        self.few_data.append(np.rot90(self.data_arr[rnd[i]+600*few_classes[s]], k=3, axes=(1, 2)).copy())
                        #self.few_data.append(self.rotate_image(self.data_arr[rnd[i]+600*few_classes[s]], 1).copy())
                        #self.few_data.append(self.rotate_image(self.data_arr[rnd[i]+600*few_classes[s]], 2).copy())
                    if self.with_train == True:
                        self.few_label.append(s+64) # +64
                        if self.data_aug:
                            self.few_label.append(s+64) # +64
                            self.few_label.append(s+64) # +64
                            self.few_label.append(s+64) # +64
                            self.few_label.append(s+64) # +64
                            self.few_label.append(s+64) # +64
                    else:
                        self.few_label.append(s)    # +64
                        if self.data_aug:
                            self.few_label.append(s) # +64
                            self.few_label.append(s) # +64
                            self.few_label.append(s) # +64
                            self.few_label.append(s) # +64
                            self.few_label.append(s) # +64

                    #if self.fewshot and (self.shot==1 or self.shot==5):
                    #    print(rnd[i]+600*few_classes[s])
            #self.data_arr = self.data_arr.transpose((0, 2, 3, 1))
        else:
            raise NotImplementedError

        self.class_encoder = {}
        for id_key, key in enumerate(self.data):
            self.class_encoder[key] = id_key

    def rotate_image(self, image, times):
        rotated_image = np.zeros(image.shape)
        for channel in range(image.shape[0]):
            rotated_image[channel, :, :] = np.rot90(image[channel, :, :], k=times)
        return rotated_image

    def get_task_batch(self, batch_size=5, n_way=20, num_shots=1, unlabeled_extra=0, cuda=False, variable=False):
        # Init variables
        batch_x = np.zeros((batch_size, self.input_channels, self.size[0], self.size[1]), dtype='float32')
        labels_x = np.zeros((batch_size, n_way), dtype='float32')
        labels_x_global = np.zeros(batch_size, dtype='int64')
        target_distances = np.zeros((batch_size, n_way * num_shots), dtype='float32')
        hidden_labels = np.zeros((batch_size, n_way * num_shots + 1), dtype='float32')
        numeric_labels = []
        batches_xi, labels_yi, oracles_yi = [], [], []
        for i in range(n_way*num_shots):
            batches_xi.append(np.zeros((batch_size, self.input_channels, self.size[0], self.size[1]), dtype='float32'))
            labels_yi.append(np.zeros((batch_size, n_way), dtype='float32'))
            oracles_yi.append(np.zeros((batch_size, n_way), dtype='float32'))
        # Iterate over tasks for the same batch

        for batch_counter in range(batch_size):
            positive_class = random.randint(0, n_way - 1)

            # Sample random classes for this TASK
            classes_ = list(self.data.keys())
            sampled_classes = random.sample(classes_, n_way)
            indexes_perm = np.random.permutation(n_way * num_shots)

            counter = 0
            for class_counter, class_ in enumerate(sampled_classes):
                if class_counter == positive_class:
                    # We take num_shots + one sample for one class
                    samples = random.sample(self.data[class_], num_shots+1)
                    # Test sample is loaded
                    batch_x[batch_counter, :, :, :] = samples[0]
                    labels_x[batch_counter, class_counter] = 1
                    labels_x_global[batch_counter] = self.class_encoder[class_]
                    samples = samples[1::]
                else:
                    samples = random.sample(self.data[class_], num_shots)

                for s_i in range(0, len(samples)):
                    batches_xi[indexes_perm[counter]][batch_counter, :, :, :] = samples[s_i]
                    if s_i < unlabeled_extra:
                        labels_yi[indexes_perm[counter]][batch_counter, class_counter] = 0
                        hidden_labels[batch_counter, indexes_perm[counter] + 1] = 1
                    else:
                        labels_yi[indexes_perm[counter]][batch_counter, class_counter] = 1
                    oracles_yi[indexes_perm[counter]][batch_counter, class_counter] = 1
                    target_distances[batch_counter, indexes_perm[counter]] = 0
                    counter += 1

            numeric_labels.append(positive_class)

        batches_xi = [torch.from_numpy(batch_xi) for batch_xi in batches_xi]
        labels_yi = [torch.from_numpy(label_yi) for label_yi in labels_yi]
        oracles_yi = [torch.from_numpy(oracle_yi) for oracle_yi in oracles_yi]

        labels_x_scalar = np.argmax(labels_x, 1)

        return_arr = [torch.from_numpy(batch_x), torch.from_numpy(labels_x), torch.from_numpy(labels_x_scalar),
                      torch.from_numpy(labels_x_global), batches_xi, labels_yi, oracles_yi,
                      torch.from_numpy(hidden_labels)]
        if cuda:
            return_arr = self.cast_cuda(return_arr)
        if variable:
            return_arr = self.cast_variable(return_arr)
        return return_arr

    def cast_cuda(self, input):
        if type(input) == type([]):
            for i in range(len(input)):
                input[i] = self.cast_cuda(input[i])
        else:
            return input.cuda()
        return input

    def cast_variable(self, input):
        if type(input) == type([]):
            for i in range(len(input)):
                input[i] = self.cast_variable(input[i])
        else:
            return Variable(input)

        return input


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        #print(index)
        if self.fewshot == True:
            img, target = self.few_data[index], self.few_label[index]

            #print("************************************************")

            # doing this so that it is consistent with all other datasets
            # to return a PIL Image
            #img = Image.fromarray(np.asarray(img),mode='F')
            #img = Image.fromarray(img)
            #img = img.transpose((1, 2, 0))
            #print("original\n",img.shape)
            img = torch.from_numpy(img)
            #print("tesnor\n",img)
            #img = img.numpy()
            #print("numpy\n",img)

            #print("-------------------------------------------------")

            if self.transform is not None:
                img = self.transform(img)


            #print("=================================================",img)

            #if self.transform is not None:
            #    target = self.transform(target)


            return img, target
        else:
            img, target = self.aug_data[index], index/80

            #print("************************************************",img)

            # doing this so that it is consistent with all other datasets
            # to return a PIL Image
            #img = Image.fromarray(np.asarray(img),mode='F')
            #img = img.transpose((1, 2, 0))
            img = torch.from_numpy(img)

            #print("-------------------------------------------------",img)

            if self.transform is not None:
                img = self.transform(img)

            #print("=================================================",img)

            #if self.transform is not None:
            #    target = self.transform(target)


            return img, target

    def __len__(self):
        if self.fewshot == True:
            if self.whole:
                return (self.way+1200)*20
            elif self.both:
                return (self.way+1200)*self.shot*4
            elif self.data_aug:
                return self.way*self.shot*4
            else:
                return self.way*self.shot
        else:
            return len(self.aug_data)
