# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 10:39:20 2019

@author: daghn
"""

import numpy as np
import random
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data
import torch.utils.model_zoo as model_zoo
import pickle
import os
#import torchvision.datasets as dsets
#import torchvision.transforms as transforms
from torch.autograd import Variable
np.set_printoptions(suppress=True)

# Choose network from list:
# 1. FNN
# 2. CNN
# 3. CNN_deep
# 4. CNN_4
# 5. CNN_blowup
# 6. CNN_mini_VGG
# 7. resnet18
active_network = 6

# CUDA 
# Change value 0 or 1
# To switch between running on cpu and GPU (CUDA)
# 0. CPU
# 1. GPU 
Acceleration_device = 0


# Indicate if images needs flattening before being fed to network. 
# They need to be flattened if we use FNN.
if active_network in [1]:
    needs_flattening = True
else:
    needs_flattening = False

# Initialize Hyper-parameters

picture_dimension = 28 # Default is 28

input_size = picture_dimension**2       # The image size = dimension squared
hidden_size = picture_dimension**2     # The number of nodes at the hidden layer

# number of circles and classes
num_circles_min = 0 
num_circles_max = 3
num_classes = num_circles_max + 1
   
num_epochs = 2         # The number of times entire dataset is trained
batch_size = 100       # The size of input data taken for one iteration
learning_rate = 0.001  # The speed of convergence
N = 10000            # Size of train dataset
V = 10000        # Size of test dataset



# Print which network we are running (more can be added)
if active_network == 1:
    print("Running FNN")
if active_network == 2:
    print("Running CNN")
if active_network == 3:
    print("Running CNN_deep")
if active_network == 4:
    print("Running CNN_4")
if active_network == 5:
    print("Running CNN_blowup")
if active_network == 6:
    print("Running CNN_mini_VGG")
if active_network == 7:
    print("Running Resnet18")


def create_image():
    area_min = 0.05
    area_max = 0.20
    
    min_circle_area = 0.2 #minimum area of circle relative to the max 1
    
    margin = 1   # margin, a number of pixels
    circle_buffer = 1.10
    
    # Create empty picture
    picture = np.zeros((picture_dimension, picture_dimension))
    
    # Randomly pick number of circles
    number_of_circles = np.random.randint(num_circles_min, num_circles_max + 1)
    
    # Randomly pick the total area of circles, relative to the picture area
    total_area_of_circles = np.random.uniform(low = area_min, high = area_max)
    
    # Create array of circle data.
    # 5 columns: area, radius, scaled radius, x-coor., y-coor.
    circle_data = np.zeros((number_of_circles, 5))
    
    # Calculate circle areas:
    # First pick random areas relative to max 1
    circle_areas = np.array(np.random.uniform(min_circle_area, 1, number_of_circles))
    
    # Then scale so they sum up to 1, and then multipy with the total area of circles and total area of picture
    # (which is relative to the full image area)
    circle_areas = (circle_areas / np.sum(circle_areas)) * total_area_of_circles
    
    # store the areas in the circle data array
    circle_data[:,0] = circle_areas
    
    # Calculate circle radii
    circle_data[:,1] = np.sqrt(circle_data[:,0]/3.1415) 
    
    # Calculate scaled circle radii
    circle_data[:,2] = circle_data[:,1] * picture_dimension
    
    # Sort circles by size
    circle_data = circle_data[circle_data[:,0].argsort()]
    circle_data = circle_data[::-1]
    
    # Place circles
    for i in range(number_of_circles):
        looking_for_centrum = True
        count = 1
        stop = 1000000
        
        while looking_for_centrum == True and count < stop:
            radius = circle_data[i,2]
            x = np.random.randint(margin + radius, picture_dimension - margin - radius)
            y = np.random.randint(margin + radius, picture_dimension - margin - radius)
            looking_for_centrum = False
            
            for j in range(0, i):
                radius_old = circle_data[j,2]
                x_old = circle_data[j,3]
                y_old = circle_data[j,4]
                
                distance = math.sqrt((x - x_old)**2 + (y - y_old)**2)
                
                if distance < circle_buffer * (radius + radius_old):
                    looking_for_centrum = True
                    break
                
            if looking_for_centrum == False:
                circle_data[i,3] = x
                circle_data[i,4] = y
                   
            count += 1
            if count == stop:
                print("couldn't place circle")
    
    # Update picture
    for i in range(number_of_circles):
        radius = circle_data[i,2]
        R = int(math.ceil(radius))
        x = int(circle_data[i,3])
        y = int(circle_data[i,4])
        
        # Checking all pixels in a sqared box around the circle. If their 
        # distance to the center is less than the radius, we color the pixel.
        box_size = 2 * R + 1        
        
        for j in range(box_size):
            for k in range(box_size):
                if (j - (R + 1))**2 + (k - (R + 1))**2 < radius**2:
                    picture[x - R + j, y - R + k] = 1        
    
#    print(circle_data)
#    print(picture.dtype)
#    
#    plt.imshow(picture)
#    plt.show()
    
    label = number_of_circles
    
    return (picture, label)

def create_dataset(N):
    
    # Create list of N pictures and corresponding list of N labels
    picture_list = []
    label_list = []
    for i in range(N):
        
        # Feedback for process on large images

        # Create one picture
        picture, label = create_image()
        
        # Convert picture from (nxn) to (1xnxn) shape (suitable for torch format)
        picture = picture[np.newaxis, ...]
        
        # Append picture and label to lists
        picture_list.append(picture)
        label_list.append(label)
    
    # Convert to np arrays    
    pictures = np.array(picture_list)
    labels = np.array(label_list)    
    
#    print(pictures.shape)
#    print(labels.shape)
    
    # Convert to torch tensors
    our_data = torch.from_numpy(pictures)
    our_labels = torch.from_numpy(labels)
    
#    print(data.shape)
#    print(labels.shape)
    
    # Encapsulate into a TensorDataset:
    dataset = torch.utils.data.TensorDataset(our_data, our_labels)
    
    return dataset
    

while True:
    input_ = input("Generate new train and test data? load data? load most recent created data? (g/l/r): ")
    if input_ == "g":
        train_dataset = create_dataset(N)
        test_dataset = create_dataset(V)
        
        NN = len(str(N).replace('.',''))-1
        VV = len(str(V).replace('.',''))-1
        np.savez("most_recent_created.npz",NN=NN,VV=VV,N=N,V=V,num_circles_max=num_circles_max)
        
        pickle_out = open("train{}e{}_{}.pickle".format(int(N/10**(NN)),NN,num_circles_max),"wb")
        pickle.dump([train_dataset,N], pickle_out)
        pickle_out.close()
        
        pickle_out = open("test{}e{}_{}.pickle".format(int(V/10**(VV)),VV,num_circles_max),"wb")
        pickle.dump([test_dataset,V], pickle_out)
        pickle_out.close()
        break
        
    if input_ == "l":
        while True: 
            dir_list = os.listdir(os.getcwd())
            dir_array = np.array([dir_list])
            comp_file_list = []
            
            for i in dir_list:
                if i[-7:] == ".pickle":
                    comp_file_list = comp_file_list + [i]
                    
            if len(comp_file_list) > 0: 
                print("\nThe directory contains the follwoing compatible files: ")
                for i in (comp_file_list):
                    print(i)
                break
                    
            else: 
                print("The directory contains no compatible files, the script will probaly crash now.")
                break
        
        while True:
            train_input = input("Write the name of the training data (without file extension): ")
            if train_input+".pickle" in comp_file_list:
                
                pickle_in = open(train_input+".pickle","rb")
                train_dataset,N = pickle.load(pickle_in)
                break
        
        while True:
            test_input = input("Write the name of the test data (without file extension): ")
            if test_input+".pickle" in comp_file_list:
                
                pickle_in = open(test_input+".pickle","rb")
                test_dataset,V = pickle.load(pickle_in)
                break

        break
    
    if input_ == "r":
        tmp = np.load("most_recent_created.npz")
        NN = int(tmp["NN"])
        VV = int(tmp["VV"])
        N = int(tmp["N"])
        V = int(tmp["V"])
        num_circles_max = int(tmp["num_circles_max"])
        
        pickle_in = open("train{}e{}_{}.pickle".format(int(N/10**(NN)),NN,num_circles_max),"rb")
        train_dataset, N = pickle.load(pickle_in)
        
        pickle_in = open("test{}e{}_{}.pickle".format(int(V/10**(VV)),VV,num_circles_max),"rb")
        test_dataset, V = pickle.load(pickle_in)
        
        print("\nTrain data file is: ")
        print("train{}e{}_{}.pickle".format(int(N/10**(NN)),NN,num_circles_max))
        print("\nTest data file is")
        print("test{}e{}_{}.pickle\n\n".format(int(V/10**(VV)),VV,num_circles_max))
        
        break
    
# Load the dataset with the DataLoader utility (giving us batches and other options)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
    
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

# FNN, Feedforward Neural Network 
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()                    # Inherited from the parent class nn.Module
        self.fc1 = nn.Linear(input_size, hidden_size)  # 1st Full-Connected Layer: 784 (input data) -> 500 (hidden node)
        self.relu = nn.ReLU()                          # Non-Linear ReLU Layer: max(0,x)
        self.fc2 = nn.Linear(hidden_size, hidden_size) # 2nd Full-Connected Layer: 500 (hidden node) -> 10 (output class)
        self.relu = nn.ReLU()                          # Non-Linear ReLU Layer: max(0,x)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):                              # Forward pass: stacking each layer together
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

# CNN, Convolutional Neural Network
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        N = 4
        K = picture_dimension // 4
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, N, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(N, 2 * N, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(K * K * (2*N), 1000)
        self.fc2 = nn.Linear(1000, num_classes)


    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out
    
# CNN_4, Convolutional Neural Network
class CNN_4(nn.Module):
       
    def __init__(self):
        super(CNN_4, self).__init__()
        
        N = 32
        K = picture_dimension // 2
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, N, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(N, N, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
            
        self.layer3 = nn.Sequential(
            nn.Conv2d(N, N, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(N, N, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
#            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.layer5 = nn.Sequential(
            nn.Conv2d(N, N, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        
        self.layer6 = nn.Sequential(
            nn.Conv2d(N, N, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(K * K * (1*N), 1000)
        self.fc2 = nn.Linear(1000, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

# CNN_deep, Convolutional Neural Network
class CNN_deep(nn.Module):
    def __init__(self):        
        super(CNN_deep, self).__init__()
        
        N = 24
        K = picture_dimension // 4
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, N, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(N, 2*N, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
            
        self.layer3 = nn.Sequential(
            nn.Conv2d(2*N, 2*N, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(2*N, 2*N, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.layer5 = nn.Sequential(
            nn.Conv2d(2*N, 4*N, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        
        self.layer6 = nn.Sequential(
            nn.Conv2d(4*N, 4*N, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(K * K * (4*N), 1000)
        self.fc2 = nn.Linear(1000, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

# CNN_blowup, Convolutional Neural Network
class CNN_blowup(nn.Module):
    def __init__(self):        
        super(CNN_blowup, self).__init__()
        
        N = 24
        K = picture_dimension // 4
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, N, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(N, 2*N, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
            
        self.layer3 = nn.Sequential(
            nn.Conv2d(2*N, 4*N, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(4*N, 8*N, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        
        self.layer5 = nn.Sequential(
            nn.Conv2d(8*N, 16*N, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        
        
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(K * K * (16*N), 1000)
        self.fc2 = nn.Linear(1000, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out
    
    # CNN_deep, Convolutional Neural Network
class CNN_mini_VGG(nn.Module):
    def __init__(self):        
        super(CNN_mini_VGG, self).__init__()
        
        N = 32
        K = picture_dimension // 4
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, N, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(N, N, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
            
        self.layer3 = nn.Sequential(
            nn.Conv2d(N, 2*N, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(2*N, 2*N, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.layer5 = nn.Sequential(
            nn.Conv2d(2*N, 4*N, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        
        self.layer6 = nn.Sequential(
            nn.Conv2d(4*N, 4*N, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(K * K * (4*N), 1000)
        self.fc2 = nn.Linear(1000, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18(num_classes=num_classes):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes)
    return model

# Instantiate the network
if active_network == 1:
    net = Net(input_size, hidden_size, num_classes)
    
if active_network == 2:
    net = CNN()
    
if active_network == 3:
    net = CNN_4()
    
if active_network == 4:
    net = CNN_deep()
    
if active_network == 5:
    net = CNN_blowup()
    
if active_network == 6:
    net = CNN_mini_VGG()
    
if active_network == 7:
    net = resnet18()

# Enable GPU
# net.cuda()    # You can comment out this line to disable GPU


# Choose the Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
#optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)


# Sets appropriate device for acceleration CPU v GPU, variable set in beginning of script
if Acceleration_device == 0:
    device = torch.device('cpu')
    tensor_type = torch.LongTensor
    
else:
    device = torch.device("cuda")
    net.cuda()
    tensor_type = torch.cuda.LongTensor


def lr_scheduler(optimizer, epoch, init_lr=learning_rate, lr_decay_epoch=1):
    """Decay learning rate by a factor of 0.3 every lr_decay_epoch epochs."""
    lr = init_lr * (0.3**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer
   
# Train the network
total_step = len(train_loader)
loss_list = []
test_loss_list = []
accuracy_list = []
test_accuracy_list = []
for epoch in range(num_epochs):
    # Decay learning rate by a factor of 0.3 every lr_decay_epoch epochs.
    lr_scheduler(optimizer, epoch, init_lr=learning_rate, lr_decay_epoch=1)
        
    for i, (images, labels) in enumerate(train_loader):
        # Flatten the images if we need to before feeding them to the network.
        if needs_flattening == True:
            images = images.view(-1, picture_dimension * picture_dimension).to(device)
        
        # Wrap with torch.autograd.variable (may have some use, but seems unnecessary at the moment)
        images = Variable(images).float()
        images = images.to(device)           
        labels = Variable(labels).type(tensor_type)
        labels = labels.to(device)
        
        # Run the forward pass
        outputs = net(images)                           # Forward pass: compute the output class given an image
        outputs = outputs.to(device)
        loss = criterion(outputs, labels)
        loss_list.append(loss.item())

        # Backprop and perform Adam optimisation
        optimizer.zero_grad()                           # Intialize the hidden weight to all zeros
        loss.backward()                                 # Backward pass: compute the weight
        optimizer.step()                                # Optimizer: update the weights of hidden nodes

        # Track the accuracy
        total = labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels.type(tensor_type)).sum().item()
        accuracy_list.append(correct / total)
        
        # Print info for every 100 steps
        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.0f} %'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                          100*correct / total))

# Test the network:
            
# net.eval() will notify all our layers that we are in eval mode,
# that way, batchnorm or dropout layers will work in eval model instead of training mode.
net.eval()

# torch.no_grad() impacts the autograd engine and deactivates it.
# It will reduce memory usage and speed up computations but we won’t be able to backprop (which we
# don't do anyway during test).
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        # Flatten the images if we need to before feeding them to the network.
        if needs_flattening == True:
            images = images.view(-1, picture_dimension * picture_dimension)
            images = images.to(device)
                
#        outputs = net(images.float().to(device))
#        outputs = outputs.to(device)
#        test_loss = criterion(outputs, labels)
#        test_loss_list.append(test_loss.item())
        _, predicted = torch.max(outputs.data, 1)  # Choose the best class from the output: The class with the best score
        total += labels.size(0)                    # Increment the total count
        correct += (predicted == labels.type(tensor_type)).sum().item()     # Increment the correct count
#        accuracy_list.append(correct / total)
        
    print('Accuracy of the network on the 10K test images: ' + str(float(100 * float(correct) / total)) + "%")

# Defining plots
#epoch_count = 
fig, (ax1, ax2) = plt.subplots(nrows=2)

ax1.plot(loss_list, 'r-')
#ax1.plot(test_loss_list, 'b-')
ax1.set(title='model loss', xlabel='epoch', ylabel='loss')
ax1.legend()

ax2.plot(accuracy_list, 'r-')
#ax2.plot(test_accuracy_list, 'b-')
ax2.set(title='Accuracy', xlabel='epoch', ylabel='accuracy')
ax2.legend()

plt.show

# Save the network and plot
#torch.save(net.state_dict(), MODEL_STORE_PATH + 'conv_net_model.ckpt')
    
def demonstrate_network():
    # Create one picture
    picture, label = create_image()
    
    # Show picture
    plt.imshow(picture)
    plt.show()
    
    # Convert picture from (nxn) to (1xnxn) shape (suitable for torch format)
    picture = picture[np.newaxis, ...]
    
    # Turn picture into torch format
    our_data = torch.from_numpy(np.array([picture]))
    our_labels = torch.from_numpy(np.array([label]))
    test_dataset = torch.utils.data.TensorDataset(our_data, our_labels)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=1,
                                          shuffle=False)
    
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            # Flatten the images if we need to before feeding them to the network.
            if needs_flattening == True:
                images = images.view(-1, picture_dimension * picture_dimension).to(device)

            outputs = net(images.float().to(device))
            outputs = outputs.to(device)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.type(tensor_type)).sum().item()


    print('Test Accuracy of the model on the 1 single test image: {} %'.format((correct / total) * 100))

    print("Ground truth is " + str(label) + " circles.")
    print("Network estimate is... " + str(int(predicted)))

    print(outputs.data)

