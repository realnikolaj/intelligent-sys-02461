# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 12:38:36 2019

@author: daghn
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 10:39:20 2019

@author: daghn
"""

import numpy as np
import random
import math
import matplotlib.pyplot as plt
from matplotlib import colors
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from collections import Counter
import torch.utils.model_zoo as model_zoo
import pickle
import os
#import torchvision.datasets as dsets
#import torchvision.transforms as transforms
from torch.autograd import Variable
np.set_printoptions(suppress=True)

torch.cuda.empty_cache()

# CUDA 
# Change value 0 or 1
# To switch between running on cpu and GPU (CUDA)
# 0. CPU
# 1. GPU 
Acceleration_device = 0

# Initialize Hyper-parameters

picture_dimension = 128 # Default is 28

input_size = picture_dimension**2       # The image size = dimension squared
hidden_size = picture_dimension**2     # The number of nodes at the hidden layer

# number of circles and classes
num_circles_min = 0
num_circles_max = 20
num_classes = num_circles_max + 1
   
num_epochs = 100           # The number of times entire dataset is trained
lr_decay_epoch = 100     # How many epochs before changing learning rate
learning_rate = 0.001  # The speed of convergence
batch_size = 100          # The size of input data taken for one iteration
N = 10000 # Size of train dataset
V = 10000  # Size of validation dataset

F = 32     # Number of filters in first convolutional layer in the net
layer_size = 5000   # size of fully connected layer before heads

# number of colors
num_colors = 10

# make a color map of fixed colors
default_colors = ['black', 'red', 'blue', 'green', 'yellow', 'darkorange', 'rebeccapurple', 'saddlebrown', 'orchid', 'gray', 'white']
cmap = colors.ListedColormap(default_colors)
bounds= (np.array([0,1,2,3,4,5,6,7,8,9,10,11]) - 0.5) / 10
norm = colors.BoundaryNorm(bounds, cmap.N)

default_categories = ['Total number of circles',
                      'red', 'blue', 'green', 'yellow', 'orange', 'purple', 'brown', 'hotpink', 'gray', 'white',
                      'Total number of colors']

# These active categories correspond to 
# 0. total number of circles
# 1 - 10. number of given color, starting from red, as above in default categories
# 11-15. number of given shape: square, circle, triangles, rectangles, half circles
# 16. number of different colors
# 17. is there at least 4 colors with more than 2 objects present? 1 for true, 0 for false.
# 18. number of red triangles
# 19. number of colors with a circle present
# 20. is there at least 3 colors with a rectangle present
# 21. is there at least 3 yellow objects?
# 22. is there at least 2 green halfcircles?

active_heads = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
#active_heads = [0,1]
primary_head = 7

#head_legends = 

def lr_scheduler(optimizer, epoch, init_lr=learning_rate, lr_decay_epoch=lr_decay_epoch):
    """Decay learning rate by a factor of 0.3 every lr_decay_epoch epochs."""
    lr = init_lr * (0.3**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer

# CNN_mini_VGG, Convolutional Neural Network
class CNN_mini_VGG(nn.Module):
    def __init__(self):        
        super(CNN_mini_VGG, self).__init__()
        
        layer_size = 5000
        
        K = picture_dimension // 8
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, F, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(F, F, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
            
        self.layer3 = nn.Sequential(
            nn.Conv2d(F, 2*F, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(2*F, 2*F, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.layer5 = nn.Sequential(
            nn.Conv2d(2*F, 4*F, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        
        self.layer6 = nn.Sequential(
            nn.Conv2d(4*F, 4*F, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.drop_out = nn.Dropout()        
        self.fc1 = nn.Linear(K * K * (4*F), layer_size)
        self.relu = nn.ReLU(inplace=True)
        
        self.head0 = nn.Linear(layer_size, num_classes)
        self.head1 = nn.Linear(layer_size, num_classes)
        self.head2 = nn.Linear(layer_size, num_classes)
        self.head3 = nn.Linear(layer_size, num_classes)
        self.head4 = nn.Linear(layer_size, num_classes)
        self.head5 = nn.Linear(layer_size, num_classes)
        self.head6 = nn.Linear(layer_size, num_classes)
        self.head7 = nn.Linear(layer_size, num_classes)
        self.head8 = nn.Linear(layer_size, num_classes)
        self.head9 = nn.Linear(layer_size, num_classes)
        self.head10 = nn.Linear(layer_size, num_classes)
        self.head11 = nn.Linear(layer_size, num_classes)
        self.head12 = nn.Linear(layer_size, num_classes)
        self.head13 = nn.Linear(layer_size, num_classes)
        self.head14 = nn.Linear(layer_size, num_classes)
        self.head15 = nn.Linear(layer_size, num_classes)
        self.head16 = nn.Linear(layer_size, num_colors + 1)
        self.head17 = nn.Linear(layer_size, 2)
        self.head18 = nn.Linear(layer_size, num_classes)
        self.head19 = nn.Linear(layer_size, num_colors + 1)
        self.head20 = nn.Linear(layer_size, 2)
        

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
        out = self.relu(out)
        
        out_total = self.head0(out)
        out_color1 = self.head1(out)
        out_color2 = self.head2(out)
        out_color3 = self.head3(out)
        out_color4 = self.head4(out)
        out_color5 = self.head5(out)
        out_color6 = self.head6(out)
        out_color7 = self.head7(out)
        out_color8 = self.head8(out)
        out_color9 = self.head9(out)
        out_color10 = self.head10(out)
        out_shape1 = self.head11(out)
        out_shape2 = self.head12(out)
        out_shape3 = self.head13(out)
        out_shape4 = self.head14(out)
        out_shape5 = self.head15(out)
        out_different_colors = self.head16(out)
        out_colors_with_at_least_2_members = self.head17(out)
        out_number_of_red_triangles = self.head18(out)
        out_number_of_colors_with_circle_present = self.head19(out) 
        out_at_least_3_colors_with_rectangle_present = self.head20(out) 
        
        return out_total, out_color1, out_color2, out_color3, out_color4, out_color5, out_color6, out_color7, out_color8, out_color9, out_color10, out_shape1, out_shape2, out_shape3, out_shape4, out_shape5, out_different_colors, out_colors_with_at_least_2_members, out_number_of_red_triangles, out_number_of_colors_with_circle_present, out_at_least_3_colors_with_rectangle_present 
        
                
# Instantiate the network
print("Instantiating the network")        
net = CNN_mini_VGG()
    
# Print which network we are running (more can be added)
print("Running CNN_mini_VGG")
 
# Sets appropriate device for acceleration CPU v GPU, variable set in beginning of script
if Acceleration_device == 0:
    device = torch.device('cpu')
    tensor_type = torch.LongTensor
    
else:
    device = torch.device("cuda")
    net.cuda()
    tensor_type = torch.cuda.LongTensor
    
# Choose the Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
#optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)


# Train the network

loss_list = []
test_loss_list = []
accuracy_list = []
test_accuracy_list = []
#for epoch in range(num_epochs):
##    # Decay learning rate by a factor of 0.3 every lr_decay_epoch epochs.
##    lr_scheduler(optimizer, epoch, init_lr=learning_rate, lr_decay_epoch=lr_decay_epoch)
#    print("")
#    print("Beginning new epoch." + ' Epoch [{}/{}].'.format(epoch + 1, num_epochs))
#    print("Making pictures")
#    train_dataset = create_dataset(N)
#    print("Done making pictures")
#    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,   
#                                               batch_size=batch_size,
#                                               shuffle=False)
#    
#    print("Training on 10k pictures, 100 steps")

    
for i in range(10):
    
    train_dataset = torch.load('dataset{}.pt'.format(i))
    test_dataset = torch.load('testset{}.pt'.format(i))

    # Load the dataset with the DataLoader utility (giving us batches and other options)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                       batch_size=batch_size,
                                       shuffle=False)
    
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                      batch_size=batch_size,
                                      shuffle=False)
    
    
    for i, (images, labels) in enumerate(train_loader):
    
        # Wrap with torch.autograd.variable (may have some use, but seems unnecessary at the moment)
        images = Variable(images).float()
        images = images.to(device)
        labels = Variable(labels).type(tensor_type)
        labels = labels.to(device)
        
        number_of_labels = labels.size(1)
        
        # Run the forward pass
        outputs = net(images)
        individual_losses = [criterion(outputs[k].to(device), labels[:,k]) for k in active_heads]
        loss = sum(individual_losses)


        loss_list.append(loss.item())
        
        # Backprop and perform Adam optimisation
        optimizer.zero_grad()                           # Intialize the hidden weight to all zeros
        loss.backward()                                 # Backward pass: compute the weight
        optimizer.step()                                # Optimizer: update the weights of hidden nodes

        # Track the accuracy every 10 steps by testing the network on 1k test dataset:
        
     
#        print("Done training {}/100 steps. Here's the data:".format(i))       
#        print("Total loss: " + str(round(loss.item(),3)))
        
        if (i + 1) % 100 == 0:   
            print("Done training 100 steps. Here's the data:")
            print("Total loss: " + str(round(loss.item(),3)))
            
            # net.eval() will notify all our layers that we are in eval mode,
            # that way, batchnorm or dropout layers will work in eval model instead of training mode.
            net.eval()
            
            # torch.no_grad() impacts the autograd engine and deactivates it.
            # It will reduce memory usage and speed up computations but we won’t be able to backprop (which we
            # don't do anyway during test).
            with torch.no_grad():
                correct = np.zeros(len(active_heads))
                
                # Counting correct outputs
                for imagess, labels in test_loader:
                    
                    imagess = Variable(imagess).float()
                    imagess = imagess.to(device)
                    outputss = net(imagess)
                    
                    # Counting correct outputs for each active head
                    for j, head in enumerate(active_heads):                        
                        outputt_j = outputss[head].to(device)
                        _, predicted = torch.max(outputt_j.data, 1)
                        correct[j] += (predicted == labels[:,head].type(tensor_type)).sum().item()
                        accuracy_list.append(100*correct[j] / V)
                # Printing accuracy for each active head
#                for j, head in enumerate(active_heads):  
#                    print('Task {} --- Loss: {:.4f}, Accuracy: {:.2f} %'.format(head, individual_losses[j].item(), 100*correct[j] / V))        
            net.train()     

