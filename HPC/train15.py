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
import pandas as pd
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
Acceleration_device = 1

# Initialize Hyper-parameters

picture_dimension = 128 # The picture width and height will be this dimension

input_size = picture_dimension**2       # The image size = dimension squared
hidden_size = picture_dimension**2     # The number of nodes at the hidden layer

# number of objects and classes
num_objects_min = 0
num_objects_max = 30
num_classes = num_objects_max + 1
   
num_epochs = 300           # The number of times 10k pictures is trained
lr_decay_epoch = 1000     # How many epochs before changing learning rate (note, ADAM is used, so no hard reason to change learning rate)
learning_rate = 0.0005     # The speed of convergence
batch_size = 100          # The number of pictures for one training iteration
N = 1000                # Number of pictures loaded (or created) into memory for training at a time
V = 10000           # Size of test dataset for accuracy tracking

F = 32     # Number of filters in first convolutional layer in the net. It is a multiplicator for all convolutional layers.
layer_size = 5000   # size of fully connected layer before heads

# number of colors
num_colors = 10    # This one shouldn't be changed
num_shapes = 5   # This one shouldn't be changed

# make a color map of fixed colors
default_colors = ['black', 'red', 'blue', 'green', 'yellow', 'darkorange', 'rebeccapurple', 'saddlebrown', 'orchid', 'gray', 'white']
cmap = colors.ListedColormap(default_colors)
bounds= (np.array([0,1,2,3,4,5,6,7,8,9,10,11]) - 0.5) / 10
norm = colors.BoundaryNorm(bounds, cmap.N)

default_categories = ['Total number of objects',
                      'red', 'blue', 'green', 'yellow', 'orange', 'purple', 'brown', 'hotpink', 'gray', 'white',
                      'Total number of colors']

# These active categories correspond to 
# 0. total number of objects
# 1-10. number of objects of given color, starting from red, as above in default categories
# 11-15. number of objects of given shape: square, circle, triangles, rectangles, half circles
# 16. number of different colors
# 17. number of different shapes

# 18. is there at least one red triangle?
# 19. is there at least one red triangle OR at least one white circle?
# 20. is there at least 2 colors with a triangle present?
# 21. is there at least 2 shapes with a red object present?

# 22. is there at least one white sharp object? (sharp = rectangle, square, triangle)
# 23. is there at least one white sharp object OR blue sharp object? 
# 24. is there at least one white sharp object AND at least one blue object?
# 25. is there at least one white sharp object AND (at least one (green OR blue) sharp object)?
# 26. is there at least one white sharp object AND NOT a blue sharp object?

# 27. is there at least 3 colors with a sharp object present?
# 28. is there at least 2 shapes with a sun color present? (sun colors: yellow, orange, red)

# 29. is there (at least one red triangle OR one white circle OR one orange rectangle)?
# 30. is there (at least one red triangle OR one white circle OR one orange rectangle) AND a blue object?

# 31. is there at least 3 colors with 3 different shapes present?
# 32. number of sun color objects
# 33. number of circles, triangles and squares
# 34. is there a blue or pink square present, as well as a purple or white circle?

# 35. number of colors with a triangle present
# 36. number of colors with a circle, or a triangle, or a rectangle present
# 37. number of colors with at least 2 different shapes present 

#active_heads = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37]
#active_heads = [34]
active_heads = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
#active_heads = [0]


#def lr_scheduler(optimizer, epoch, init_lr=learning_rate, lr_decay_epoch=lr_decay_epoch):
#    """Decay learning rate by a factor of 0.3 every lr_decay_epoch epochs."""
#    lr = init_lr * (0.3**(epoch // lr_decay_epoch))
#
#    if epoch % lr_decay_epoch == 0:
#        print('LR is set to {}'.format(lr))
#
#    for param_group in optimizer.param_groups:
#        param_group['lr'] = lr
#
#    return optimizer


# measures the fraction of images with a positive label ( =1 ) in a random 10k set
def count_positives(label_index):
    count = 0
    for i in range(10000):
        value = create_image()[1][label_index]
        if value == 1:
            count += 1
    return count / 10000    


# Download dataset (we actually create it on the fly)
#train_dataset = create_dataset(N)
#test_dataset = create_dataset(V)
    

# Load the dataset with the DataLoader utility (giving us batches and other options)
#train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
#                                           batch_size=batch_size,
#                                           shuffle=False)
#    
#test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
#                                          batch_size=batch_size,
#                                          shuffle=False)

# CNN_mini_VGG, Convolutional Neural Network
class CNN_mini_VGG(nn.Module):
    def __init__(self):        
        super(CNN_mini_VGG, self).__init__()
        
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
        self.head17 = nn.Linear(layer_size, num_shapes + 1)
        self.head18 = nn.Linear(layer_size, 2)
        self.head19 = nn.Linear(layer_size, 2)
        self.head20 = nn.Linear(layer_size, 2)
        self.head21 = nn.Linear(layer_size, 2)
        self.head22 = nn.Linear(layer_size, 2)
        self.head23 = nn.Linear(layer_size, 2)
        self.head24 = nn.Linear(layer_size, 2)
        self.head25 = nn.Linear(layer_size, 2)
        self.head26 = nn.Linear(layer_size, 2)
        self.head27 = nn.Linear(layer_size, 2)
        self.head28 = nn.Linear(layer_size, 2)
        self.head29 = nn.Linear(layer_size, 2)
        self.head30 = nn.Linear(layer_size, 2)
        self.head31 = nn.Linear(layer_size, 2)
        self.head32 = nn.Linear(layer_size, num_classes)
        self.head33 = nn.Linear(layer_size, num_classes)
        self.head34 = nn.Linear(layer_size, 2)
        self.head35 = nn.Linear(layer_size, num_colors + 1)
        self.head36 = nn.Linear(layer_size, num_colors + 1)
        self.head37 = nn.Linear(layer_size, num_colors + 1)
        
        
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
        out_color_1 = self.head1(out)
        out_color_2 = self.head2(out)
        out_color_3 = self.head3(out)
        out_color_4 = self.head4(out)
        out_color_5 = self.head5(out)
        out_color_6 = self.head6(out)
        out_color_7 = self.head7(out)
        out_color_8 = self.head8(out)
        out_color_9 = self.head9(out)
        out_color_10 = self.head10(out)
        out_shape_1 = self.head11(out)
        out_shape_2 = self.head12(out)
        out_shape_3 = self.head13(out)
        out_shape_4 = self.head14(out)
        out_shape_5 = self.head15(out)
        out_different_colors = self.head16(out)
        out_different_shapes = self.head17(out)
        out_head_18 = self.head18(out)
        out_head_19 = self.head19(out)
        out_head_20 = self.head20(out)
        out_head_21 = self.head21(out)
        out_head_22 = self.head22(out)
        out_head_23 = self.head23(out)
        out_head_24 = self.head24(out)
        out_head_25 = self.head25(out)
        out_head_26 = self.head26(out)
        out_head_27 = self.head27(out)
        out_head_28 = self.head28(out)
        out_head_29 = self.head29(out)
        out_head_30 = self.head30(out)
        out_head_31 = self.head31(out)
        out_head_32 = self.head32(out)
        out_head_33 = self.head33(out)
        out_head_34 = self.head34(out)
        out_head_35 = self.head35(out)
        out_head_36 = self.head36(out)
        out_head_37 = self.head37(out)
        
        
        return (out_total, 
                out_color_1,
                out_color_2, 
                out_color_3,
                out_color_4, 
                out_color_5,
                out_color_6,
                out_color_7,
                out_color_8,
                out_color_9,
                out_color_10,
                out_shape_1,
                out_shape_2,
                out_shape_3,
                out_shape_4,
                out_shape_5,
                out_different_colors, 
                out_different_shapes,
                out_head_18, 
                out_head_19, 
                out_head_20, 
                out_head_21,
                out_head_22,
                out_head_23,
                out_head_24,
                out_head_25, 
                out_head_26, 
                out_head_27,
                out_head_28, 
                out_head_29,
                out_head_30,
                out_head_31,
                out_head_32,
                out_head_33,
                out_head_34,
                out_head_35,
                out_head_36,
                out_head_37) 
        
                
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
loss_list = np.empty([40, num_epochs])
accuracy_list = np.empty([40, num_epochs])

test_dataset = torch.load('testset.pt') 
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                      batch_size=batch_size,
                                      shuffle=False)
            
for epoch in range(num_epochs):    
    
    train_dataset = torch.load('dataset{}.pt'.format(epoch))
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,   
                                                   batch_size=batch_size,
                                                   shuffle=False)
    print("")
    print("Loading file: [{}/300].".format(epoch))
        
    # Train the network on 10k pictures
    print("")
    print("Beginning new epoch." + ' Epoch [{}/{}].'.format(epoch + 1, num_epochs))
    print("Training on 10000 pictures, 100 steps of 100 pictures")
        
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

   #        loss_list.append(loss.item())
        
        # Backprop and perform Adam optimisation
        optimizer.zero_grad()                           # Intialize the hidden weight to all zeros
        loss.backward()                                 # Backward pass: compute the weight
        optimizer.step()                                # Optimizer: update the weights of hidden nodes
            
    # Track the accuracy every epoch (10k pictures) by testing the network on a fixed 10k test dataset:
    print("")
    print("Done training 100 batches of 100 pictures. Here's the data:")
    print("")
    print('----- Epoch [{}/{}] -----'.format(epoch + 1, num_epochs))
   
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
        
        # Printing and storing accuracy for each active head, as well as the loss from the last training batch
        for j, head in enumerate(active_heads):  
            print('Task {:>2} --- Loss: {:.3f}, Accuracy: {:.2f} %'.format(head, individual_losses[j].item(), 100*correct[j] / V)) 
            loss_list[head][epoch] = individual_losses[j].item()
            accuracy_list[head][epoch] = correct[j] / V
    net.train()                   

# now loss_list and accuracy_list are done and can be saved to file
           
np.save("_0_to_15_accuracy_list",accuracy_list)
np.save("_0_to_15_loss_list",loss_list)    

fig, (ax1, ax2) = plt.subplots(nrows=2)

for i in range(22):
    ax1.plot(loss_list[i], label='loss')
ax1.set(title='model loss', xlabel='epoch', ylabel='loss')

for i in range(22):
    ax2.plot(accuracy_list[i], label='accuracy')
ax2.set(title='Accuracy', xlabel='batches', ylabel='Accuracy')


plt.show()



fig.savefig('network_result15.png', transparent=False, dpi=80, bbox_inches="tight")

 
            
