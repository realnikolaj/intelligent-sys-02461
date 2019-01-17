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
Acceleration_device = 1

# Initialize Hyper-parameters

picture_dimension = 128 # Default is 28

input_size = picture_dimension**2       # The image size = dimension squared
hidden_size = picture_dimension**2     # The number of nodes at the hidden layer

# number of circles and classes
num_circles_min = 0
num_circles_max = 20
num_classes = num_circles_max + 1
   
num_epochs = 10           # The number of times entire dataset is trained
lr_decay_epoch = 100     # How many epochs before changing learning rate
learning_rate = 0.001  # The speed of convergence
batch_size = 100          # The size of input data taken for one iteration
N = 10000 # Size of train dataset
V = 10000  # Size of tracking test dataset

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
# 1 - N. colors, starting from red, as above in default categories
# N+1. total number of colors
active_heads = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
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

def create_image():
    area_min = 0.10
    area_max = 0.20
    
    min_circle_area = 0.2 #minimum area of circle relative to the max 1
    
    margin = 1   # margin, a number of pixels
    circle_buffer = 1.10
    
    # Create empty picture
    picture = np.zeros((picture_dimension, picture_dimension))
    
    # Randomly pick total number of circles
    total_number_of_circles = np.random.randint(num_circles_min, num_circles_max + 1)
    
    # Create array of circle data.
    # 7 columns: area, radius, scaled radius, x-coor., y-coor., color, shape
    circle_data = np.zeros((total_number_of_circles, 7))
    
    # Randomly pick number of colors in palette (how many colors we can choose from for this picture)
    number_of_colors_in_palette = np.random.randint(1, num_colors + 1)
    
    # Randomly pick which colors to include from the palette
    colors_in_picture = random.sample(range(1, num_colors + 1), number_of_colors_in_palette)
    
    # Assign random color to each circle
    circle_data[:,5] = np.array(random.choices(population=colors_in_picture, k=total_number_of_circles))
    
    # Randomly pick the total area of circles, relative to the picture area
    total_area_of_circles = np.random.uniform(low = area_min, high = area_max)
    
    # Calculate circle areas:
    # First pick random areas relative to max 1
    circle_areas = np.array(np.random.uniform(min_circle_area, 1, total_number_of_circles))
    
    # Then scale areas so they sum up to 1,
    # and then multipy with the total area of circles
    # (which is in [0,1], as in relative to the total picture area)
    circle_areas = (circle_areas / np.sum(circle_areas)) * total_area_of_circles
    
    # store the relative areas in the circle data array
    circle_data[:,0] = circle_areas
    
    # Calculate relative circle radii
    circle_data[:,1] = np.sqrt(circle_data[:,0]/3.1415) 
    
    # Calculate scaled circle radii
    circle_data[:,2] = circle_data[:,1] * picture_dimension
    
    # Sort circles by size
    circle_data = circle_data[circle_data[:,0].argsort()]
    circle_data = circle_data[::-1]
    
    # Place circles
    for i in range(total_number_of_circles):
        looking_for_centrum = True
        count = 1
        stop = 1000
        
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
    for i in range(total_number_of_circles):
        radius = circle_data[i,2]
        x = int(circle_data[i,3])
        y = int(circle_data[i,4])
        R = int(math.ceil(radius))
        
        # Pick random shape
        shape = random.randint(1,5)
        
        # Store in circle data
        circle_data[i, 6] = shape
        
        # Squares
        if shape == 1:
            RR = math.ceil(R*0.7)
            picture[x-RR:x+1+RR,y-RR:y+1+RR] = circle_data[i, 5] / 10
        
        # Circles
        if shape == 2:
            # Checking all pixels in a sqared box around the circle. If their 
            # distance to the center is less than the radius, we color the pixel.
            box_size = 2 * R + 1        
            
            for j in range(box_size):
                for k in range(box_size):
                    if (j - (R + 1))**2 + (k - (R + 1))**2 < radius**2:
                        picture[x - R + j, y - R + k] = circle_data[i, 5] / 10
                        
        # Triangles
        if shape == 3:
            RR = math.ceil(R*0.8)
            for j in range(R):
                picture[(x+j),y-R+j:y+R-j] = circle_data[i, 5] / 10         
        
        # Rectangles        
        if shape == 4:
            picture[x-math.ceil(R*0.4):x+math.ceil(R*0.4),y-R:y+R] = circle_data[i, 5] / 10                
                        
        # Half circles    
        if shape == 5:
            box_size = (2 * R + 1) 
            
            for j in range(R + 1):
                for k in range(box_size):
                    if (j - (R + 1))**2 + (k - (R + 1))**2 < radius**2:
                        picture[x - R + j, y - R + k] = circle_data[i, 5] / 10
    
    # Turn data into variable name
    object_colors = circle_data[:,5]
    
    # Count the number of objects of the various colors, and put in a dictionary
    dictionary_for_number_of_color_occurrences = Counter(object_colors)
    
    # Make a simple vector of color count labels, using the dictionary
    color_label_vector = np.zeros(10)
    for i in range(10):
        color_label_vector[i] = dictionary_for_number_of_color_occurrences[i+1]
        
    # Turn data into variable name
    object_shapes = circle_data[:,6]
    
    # Count the number of objects of the various shapes, and put in a dictionary
    dictionary_for_number_of_shape_occurrences = Counter(object_shapes)
    
    # Make a simple vector of shape count labels, using the dictionary
    shape_label_vector = np.zeros(5)
    for i in range(5):
        shape_label_vector[i] = dictionary_for_number_of_shape_occurrences[i+1]    
    
    # Determine the number of different colors in the picture
    number_of_colors_in_picture = len(np.unique(object_colors))
    
    # Determine number of colors with more than 2 objects present:
    number_of_at_least_2_members = int( len([x for x in color_label_vector if x>=2]) >=4 )
    
    # Determine number of red triangles
    number_of_red_triangles = len([x for x in circle_data if x[5] == 1 and x[6] == 3])
    
    # Turn single labels into arrays for concatenation
    total_number_of_circles = np.array([total_number_of_circles])
    number_of_colors_in_picture = np.array([number_of_colors_in_picture])
    number_of_at_least_2_members = np.array([number_of_at_least_2_members])
    number_of_red_triangles = np.array([number_of_red_triangles])
    
    # Create vector of all labels
    all_labels_vector = np.concatenate((total_number_of_circles, color_label_vector, shape_label_vector, number_of_colors_in_picture, number_of_at_least_2_members, number_of_red_triangles))
    
    #    print(circle_data)
#    print(picture.dtype)
#    
#    plt.imshow(picture, interpolation='nearest', origin='lower',
#                    cmap=cmap, norm=norm)
#    plt.show()
    
    return (picture, all_labels_vector)

def create_dataset(N):
    # Create list of N pictures and corresponding list of N labels
    picture_list = []
    label_list = []
    
    for i in range(N):
        
        # Create one picture
        picture, all_labels_vector = create_image()
        
        # Convert picture from (nxn) to (1xnxn) shape (suitable for torch format)
        picture = picture[np.newaxis, ...]
        
        # Append picture and label to lists
        picture_list.append(picture)
        label_list.append(all_labels_vector)
    
    # Convert to np arrays    
    pictures = np.array(picture_list)
    labels = np.array(label_list)    
    
    # Convert to torch tensors
    our_data = torch.from_numpy(pictures)
    our_labels = torch.from_numpy(labels)
    
    # Encapsulate into a TensorDataset:
    dataset = torch.utils.data.TensorDataset(our_data, our_labels)
    
    return dataset

# Download dataset (we actually create it on the fly)
print(N)
train_dataset = create_dataset(N)
test_dataset = create_dataset(V)
    
# Load the dataset with the DataLoader utility (giving us batches and other options)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=False)
    
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

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
        out_at_least_2_members = self.head17(out)
        number_of_red_triangles = self.head18(out)
        
        return out_total, out_color1, out_color2, out_color3, out_color4, out_color5, out_color6, out_color7, out_color8, out_color9, out_color10, out_shape1, out_shape2, out_shape3, out_shape4, out_shape5, out_different_colors, out_at_least_2_members, number_of_red_triangles
        
                
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

total_step = len(train_loader)
loss_list = []
test_loss_list = []
accuracy_list = []
test_accuracy_list = []
for epoch in range(num_epochs):
#    # Decay learning rate by a factor of 0.3 every lr_decay_epoch epochs.
#    lr_scheduler(optimizer, epoch, init_lr=learning_rate, lr_decay_epoch=lr_decay_epoch)
    print("")
    print("Beginning new epoch." + ' Epoch [{}/{}].'.format(epoch + 1, num_epochs))
    print("Making pictures")
    train_dataset = create_dataset(N)
    print("Done making pictures")
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,   
                                               batch_size=batch_size,
                                               shuffle=False)
    
    print("Training on 10k pictures, 100 steps")
    
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
        
#        output_0 = outputs[0].to(device)
#        output_1 = outputs[1].to(device)
#        output_2 = outputs[2].to(device)
#        output_3 = outputs[3].to(device)
#        output_4 = outputs[4].to(device)
#        output_5 = outputs[5].to(device)
#        output_6 = outputs[6].to(device)
#        output_7 = outputs[7].to(device)
#        output_8 = outputs[8].to(device)
#        output_9 = outputs[9].to(device)
#        output_10 = outputs[10].to(device)
#        output_11 = outputs[11].to(device)
#        output_12 = outputs[12].to(device)
#        output_13 = outputs[13].to(device)
#        output_14 = outputs[14].to(device)
#        output_15 = outputs[15].to(device)
#        output_16 = outputs[16].to(device)
#        output_17 = outputs[17].to(device)
#        
#        loss_0 = criterion(output_0, labels[:,0])
#        loss_1 = criterion(output_1, labels[:,1])
#        loss_2 = criterion(output_2, labels[:,2])
#        loss_3 = criterion(output_3, labels[:,3])
#        loss_4 = criterion(output_4, labels[:,4])
#        loss_5 = criterion(output_5, labels[:,5])
#        loss_6 = criterion(output_6, labels[:,6])
#        loss_7 = criterion(output_7, labels[:,7])
#        loss_8 = criterion(output_8, labels[:,8])
#        loss_9 = criterion(output_9, labels[:,9])
#        loss_10 = criterion(output_10, labels[:,10])
#        loss_11 = criterion(output_11, labels[:,11])
#        loss_12 = criterion(output_12, labels[:,12])
#        loss_13 = criterion(output_13, labels[:,13])
#        loss_14 = criterion(output_14, labels[:,14])
#        loss_15 = criterion(output_15, labels[:,15])
#        loss_16 = criterion(output_16, labels[:,16])
#        loss_17 = criterion(output_17, labels[:,17])
   

#        loss_list.append(loss.item())
        
        # Backprop and perform Adam optimisation
        optimizer.zero_grad()                           # Intialize the hidden weight to all zeros
        loss.backward()                                 # Backward pass: compute the weight
        optimizer.step()                                # Optimizer: update the weights of hidden nodes
        
        # Track the accuracy every 10 steps by testing the network on 1k test dataset:
        
        if (i + 1) % 100 == 0:   
            print("Done training 100 steps. Here's the data:")
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
                
                # Printing accuracy for each active head
                for j, head in enumerate(active_heads):  
                    print('Task {} --- Loss: {:.4f}, Accuracy: {:.2f} %'.format(head, individual_losses[j].item(), 100*correct[j] / V))        
            net.train()       
            
def demonstrate_network():
    picture_list = []
    label_list = []
    
    # Create one picture
    picture, all_labels_vector = create_image()
    
    # Show picture
    plt.imshow(picture, interpolation='nearest', origin='lower',
                    cmap=cmap, norm=norm)

    plt.show()
    
    # Convert picture from (nxn) to (1xnxn) shape (suitable for torch format)
    picture = picture[np.newaxis, ...]
    
    # Append picture and label to lists
    picture_list.append(picture)
    label_list.append(all_labels_vector)
    
    # Convert to np arrays    
    pictures = np.array(picture_list)
    labels = np.array(label_list)    
    
    # Convert to torch tensors
    our_data = torch.from_numpy(pictures)
    our_labels = torch.from_numpy(labels)
    
    test_dataset = torch.utils.data.TensorDataset(our_data, our_labels)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=1,
                                          shuffle=False)
    
    with torch.no_grad():
        for images, labels in test_loader:
            
            # Wrap with torch.autograd.variable (may have some use, but seems unnecessary at the moment)
            images = Variable(images).float()
            images = images.to(device)
            labels = Variable(labels).type(tensor_type)
            labels = labels.to(device)
        
            outputs = net(images)
            
            softmax = nn.Softmax(dim=1)
            
            output_0 = outputs[0].to(device)
            output_1 = outputs[1].to(device)
            output_2 = outputs[2].to(device)
            output_3 = outputs[3].to(device)
            output_4 = outputs[4].to(device)
            output_5 = outputs[5].to(device)
            output_6 = outputs[6].to(device)
            output_7 = outputs[7].to(device)
            output_8 = outputs[8].to(device)
            output_9 = outputs[9].to(device)
            output_10 = outputs[10].to(device)
            output_11 = outputs[11].to(device)
            output_12 = outputs[12].to(device)
            output_13 = outputs[13].to(device)
            output_14 = outputs[14].to(device)
            output_15 = outputs[15].to(device)
            output_16 = outputs[16].to(device)
            output_17 = outputs[17].to(device)
            output_18 = outputs[18].to(device)
            
            
            output_0 = softmax(output_0)       # applies softmax to outputs for easy interpretation
            _, predicted_0 = torch.max(output_0.data, 1)
            output_0 = output_0.cpu()          # loads the output back to the cpu so it can be changed to numpy
            output_0 = output_0.data.numpy()   # unwrap from Variable with .data, then transform to numpy so we get scientific suppression
            print("----- Task 0 -----")
            print("Ground truth is " + str(int(all_labels_vector[0])) )
            print("Network estimate is... " + str(int(predicted_0)))
            print("Based on these softmax values:")
            print(output_0)
            print("")
            
            output_1 = softmax(output_1)       # applies softmax to outputs for easy interpretation
            _, predicted_1 = torch.max(output_1.data, 1)
            output_1 = output_1.cpu()          # loads the output back to the cpu so it can be changed to numpy
            output_1 = output_1.data.numpy()   # unwrap from Variable with .data, then transform to numpy so we get scientific suppression
            print("----- Task 1 -----")
            print("Ground truth is " + str(int(all_labels_vector[1])) )
            print("Network estimate is... " + str(int(predicted_1)))
            print("Based on these softmax values:")
            print(output_1)
            print("")
            
            print("Task 0. Ground truth and estimate: "+ str(int(all_labels_vector[0])) + "  " + str(int(predicted_0)))
            print("Task 1. Ground truth and estimate: "+ str(int(all_labels_vector[1])) + "  " + str(int(predicted_1)))