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
    
    # Determine if there is at least 4 colors with more than 2 objects present:
    number_of_at_least_2_members = int( len([x for x in color_label_vector if x>=2]) >=4 )
    
    # Determine number of red triangles
    number_of_red_triangles = len([x for x in circle_data if x[5] == 1 and x[6] == 3])
    
    # Determine number of colors with a circle present    
    circle_objects = circle_data[circle_data[:,6] == 2]
    colors_of_circle_objects = circle_objects[:,5]
    number_of_colors_with_circle_present = len(np.unique(colors_of_circle_objects))
    
    # Determine if number of colors with a rectangle present is at least 3
    rectangle_objects = circle_data[circle_data[:,6] == 4]
    colors_of_rectangle_objects = rectangle_objects[:,5]
    number_of_colors_with_rectangle_present = len(np.unique(colors_of_rectangle_objects))
    is_there_at_least_3_colors_with_rectangle_present = int( number_of_colors_with_rectangle_present >= 3)
    
    
    
    # Turn single labels into arrays for concatenation
    total_number_of_circles = np.array([total_number_of_circles])
    number_of_colors_in_picture = np.array([number_of_colors_in_picture])
    number_of_at_least_2_members = np.array([number_of_at_least_2_members])
    number_of_red_triangles = np.array([number_of_red_triangles])
    number_of_colors_with_circle_present = np.array([number_of_colors_with_circle_present])
    is_there_at_least_3_colors_with_rectangle_present = np.array([is_there_at_least_3_colors_with_rectangle_present])
    
    # Create vector of all labels
    all_labels_vector = np.concatenate((total_number_of_circles,
                                        color_label_vector,
                                        shape_label_vector,
                                        number_of_colors_in_picture, 
                                        number_of_at_least_2_members, 
                                        number_of_red_triangles, 
                                        number_of_colors_with_circle_present, 
                                        is_there_at_least_3_colors_with_rectangle_present))
    
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
    picture_1, all_labels_vector = create_image()
    
    # Show picture
    plt.imshow(picture_1, interpolation='nearest', origin='lower',
                    cmap=cmap, norm=norm)
    plt.show()
    
    # Convert picture from (nxn) to (1xnxn) shape (suitable for torch format)
    picture = picture_1[np.newaxis, ...]
    
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
            output_19 = outputs[19].to(device)
            output_20 = outputs[20].to(device)
            
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
            
            output_2 = softmax(output_2)       # applies softmax to outputs for easy interpretation
            _, predicted_2 = torch.max(output_2.data, 1)
            output_2 = output_2.cpu()          # loads the output back to the cpu so it can be changed to numpy
            output_2 = output_2.data.numpy()   # unwrap from Variable with .data, then transform to numpy so we get scientific suppression
            print("----- Task 2 -----")
            print("Ground truth is " + str(int(all_labels_vector[2])) )
            print("Network estimate is... " + str(int(predicted_2)))
            print("Based on these softmax values:")
            print(output_2)
            print("")
            
            output_3 = softmax(output_3)       # applies softmax to outputs for easy interpretation
            _, predicted_3 = torch.max(output_3.data, 1)
            output_3 = output_3.cpu()          # loads the output back to the cpu so it can be changed to numpy
            output_3 = output_3.data.numpy()   # unwrap from Variable with .data, then transform to numpy so we get scientific suppression
            print("----- Task 3 -----")
            print("Ground truth is " + str(int(all_labels_vector[3])) )
            print("Network estimate is... " + str(int(predicted_3)))
            print("Based on these softmax values:")
            print(output_3)
            print("")
            
            output_4 = softmax(output_4)       # applies softmax to outputs for easy interpretation
            _, predicted_4 = torch.max(output_4.data, 1)
            output_4 = output_4.cpu()          # loads the output back to the cpu so it can be changed to numpy
            output_4 = output_4.data.numpy()   # unwrap from Variable with .data, then transform to numpy so we get scientific suppression
            print("----- Task 4 -----")
            print("Ground truth is " + str(int(all_labels_vector[4])) )
            print("Network estimate is... " + str(int(predicted_4)))
            print("Based on these softmax values:")
            print(output_4)
            print("")
            
            output_5 = softmax(output_5)       # applies softmax to outputs for easy interpretation
            _, predicted_5 = torch.max(output_5.data, 1)
            output_5 = output_5.cpu()          # loads the output back to the cpu so it can be changed to numpy
            output_5 = output_5.data.numpy()   # unwrap from Variable with .data, then transform to numpy so we get scientific suppression
            print("----- Task 5 -----")
            print("Ground truth is " + str(int(all_labels_vector[5])) )
            print("Network estimate is... " + str(int(predicted_5)))
            print("Based on these softmax values:")
            print(output_5)
            print("")
            
            output_6 = softmax(output_6)       # applies softmax to outputs for easy interpretation
            _, predicted_6 = torch.max(output_6.data, 1)
            output_6 = output_6.cpu()          # loads the output back to the cpu so it can be changed to numpy
            output_6 = output_6.data.numpy()   # unwrap from Variable with .data, then transform to numpy so we get scientific suppression
            print("----- Task 6 -----")
            print("Ground truth is " + str(int(all_labels_vector[6])) )
            print("Network estimate is... " + str(int(predicted_6)))
            print("Based on these softmax values:")
            print(output_6)
            print("")
            
            output_7 = softmax(output_7)       # applies softmax to outputs for easy interpretation
            _, predicted_7 = torch.max(output_7.data, 1)
            output_7 = output_7.cpu()          # loads the output back to the cpu so it can be changed to numpy
            output_7 = output_7.data.numpy()   # unwrap from Variable with .data, then transform to numpy so we get scientific suppression
            print("----- Task 7 -----")
            print("Ground truth is " + str(int(all_labels_vector[7])) )
            print("Network estimate is... " + str(int(predicted_7)))
            print("Based on these softmax values:")
            print(output_7)
            print("")
            
            output_8 = softmax(output_8)       # applies softmax to outputs for easy interpretation
            _, predicted_8 = torch.max(output_8.data, 1)
            output_8 = output_8.cpu()          # loads the output back to the cpu so it can be changed to numpy
            output_8 = output_8.data.numpy()   # unwrap from Variable with .data, then transform to numpy so we get scientific suppression
            print("----- Task 8 -----")
            print("Ground truth is " + str(int(all_labels_vector[8])) )
            print("Network estimate is... " + str(int(predicted_8)))
            print("Based on these softmax values:")
            print(output_8)
            print("")
            
            output_9 = softmax(output_9)       # applies softmax to outputs for easy interpretation
            _, predicted_9 = torch.max(output_9.data, 1)
            output_9 = output_9.cpu()          # loads the output back to the cpu so it can be changed to numpy
            output_9 = output_9.data.numpy()   # unwrap from Variable with .data, then transform to numpy so we get scientific suppression
            print("----- Task 9 -----")
            print("Ground truth is " + str(int(all_labels_vector[9])) )
            print("Network estimate is... " + str(int(predicted_9)))
            print("Based on these softmax values:")
            print(output_9)
            print("")
            
            output_10 = softmax(output_10)       # applies softmax to outputs for easy interpretation
            _, predicted_10 = torch.max(output_10.data, 1)
            output_10 = output_10.cpu()          # loads the output back to the cpu so it can be changed to numpy
            output_10 = output_10.data.numpy()   # unwrap from Variable with .data, then transform to numpy so we get scientific suppression
            print("----- Task 10 -----")
            print("Ground truth is " + str(int(all_labels_vector[10])) )
            print("Network estimate is... " + str(int(predicted_10)))
            print("Based on these softmax values:")
            print(output_10)
            print("")
            
            output_11 = softmax(output_11)       # applies softmax to outputs for easy interpretation
            _, predicted_11 = torch.max(output_11.data, 1)
            output_11 = output_11.cpu()          # loads the output back to the cpu so it can be changed to numpy
            output_11 = output_11.data.numpy()   # unwrap from Variable with .data, then transform to numpy so we get scientific suppression
            print("----- Task 11 -----")
            print("Ground truth is " + str(int(all_labels_vector[11])) )
            print("Network estimate is... " + str(int(predicted_11)))
            print("Based on these softmax values:")
            print(output_11)
            print("")
            
            output_12 = softmax(output_12)       # applies softmax to outputs for easy interpretation
            _, predicted_12 = torch.max(output_12.data, 1)
            output_12 = output_12.cpu()          # loads the output back to the cpu so it can be changed to numpy
            output_12 = output_12.data.numpy()   # unwrap from Variable with .data, then transform to numpy so we get scientific suppression
            print("----- Task 12 -----")
            print("Ground truth is " + str(int(all_labels_vector[12])) )
            print("Network estimate is... " + str(int(predicted_12)))
            print("Based on these softmax values:")
            print(output_12)
            print("")
            
            output_13 = softmax(output_13)       # applies softmax to outputs for easy interpretation
            _, predicted_13 = torch.max(output_13.data, 1)
            output_13 = output_13.cpu()          # loads the output back to the cpu so it can be changed to numpy
            output_13 = output_13.data.numpy()   # unwrap from Variable with .data, then transform to numpy so we get scientific suppression
            print("----- Task 13 -----")
            print("Ground truth is " + str(int(all_labels_vector[13])) )
            print("Network estimate is... " + str(int(predicted_13)))
            print("Based on these softmax values:")
            print(output_13)
            print("")
            
            output_14 = softmax(output_14)       # applies softmax to outputs for easy interpretation
            _, predicted_14 = torch.max(output_14.data, 1)
            output_14 = output_14.cpu()          # loads the output back to the cpu so it can be changed to numpy
            output_14 = output_14.data.numpy()   # unwrap from Variable with .data, then transform to numpy so we get scientific suppression
            print("----- Task 14 -----")
            print("Ground truth is " + str(int(all_labels_vector[14])) )
            print("Network estimate is... " + str(int(predicted_14)))
            print("Based on these softmax values:")
            print(output_14)
            print("")
            
            output_15 = softmax(output_15)       # applies softmax to outputs for easy interpretation
            _, predicted_15 = torch.max(output_15.data, 1)
            output_15 = output_15.cpu()          # loads the output back to the cpu so it can be changed to numpy
            output_15 = output_15.data.numpy()   # unwrap from Variable with .data, then transform to numpy so we get scientific suppression
            print("----- Task 15 -----")
            print("Ground truth is " + str(int(all_labels_vector[15])) )
            print("Network estimate is... " + str(int(predicted_15)))
            print("Based on these softmax values:")
            print(output_15)
            print("")
            
            output_16 = softmax(output_16)       # applies softmax to outputs for easy interpretation
            _, predicted_16 = torch.max(output_16.data, 1)
            output_16 = output_16.cpu()          # loads the output back to the cpu so it can be changed to numpy
            output_16 = output_16.data.numpy()   # unwrap from Variable with .data, then transform to numpy so we get scientific suppression
            print("----- Task 16 -----")
            print("Ground truth is " + str(int(all_labels_vector[16])) )
            print("Network estimate is... " + str(int(predicted_16)))
            print("Based on these softmax values:")
            print(output_16)
            print("")
            
            output_17 = softmax(output_17)       # applies softmax to outputs for easy interpretation
            _, predicted_17 = torch.max(output_17.data, 1)
            output_17 = output_17.cpu()          # loads the output back to the cpu so it can be changed to numpy
            output_17 = output_17.data.numpy()   # unwrap from Variable with .data, then transform to numpy so we get scientific suppression
            print("----- Task 17 -----")
            print("Ground truth is " + str(int(all_labels_vector[17])) )
            print("Network estimate is... " + str(int(predicted_17)))
            print("Based on these softmax values:")
            print(output_17)
            print("")
            
            output_18 = softmax(output_18)       # applies softmax to outputs for easy interpretation
            _, predicted_18 = torch.max(output_18.data, 1)
            output_18 = output_18.cpu()          # loads the output back to the cpu so it can be changed to numpy
            output_18 = output_18.data.numpy()   # unwrap from Variable with .data, then transform to numpy so we get scientific suppression
            print("----- Task 18 -----")
            print("Ground truth is " + str(int(all_labels_vector[18])) )
            print("Network estimate is... " + str(int(predicted_18)))
            print("Based on these softmax values:")
            print(output_18)
            print("")
            
            output_19 = softmax(output_19)       # applies softmax to outputs for easy interpretation
            _, predicted_19 = torch.max(output_19.data, 1)
            output_19 = output_19.cpu()          # loads the output back to the cpu so it can be changed to numpy
            output_19 = output_19.data.numpy()   # unwrap from Variable with .data, then transform to numpy so we get scientific suppression
            print("----- Task 19 -----")
            print("Ground truth is " + str(int(all_labels_vector[19])) )
            print("Network estimate is... " + str(int(predicted_19)))
            print("Based on these softmax values:")
            print(output_19)
            print("")
            
            output_20 = softmax(output_20)       # applies softmax to outputs for easy interpretation
            _, predicted_20 = torch.max(output_20.data, 1)
            output_20 = output_20.cpu()          # loads the output back to the cpu so it can be changed to numpy
            output_20 = output_20.data.numpy()   # unwrap from Variable with .data, then transform to numpy so we get scientific suppression
            print("----- Task 20 -----")
            print("Ground truth is " + str(int(all_labels_vector[20])) )
            print("Network estimate is... " + str(int(predicted_20)))
            print("Based on these softmax values:")
            print(output_20)
            print("")
            
            # Show picture
            plt.imshow(picture_1, interpolation='nearest', origin='lower',
                    cmap=cmap, norm=norm)
            plt.show()
            
            print("Task 0. Ground truth and estimate: "+ str(int(all_labels_vector[0])) + "  " + str(int(predicted_0)))
            print("")
            print("Task 1. Ground truth and estimate: "+ str(int(all_labels_vector[1])) + "  " + str(int(predicted_1)))
            print("Task 2. Ground truth and estimate: "+ str(int(all_labels_vector[2])) + "  " + str(int(predicted_2)))
            print("Task 3. Ground truth and estimate: "+ str(int(all_labels_vector[3])) + "  " + str(int(predicted_3)))
            print("Task 4. Ground truth and estimate: "+ str(int(all_labels_vector[4])) + "  " + str(int(predicted_4)))
            print("Task 5. Ground truth and estimate: "+ str(int(all_labels_vector[5])) + "  " + str(int(predicted_5)))
            print("Task 6. Ground truth and estimate: "+ str(int(all_labels_vector[6])) + "  " + str(int(predicted_6)))
            print("Task 7. Ground truth and estimate: "+ str(int(all_labels_vector[7])) + "  " + str(int(predicted_7)))
            print("Task 8. Ground truth and estimate: "+ str(int(all_labels_vector[8])) + "  " + str(int(predicted_8)))
            print("Task 9. Ground truth and estimate: "+ str(int(all_labels_vector[9])) + "  " + str(int(predicted_9)))
            print("Task 10. Ground truth and estimate: "+ str(int(all_labels_vector[10])) + "  " + str(int(predicted_10)))
            print("")
            print("Task 11. Ground truth and estimate: "+ str(int(all_labels_vector[11])) + "  " + str(int(predicted_11)))
            print("Task 12. Ground truth and estimate: "+ str(int(all_labels_vector[12])) + "  " + str(int(predicted_12)))
            print("Task 13. Ground truth and estimate: "+ str(int(all_labels_vector[13])) + "  " + str(int(predicted_13)))
            print("Task 14. Ground truth and estimate: "+ str(int(all_labels_vector[14])) + "  " + str(int(predicted_14)))
            print("Task 15. Ground truth and estimate: "+ str(int(all_labels_vector[15])) + "  " + str(int(predicted_15)))
            print("")
            print("Task 16. Ground truth and estimate: "+ str(int(all_labels_vector[16])) + "  " + str(int(predicted_16)))
            print("Task 17. Ground truth and estimate: "+ str(int(all_labels_vector[17])) + "  " + str(int(predicted_17)))
            print("Task 18. Ground truth and estimate: "+ str(int(all_labels_vector[18])) + "  " + str(int(predicted_18)))
            print("Task 19. Ground truth and estimate: "+ str(int(all_labels_vector[19])) + "  " + str(int(predicted_19)))
            print("Task 20. Ground truth and estimate: "+ str(int(all_labels_vector[20])) + "  " + str(int(predicted_20)))
            