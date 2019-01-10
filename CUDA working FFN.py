# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import random
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data
import pickle
import os
#import torchvision.datasets as dsets
#import torchvision.transforms as transforms
from torch.autograd import Variable
np.set_printoptions(suppress=True)

#Set device to CUDA
device = torch.device("cuda:0")

# Initialize Hyper-parameters
picture_dimension = 28
input_size = picture_dimension**2       # The image size = 28 x 28 = 784
hidden_size = 784      # The number of nodes at the hidden layer

# number of circles and classes
num_circles_min = 0 
num_circles_max = 20
num_classes = num_circles_max - num_circles_min + 1    
   
num_epochs = 5        # The number of times entire dataset is trained
batch_size = 100       # The size of input data took for one iteration
learning_rate = 0.001  # The speed of convergence
N = 10000          # Size of train dataset
V = 1000                # Size of test dataset

def create_image():
    area_min = 0.05
    area_max = 0.10
    
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
    
# make new dataset, load old, or load most recent
while True:
    input_ = input("Generate new train and test data? load data? load most recent created data? (g/l/r): ")
    if input_ == "g":
        train_dataset = create_dataset(N)
        test_dataset = create_dataset(V)
        
        NN = len(str(N).replace('.',''))-1
        VV = len(str(V).replace('.',''))-1
        np.savez("most_recent_created.npz",NN=NN,VV=VV,N=N,V=V,num_circles_max=num_circles_max)
        
        pickle_out = open("train{}e{}_{}.pickle".format(int(N/10**(NN)),NN,num_circles_max),"wb")
        pickle.dump(train_dataset, pickle_out)
        pickle_out.close()
        
        pickle_out = open("test{}e{}_{}.pickle".format(int(V/10**(VV)),VV,num_circles_max),"wb")
        pickle.dump(test_dataset, pickle_out)
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
                train_dataset = pickle.load(pickle_in)
                break
        
        while True:
            test_input = input("Write the name of the test data (without file extension): ")
            if test_input+".pickle" in comp_file_list:
                
                pickle_in = open(test_input+".pickle","rb")
                test_dataset = pickle.load(pickle_in)
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
        train_dataset = pickle.load(pickle_in)
        
        pickle_in = open("test{}e{}_{}.pickle".format(int(V/10**(VV)),VV,num_circles_max),"rb")
        test_dataset = pickle.load(pickle_in)
        
        print("\nTrain data file is: ")
        print("train{}e{}_{}.pickle".format(int(N/10**(NN)),NN,num_circles_max))
        print("\nTest data file is")
        print("test{}e{}_{}.pickle\n\n".format(int(V/10**(VV)),VV,num_circles_max))
        
        break
        
        
    
# Load the dataset with the DataLoader utility (giving us batches and other options)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=False)
    
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

# Feedforward Neural Network Model Structure
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

##Convolutional Neural Network
#class ConvNet(nn.Module):
#    def __init__(self):
#        super(ConvNet, self).__init__()
#        self.layer1 = nn.Sequential(
#            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
#            nn.ReLU(),
#            nn.MaxPool2d(kernel_size=2, stride=2))
#        self.layer2 = nn.Sequential(
#            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
#            nn.ReLU(),
#            nn.MaxPool2d(kernel_size=2, stride=2))
#        self.drop_out = nn.Dropout()
#        self.fc1 = nn.Linear(7 * 7 * 64, 1000)
#        self.fc2 = nn.Linear(1000, num_classes)
#
#    def forward(self, x):
#        out = self.layer1(x)
#        out = self.layer2(out)
#        out = out.reshape(out.size(0), -1)
#        out = self.drop_out(out)
#        out = self.fc1(out)
#        out = self.fc2(out)
#        return out

# Instantiate the network
net = Net(input_size, hidden_size, num_classes)
#net = net.to(device)
#net = ConvNet()

# Enable GPU
net.cuda()    # You can comment out this line to disable GPU

# Choose the Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
#optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

# Train the FNN Model
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):   # Load a batch of images with its (index, data, class)
        images = Variable(images.view(-1, 28*28)).float()           # Convert torch tensor to Variable: change image from a vector of size 784 to a matrix of 28 x 28
        images = images.to(device)
        labels = Variable(labels).type(torch.cuda.LongTensor)
        labels = labels.to(device)
        optimizer.zero_grad()                             # Intialize the hidden weight to all zeros
        outputs = net(images)                             # Forward pass: compute the output class given a image
        loss = criterion(outputs, labels)                 # Compute the loss: difference between the output class and the pre-given label
        loss.backward()                                   # Backward pass: compute the weight
        optimizer.step()                                  # Optimizer: update the weights of hidden nodes
        
        if (i+1) % 100 == 0:                              # Logging
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                 %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data))

## Train the CNN model
#total_step = len(train_loader)
#loss_list = []
#acc_list = []
#for epoch in range(num_epochs):
#    for i, (images, labels) in enumerate(train_loader):
#        # Run the forward pass
#        outputs = net(images.float())
#        loss = criterion(outputs, labels.type(torch.LongTensor))
#        loss_list.append(loss.item())
#
#        # Backprop and perform Adam optimisation
#        optimizer.zero_grad()
#        loss.backward()
#        optimizer.step()
#
#        # Track the accuracy
#        total = labels.size(0)
#        _, predicted = torch.max(outputs.data, 1)
#        correct = (predicted == labels.type(torch.LongTensor)).sum().item()
#        acc_list.append(correct / total)
#
#        if (i + 1) % 100 == 0:
#            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:2f} %'
#                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
#                          100*correct / total))

# Testing the FNN Model
correct = 0
total = 0
for images, labels in test_loader:
    images = Variable(images.view(-1, 28*28)).float() 
    images = images.to(device)
    outputs = net(images)
    _, predicted = torch.max(outputs.data, 1)  # Choose the best class from the output: The class with the best score
    total += labels.size(0)                    # Increment the total count
    correct += (predicted == labels.type(torch.cuda.LongTensor)).sum()     # Increment the correct count
    
print('Accuracy of the network on the {}K test images: '.format(int(N/1000)) + str(float(100 * float(correct) / total)) + "%")

## Testing the CNN model
#net.eval()
#with torch.no_grad():
#    correct = 0
#    total = 0
#    for images, labels in test_loader:
#        outputs = net(images.float())
#        _, predicted = torch.max(outputs.data, 1)
#        total += labels.size(0)
#        correct += (predicted == labels.type(torch.LongTensor)).sum().item()
#
#    print('Test Accuracy of the model on the 10000 test images: {} %'.format((correct / total) * 100))

# Save the model and plot
#torch.save(net.state_dict(), MODEL_STORE_PATH + 'conv_net_model.ckpt')

def demonstrate_FFN():
    # Create one picture
    picture, label = create_image()
    
    # Show picture
    plt.imshow(picture)
    plt.show()
    
    # Turn picture into torch format
    our_data = torch.from_numpy(np.array([picture]))
    our_labels = torch.from_numpy(np.array([label]))
    test_dataset = torch.utils.data.TensorDataset(our_data, our_labels)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=1,
                                          shuffle=False)
    
    correct = 0
    total = 0
    predicted = 0
    for images, labels in test_loader:
        images = Variable(images.view(-1, 28*28)).float() 
        images = images.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)  # Choose the best class from the output: The class with the best score
        
        total += labels.size(0)                    # Increment the total count
        correct += (predicted == labels.type(torch.cuda.LongTensor)).sum()     # Increment the correct count
        
    print('Accuracy of the network on the 1 test image: %f %%' % (100. * correct / total)) 
    
    print("Ground truth is " + str(label) + " circles.")
    print("Network estimate is... " + str(int(predicted)))
    print(outputs.data)
    
#def demonstrate_CNN():
#    # Create one picture
#    picture, label = create_image()
#    
#    # Show picture
#    plt.imshow(picture)
#    plt.show()
#    
#    # Turn picture into torch format
#    our_data = torch.from_numpy(np.array([picture]))
#    our_labels = torch.from_numpy(np.array([label]))
#    test_dataset = torch.utils.data.TensorDataset(our_data, our_labels)
#    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
#                                          batch_size=1,
#                                          shuffle=False)
#    
#    with torch.no_grad():
#        correct = 0
#        total = 0
#        for images, labels in test_loader:
#            outputs = net(images.float())
#            _, predicted = torch.max(outputs.data, 1)
#            total += labels.size(0)
#            correct += (predicted == labels.type(torch.LongTensor)).sum().item()
#
#    print('Test Accuracy of the model on the 10000 test images: {} %'.format((correct / total) * 100))
#
#    print("Ground truth is " + str(label) + " circles.")
#    print("Network estimate is... " + str(int(predicted)))
#    print(outputs.data)
 