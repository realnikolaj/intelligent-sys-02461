import numpy as np
import random
import math
import matplotlib.pyplot as plt
from matplotlib import colors
import torch
import torch.nn as nn
import torch.utils.data
from collections import Counter
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
Acceleration_device = 1

# Indicate if images needs flattening before being fed to network. 
# They need to be flattened if we use FNN.
if active_network in [1]:
    needs_flattening = True
else:
    needs_flattening = False

# Initialize Hyper-parameters

picture_dimension = 100 # Default is 28

input_size = picture_dimension**2       # The image size = dimension squared
hidden_size = picture_dimension**2     # The number of nodes at the hidden layer

# number of circles and classes
num_circles_min = 0 
num_circles_max = 20
num_classes = num_circles_max + 1

# 1 for yes (beta)
overlapping_circles = 0
   
num_epochs = 3         # The number of times entire dataset is trained
batch_size = 100       # The size of input data taken for one iteration
learning_rate = 0.001  # The speed of convergence
N = 10000              # Size of train dataset
V = 1000    # Size of test dataset

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
active_categories = np.arange(0, num_colors + 2)

# use this for manually choosing active categories (ie. what to count) (so far, only choose 1 category or things will break down):
active_categories = np.array([1])

# Print which network we are running (more can be added)
#if active_network == 1:
#    print("Running FNN")
#if active_network == 2:
#    print("Running CNN")
#if active_network == 3:
#    print("Running CNN_deep")
#if active_network == 4:
#    print("Running CNN_4")
#if active_network == 5:
#    print("Running CNN_blowup")
#if active_network == 6:
#    print("Running CNN_mini_VGG")
#if active_network == 7:
#    print("Running Resnet18")


def create_image():
    area_min = 0.10
    area_max = 0.40
    
    min_circle_area = 0.2 #minimum area of circle relative to the max 1
    
    margin = 1   # margin, a number of pixels
    circle_buffer = 1.2
    
    # Create empty picture
    picture = np.zeros((picture_dimension, picture_dimension))
    
    # Randomly pick total number of circles
    total_number_of_circles = np.random.randint(num_circles_min, num_circles_max + 1)
    
    # Create array of circle data.
    # 6 columns: area, radius, scaled radius, x-coor., y-coor., color, object shape
    circle_data = np.zeros((total_number_of_circles, 7))
    
    # Randomly pick number of colors in palette (how many colors we can choose from for this picture)
    number_of_colors_in_palette = np.random.randint(1, num_colors + 1)
    
    # Randomly pick which colors to include in the palette
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
    invalid_circles = []
    for i in range(total_number_of_circles):
        looking_for_centrum = True
        count = 1
        stop = 20
        
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
                
                #50% overlapping cirkles
                if overlapping_circles == 1:
                    distance = distance*2
                    
                
                if distance < (circle_buffer * (radius_old+radius)):
#                and circle_data[j,5] == circle_data[i,5]):
                    looking_for_centrum = True
                    break
                
            if looking_for_centrum == False:
                circle_data[i,3] = x
                circle_data[i,4] = y
                   
            count += 1
            
            if count == stop:
#                print("couldn't place 1 circle")
                invalid_circles = invalid_circles + [i]
                
                break

    
    #delete invalid circles
    for i in reversed(invalid_circles):
        circle_data = np.delete(circle_data,i,0)
        

    # Update picture
    for i in range(len(circle_data[:,0])):
        radius = circle_data[i,2]
        x = math.ceil(circle_data[i,3])
        y = math.ceil(circle_data[i,4])
        R = math.ceil(math.ceil(radius))
        
        # Checking all pixels in a sqared box around the circle. If their 
        # distance to the center is less than the radius, we color the pixel.
        shape = random.randint(1,10)
        
        # squares
        # object shape = 1
        if shape == 1:
            
            RR = math.ceil(R*0.75)
            picture[x-RR:x+1+RR,y-RR:y+1+RR] = circle_data[i, 5] / 10
            circle_data[i,6] = 1
                    
        # cirkles 
        # object shape = 2
        if shape == 2:
                        
            box_size = (2 * R + 1)        
            
            for j in range(box_size):
                for k in range(box_size):
                    if (j - (R + 1))**2 + (k - (R + 1))**2 < radius**2:
                        picture[x - R + j, y - R + k] = circle_data[i, 5] / 10
                        circle_data[i,6] = 2
        
        # triangles
        # object shape = 3
        if shape == 3:

            for j in range(2*R):
                picture[(x-R+j),y-R+j:y+R-j]= circle_data[i, 5] / 10
                circle_data[i,6] = 3
        
        # rectangles 
        # object shape = 4
        if shape == 4:
            
            picture[x-math.ceil(R*0.4):x+math.ceil(R*0.4),y-R:y+R] = circle_data[i, 5] / 10
            circle_data[i,6] = 4

                    
        # half cirkles 
        # object shape = 5
        if shape == 5:
            box_size = (2 * R + 1) 
            
            for j in range(box_size):
                for k in range(math.ceil(box_size*0.5)):
                    if (j - (R + 1))**2 + (k - (R + 1))**2 < radius**2:

                        picture[x - R + j, y - R + k] = circle_data[i, 5] / 10
            
            circle_data[i,6] = 5
            
        #cross
        if shape == 6:
            
            
            picture[x-math.ceil(R*0.2):x+math.ceil(R*0.2),y-R:y+R] = circle_data[i, 5] / 10
            picture[x-R:x+R,y-math.ceil(R*0.2):y+math.ceil(R*0.2)] = circle_data[i, 5] / 10
            
            
            circle_data[i,6] = 6
        
        # capital i    
        if shape == 7:
            
            
            picture[x+math.ceil(R*0.5):x+math.ceil(R),y-R:y+R] = circle_data[i, 5] / 10
            picture[x-math.ceil(R):x-math.ceil(R*0.5),y-R:y+R] = circle_data[i, 5] / 10
            picture[x-R:x+R,y-math.ceil(R*0.25):y+math.ceil(R*0.25)] = circle_data[i, 5] / 10
            
            
            circle_data[i,6] = 7
            
        # z   
        if shape == 8:
            
            
            picture[x+math.ceil(R*0.5):x+math.ceil(R),y-R:y+R] = circle_data[i, 5] / 10
            picture[x-math.ceil(R):x-math.ceil(R*0.5),y-R:y+R] = circle_data[i, 5] / 10
            
            for v in range (math.ceil(R*1.2)):
                picture[ (x-math.ceil(R*0.5)+v)  :  (x-math.ceil(R*0.5)+1+v)  ,  (y-R+v) : (y-R+math.ceil(R*0.7)+v) ] = circle_data[i, 5] / 10
            
            circle_data[i,6] = 8
          
        # hourglass    
        if shape == 9:

            for j in range(R):
                picture[(x-math.ceil(R*0.8)+j),y-R+j:y+R-j]= circle_data[i, 5] / 10
                picture[(x+math.ceil(R*0.8)-j),y-R+j:y+R-j]= circle_data[i, 5] / 10
                
                
            circle_data[i,6] = 9
            
            
        # cross
        if shape == 10:
            
            picture[x+math.ceil(R*0.5):x+math.ceil(R),y-R:y+R] = circle_data[i, 5] / 10
            picture[x-math.ceil(R*0.7):x+R,y-R:y-math.ceil(R*0.5)] = circle_data[i, 5] / 10
            
            for v in range (math.ceil(R*1.2)):
                w = random.randint(1,2)
                picture[ (x+math.ceil(R*0.5)-v)  :  (x+math.ceil(R*0.5)+1-v)  ,  (y-R+v) : (y-R+math.ceil(R*0.7)+v) ] = circle_data[i, 5] / 10

            circle_data[i,6] = 10
        
            
        
            
            
            
                
                
                
        
        
    
#    print(circle_data)
#    print(picture.dtype)
#    
    plt.imshow(picture, interpolation='nearest', origin='lower',
                    cmap=cmap, norm=norm)
    plt.show()
    
    object_properties = circle_data[:,5]
    dictionary_for_number_of_occurrences = Counter(object_properties)
    
    number_of_colors_in_picture = len(np.unique(circle_data[:,5]))
    
    return (picture, object_properties, total_number_of_circles, dictionary_for_number_of_occurrences, number_of_colors_in_picture)
