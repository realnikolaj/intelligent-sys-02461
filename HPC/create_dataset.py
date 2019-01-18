# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 10:39:20 2019

@author: daghn
"""

import numpy as np
import random
import math
import torch
import torch.utils.data
from matplotlib import colors
from collections import Counter
#np.set_printoptions(suppress=True)

torch.cuda.empty_cache()
# Initialize Hyper-parameters

picture_dimension = 64 # Default is 28

input_size = picture_dimension**2       # The image size = dimension squared
hidden_size = picture_dimension**2     # The number of nodes at the hidden layer

# number of circles and classes
num_circles_min = 0 
num_circles_max = 20
num_classes = num_circles_max + 1
   
# Size of datasets
N = 10000            # Size of train dataset
V = 1      # Size of test dataset

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
    
    
    # Split data!    
    D = 0
    for i in range(2):
        C = (N/2)*(i+1)
        
        
        # Convert to torch tensors
        our_data = torch.from_numpy(pictures[int(D):int(C)])
        our_labels = torch.from_numpy(labels[int(D):int(C)])
    
        # Encapsulate into a TensorDataset:
        dataset = torch.utils.data.TensorDataset(our_data, our_labels)
        torch.save(dataset, 'testset{}.pt'.format(i), pickle_protocol=4)     
        D = C+1
   
    return


