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
np.set_printoptions(suppress=True)

np.set_printoptions(suppress=True)

torch.cuda.empty_cache()

# CUDA 
# Change value 0 or 1
# To switch between running on cpu and GPU (CUDA)
# 0. CPU
# 1. GPU 
Acceleration_device = 1

# Initialize Hyper-parameters

picture_dimension = 128    # The picture width and height will be this dimension

input_size = picture_dimension**2       # The image size = dimension squared
hidden_size = picture_dimension**2     # The number of nodes at the hidden layer

# number of objects and classes
num_objects_min = 0
num_objects_max = 30
num_classes = num_objects_max + 1
   
num_epochs = 200           # The number of times 10k pictures is trained
lr_decay_epoch = 1000     # How many epochs before changing learning rate (note, ADAM is used, so no hard reason to change learning rate)
learning_rate = 0.0005     # The speed of convergence
batch_size = 100          # The number of pictures for one training iteration
N = 1000000                # Number of pictures loaded (or created) into memory for training at a time
V = 10000           # Size of test dataset for accuracy tracking

F = 32     # Number of filters in first convolutional layer in the net. It is a multiplicator for all convolutional layers.
layer_size = 5000  # size of fully connected layer before heads

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
# 1-10. number of given color, starting from red, as above in default categories
# 11-15. number of given shape: square, circle, triangles, rectangles, half circles
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

# 38. number of colors with at least 2 different shapes present

## 40. number of shapes with a sun color present
 

#active_heads = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34]
#active_heads = [18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34]
#active_heads = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,23]
active_heads = [16]


#head_legends = 

def lr_scheduler(optimizer, epoch, init_lr=learning_rate, lr_decay_epoch=lr_decay_epoch):
    """Decay learning rate by a factor of 0.3 every lr_decay_epoch epochs."""
    lr = init_lr * (0.3**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer

# measures the fraction of images with a positive label ( =1 ) in a random 10k set
def count_positives(label_index):
    count = 0
    for i in range(10000):
        value = create_image()[1][label_index]
        if value == 1:
            count += 1
    return count / 10000    

def create_image():
    area_min = 0.10
    area_max = 0.20
    
    min_object_area = 0.2 #minimum area of object relative to the max 1
    
    margin = 1   # margin, a number of pixels
    object_buffer = 1.10
    
    # Create empty picture
    picture = np.zeros((picture_dimension, picture_dimension))
    
    # Randomly pick total number of objects
    total_number_of_objects = np.random.randint(num_objects_min, num_objects_max + 1)
    
    # Create array of object data.
    # 7 columns: area, radius, scaled radius, x-coor., y-coor., color, shape
    object_data = np.zeros((total_number_of_objects, 7))
    
    
    # ASSIGN COLORS
    # Randomly pick number of colors in palette (how many colors we can choose from for this picture)
    number_of_colors_in_palette = np.random.randint(1, num_colors + 1)
    
    # Randomly pick which colors to include from the palette
    colors_in_picture = random.sample(range(1, num_colors + 1), number_of_colors_in_palette)
    
    # Assign random color from the palette to each object
    object_data[:,5] = np.array(random.choices(population=colors_in_picture, k=total_number_of_objects))
    
    
    # ASSIGN SHAPES
    # Randomly pick number of shapes in the shape palette (how many shapes we can choose from for this picture)
    number_of_shapes_in_palette = np.random.randint(1, 6)
    
    # Randomly pick which shapes to include from the shape palette
    shapes_in_picture = random.sample(range(1, 6), number_of_shapes_in_palette)
    
    # Assign random shape from the shape palette to each object
    object_data[:,6] = np.array(random.choices(population=shapes_in_picture, k=total_number_of_objects))
    
    
    # Point to data with variable names
    object_colors = object_data[:,5]
    object_shapes = object_data[:,6]
    
    
    # GENERATE AREA DATA
    # Randomly pick the total area of objects, relative to the picture area
    total_area_of_objects = np.random.uniform(low = area_min, high = area_max)
    
    # Calculate object areas:
    # First pick random areas relative to max 1
    object_areas = np.array(np.random.uniform(min_object_area, 1, total_number_of_objects))
    
    # Then scale areas so they sum up to 1,
    # and then multipy with the total area of objects
    # (which is in [0,1], as in relative to the total picture area)
    object_areas = (object_areas / np.sum(object_areas)) * total_area_of_objects
    
    # store the relative areas in the object data array
    object_data[:,0] = object_areas
    
    # Calculate relative object radii
    object_data[:,1] = np.sqrt(object_data[:,0]/3.1415) 
    
    # Calculate scaled object radii
    object_data[:,2] = object_data[:,1] * picture_dimension
    
    # Sort objects by size
    object_data = object_data[object_data[:,0].argsort()]
    object_data = object_data[::-1]
    
    
    # PLACE OBJECTS
    for i in range(total_number_of_objects):
        looking_for_centrum = True
        count = 1
        stop = 1000
        
        while looking_for_centrum == True and count < stop:
            radius = object_data[i,2]
            x = np.random.randint(margin + radius, picture_dimension - margin - radius)
            y = np.random.randint(margin + radius, picture_dimension - margin - radius)
            looking_for_centrum = False
            
            for j in range(0, i):
                radius_old = object_data[j,2]
                x_old = object_data[j,3]
                y_old = object_data[j,4]
                
                distance = math.sqrt((x - x_old)**2 + (y - y_old)**2)
                
                if distance < object_buffer * (radius + radius_old):
                    looking_for_centrum = True
                    break
                
            if looking_for_centrum == False:
                object_data[i,3] = x
                object_data[i,4] = y
                   
            count += 1
            
            if count == stop:
                print("couldn't place object")
    
    # DRAW PICTURE
    for i in range(total_number_of_objects):
        radius = object_data[i,2]
        x = int(object_data[i,3])
        y = int(object_data[i,4])
        R = int(math.ceil(radius))
    
        color = object_colors[i]
        shape = object_shapes[i] 
        

        # Squares
        if shape == 1:
            RR = math.ceil(R*0.7)
            picture[x-RR:x+1+RR,y-RR:y+1+RR] = color / 10
        
        # Circles
        if shape == 2:
            # Checking all pixels in a sqared box around the circle. If their 
            # distance to the center is less than the radius, we color the pixel.
            box_size = 2 * R + 1        
            
            for j in range(box_size):
                for k in range(box_size):
                    if (j - (R + 1))**2 + (k - (R + 1))**2 < radius**2:
                        picture[x - R + j, y - R + k] = color / 10
                        
        # Triangles
        if shape == 3:
            RR = math.ceil(R*0.8)
            for j in range(R):
                picture[(x+j),y-R+j:y+R-j] = color / 10         
        
        # Rectangles        
        if shape == 4:
            picture[x-math.ceil(R*0.4):x+math.ceil(R*0.4),y-R:y+R] = color / 10                
                        
        # Half circles    
        if shape == 5:
            box_size = (2 * R + 1) 
            
            for j in range(box_size):
                for k in range(R + 1):
                    if (j - (R + 1))**2 + (k - (R + 1))**2 < radius**2:
                        picture[x - R + j, y - R + k] = color / 10
    
    
    # LABELS:
    
    # 1-10. OBJECT COLORS
    # Count the number of objects of the various colors, and put in a dictionary
    dictionary_for_number_of_color_occurrences = Counter(object_colors)
    
    # Make a simple vector of color count labels, using the dictionary
    color_label_vector = np.zeros(10)
    for i in range(10):
        color_label_vector[i] = dictionary_for_number_of_color_occurrences[i+1]
     
        
    # 11-15. OBJECT SHAPES       
    # Count the number of objects of the various shapes, and put in a dictionary
    dictionary_for_number_of_shape_occurrences = Counter(object_shapes)
    
    # Make a simple vector of shape count labels, using the dictionary
    shape_label_vector = np.zeros(5)
    for i in range(5):
        shape_label_vector[i] = dictionary_for_number_of_shape_occurrences[i+1]    
    
    
    # NUMBER OF SHAPES AND COLORS
    # 16. Determine the number of different colors in the picture
    number_of_colors_in_picture = len(np.unique(object_colors))
    
    # 17. Determine the number of different shapes in the picture
    number_of_shapes_in_picture = len(np.unique(object_shapes))
    
    
    # COMPLEX LABELS    
    # 18. is there a red triangle?
    red_triangle = int(any( (x[5] == 1) and (x[6] == 3) for x in object_data))
    
    # 19. is there a red triangle or a white circle? 
    white_circle = int(any( (x[5] == 10) and (x[6] == 2) for x in object_data))
    red_triangle_or_white_circle = (red_triangle == 1) or (white_circle == 1)
    
    # 20. is there at least 2 colors with a triangle present?
    triangle_objects = object_data[object_data[:,6] == 3]
    colors_of_triangle_objects = triangle_objects[:,5]
    number_of_colors_with_triangle_present = len(np.unique(colors_of_triangle_objects))
    _2_colors_with_triangle = int( number_of_colors_with_triangle_present >= 2 )
    
    # 21. is there at least 2 shapes with a red object present?
    red_objects = object_data[object_data[:,5] == 1]
    shape_of_red_objects = red_objects[:,6]
    number_of_shapes_with_red_objects = len(np.unique(shape_of_red_objects))
    _2_shapes_with_red = int( number_of_shapes_with_red_objects >= 2 )
    
    # Define sharp objects (square, rectangle, triangle)
    sharp_objects = object_data[np.isin(object_data[:,6], np.array([1,3,4]))]
    
    # 22. is there at least one white sharp object? 
    white_sharp_object = int(any( (x[5] == 10) for x in sharp_objects))
    
    # 23. is there at least one white sharp object OR blue sharp object? 
    white_or_blue_sharp_object = int(any( (x[5] == 10) or (x[5] == 2) for x in sharp_objects))
    
    # 24. is there at least one white sharp object AND at least one blue object?
    any_blue = int(any( (x == 2) for x in object_colors))
    white_sharp_and_any_blue = (white_sharp_object == 1) and (any_blue == 1)
    
    # 25. is there at least one white sharp object AND (at least one (green OR blue) sharp object)?
    green_sharp_object = int(any( (x[5] == 3) for x in sharp_objects))
    blue_sharp_object = int(any( (x[5] == 2) for x in sharp_objects))
    white_sharp_and_green_or_blue_sharp = (white_sharp_object == 1) and ( (green_sharp_object == 1) or (blue_sharp_object == 1) )
    
    # 26. is there at least one white sharp object AND NOT a blue sharp object?
    white_sharp_and_not_blue_sharp = (white_sharp_object == 1) and (blue_sharp_object == 0)
    
    # 27. is there at least 3 colors with a sharp object present?
    colors_of_sharp_objects = sharp_objects[:,5]
    number_of_colors_with_sharp_present = len(np.unique(colors_of_sharp_objects))
    _3_colors_with_sharp = int( number_of_colors_with_sharp_present >= 3 )
    
    # 28. is there at least 2 shapes with a sun color present? (sun colors: yellow, orange, red)
    sun_color_objects = object_data[np.isin(object_data[:,5], np.array([1,4,5]))]
    shapes_of_sun_color_objects = sun_color_objects[:,6]
    number_of_shapes_with_sun_color = len(np.unique(shapes_of_sun_color_objects))
    _2_shapes_with_sun_color = int( number_of_shapes_with_sun_color >= 2 )
    
    # 29. is there (at least one red triangle OR one white circle OR one orange rectangle)?
    orange_rectangle = int(any( (x[5] == 5) and (x[6] == 4) for x in object_data))
    red_t_or_white_c_or_orange_r = (red_triangle == 1) or (white_circle == 1) or (orange_rectangle == 1)
    
    # 30. is there (at least one red triangle OR one white circle OR one orange rectangle) AND a blue object?
    red_t_or_white_c_or_orange_r_and_blue = (red_t_or_white_c_or_orange_r == 1) and (any_blue == 1)
    
    # 31. is there at least 3 colors with 2 different shapes present?
    count = 0
    for i in range(1,11):
        objects = object_data[object_data[:,5] == i]
        shapes = objects[:,6]
        number_of_shapes = len(np.unique(shapes))
        if number_of_shapes >= 3:
            count +=1
    _3_colors_with_2_shapes = int(count>=2)    

    # 32. number of sun color objects   
    number_of_sun_color_objects = len(sun_color_objects)   

    # 33. number of circles, triangles and squares
    c_plus_t_plus_s = shape_label_vector[1] + shape_label_vector[2] + shape_label_vector[3]
    
    # 34. is there (a blue OR pink square present) AND (a purple OR white circle)?     
    blue_square = int(len([x for x in object_data if x[5] == 2 and x[6] == 1]) >= 1)
    pink_square = int(len([x for x in object_data if x[5] == 8 and x[6] == 1]) >= 1)
    purple_circle = int(len([x for x in object_data if x[5] == 5 and x[6] == 2]) >= 1)
    white_circle = int(len([x for x in object_data if x[5] == 10 and x[6] == 2]) >= 1)
    disease_1 = int( (blue_square or pink_square) and (purple_circle or white_circle))
    
    # 35. number of colors with a triangle present
    triangle_objects = object_data[object_data[:,6] == 3]
    colors_of_triangle_objects = triangle_objects[:,5]
    number_of_colors_with_triangle_present = len(np.unique(colors_of_triangle_objects))
    
    # 36. number of colors with a circle, or a triangle, or a rectangle present
    virus_objects = object_data[np.isin(object_data[:,6], np.array([2,3,4]))]
    colors_of_virus_objects = virus_objects[:,5]
    number_of_colors_with_virus_present = len(np.unique(colors_of_virus_objects))
    
    # 37. number of colors with at least 2 different shapes present 
    infected = 0
    for i in range(1,11):
        objects = object_data[object_data[:,5] == i]
        shapes = objects[:,6]
        number_of_shapes = len(np.unique(shapes))
        if number_of_shapes >= 2:
            infected +=1
    
    # Turn single labels into arrays for concatenation with color and shape vectors
    total_number_of_objects = np.array([total_number_of_objects])
    number_of_colors_in_picture = np.array([number_of_colors_in_picture])
    number_of_shapes_in_picture = np.array([number_of_shapes_in_picture])
    red_triangle = np.array([red_triangle])
    red_triangle_or_white_circle = np.array([red_triangle_or_white_circle])
    _2_colors_with_triangle = np.array([_2_colors_with_triangle])
    _2_shapes_with_red = np.array([_2_shapes_with_red])
    white_sharp_object = np.array([white_sharp_object])
    white_or_blue_sharp_object = np.array([white_or_blue_sharp_object])
    white_sharp_and_any_blue = np.array([white_sharp_and_any_blue])
    white_sharp_and_green_or_blue_sharp = np.array([white_sharp_and_green_or_blue_sharp])
    white_sharp_and_not_blue_sharp = np.array([white_sharp_and_not_blue_sharp])
    _3_colors_with_sharp = np.array([_3_colors_with_sharp])
    _2_shapes_with_sun_color = np.array([_2_shapes_with_sun_color])
    red_t_or_white_c_or_orange_r = np.array([red_t_or_white_c_or_orange_r])
    red_t_or_white_c_or_orange_r_and_blue = np.array([red_t_or_white_c_or_orange_r_and_blue])
    _3_colors_with_2_shapes = np.array([_3_colors_with_2_shapes])
    number_of_sun_color_objects = np.array([number_of_sun_color_objects])
    c_plus_t_plus_s = np.array([c_plus_t_plus_s])
    disease_1 = np.array([disease_1])
    number_of_colors_with_triangle_present = np.array([number_of_colors_with_triangle_present])
    number_of_colors_with_virus_present = np.array([number_of_colors_with_virus_present])
    infected = np.array([infected])
    
    # Create vector of all labels
    all_labels_vector = np.concatenate((total_number_of_objects,
                                        color_label_vector,
                                        shape_label_vector,
                                        number_of_colors_in_picture,
                                        number_of_shapes_in_picture,
                                        red_triangle,
                                        red_triangle_or_white_circle, 
                                        _2_colors_with_triangle,
                                        _2_shapes_with_red,
                                        white_sharp_object,
                                        white_or_blue_sharp_object,
                                        white_sharp_and_any_blue,
                                        white_sharp_and_green_or_blue_sharp,
                                        white_sharp_and_not_blue_sharp,
                                        _3_colors_with_sharp,
                                        _2_shapes_with_sun_color,
                                        red_t_or_white_c_or_orange_r,
                                        red_t_or_white_c_or_orange_r_and_blue,
                                        _3_colors_with_2_shapes,
                                        number_of_sun_color_objects,
                                        c_plus_t_plus_s,
                                        disease_1,
                                        number_of_colors_with_triangle_present,
                                        number_of_colors_with_virus_present,
                                        infected
                                        ))
#     
#    plt.imshow(picture, interpolation='nearest', origin='lower',
#                    cmap=cmap, norm=norm)
#    plt.show()
#    
    return (picture, all_labels_vector)

#exit()

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
    
    # Split the data into files
    D = 0
    for i in range(100):
        C = (N/100)*(i+1)


        # Convert to torch tensors
        our_data = torch.from_numpy(pictures[int(D):int(C)])
        our_labels = torch.from_numpy(labels[int(D):int(C)])

        # Encapsulate into a TensorDataset:
        dataset = torch.utils.data.TensorDataset(our_data, our_labels)
        torch.save(dataset, 'dataset{}.pt'.format(200+i), pickle_protocol=4)
        D = C+1
        
    return
create_dataset(N)
