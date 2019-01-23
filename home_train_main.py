"""

In order to test run this training script, we suggest changing a few parameters to make it less GPU memory
intensive:
    
F = 32 is default, it can be changed to, say, 8, with a huge hit in network performance though.   

layer_size = 5000 is default, a lower value can be tried.

picture_dimension = 128 can be lowered to, say, 64 or 32.

num_objects_max = 30 can be lowered.

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

# Initialize Hyper-parameters

# CUDA 
# Change value 0 or 1
# To switch between running on cpu and GPU (CUDA)
# 0. CPU
# 1. GPU 
Acceleration_device = 1

# DATA
picture_dimension = 128     # The picture width and height will be this dimension

# number of objects and classes
num_objects_min = 0
num_objects_max = 30
num_classes = num_objects_max + 1

# number of colors and shapes
num_colors = 10             # This one shouldn't be changed
num_shapes = 5              # This one shouldn't be changed

# MODEL AND TRAINING 
num_epochs = 300            # The number of times 10k pictures is trained
lr_decay_epoch = 1000       # How many epochs before changing learning rate (note, ADAM is used, so no hard reason to change learning rate)
learning_rate = 0.0005     # The speed of convergence
batch_size = 100            # The number of pictures for one training iteration
N = 1000                    # Number of pictures created at a time for training (when not using a saved dataset)
V = 10000                   # Size of test dataset for accuracy tracking
F = 32                      # Number of filters in first convolutional layer in the net. It is a multiplicator for all convolutional layers.
layer_size = 5000           # Size of fully connected layer before heads

# Color map of fixed colors
default_colors = ['black', 'red', 'blue', 'green', 'yellow', 'darkorange', 'rebeccapurple', 'saddlebrown', 'orchid', 'gray', 'white']
cmap = colors.ListedColormap(default_colors)
bounds= (np.array([0,1,2,3,4,5,6,7,8,9,10,11]) - 0.5) / 10
norm = colors.BoundaryNorm(bounds, cmap.N)


# Active heads correspond to: 
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
 
active_heads = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37]
#active_heads = [18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34]
#active_heads = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,23]
#active_heads = [0]

# Learning rate scheduler, not used, but may be useful in some cases for helping the optimizer (here Adam)
def lr_scheduler(optimizer, epoch, init_lr=learning_rate, lr_decay_epoch=lr_decay_epoch):
    """Decay learning rate by a factor of 0.3 every lr_decay_epoch epochs."""
    lr = init_lr * (0.3**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer

# Measures the fraction of images with a positive label ( =1 ) in a random 10k set.
def count_positives(label_index):
    count = 0
    for i in range(10000):
        value = create_image()[1][label_index]
        if value == 1:
            count += 1
    return count / 10000    

# Create one image and its labels.
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
    
    # Randomly pick which colors to include in the palette
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
    
    # Convert to torch tensors
    our_data = torch.from_numpy(pictures)
    our_labels = torch.from_numpy(labels)
    
    # Encapsulate into a TensorDataset:
    dataset = torch.utils.data.TensorDataset(our_data, our_labels)
    
    return dataset

# Download dataset (we actually create it on the fly on home computers)
train_dataset = create_dataset(N)
test_dataset = create_dataset(V)
    
# Load the dataset with the DataLoader utility (giving us batches and other options)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=False)
    
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

# CNN_mini_VGG, Convolutional Neural Network.
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
    
# Print which network we are running.
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


# TRAIN THE NETWORK

loss_list = np.empty([40, num_epochs])
accuracy_list = np.empty([40, num_epochs])

for epoch in range(num_epochs):
#    # Decay learning rate by a factor of 0.3 every lr_decay_epoch epochs.
#    lr_scheduler(optimizer, epoch, init_lr=learning_rate, lr_decay_epoch=lr_decay_epoch)
    
    # Train the network on 10k pictures
    for p in range(10):
        if p == 0:
            print("")
            print("Beginning new epoch." + ' Epoch [{}/{}].'.format(epoch + 1, num_epochs))
                
        train_dataset = create_dataset(N)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,   
                                                   batch_size=batch_size,
                                                   shuffle=False)
        
#        print("Training on 1000 pictures, 10 steps of 100 pictures")
        
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
   
# Now loss_list and accuracy_list are done and can be saved to file  
np.save("0_with_40_objects_accuracy_list",accuracy_list)
np.save("0_with_40_objects_loss_list",loss_list)  

# Needs some more heads added. there is some technical issue which makes it hard to make a loop and use index.            
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
            output_21 = outputs[21].to(device)
            output_22 = outputs[22].to(device)
            output_23 = outputs[23].to(device)
            
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
            
            output_21 = softmax(output_21)       # applies softmax to outputs for easy interpretation
            _, predicted_21 = torch.max(output_21.data, 1)
            output_21 = output_21.cpu()          # loads the output back to the cpu so it can be changed to numpy
            output_21 = output_21.data.numpy()   # unwrap from Variable with .data, then transform to numpy so we get scientific suppression
            print("----- Task 21 -----")
            print("Ground truth is " + str(int(all_labels_vector[21])) )
            print("Network estimate is... " + str(int(predicted_21)))
            print("Based on these softmax values:")
            print(output_21)
            print("")
            
            output_22 = softmax(output_22)       # applies softmax to outputs for easy interpretation
            _, predicted_22 = torch.max(output_22.data, 1)
            output_22 = output_22.cpu()          # loads the output back to the cpu so it can be changed to numpy
            output_22 = output_22.data.numpy()   # unwrap from Variable with .data, then transform to numpy so we get scientific suppression
            print("----- Task 22 -----")
            print("Ground truth is " + str(int(all_labels_vector[22])) )
            print("Network estimate is... " + str(int(predicted_22)))
            print("Based on these softmax values:")
            print(output_22)
            print("")
            
            output_23 = softmax(output_23)       # applies softmax to outputs for easy interpretation
            _, predicted_23 = torch.max(output_23.data, 1)
            output_23 = output_23.cpu()          # loads the output back to the cpu so it can be changed to numpy
            output_23 = output_23.data.numpy()   # unwrap from Variable with .data, then transform to numpy so we get scientific suppression
            print("----- Task 23 -----")
            print("Ground truth is " + str(int(all_labels_vector[23])) )
            print("Network estimate is... " + str(int(predicted_23)))
            print("Based on these softmax values:")
            print(output_23)
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
            print("Task 21. Ground truth and estimate: "+ str(int(all_labels_vector[21])) + "  " + str(int(predicted_21)))
            print("Task 22. Ground truth and estimate: "+ str(int(all_labels_vector[22])) + "  " + str(int(predicted_22)))
            print("Task 23. Ground truth and estimate: "+ str(int(all_labels_vector[23])) + "  " + str(int(predicted_23)))
            