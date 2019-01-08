# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 10:39:20 2019

@author: daghn
"""

import numpy as np
import random
import math
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)

def create_image():
    area_min = 0.10
    area_max = 0.20
    
    min_circle_area = 0.5 #minimum area of circle relative to the max 1
    
    num_circles_min = 1
    num_circles_max = 3
    
    # x and y dimension
    picture_dimension = 100
    
    picture = np.zeros((picture_dimension, picture_dimension))
    
    margin = 5
    circle_buffer = 1.10
    
    number_of_circles = 0
    total_area_of_circles = 0
    area_of_each_circle = 0
    
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
    
    # Then scale so they sum up to 1, and then multipy with the set total area
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
        stop = 10000
        
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
    
    print(circle_data)
    print(picture)
    
    plt.imshow(picture)
    plt.show()