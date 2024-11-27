import os
import sys
import math
import time
import logging
import functools
import random
from tqdm import tqdm
from typing import List, Tuple
from argparse import ArgumentParser
from termcolor import cprint

# Computer vision, Data science, and plotting imports:
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import scipy as sc
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def pad_image(image: np.ndarray, pad_num: int = 5) -> np.ndarray:
    return np.pad(image, 2*[(pad_num//2, pad_num//2)], mode='constant', constant_values=0)

def calculate_disparity_map(left_image: np.ndarray, right_image: np.ndarray, window_size: int, max_disparity: np.uint8) -> np.ndarray:
    # Padding:
    left_padded_image = pad_image(left_image, window_size)
    right_padded_image = pad_image(right_image, window_size)
    
    disparity_map = list()
    for row in tqdm(range(left_image.shape[0])):
        disp_row_list = list()
        temp_map = np.array(range(max_disparity))
        
        for col in range(left_image.shape[1] - window_size):
            right_columns = [(col - temp) >= 0 and (col - temp) < (right_image.shape[1] - window_size) for temp in temp_map]
            trial_disparity = temp_map[right_columns]
            left_window = left_padded_image[row:row+window_size, col:col+window_size]
            left_window_bit_map = (left_window > left_window[left_window.shape[0]//2, left_window.shape[1]//2]) 
            
            # Check disparity:
            temp = list()
            for disparity in trial_disparity:
                right_window = right_padded_image[row:row+window_size, col-disparity:col-disparity+window_size]
                right_window_bit_map = (right_window > right_window[right_window.shape[0]//2, right_window.shape[1]//2]) 
                temp.append((disparity,np.sum(left_window_bit_map != right_window_bit_map)))
            disp_row_list.append(min(temp, key=lambda x: x[1])[0])
        disparity_map.append(np.array(disp_row_list))
    return np.array(disparity_map)

# Function to calculate the binary mask and accuracy
def calculate_binary_mask(disparity_map: np.array, ground_truth: np.array, disparity_threshold: int = 3) -> Tuple[np.array, float]:
    # Calculate the absolute error b/w disparity map and ground truth:
    ground_truth_valid_mask = (ground_truth > 0).astype(np.uint8)
    error = np.abs(ground_truth - disparity_map)

    # cprint(f"Valid mask shape: {ground_truth_valid_mask.shape}", "cyan") # DEBUG STATEMENT!!!
    # cprint(f"Valid mask shape: {ground_truth_valid_mask[:10]}", "cyan") # DEBUG STATEMENT!!!

    # Create binary mask error <= disparity_threshold:
    binary_mask = ((error <= disparity_threshold) & ground_truth_valid_mask.astype(np.uint8))
    # cprint(f"Binary mask shape: {binary_mask.shape}", "cyan") # DEBUG STATEMENT!!!
    # cprint(f"Binary mask shape: {binary_mask[:10]}", "cyan") # DEBUG STATEMENT!!!

    # Calculate accuracy:
    accuracy = np.sum(binary_mask) / np.sum(ground_truth_valid_mask)
    binary_mask *= 255
    return binary_mask, accuracy    
    
def task_3_main(window_size=7):
    assert window_size % 2, "Window size must be odd"
    
    #read in the images 
    left_image = cv2.imread('Task3Images/Task3Images/im2.png',cv2.IMREAD_GRAYSCALE)
    right_image = cv2.imread('Task3Images/Task3Images/im6.png',cv2.IMREAD_GRAYSCALE)
    left_ground_truth_image = cv2.imread('Task3Images/Task3Images/disp2.png',cv2.IMREAD_GRAYSCALE)
    left_ground_truth_image = (left_ground_truth_image.astype(np.float32)/4).astype(np.uint8)
    right_ground_truth_image = cv2.imread('Task3Images/Task3Images/disp6.png',cv2.IMREAD_GRAYSCALE)
    
    max_disparity = np.max(left_ground_truth_image)
    # Calculate the disparity map:
    disparity_map = calculate_disparity_map(left_image, right_image, window_size, max_disparity)
    
    # cprint(f"{disparity_map[:10]}", "cyan") # DEBUG STATEMENT!!!
    # cprint(f"Disparity map type: {type(disparity_map)}", "cyan") # DEBUG STATEMENT!!!
    # cprint(f"Disparity map shape: {np.shape(disparity_map)}", "cyan") # DEBUG STATEMENT!!!
    # cprint(f"Disparity map shape: {len(disparity_map), len(disparity_map[0])}", "cyan") # DEBUG STATEMENT!!!
    cv2.imwrite(f'TASK_3/disparity_map_with_window_{window_size}.jpg', disparity_map)

    # Calculate the binary mask and accuracy using the resized ground truth and disparity map:
    left_ground_truth_resized = cv2.resize(left_ground_truth_image, (disparity_map.shape[1], disparity_map.shape[0]))
    binary_mask, accuracy = calculate_binary_mask(disparity_map, left_ground_truth_resized,disparity_threshold=2)

    # Save the binary mask and print the accuracy
    cv2.imwrite(f'TASK_3/binary_mask_with_window_{window_size}.jpg', binary_mask)
    cprint(f"Accuracy with a window of {window_size} by {window_size}: {accuracy*100:.2f}%", "green")
    return 

if __name__ == "__main__":
    for window_size in range(3, 16, 2):
        cprint(f"Running Task 3 with window size: {window_size}", "yellow")
        task_3_main(window_size=window_size)
    cprint("Task 3 completed successfully", "green")
