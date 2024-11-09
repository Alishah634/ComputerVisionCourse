import os
import sys
import time
import logging
import functools
from tqdm import tqdm
from typing import List, Tuple
from argparse import ArgumentParser
from termcolor import cprint
import pickle
import math
import BitVector

# Computer vision, Data science, and plotting imports:
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.feature import local_binary_pattern
from skimage import transform
from scipy.ndimage import convolve
from scipy.optimize import least_squares
from sklearn import svm
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Folder Paths:
DATASET1_PATH = "HW8-Files/HW8-Files/Dataset1/"
DATASET2_PATH = "HW8-Files/HW8-Files/Dataset2/"

'''START of Utility functions '''
# Utility functions
def ensure_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def display_image(img, title="Image"):
    plt.figure(figsize=(8, 6))
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()
''' END of Utility functions '''
    
    
'''========================================================= TASK 3.2.1 ========================================================='''
# Group lines by proximity based on rho values
def group_lines(lines, distance_threshold=0.5):
    if not lines:
        return []
    lines = sorted(lines, key=lambda line: line[0])  # Sort by rho
    grouped_lines = []
    current_group = [lines[0]]

    for i in range(1, len(lines)):
        dist = np.linalg.norm(np.array(lines[i]) - np.array(current_group[-1]))
        if dist < distance_threshold:
            current_group.append(lines[i])
        else:
            mean_rho = np.mean([line[0] for line in current_group])
            mean_theta = np.mean([line[1] for line in current_group])
            grouped_lines.append([mean_rho, mean_theta])
            current_group = [lines[i]]

    if current_group:
        mean_rho = np.mean([line[0] for line in current_group])
        mean_theta = np.mean([line[1] for line in current_group])
        grouped_lines.append([mean_rho, mean_theta])

    return grouped_lines

# Helper function to calculate intersections (corners)
def find_intersection_polar_coords(line1, line2):
    rho1, theta1 = line1
    rho2, theta2 = line2
    if theta1 == theta2:
        return None  # Parallel lines (no intersection)
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([rho1, rho2])
    x, y = np.linalg.solve(A, b)
    return int(x), int(y)

# Function to draw a line given rho and theta in polar coordinates
def draw_polar_line(img, rho, theta, color=(0, 255, 0), thickness=2):
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 2500 * (-b))
    y1 = int(y0 + 2500 * (a))
    x2 = int(x0 - 2500 * (-b))
    y2 = int(y0 - 2500 * (a))
    cv2.line(img, (x1, y1), (x2, y2), color, thickness)

# Function to convert a line to homogeneous coordinates
def to_homogeneous(rho, theta):
    a = np.cos(theta)
    b = np.sin(theta)
    return np.array([a, b, -rho])

# Function to find the intersection of two homogeneous lines
def find_intersection(line1, line2):
    x, y, w = np.cross(line1, line2)  # Cross product
    if w == 0:  # Lines are parallel
        cprint("Lines are parallel, no intersection!!!", "red")
        return None
    return int(x / w), int(y / w)

# Calculate homogeneous form from two rho values representing x-intercepts
def calculate_homogeneous_line(rho1, rho2, image_height):
    # Define two points on the line
    # X intercept and then the second point uses a large y-value (approaching infinity i.e image height):
    x1, y1 = rho1, 0
    x2, y2 = rho2, image_height  
    # Compute homogeneous coordinates (a, b, c) for the line
    a = y1 - y2
    b = x2 - x1
    c = x1 * y2 - x2 * y1
    return np.array([a, b, c])

# Detect corners (intersections) from image
def detect_corners(image, image_num, corner_params=[400, 300, 50, 10, np.pi / 90, 10], pic_num=None):
    canny_thresh1, canny_thresh2, houghs_thresh, rho_lines_thresh, theta_thresh, corner_thresh = corner_params

    # Convert to grayscale and apply edge detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, canny_thresh1, canny_thresh2)
    dilated_edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    # Save edge images
    if pic_num is not None:
        ensure_directory("MyResults/Edges")
        cv2.imwrite(f'MyResults/Edges/Pic_{pic_num}edges.jpg', edges)
        cv2.imwrite(f'MyResults/Edges/Pic_{pic_num}dilated_edges.jpg', dilated_edges)

    # Hough Transform to detect lines in polar coordinates
    lines = cv2.HoughLines(edges, 1, np.pi / 180, houghs_thresh, None, 0, 0)
    line_img = image.copy()
    line_test_img = image.copy()

    # Separate lines into vertical and horizontal lines:
    vertical_lines, horizontal_lines = [], []
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            if np.abs(theta) < np.pi / 4 or np.abs(theta) > 3 * np.pi / 4:
                vertical_lines.append([rho, theta])
            else:
                horizontal_lines.append([rho, theta])
    
    # Initial clustering of lines (not perfect but groups lines approx. together):
    vertical_lines = group_lines(vertical_lines, distance_threshold=10)
    horizontal_lines = group_lines(horizontal_lines, distance_threshold=10)

    # Cluster lines by their x_intercept to get exactly 8 vertical lines: and 10 horizontal lines
    base_pts = list()
    for line in vertical_lines:
        rho, theta = line
        # (x_intercept, 0):
        x_intercept = rho / np.cos(theta)
        # x at infinity:
        x_upper = -line_test_img.shape[0] * np.tan(theta) + (rho / np.cos(theta))
        # cv2.circle(line_test_img, (int(x_upper), image.shape[0]), 5, (0, 255, 0), -1)
        # cv2.line(line_test_img, (int(x_intercept), 0), (int(x_upper), line_test_img.shape[0]), (0, 255, 0), 2)  
        # Append as a 2D point with base and upper x-coordinates
        base_pts.append([x_intercept, x_upper])
    
    # Check if we have enough points for clustering
    if len(base_pts) >= 8:
        # Convert to a 2D array and apply KMeans
        base_pts_array = np.array(base_pts)
        clustered_pts = KMeans(n_clusters=8, random_state=0).fit(base_pts_array)
        # Extract cluster centers
        test_lines = list()
        # print(clustered_pts.cluster_centers_)
        for p1, p2 in clustered_pts.cluster_centers_:
            # Draw the clustered line
            # cv2.circle(line_test_img, (int(p1), 0), 5, (255, 255, 0), -1)
            # cv2.circle(line_test_img, (int(p2), line_test_img.shape[0]), 5, (255, 255, 0), -1)
            cv2.line(line_test_img, (int(p1), 0), (int(p2), line_test_img.shape[0]), (0, 0, 255), 2)
    else:
        print("Not enough points for clustering. Found only", len(base_pts))
    
    vertical_lines = clustered_pts.cluster_centers_
    # cprint(f"Vertical Lines : {vertical_lines}", "cyan")
    
    # Cluster lines to get exactly 10 horizontal lines: (Apparently dont need to do this as it has the perfect numer already)
    # for line in vertical_lines:
    #     rho, theta = line
    #     draw_polar_line(line_test_img, rho, theta)
    
    for line in horizontal_lines:
        rho, theta = line
        if pic_num is not None:
            draw_polar_line(line_test_img, rho, theta, color=(255, 0, 0))
    if len(horizontal_lines) != 10:
        cprint(f"Image {image_num} has {len(horizontal_lines)} horizontal lines", "red")
    if len(vertical_lines) != 8:
        cprint(f"Image {image_num} has {len(vertical_lines)} vertical lines", "red")

    
    # Save the image with lines drawn:
    if pic_num is not None:
        ensure_directory("MyResults/HoughLines")
        cv2.imwrite(f'MyResults/HoughLines/Pic_{pic_num}_lines.jpg', line_test_img)
        # time.sleep(1)
        
    # Convert lines to homogeneous form for intersection calculation:
    horizontal_homogeneous_lines = [to_homogeneous(np.array(h[0]), np.array(h[1])) for h in horizontal_lines]
    # Vertical lines will be specified by their x-intercepts, because the cluster_centers_ are in that form (x_intercept, x_at_infinity) i.e (rho1, rho2):
    # Need to sort so labeling is top-left to bottom-right:
    # Since x[0] is the upper x-intercept, since 0,0 is at the top left corner sort according to the x-intercept:
    sorted_v_lines = sorted(vertical_lines, key=lambda x: x[0]) 
    vertical_homogeneous_lines = list()
    # for rho1, rho2 in clustered_pts.cluster_centers_:
    for rho1, rho2 in sorted_v_lines:
        vertical_homogeneous_lines.append(calculate_homogeneous_line(rho1, rho2, line_test_img.shape[0]))
    
    # cprint(f"Vertical Homogeneous line representation:\n {vertical_homogeneous_lines}", "cyan")
    # print()
    # cprint(f"Horizontal Homogeneous line representation:\n {horizontal_homogeneous_lines}", "cyan")
    
    # Using the homogeneous lines, find the intersections:
    intersections = []
    corner_img = image.copy()
    for h in horizontal_homogeneous_lines:
        for v in vertical_homogeneous_lines:
            intersection = find_intersection(h, v)
            if intersection:
                intersections.append(intersection)
                if pic_num is not None:
                    cv2.circle(line_test_img, intersection, 5, (0, 255, 255), -1)
                    cv2.putText(line_test_img, str(len(intersections)), (intersection[0]+5, intersection[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    # cv2.circle(corner_img, intersection, 5, (0, 255, 255), -1)
                    # cv2.putText(corner_img, str(len(intersections)), (intersection[0]+5, intersection[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # cprint(f"Intersections: {intersections}", "cyan")

    # Save the image with and lines corners drawn:
    if pic_num is not None:            
        ensure_directory("MyResults/Corners")
        # cv2.imwrite(f'MyResults/Corners/Pic_{pic_num}_corners.jpg', corner_img)
        cv2.imwrite(f'MyResults/Corners/Pic_{pic_num}_corners.jpg', line_test_img)
        # time.sleep(1)  # To prevent overwriting of images

    # ensure_directory("TrialResults/ALL_LINES")
    # cv2.imwrite(f'TrialResults/ALL_LINES/TEST_{pic_num}_lines.jpg', line_test_img)
    # time.sleep(1)  # To prevent overwriting of images
        
    return intersections

# Process images and detect corners
def process_images(dataset_path):
    image_paths = [os.path.join(dataset_path, img) for img in os.listdir(dataset_path)]
    all_image_corners = []
    for idx in tqdm(range(len(image_paths))):
        image = cv2.imread(image_paths[idx])
        corners = detect_corners(image, idx, pic_num=idx+1)
        # corners = detect_corners(image, idx)
        all_image_corners.append(corners)
    return all_image_corners


"""========================================================= END of TASK 3.2.1 ========================================================="""

"""========================================================= START of TASK 3.2 ========================================================="""
# 3.2.2 Camera Calibration

def calc_world_cords(grid_length:int = 10, num_vert: int = 8, num_horiz: int = 10) -> List[List[int]]:
    # All of these input params are in inches.
    # Generate x and y coordinates using lists
    x = [i * grid_length for i in range(num_vert)]
    y = [i * grid_length for i in range(num_horiz)]
    # Manually create the mesh grid as lists of coordinate pairs
    world_coords = []
    for yi in y:
        for xi in x:
            world_coords.append([xi, yi])
    return world_coords
            
def calc_intrinsic_params():
    pass

def calc_extrinsic_params():
    pass

def calc_radial_distortion_params():
    pass

def zhangs_algo(all_image_corners):
    for row in (world_cords:= calc_world_cords()): print(row)
    
    pass



"""========================================================= END of TASK 3.2 ========================================================="""
def task1():
    # Task 3.2.1: Detect corners of calibration pattern in images
    temp = process_images(DATASET1_PATH)
    all_image_corners = []
    for corners in temp:
        if len(corners) == 80:  # We expect 80 corners
            all_image_corners.append(corners)
    cprint(f"Number of images with 80 corners: {len(all_image_corners)}", "white")
    
    # Task 3.2.2: Zhang's algo for camera calibration:
    zhangs_algo(all_image_corners)
    
    
    
    # print(calc_world_cords())
    
    cprint("Task 1 completed successfully!", "green")


def task2():
    pass


if __name__ == "__main__":
    parser = ArgumentParser(description="Script to run Task 1 or Task 2.")
    parser.add_argument('--task', '-t', type=int, required=True, choices=[1, 2], help="Select task: 1 for Task 1, 2 for Task 2")
    args = parser.parse_args()

    cprint("Make sure you are running from HW8 directory!!!", "red")
    ensure_directory("MyResults")

    if args.task == 1:
        task1()
    elif args.task == 2:
        task2()
