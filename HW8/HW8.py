import os
import sys
import time
import logging # For logging and decorators
import functools # For logging and decorators
from tqdm import tqdm # Progress bar
from typing import List, Tuple # Format typing
from argparse import ArgumentParser # Parsing arguments
from termcolor import cprint # Formatting prints
import pickle # For saving intermediate values to avoid recomputing
import math # 
import BitVector

# Other Computer vision, Data science, Math, plotting imports:
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

# Folder Paths:
DATASET1_PATH = "HW8-Files/HW8-Files/Dataset1/" # Images are in the format Pic_#.jpg
DATASET2_PATH = "HW8-Files/HW8-Files/Dataset2/" # Images are in the format Pic_#.jpg

'''Util functions '''
# Create a folder if it does not exist
def ensure_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def display_image(img, title="Image"):
    plt.figure(figsize=(8, 6))
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')
    # plt.savefig(f"MyResults/test.png")
    plt.show()
    # time.sleep()
'''Util functions^^^ '''

"""========================================================= START of TASK 3.1 ========================================================="""
# Load all images from Dataset1
image_paths = [ DATASET1_PATH+name for name in (os.listdir(DATASET1_PATH)) ]
images = [cv2.imread(image_path) for image_path in image_paths]

# Detect and visualize corners and lines using polar coordinates
# def detect_corners(image, save_img=False, output_name="output", canny_thresh1=225, canny_thresh2=225, houghs_thresh=43, rho_lines_thresh=10, theta_thresh = np.pi / 90, corner_thresh=10):
def detect_corners(image, save_img=False, output_name="output",corner_params=[225, 225, 43,  10,  np.pi / 90,  10]):
    canny_thresh1, canny_thresh2, houghs_thresh, rho_lines_thresh, theta_thresh, corner_thresh= corner_params
    
    # Convert to grayscale and apply Gaussian blur
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny edge detection
    edges = cv2.Canny(blurred, canny_thresh1, canny_thresh2)
    dilated_edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
    if save_img:
        ensure_directory("MyResults")  # Ensure 'MyResults' directory exists
        cv2.imwrite(f'MyResults/{output_name}_edges.jpg', edges)
        cv2.imwrite(f'MyResults/{output_name}_dilated_edges.jpg', dilated_edges)

    # Apply Hough Line Transform to detect lines in polar coordinates
    lines = cv2.HoughLines(edges, 1, np.pi / 180, houghs_thresh, None, 0, 0)

    # Create a copy of the original image to draw the lines on
    line_img = image.copy()

    vertical_lines, horizontal_lines = [], []
    if lines is not None:
        for line in lines:
            rho, theta = line[0]

            # Classify lines as vertical or horizontal based on theta
            if np.abs(theta) < np.pi / 4 or np.abs(theta) > 3 * np.pi / 4:
                vertical_lines.append((rho, theta))
            else:
                horizontal_lines.append((rho, theta))
    hthresh = sum([horizontal_lines[i][0] for i in range(len(horizontal_lines))])/len(horizontal_lines)
    vthresh = sum([vertical_lines[i][0] for i in range(len(vertical_lines))])/len(vertical_lines)
    hthresh = abs(hthresh)
    vthresh = abs(vthresh)
    # print(f"Threshold of Horizontal lines: {hthresh}")
    # print(f"Threshold of Vertical lines: {vthresh}")
    
    # Helper function to group lines based on their rho value
    # def group_lines(lines, rho_threshold=10, theta_threshold=np.pi / 90):
    def group_lines(lines, rho_threshold=10, theta_threshold=np.pi / 90):
        if not lines:
            return []

        # Sort lines by rho for easier clustering
        lines = sorted(lines, key=lambda line: (line[0], line[1]))
        grouped_lines = []
        current_group = [lines[0]]

        for i in range(1, len(lines)):
            rho, theta = lines[i]
            prev_rho, prev_theta = current_group[-1]

            # Check if both rho and theta differences are within thresholds
            if abs(rho - prev_rho) < rho_threshold and abs(theta - prev_theta) < theta_threshold:
                current_group.append((rho, theta))
            else:
                # Average the current group and add it as a single line
                avg_rho = np.mean([line[0] for line in current_group])
                avg_theta = np.mean([line[1] for line in current_group])
                grouped_lines.append((avg_rho, avg_theta))
                
                # Start a new group with the current line
                current_group = [(rho, theta)]

        # Handle the last group
        if current_group:
            avg_rho = np.mean([line[0] for line in current_group])
            avg_theta = np.mean([line[1] for line in current_group])
            grouped_lines.append((avg_rho, avg_theta))

        return grouped_lines


    grouped_horizontal = group_lines(horizontal_lines)
    grouped_vertical = group_lines(vertical_lines)

    for avg_rho, avg_theta in grouped_horizontal:
        draw_polar_line(line_img, avg_rho, avg_theta, color=(255, 0, 0))  # Horizontal in blue

    for avg_rho, avg_theta in grouped_vertical:
        draw_polar_line(line_img, avg_rho, avg_theta, color=(0, 0, 255))  # Vertical in red

    if save_img:
        cv2.imwrite(f'MyResults/{output_name}_lines.jpg', line_img)

    # Calculate intersection points as corners based on the grouped lines
    corner_img = image.copy()
    corners = []
    for hline in grouped_horizontal:
        for vline in grouped_vertical:
            intersect = find_intersection_polar(hline, vline)
            if intersect is not None:
                x, y = int(intersect[0]), int(intersect[1])
                if 0 <= x < corner_img.shape[1] and 0 <= y < corner_img.shape[0]:
                    corners.append((x, y))


    def group_corners(corners, distance_threshold=0.5):
        if not corners:
            return []

        # Sort corners by x-coordinate for easier clustering
        corners = sorted(corners, key=lambda pt: (pt[0], pt[1]))
        grouped_corners = []
        current_group = [corners[0]]

        for i in range(1, len(corners)):
            # Calculate the Euclidean distance between the current point and the last point in the current group
            dist = np.linalg.norm(np.array(corners[i]) - np.array(current_group[-1]))
            
            # If within threshold, add to the current group
            if dist < distance_threshold:
                current_group.append(corners[i])
            else:
                # Average the current group and add it as a single point
                mean_x = int(np.mean([pt[0] for pt in current_group]))
                mean_y = int(np.mean([pt[1] for pt in current_group]))
                grouped_corners.append((mean_x, mean_y))
                
                # Start a new group with the current point
                current_group = [corners[i]]
        
        # Average and append the last group if there is one
        if current_group:
            mean_x = int(np.mean([pt[0] for pt in current_group]))
            mean_y = int(np.mean([pt[1] for pt in current_group]))
            grouped_corners.append((mean_x, mean_y))

        return grouped_corners

    grouped_corners = group_corners(corners) 
    grouped_corners = group_corners(grouped_corners, distance_threshold=corner_thresh) 
    for corner in grouped_corners:
        x, y = int(corner[0]), int(corner[1])
        if 0 <= x < corner_img.shape[1] and 0 <= y < corner_img.shape[0]:
            # corners.append((x, y))
            cv2.circle(corner_img, (x, y), 5, (0, 255, 255), -1)

    if save_img:
        # cv2.imwrite(f'MyResults/{output_name}_corners.jpg', corner_img)
        cv2.imwrite(f'MyResults/{output_name}_corners.jpg', corner_img)
        time.sleep(1)
    # return corners
    return grouped_corners

# Helper function to draw a line given rho and theta in polar coordinates
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

# Helper function to find intersection of two lines in polar coordinates
def find_intersection_polar(line1, line2):
    rho1, theta1 = line1
    rho2, theta2 = line2
    # Lines are parallel if their angles are the same
    if theta1 == theta2:
        return None
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([rho1, rho2])
    # Solve the linear system to find the intersection point
    x, y = np.linalg.solve(A, b)
    return x, y


# Main function to process each image in the dataset
def process_images(dataset_path):
    image_paths = [os.path.join(dataset_path, img) for img in os.listdir(dataset_path)]
    all_image_corners = list()
    temp = list()
    i = 0
    for idx, img_path in enumerate(image_paths):
        # print(f"Processing image {idx + 1}/{len(image_paths)}...")
        image = cv2.imread(img_path)
        # corners = detect_corners(image, save_img=True, output_name=f"Result_{idx + 1}")
        corners = detect_corners(image, save_img=True, output_name=f"TEST")
        all_image_corners.append(corners)
        if len(corners) ==80:
            temp.append(str(idx+1))
            cprint(f"{i+1}. Image {idx+1} has {len(corners)} corners", "green")
            i += 1
    ensure_directory("MyResults/TestFiles/")
    with open("MyResults/TestFiles/valid_image.txt", "w") as f:
        for i in temp:
            f.write(f"Pic_{str(i)}.jpg\n")
            
    return all_image_corners
"""========================================================= END of TASK 3.1  =========================================================="""


def task1():
    temp  = process_images(DATASET1_PATH)
    all_image_corners = list()
    for corners in temp:
        # cprint(f"Number of Detected corners: {len(corners)}", "cyan")
        if len(corners) == 80:
            all_image_corners.append(corners)
    # cprint(f"Number of images with 80 corners: {len(all_image_corners)}", "white")


    cprint("Task 1 completed successfully!", "green") 

def task2():
    pass



if __name__ == "__main__":
    # Create argument parser
    parser = ArgumentParser(
        description="Script to run Task 1 or Task 2. Use the arguments below to select the task.",
        epilog="Example usage: python HW7.py --task 1"
    )
    # Add arguments to select task
    parser.add_argument(
        '--task', '-t', type=int, required=True, choices=[1, 2],
        help="Select which task to run: 1 for Task 1, 2 for Task 2"
    )

    args = parser.parse_args()
    # Ensure you're in the correct directory
    cprint("Make sure you are running from HW8 directory!!!\n", "red")
    ensure_directory(f"MyResults")
    # Execute the task based on the argument provided
    if args.task == 1:
        task1()
    elif args.task == 2:
        task2()