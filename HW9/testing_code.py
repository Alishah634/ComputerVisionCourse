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

# Folder paths:
from Task3 import task_3_main
# Path to the depth_map directory is HW9/depth_map
# Ensure the correct path to the depth_map directory is added
sys.path.append('/mnt/c/Users/syeda/Documents/Masters/Fall2024/ECE661/HW9/depth_map')
sys.path.append('/mnt/c/Users/syeda/Documents/Masters/Fall2024/ECE661/HW9/depth_map/data/scene_info/1589_subset.pkl')
sys.path.append('/mnt/c/Users/syeda/Documents/Masters/Fall2024/ECE661/HW9/data/MegaDepth_v1/1589/dense0/imgs/')
from depth_map.plot_depth_check import task4_main


'''START of Utility functions '''
# Utility functions
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def time_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"\033[33m{func.__name__} took {elapsed_time:.4f} seconds to execute\n\033[0m")
        return result
    return wrapper

def ensure_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
''' END of Utility functions '''

# Ensure the required directories are created    
# Folder Paths:
MyDATASET_PATH = "MyDataset/"
DATASET1_PATH = "Dataset1/"
ensure_directory("MyDataset/")
RESULTS_PATH = "MyResults/"
ensure_directory(RESULTS_PATH)
ensure_directory("TASK_1/")
ensure_directory("TASK_2/")
ensure_directory("TASK_3/")
ensure_directory("TASK_4/")


def calculate_image_rotation_matrix(angle, img):
    # Convert angle to radians
    theta = np.radians(angle)
    img_width, img_height = img.shape[1], img.shape[0]
    # Image center
    cx, cy = img_width / 2, img_height / 2

    # Compute components of the rotation matrix
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # Translation to move center to origin
    T_to_origin = np.array([
        [1, 0, -cx],
        [0, 1, -cy],
        [0, 0, 1]
    ])

    # Rotation matrix around origin
    R = np.array([
        [cos_theta, -sin_theta, 0],
        [sin_theta, cos_theta, 0],
        [0, 0, 1]
    ])

    # Translation back to center
    T_back = np.array([
        [1, 0, cx],
        [0, 1, cy],
        [0, 0, 1]
    ])

    # Full transformation
    M = T_back @ R @ T_to_origin
    return M


# From previous homeworks:
def extract_patch(image, point, patch_size=5):
    # Extract a patch around a given point in an image.
    y, x = point
    half_size = patch_size // 2
    if y - half_size < 0 or y + half_size >= image.shape[0] or x - half_size < 0 or x + half_size >= image.shape[1]:
        return None  
    return image[y - half_size:y + half_size + 1, x - half_size:x + half_size + 1]

def calc_SSD(patcH_left, patcH_prime):
    # Compute the SSD (Sum of Squared Differences) between two patches.
    return np.sum((patcH_left - patcH_prime) ** 2)

def find_SSD_correspondences(img1, img2, points1, points2, patch_size=5):
    # Find correspondences between points1 and points2 using SSD metric.
    correspondences = []
    # To ensure uniquene points:
    matched_points = set()  
    for point1 in tqdm(points1):
        patcH_left = extract_patch(img1, point1, patch_size)
        if patcH_left is None:
            continue  # Skip if the patch goes out of bounds

        best_match, best_score = None, float('inf')
        for point2 in points2:
            if tuple(point2) in matched_points:
                continue  # Skip already matched points
            patcH_prime = extract_patch(img2, point2, patch_size)
            if patcH_prime is None or patcH_left.shape != patcH_prime.shape:
                continue  # Skip if patch is invalid or shapes don't match
            score = calc_SSD(patcH_left, patcH_prime)
            if score < best_score:
                best_score, best_match = score, point2
        if best_match is not None:
            matched_points.add(tuple(best_match))
            correspondences.append((tuple(point1), tuple(best_match)))
    return correspondences

def visualize_correspondences(img1, img2, correspondences):
    # Ensure images are in BGR format:
    if len(img1.shape) == 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    if len(img2.shape) == 2:
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    h1, w1 = img1.shape[:2]
    h2 , w2 = img2.shape[:2]

    # Combine the two images side by side
    combined_height = max(h1, h2)
    combined_width = w1 + w2
    combined_image = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)

    # Place the two images side by side
    combined_image[:h1, :w1, :] = img1
    combined_image[:h2, w1:w1 + w2, :] = img2

    # Draw correspondences
    for correspondence in correspondences:
        if correspondence is None or correspondence[1] is None:
            continue  # Skip invalid points

        point1, point2 = correspondence

        # First image point
        pt1 = (int(point1[1]), int(point1[0]) + int(h1//2) - 200)  # (x, y)

        # Second image point with x-offset
        pt2 = (int(point2[1] + w1), int(point2[0]) + h1//2)  # (x + offset, y)

        # Random color for the correspondence
        color = tuple(np.random.randint(0, 255, 3).tolist())

        # Draw circles and lines for correspondence
        cv2.circle(combined_image, pt1, 5, color, -1) 
        cv2.circle(combined_image, pt2, 5, color, -1) 
        cv2.line(combined_image, pt1, pt2, color, 2)  

    return combined_image

'''END of functions from previous homeworks^^^'''

################################################## TASK 1 Crap: ###################################
def estimate_fundamental_matrix(points1, points2):
    assert len(points1) >= 8 and len(points2) >= 8, "Need at least 8 points for 8-point algorithm."

    points1 = np.array(points1)
    points2 = np.array(points2)

    # Normalize the points
    def normalize_points(points):
        mean = np.mean(points, axis=0)
        std = np.std(points)
        T = np.array([[1 / std, 0, -mean[0] / std],
                      [0, 1 / std, -mean[1] / std],
                      [0, 0, 1]])
        points_h = np.hstack((points, np.ones((len(points), 1))))
        points_norm = (T @ points_h.T).T
        return points_norm, T

    points1_norm, T1 = normalize_points(points1)
    points2_norm, T2 = normalize_points(points2)

    # Create the matrix A
    A = np.zeros((len(points1), 9))
    for i in range(len(points1)):
        x1, y1 = points1_norm[i, :2]
        x2, y2 = points2_norm[i, :2]
        A[i] = [x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, 1]

    # Solve for F using SVD
    _, _, Vt = np.linalg.svd(A)
    F_hat = Vt[-1].reshape(3, 3)

    # Enforce rank-2 constraint by seting the smallest singular value to 0
    U, D, Vt = np.linalg.svd(F_hat)
    D[2] = 0  
    F_hat = U @ np.diag(D) @ Vt

    # Denormalize F
    F = T2.T @ F_hat @ T1
    return F / F[-1, -1]


def calculate_epipoles(F):
    e = sc.linalg.null_space(F).flatten()
    e_prime = sc.linalg.null_space(F.T).flatten()
    return e , e_prime

def calculate_projection_matrices(F, e_prime):
    # Left projection matrix:
    P = np.hstack((np.eye(3), np.zeros((3, 1))))  
    e_prime_cross = np.array([[0, -e_prime[2], e_prime[1]],
                              [e_prime[2], 0, -e_prime[0]],
                              [-e_prime[1], e_prime[0], 0]]) 
    P_prime = np.hstack((e_prime_cross @ F, e_prime.reshape(-1, 1)))  
    return P, P_prime

def calculate_rectification_homographies(img1, img2, F, view_1_corners, view_2_corners):
    # Calculate epipoles
    e_prime, e = calculate_epipoles(F)

    # Compute the rotation for aligning epipole to 'infinity'
    def compute_rotation_matrix(e):
        e = e[:2]  
        theta = np.arctan2(e[1], e[0])  
        R = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
        return R

    # Compute translations to center:
    def compute_translation_matrix(e):
        return np.array([
            [1, 0, -img1.shape[0]],
            [0, 1, -img1.shape[1]],
            [0, 0, 1]
        ])

    # H_prime = T1 * R1 * G * T1
    
    # Left image homography (H)
    T1 = compute_translation_matrix(e)
    R1 = compute_rotation_matrix(e)
    H_left = R1 @ T1

    # Right image homography (H_prime)
    T2 = compute_translation_matrix(e_prime)
    R2 = compute_rotation_matrix(e_prime)
    H_prime = R2 @ T2
    
    cprint(f"H_left_refined: {H_left} \n", "cyan") #DEBUG STATMENTS!!!
    cprint(f"H_prime_refined: {H_prime} \n", "cyan") #DEBUG STATMENTS!!!
    return H_left, H_prime

def rectify_images(img1, img2, H_left, H_prime):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    img1_rectified = cv2.warpPerspective(img1, H_left, (w1, h1))
    img2_rectified = cv2.warpPerspective(img2, H_prime, (w2, h2))
    return img1_rectified, img2_rectified

# Function to triangulate a 3D point
def triangulate_point(x, x_prime, P, P_prime):
    A = np.zeros((4, 4))
    A[0] = x[0] * P[2] - P[0]
    A[1] = x[1] * P[2] - P[1]
    A[2] = x_prime[0] * P_prime[2] - P_prime[0]
    A[3] = x_prime[1] * P_prime[2] - P_prime[1]
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    return X / X[-1]  # Homogeneous coordinates

# Function to project a 3D point into an image plane
def project_point(X, P):
    projected = P @ X
    return projected[:2] / projected[2]  # Normalize

# Cost function for reprojection error
def reprojection_residuals(params, points1, points2, P, P_prime_init):
    # Reshape the parameter vector to get P'
    P_prime = P_prime_init.copy()
    P_prime[:3, :] = params.reshape(3, 4)

    residuals = []
    for x, x_prime in zip(points1, points2):
        # Triangulate the 3D point
        X = triangulate_point(x, x_prime, P, P_prime)
        # Compute the reprojection residuals
        x_proj = project_point(X, P)
        x_prime_proj = project_point(X, P_prime)
        residuals.extend(x[:2] - x_proj)
        residuals.extend(x_prime[:2] - x_prime_proj)
    return np.array(residuals)

# Function to refine the projection matrix using LM optimization
def refine_projection_matrix_lm(P, P_prime, points1, points2):
    initial_params = P_prime[:3, :].flatten()

    # Optimize the parameters using LM
    result = least_squares(
        reprojection_residuals,
        initial_params,
        args=(points1, points2, P, P_prime),
        method='lm',
        verbose=True
    )
    optimized_P_prime = P_prime.copy()
    optimized_P_prime[:3, :] = result.x.reshape(3, 4)
    return optimized_P_prime


def calculate_fundamental_matrix(P, P_prime, epipole_prime):
    p_inverse = np.linalg.pinv(P)
    F = np.array([[0, -epipole_prime[2], epipole_prime[1]], 
                  [epipole_prime[2], 0, -epipole_prime[0]], 
                  [-epipole_prime[1], epipole_prime[0], 0]]) @ P_prime @ p_inverse
    return F 

'''========================================================= Start of TASK 1: Projective Stero Reconstruction: ========================================================='''

'''========================================================= End of TASK 1: Projective Stero Reconstruction: ========================================================='''
    
# Hardcode manually selected corners for the two views (points are in order of top-left ->  bottom-left, top-right ->bottom-right):    
# view_1_corners = [ [226.5, 736.5], [270, 1192.5], [324, 1974], [723, 1440], [1071, 675], [1059, 1023], [1050, 1341], [1045, 1645.5]]
# view_2_corners = [ [196, 1018], [226, 1270], [504, 1516], [286, 1748], [744, 992], [756, 1280], [778, 1536], [804, 1806]]


@time_decorator
def task1():
    '''TASK 3.1: Image Rectification'''
    cprint(f"Running TASK 3.1: Image Rectification...", "yellow")
    # Book images (These sucked had to take new images then it worked:
    # Read the images and resize them to (256, 256) :
    img1 = cv2.imread(MyDATASET_PATH + "view1.jpg")
    img2 = cv2.imread(MyDATASET_PATH + "view2.jpg")
    view_1_corners = [ [384, 1556], [434, 1558], [395, 1447], [520, 1447], [480, 1332], [422, 1222], [282, 827] ,   [1651, 1255], [1717, 1122], [1560, 934], [1457, 807], [1114, 558]]
    view_2_corners = [ [791, 1819], [854, 1789], [831, 1733], [938, 1621], [862, 1512], [769, 1441], [210, 1131] ,[1736, 1023], [1770, 890], [1575, 770], [1408, 692], [884, 599]] 
    
    # img1 = cv2.resize(img1, (512, 512))
    # img2 = cv2.resize(img2, (512, 512))
    cv2.imwrite(RESULTS_PATH + "test_img1_resize.jpg", img1)
    cv2.imwrite(RESULTS_PATH + "test_img2_resize.jpg", img2)

    # Trash images:
    img1 = cv2.imread(MyDATASET_PATH + "view3.jpg")
    img2 = cv2.imread(MyDATASET_PATH + "view4.jpg")
    view_1_corners = [ [348, 206], [1156, 148], [1422, 344], [1190, 644], [1038, 1194], [570, 1320], [426, 648], [322, 556], [402, 436], [470, 378], [558, 364], [932, 324]  ]
    view_2_corners = [ [320, 236], [1034, 148], [1302, 306], [1126, 592], [988, 1120], [570, 1302], [462, 638], [380, 568], [458, 454], [442, 398], [522, 372], [852, 316] ]
    
    print(len(view_1_corners), len(view_2_corners))
    
    # Apply canny edge detection to the images:
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Compute the fundamental matrix
    F = estimate_fundamental_matrix(view_1_corners, view_2_corners)
    print("Fundamental Matrix:\n", F)
    assert F.shape == (3, 3), "Fundamental matrix should be a 3x3 matrix."
    assert np.linalg.matrix_rank(F) == 2, "Fundamental matrix should have rank 2."
        
    left_epipole, right_epipole = calculate_epipoles(F)
    print("Left Epipole (e):", left_epipole)
    print("Right Epipole (e'):", right_epipole)

        
    P, P_prime = calculate_projection_matrices(F, right_epipole)
    print("Left Projection Matrix (P):\n", P)
    print("Right Projection Matrix (P'):\n", P_prime)

    #TODO NEED TO REFINE USING LM!!!:
    # Refine P'
    ensure_directory("REFINED/")
    P_prime_refined = refine_projection_matrix_lm(P, P_prime, view_1_corners, view_2_corners)
    print("Refined Right Projection Matrix (P'):\n", P_prime_refined)
    
    # Refined fundamental matrix
    F_refined = calculate_fundamental_matrix(P, P_prime_refined, right_epipole)
    # left_epipole, right_epipole = calculate_epipoles(F_refined) # Redundant for testing purposes!!!

    # Compute rectification homographies
    H, H_prime = calculate_rectification_homographies(img1, img2, F_refined, view_1_corners, view_2_corners)
    
    H_test = H_prime.copy()
    # H_test = H_test @ shift_translation
    angle = -20  # Rotate by 45 degrees
    rotation_matrix = calculate_image_rotation_matrix(angle, img2)
    H_test = H_test @ rotation_matrix

    cprint(f"(H, W) of left image: {img1.shape}" , "cyan")
    cprint(f"(H, W) of right image: {img2.shape}", "cyan")
    shift_translation = np.array([[1, 0, -250], [0, 1, 0], [0, 0, 1]])
    # # Rectify images
    print("Starting rectification:")
    # img1_rectified, img2_rectified = rectify_images(img1, img2, H@shift_translation, H_prime@shift_translation)
    # img1_rectified, img2_rectified = rectify_images(img1, img2, H@shift_translation, H_test)
    img1_rectified, img2_rectified = rectify_images(img1, img2, H, H_prime@rotation_matrix)

    # Save rectified images
    cv2.imwrite("REFINED/"+"test_img1_rectified.jpg", img1_rectified)
    cv2.imwrite("REFINED/"+"test_img2_rectified.jpg", img2_rectified)

    '''TASK 3.2: Interest Point Detection'''
    cprint(f"Running TASK 3.2: Interest Point Detection...", "yellow")

    img1_gray = cv2.GaussianBlur(img1_rectified, (5, 5), 0)
    img2_gray = cv2.GaussianBlur(img2_rectified, (5, 5), 0)
    img1_gray = cv2.medianBlur(img1_rectified, 7)
    img2_gray = cv2.medianBlur(img1_rectified, 7)

    edges1 = cv2.Canny(img1_gray, 100, 150)
    edges2 = cv2.Canny(img2_gray, 100, 150)
    
    kernel = np.ones((5,5), np.uint8)  
    dilated_edges1 = cv2.dilate(edges1, kernel, iterations=1)
    dilated_edges2 = cv2.dilate(edges2, kernel, iterations=1)
    both_de_edges1 = cv2.erode(dilated_edges1, kernel, iterations=1)
    both_de_edges2 = cv2.erode(dilated_edges1, kernel, iterations=1)
    both_ed_edges1 = cv2.dilate(cv2.erode(cv2.dilate(edges1, np.ones((3,3), np.uint8) , iterations=1), kernel, iterations=1), kernel, iterations=2)
    both_ed_edges2 = cv2.dilate(cv2.erode(cv2.dilate(edges2, np.ones((3,3), np.uint8) , iterations=1), kernel, iterations=1), kernel, iterations=2)

    cv2.imwrite(RESULTS_PATH + "img1_edges.jpg", edges1)
    cv2.imwrite(RESULTS_PATH + "img2_edges.jpg", edges2)
    cv2.imwrite(RESULTS_PATH + "img1_dilated_edges.jpg", dilated_edges1)
    cv2.imwrite(RESULTS_PATH + "img2_dilated_edges.jpg", dilated_edges2)

    # Testing canny:    
    cv2.imwrite("TEST_DATASET/" + "test_img1_edges.jpg", edges1)
    cv2.imwrite("TEST_DATASET/" + "test_img2_edges.jpg", edges2)
    cv2.imwrite("TEST_DATASET/" + "test_img1_dilated_edges.jpg", dilated_edges1)
    cv2.imwrite("TEST_DATASET/" + "test_img2_dilated_edges.jpg", dilated_edges2)
    cv2.imwrite("TEST_DATASET/" + "img1_both_dilate_errode_edges.jpg", both_de_edges1)
    cv2.imwrite("TEST_DATASET/" + "img2_both_dilate_errode_edges.jpg", both_de_edges2)
    cv2.imwrite("TEST_DATASET/" + "img1_both_errode_dilate_edges.jpg", both_ed_edges1)
    cv2.imwrite("TEST_DATASET/" + "img2_both_errode_dilate_edges.jpg", both_ed_edges2)


    # Find coordinates of edge pixels (interest points i.e (y, x) coordinates of edge pixels)
    points1 = np.column_stack(np.where(edges1 > 0))[:100]  
    points2 = np.column_stack(np.where(edges2 > 0))[:100]  
    print(points1)
    print(points2)

    # img1_gray_rectified = cv2.imread("REFINED/img1_rectified.jpg")
    # img2_gray_rectified = cv2.imread("REFINED/img2_rectified.jpg")

    correspondences = find_SSD_correspondences(dilated_edges1, dilated_edges2, points1, points2)
    combined_image = visualize_correspondences(img1_rectified, img2_rectified, correspondences)
    cv2.imwrite("REFINED/correspondences.jpg", combined_image)
    
    
    '''TASK 3.3: Projective Reconstruction'''
    '''TASK 3.4: 3D Visual Inspection'''

    cprint("Task 1 completed successfully!", "green")
    pass

@time_decorator
def task2():
    cprint("Task 2 completed successfully!", "green")
    pass

# Dense Stereo Matching:
@time_decorator
def task3():
    # Best window size is 13:
    cprint(f"Running Task 3 with window size: {13}", "yellow")
    task_3_main(13) 
    cprint("Task 3 completed successfully!", "green")
    return

@time_decorator
def task4():
    task4_main()
    cprint("Task 4 completed successfully!", "green")
    pass

    
if __name__ == "__main__":
    parser = ArgumentParser(description="Script to run Task 1 or Task 2.")
    parser.add_argument('--task', '-t', type=int, required=True, choices=[1, 2, 3, 4], help="Select task: 1 for Task 1, 2 for Task 2")
    args = parser.parse_args()

    cprint("Make sure you are running from HW8 directory!!!", "red")
    ensure_directory("MyResults")

    if args.task == 1:
        task1()
    elif args.task == 2:
        task2()
    elif args.task == 3:
        task3()
    elif args.task == 4:
        task4()
