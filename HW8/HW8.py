import os
import sys
import time
import logging
import functools
from tqdm import tqdm
from typing import List, Tuple
from argparse import ArgumentParser
from termcolor import cprint

# Computer vision, Data science, and plotting imports:
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Folder Paths:
DATASET1_PATH = "HW8-Files/HW8-Files/Dataset1/"
DATASET2_PATH = "HW8-Files/HW8-Files/Dataset2/"

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
    
    
'''========================================================= TASK 3.2.1 ========================================================='''
# Group lines by proximity based on rho values
def group_lines(lines, distance_threshold=0.5):
    if not lines:
        return []
    # Sort by rho:
    lines = sorted(lines, key=lambda line: line[0])  
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

def find_intersection_polar_coords(line1, line2):
    rho1, theta1 = line1
    rho2, theta2 = line2
    if theta1 == theta2:
        cprint("Lines are parallel, no intersection!!!", "red")
        return None  
    A = np.array([[np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([rho1, rho2])
    x, y = np.linalg.solve(A, b)
    return int(x), int(y)

# Function to draw a line given rho and theta in polar coordinates
def draw_polar_line(img, rho, theta, color=(0, 255, 0), thickness=2):
    a, b = np.cos(theta), np.sin(theta)
    x0, y0 = a * rho, b * rho
    x1, y1 = int(x0 + 5000 * (-b)), int(y0 + 5000 * (a))
    x2, y2 = int(x0 - 5000 * (-b)), int(y0 - 5000 * (a))
    cv2.line(img, (x1, y1), (x2, y2), color, thickness)

# Function to convert a line to homogeneous coordinates
def to_homogeneous(rho, theta):
    return np.array([np.cos(theta), np.sin(theta), -rho])

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
    a, b, c = (y1 - y2), (x2 - x1), (x1 * y2 - x2 * y1)
    return np.array([a, b, c])

# Detect corners (intersections) from image
def detect_corners(image, image_num, corner_params=(None|list), folder_name=["Edges", "HoughLines", "Corners"], pic_num=None):
    if folder_name is None:
        folder_name = ["Edges", "HoughLines", "Corners"]
    if corner_params is None:
        canny_thresh1, canny_thresh2, houghs_thresh, rho_lines_thresh, theta_thresh, corner_thresh = [400, 300, 50, 10, np.pi / 90, 10]
    else:
        canny_thresh1, canny_thresh2, houghs_thresh, rho_lines_thresh, theta_thresh, corner_thresh = corner_params
        
    # Convert to grayscale and apply edge detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, canny_thresh1, canny_thresh2)
    
    
    # blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # edges = cv2.Canny(blurred, canny_thresh1, canny_thresh2)
    dilated_edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)

    # Save edge images
    if pic_num is not None:
        ensure_directory(f"MyResults/{folder_name[0]}")
        cv2.imwrite(f'MyResults/{folder_name[0]}/Pic_{pic_num}_edges.jpg', edges)
        cv2.imwrite(f'MyResults/{folder_name[0]}/Pic_{pic_num}_dilated_edges.jpg', dilated_edges)

    # Hough Transform to detect lines in polar coordinates
    lines = cv2.HoughLines(edges, 1, np.pi / 180, houghs_thresh, None, 0, 0)
    # lines = cv2.HoughLines(dilated_edges, 1, np.pi / 180, houghs_thresh, None, 0, 0)
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
    vertical_lines = group_lines(vertical_lines, distance_threshold=rho_lines_thresh)
    horizontal_lines = group_lines(horizontal_lines, distance_threshold=rho_lines_thresh)

    # for line in horizontal_lines:
    #     rho, theta = line
    #     draw_polar_line(line_test_img, rho, theta, color=(0, 255, 0))
    #     ensure_directory("MyResults/MyDataset/Test/")
    #     cv2.imwrite(f'MyResults/MyDataset/Test/Pic_{pic_num}_horizontal_lines.jpg', line_test_img)
        
    # Cluster lines by their x_intercept to get exactly 8 vertical lines: and 10 horizontal lines
    base_pts = list()
    for line in vertical_lines:
        rho, theta = line
        # (x_intercept, 0):
        x_intercept = rho / np.cos(theta)
        # x at infinity:
        x_upper = -line_test_img.shape[0] * np.tan(theta) + (rho / np.cos(theta))
        # Append as a 2D point with base and upper x-coordinates, to cluster the lines together using Kmeans later
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
            cv2.line(line_test_img, (int(p1), 0), (int(p2), line_test_img.shape[0]), (0, 0, 255), 5) # Task 1
            # cv2.line(line_test_img, (int(p1), 0), (int(p2), line_test_img.shape[0]), (0, 0, 255), 10) # Task 2
    else:
        cprint(f"Not enough vertical points for clustering. Found only:{len(base_pts)} !!!", "red")
    
    vertical_lines = clustered_pts.cluster_centers_
    # cprint(f"Vertical Lines : {vertical_lines}", "cyan")

    
    # Cluster horizontal lines by their y-intercepts
    base_pts_horizontal = []
    for line in horizontal_lines:
        rho, theta = line
        y_intercept = rho / np.sin(theta)  # Y-intercept
        y_far = -line_test_img.shape[1] * (1 / np.tan(theta)) + y_intercept  # Y-coordinate far along the line
        # X-coordinate far along the line
        # x_far = (y_intercept-line_test_img.shape[1]) * np.tan(theta)  
        # y_far = line_test_img.shape[1]
        base_pts_horizontal.append([y_intercept, y_far])
    
    if len(base_pts_horizontal) >= 10:
        base_pts_array_horizontal = np.array(base_pts_horizontal)
        clustered_pts_horizontal = KMeans(n_clusters=10, random_state=0).fit(base_pts_array_horizontal)
        
        for p1, p2 in clustered_pts_horizontal.cluster_centers_:
            # Draw the clustered line
            cv2.line(line_test_img, (0, int(p1)), (line_test_img.shape[1], int(p2), ), (255, 0, 0), 5) # Task 1
            # cv2.line(line_test_img, (0, int(p1)), (line_test_img.shape[1], int(p2), ), (255, 0, 0), 10) # Task 2
        
    else:
        cprint(f"Not enough horizontal points for clustering. Found only: {len(base_pts_horizontal)} !!!", "red")
    
    horizontal_lines = clustered_pts_horizontal.cluster_centers_
    
    
    # Cluster lines to get exactly 10 horizontal lines: (Apparently dont need to do this as it has the perfect numer already)
    # for line in horizontal_lines:
        # rho, theta = line
        # if pic_num is not None:
            # draw_polar_line(line_test_img, rho, theta, color=(255, 0, 0))
    if len(horizontal_lines) != 10:
        cprint(f"Image {image_num} has {len(horizontal_lines)} horizontal lines", "red")
    if len(vertical_lines) != 8:
        cprint(f"Image {image_num} has {len(vertical_lines)} vertical lines", "red")
    # Save the image with lines drawn:
    if pic_num is not None:
        ensure_directory(f"MyResults/{folder_name[1]}")
        cv2.imwrite(f'MyResults/{folder_name[1]}/Pic_{pic_num}_lines.jpg', line_test_img)
        
    # Convert lines to homogeneous form for intersection calculation:
    horizontal_homogeneous_lines = [to_homogeneous(np.array(h[0]), np.array(h[1])) for h in horizontal_lines]
    # Vertical lines will be specified by their x-intercepts, because the cluster_centers_ are in that form (x_intercept, x_at_infinity) i.e (rho1, rho2):
    # Need to sort so labeling is top-left to bottom-right:
    # Since x[0] is the upper x-intercept, since 0,0 is at the top left corner sort according to the x-intercept:
    sorted_v_lines = sorted(vertical_lines, key=lambda x: x[0]) 
    vertical_homogeneous_lines = list()
    for rho1, rho2 in sorted_v_lines:
        vertical_homogeneous_lines.append(calculate_homogeneous_line(rho1, rho2, line_test_img.shape[0]))
    
    sorted_h_lines = sorted(horizontal_lines, key=lambda x: x[1]) 
    horizontal_homogeneous_lines = list()
    for rho1, rho2 in sorted_h_lines:
        def calculate_homogeneous_line_horizontal(rho1, rho2, image_width):
            # Define two points on the line
            # Y intercept and then the second point uses a large x-value (approaching infinity, i.e., image width)
            x1, y1 = 0, rho1  # Start at the y-intercept
            x2, y2 = image_width, rho2  # A point far along the x-direction

            # Compute homogeneous coordinates (a, b, c) for the line
            a, b, c = (y1 - y2), (x2 - x1), (x1 * y2 - x2 * y1)
            return np.array([a, b, c])
        horizontal_homogeneous_lines.append(calculate_homogeneous_line_horizontal(rho1, rho2, line_test_img.shape[1]))

    # horizontal_homogeneous_lines.remove(horizontal_homogeneous_lines[-1])
    
    
    
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
                    cv2.circle(line_test_img, intersection, 5, (0, 255, 255), -1) # For Task 1
                    # cv2.circle(line_test_img, intersection, 20, (0, 255, 255), -1) # For Task 2
                    cv2.putText(line_test_img, str(len(intersections)), (intersection[0]+5, intersection[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1) # Task 1
                    # cv2.putText(line_test_img, str(len(intersections)), (intersection[0]+10, intersection[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255,255), 7) # Task 2

    # cprint(f"Intersections: {intersections}", "cyan")

    # Save the image with and lines corners drawn:
    if pic_num is not None:            
        ensure_directory(f"MyResults/{folder_name[2]}")
        cv2.imwrite(f'MyResults/{folder_name[2]}/Pic_{pic_num}_corners.jpg', line_test_img)

    # ensure_directory("TrialResults/ALL_LINES")
    # cv2.imwrite(f'TrialResults/ALL_LINES/TEST_{pic_num}_lines.jpg', line_test_img)
    # time.sleep(1)  
    return intersections

# Process images and detect corners
def process_images(dataset_path, f_name:(None |list) = None, params: (None|list) = None):
    image_paths = [os.path.join(dataset_path, img) for img in os.listdir(dataset_path)]
    all_image_corners = []
    for idx in tqdm(range(len(image_paths))):
        image = cv2.imread(image_paths[idx])
        
        corners = detect_corners(image, idx, folder_name=f_name, pic_num=idx+1, corner_params=params)
        all_image_corners.append(corners)
    return all_image_corners
"""========================================================= END of TASK 3.2.1 ========================================================="""

"""========================================================= START of TASK 3.2 ========================================================="""
# 3.2.2 Camera Calibration
'''(From previous HW) Compute the Homography H using the concept Ax = b -> x = A^-1*b :'''    
def compute_homography(src_pts: np.ndarray, dest_pts: np.ndarray) -> np.ndarray:
    A = list()
    for i in range(len(dest_pts)):
        x, y = src_pts[i][0], src_pts[i][1]
        x_prime, y_prime = dest_pts[i][0], dest_pts[i][1]
        A.append([0, 0, 0, -x, -y, -1, y_prime * x, y_prime * y, y_prime])
        A.append([x, y, 1, 0, 0, 0, -x_prime * x, -x_prime * y, -x_prime])
    A = np.array(A)
    H = np.linalg.svd(np.matmul(A.T, A))[-1][-1]
    H = np.reshape(H, (3, 3))
    return H

def compute_v_ij(h: np.ndarray , i: int, j: int) -> np.ndarray:
    return np.array([
        h[0, i] * h[0, j],                        # h_i1 * h_j1
        h[0, i] * h[1, j] + h[1, i] * h[0, j],    # h_i1 * h_j2 + h_i2 * h_j1
        h[1, i] * h[1, j],                        # h_i2 * h_j2
        h[2, i] * h[0, j] + h[0, i] * h[2, j],    # h_i3 * h_j1 + h_i1 * h_j3
        h[2, i] * h[1, j] + h[1, i] * h[2, j],    # h_i3 * h_j2 + h_i2 * h_j3
        h[2, i] * h[2, j]                         # h_i3 * h_j3
    ])

def compute_omega(homographies: List[np.ndarray]) -> np.ndarray:
    # Construct the matrix B from all homographies
    B = []
    for h in homographies:
        # Each homography provides two constraints
        # First row, Second Row, First and second rows
        v12 = compute_v_ij(h, 0, 1)
        v11 = compute_v_ij(h, 0, 0)
        v22 = compute_v_ij(h, 1, 1)
        B.append(v12)            # (orthogonality)
        B.append(v11 - v22)      # (scale)
    B = np.array(B)
    # Solve for omega by finding the null space of B using SVD
    # The last row of V from SVD corresponds to the smallest singular value
    omega_vec = np.linalg.svd(B)[-1][-1]  

    # Reshape omega_vec into the 3x3 symmetric matrix omega:
    omega = np.array([
        [omega_vec[0], omega_vec[1], omega_vec[3]],
        [omega_vec[1], omega_vec[2], omega_vec[4]],
        [omega_vec[3], omega_vec[4], omega_vec[5]]
    ])
    return omega


def calc_world_cords(grid_length:int = 10, num_vert: int = 8, num_horiz: int = 10) -> List[List[int]]:
    # All of these input params are in inches!!!
    # Generate x and y coordinates using lists
    x = [i * grid_length for i in range(num_vert)]
    y = [i * grid_length for i in range(num_horiz)]
    # Manually create the mesh grid as lists of coordinate pairs
    world_coords = []
    for yi in y:
        for xi in x:
            world_coords.append([xi, yi])
    return world_coords
            
# Calculate the intrinsic matrix K from the image of the absolute conic omega.
def calc_intrinsic_params(omega: np.ndarray) -> np.ndarray:
    # Elements of the omega matrix
    omega_11, omega_12, omega_13 = omega[0, 0], omega[0, 1], omega[0, 2]
    omega_22, omega_23, omega_33 = omega[1, 1], omega[1, 2], omega[2, 2]

    # Compute principal point (c_x, c_y)
    denominator = omega_11 * omega_22 - omega_12**2
    c_x = (omega_12 * omega_23 - omega_13 * omega_22) / denominator
    c_y = (omega_12 * omega_13 - omega_11 * omega_23) / denominator

    # Compute focal lengths (f_x, f_y)
    f_x = np.sqrt((omega_33 * omega_11 - omega_13**2) / denominator)
    f_y = np.sqrt((omega_33 * omega_22 - omega_23**2) / denominator)

    # Compute skew (s)
    s = (omega_12 - c_x * c_y) / (f_x * f_y)

    # Construct the intrinsic matrix K
    K = np.array([
        [f_x, s,   c_x],
        [0,   f_y, c_y],
        [0,   0,   1  ]
    ])

    return K

def calc_extrinsic_params(H: np.ndarray, K: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Compute K^(-1) * H
    K_inv = np.linalg.inv(K)
    h1 = H[:, 0]
    h2 = H[:, 1]
    h3 = H[:, 2]
    
    # Compute the normalized rotation vectors and translation vector
    r1 = np.dot(K_inv, h1)
    r2 = np.dot(K_inv, h2)
    t = np.dot(K_inv, h3)
    
    # Normalize r1 and r2 to get unit vectors
    norm_r1 = np.linalg.norm(r1)
    norm_r2 = np.linalg.norm(r2)

    r1 /= (norm_r1 + norm_r2) / 2.0
    r2 /= (norm_r1 + norm_r2) / 2.0
    t /= (norm_r1 + norm_r2) / 2.0
    
    # Compute r3 as the cross product of r1 and r2
    r3 = np.cross(r1, r2)
    
    # Construct the rotation matrix R
    R = np.column_stack((r1, r2, r3))
    
    # Use SVD to ensure R is a valid rotation matrix(ortho-normal)
    U, _, Vt = np.linalg.svd(R)
    R = np.dot(U, Vt)
    return R, t

def calc_projection_matrix(K, extrinsic_parameters):
    P = list()
    for i, Rt in enumerate(extrinsic_parameters):
        P.append(np.dot(K, Rt))
    return P

# Extra Credit: Calculate radial distortion parameters:
def calc_radial_distortion_params(K, extrinsic_params, all_image_corners, world_coords):
    # Initial radial distortion parameters [k1, k2]
    initial_k = np.zeros(2)
    # Define the residual function for least squares
    def residuals(params: np.ndarray) -> np.ndarray:
        K_refined = params[:9].reshape((3, 3))
        k1, k2 = params[9:11]
        refined_extrinsics = [params[11 + i*12:11 + (i+1)*12].reshape(3, 4) for i in range(len(all_image_corners))]
        projection_matrices = calc_projection_matrix(K_refined, refined_extrinsics)
        residuals_list = []
        for i, corners in enumerate(all_image_corners):
            reprojected_points = reproject_points_with_distortion(projection_matrices[i], world_coords, K_refined, k1, k2)
            error = reprojected_points - corners
            residuals_list.extend(error.flatten())
        return np.array(residuals_list)

    # Combine K, extrinsic parameters, and initial_k into the initial parameter array
    initial_params = np.concatenate([K.flatten(), [initial_k[0], initial_k[1]], np.hstack([param.flatten() for param in extrinsic_params])])

    # Run least squares optimization to find optimal K, extrinsic parameters, and radial distortion
    result = least_squares(residuals, initial_params, method='lm', verbose=1)
    optimized_K = result.x[:9].reshape((3, 3))
    optimized_k = result.x[9:11]
    optimized_extrinsics = [result.x[11 + i*12:11 + (i+1)*12].reshape(3, 4) for i in range(len(all_image_corners))]
    return optimized_K, optimized_extrinsics, optimized_k

def reproject_points_with_distortion(P, world_coords, K, k1, k2):
    world_coords_homogeneous = np.hstack((world_coords, np.zeros((len(world_coords), 1)), np.ones((len(world_coords), 1)))).T
    reprojected_corners_homogeneous = np.dot(P, world_coords_homogeneous)
    reprojected_corners = (reprojected_corners_homogeneous[:2] / reprojected_corners_homogeneous[2]).T
    
    # Apply radial distortion
    x = reprojected_corners[:, 0]
    y = reprojected_corners[:, 1]
    cx, cy = K[0, 2], K[1, 2]
    r2 = (x - cx) ** 2 + (y - cy) ** 2
    x_distorted = x + (x - cx) * (k1 * r2 + k2 * r2 ** 2)
    y_distorted = y + (y - cy) * (k1 * r2 + k2 * r2 ** 2)
    reprojected_corners_distorted = np.column_stack((x_distorted, y_distorted))
    return reprojected_corners_distorted


def zhangs_algo(all_image_corners):
    # Get world coordinates:
    world_cords = calc_world_cords()
    for row in (world_cords:= calc_world_cords()): pass #print(row)
    
    # Calculate Homography for each corner:
    homography_matrices = []
    for corners in all_image_corners:   
        homography_matrices.append(compute_homography(world_cords, corners))
    
    # Calculate Omega H^-TIH^T:        
    omega = compute_omega(homography_matrices)
    # cprint(f"Omega: {omega}", "cyan")
    
    # Calculate Intrinsic Parameters:
    K = calc_intrinsic_params(omega)
    # cprint(f"Intrinsic Parameters: {K}", "cyan")
    
    # Calculate Extrinsic Parameters:
    extrinsic_parameters = list() # Where each element is the combination of R and t for each image
    for H in homography_matrices:
        R, t = calc_extrinsic_params(H, K)
        R_and_t = np.hstack((R, t.reshape(-1, 1)))  # reshape r to ensure it is a (3, 1)
        assert R_and_t.shape == (3, 4), "R_and_t matrix should have dimensions (3, 4)"
        # R_and_t = np.array(R_and_t) # Convert to numpy array
        extrinsic_parameters.append((R_and_t))
        # print(f"\b Extrinsic Parameter:\n {R_and_t}")
    # for i, Rt in enumerate(extrinsic_parameters):
        # cprint(f"Extrinsic Matrix [R | t] for Image {i + 1}:\n {Rt}\n", "cyan")

    # Calculate Projection Matrix:
    projection_matrix = calc_projection_matrix(K, extrinsic_parameters)
    
    return homography_matrices, K, extrinsic_parameters, projection_matrix

def reprojection_error(all_image_corners, projection_matrix):
    reprojection_errors = []
    world_coords = calc_world_cords()
    world_coords_homogeneous = np.hstack((world_coords, np.zeros((80, 1)), np.ones((80, 1)))).T  
    assert world_coords_homogeneous.shape == (4, 80), "World coordinates should have shape (4, 80)"
    
    # For all the image corners, calculate the reprojected points and compare with the detected corners:
    for i, corners in enumerate(all_image_corners):
        # Project the world coordinates using the projection matrix
        reprojected_corners_homogeneous = np.dot(projection_matrix[i], world_coords_homogeneous)
        
        # Normalize to convert from homogeneous to 2D
        reprojected_corners = (reprojected_corners_homogeneous[:2] / reprojected_corners_homogeneous[2]).T
    
        # Calculate Euclidean distance between reprojected and detected corners
        error = np.linalg.norm(reprojected_corners - corners, axis=1)/len(corners)
        reprojection_errors.append(error)  # Mean error for this image

    # Average and variance over all images
    cprint(f"Average reprojection error: {(avg_reprojection_error:=np.mean(reprojection_errors))}")
    cprint(f"Reprojection Variance: {(variance_reprojection_error:=np.var(reprojection_errors))}")
    return avg_reprojection_error, variance_reprojection_error, reprojection_errors

def overlay_projections_on_image(image, reprojected_points, detected_corners, reprojection_name: str = "Reprojected Corners", detect_color=(0, 255, 255), reproject_color=(0, 0, 255)):
    image_with_projections = image.copy()
    reprojected_points = np.int32(reprojected_points)
    detected_corners = np.int32(detected_corners)

    # Display legend for colors representing each type of point
    cv2.putText(image_with_projections, "Detected Corners", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, detect_color, 1) # Task 1
    cv2.putText(image_with_projections, reprojection_name, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, reproject_color, 1) # Task 1
    
    
    # cv2.putText(image_with_projections, "Detected Corners", (6*10, 6*20), cv2.FONT_HERSHEY_SIMPLEX, 3, detect_color, 3) # Task 2
    # cv2.putText(image_with_projections, reprojection_name, (10*10, 10*40), cv2.FONT_HERSHEY_SIMPLEX, 3, reproject_color, 3) # Task 2

    
    

    point_thickness = 5 # Task 1 
    # point_thickness = 20 # Task 2 
    # Draw detected corners
    for idx, corner in enumerate(detected_corners):
        cv2.circle(image_with_projections, tuple(corner), point_thickness, detect_color, -1)
        cv2.putText(image_with_projections, str(idx + 1), (corner[0] + 5, corner[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1) # Task 1
        # cv2.putText(image_with_projections, str(idx + 1), (corner[0] + 10, corner[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 7) # Task 2
    # Draw reprojected points
    for idx, point in enumerate(reprojected_points):
        cv2.circle(image_with_projections, tuple(point), point_thickness, reproject_color, -1)

    return image_with_projections

def plot_homography_projections(all_image_corners, projection_matrix, dataset_path, folder_name="Homography_Projections", base_file_name="Pic", reproject_name="Reprojected Corners"):
    # Get image paths from dataset folder
    image_paths = [os.path.join(dataset_path, img) for img in os.listdir(dataset_path)]
    images = [cv2.imread(path) for path in image_paths]

    # Ensure target directory exists
    ensure_directory(f"MyResults/{folder_name}")
    # Process each image and save with overlaid projections
    for i, (corners, image) in enumerate(zip(all_image_corners, images)):
        reprojected_points = reproject_points(projection_matrix[i], calc_world_cords())
        image_with_projections = overlay_projections_on_image(image, reprojected_points, corners, reprojection_name=reproject_name)
        cv2.imwrite(f"MyResults/{folder_name}/{base_file_name}_{i+1}.jpg", image_with_projections)

def reproject_points(P, world_coords):
    # Convert world coordinates to homogeneous form (4xN)
    world_coords_homogeneous = np.hstack((world_coords, np.zeros((len(world_coords), 1)), np.ones((len(world_coords), 1)))).T  
    assert world_coords_homogeneous.shape[0] == 4, "World coordinates should have shape (4, N)"
    # Perform projection
    reprojected_corners_homogeneous = np.dot(P, world_coords_homogeneous)  
    assert reprojected_corners_homogeneous.shape[0] == 3, "Reprojected corners should have shape (3, N)"
    # Normalize to convert from homogeneous to 2D
    reprojected_corners = (reprojected_corners_homogeneous[:2] / reprojected_corners_homogeneous[2]).T 
    assert reprojected_corners.shape[0] == len(world_coords), "Reprojected corners should have the same number of points as world coordinates"
    return reprojected_corners

@time_decorator
def lm_optimization(K, extrinsic_params, all_image_corners, world_coords):
    # Flatten initial K and extrinsic parameters for least squares
    initial_params = np.concatenate([K.flatten(), np.hstack([param.flatten() for param in extrinsic_params])])
    assert initial_params.shape == (9 + 12 * len(all_image_corners),), "Initial parameters should have the correct shape (9 + 12 * num_images)"
    def residuals(params: np.ndarray) -> np.ndarray:
        # Separate K and extrinsic parameters from the parameter array
        K_refined = params[:9].reshape((3, 3))
        assert K_refined.shape == (3, 3), "Intrinsic matrix should have shape (3, 3)"
        refined_extrinsics = [params[9 + i*12:9 + (i+1)*12].reshape(3, 4) for i in range(len(all_image_corners))]
        assert all([param.shape == (3, 4) for param in refined_extrinsics]), "Extrinsic parameters should have shape (3, 4)"
        # Calculate projection matrices using refined parameters:
        projection_matrices = calc_projection_matrix(K_refined, refined_extrinsics)
        # Calculate residuals for all points across all images
        residuals_list = []
        for i, corners in enumerate(all_image_corners):
            reprojected_points = reproject_points(projection_matrices[i], world_coords)
            error = reprojected_points - corners
            residuals_list.extend(error.flatten())  
        return np.array(residuals_list)  
    
    # Run least squares optimization
    result = least_squares(residuals, initial_params, method='lm',verbose=True)
    optimized_K = result.x[:9].reshape((3, 3))
    optimized_extrinsics = [result.x[9 + i*12:9 + (i+1)*12].reshape(3, 4) for i in range(len(all_image_corners))]
    return optimized_K, optimized_extrinsics
"""========================================================= END of TASK 3.2 ========================================================="""

"""========================================================= START of TASK 3.3 ========================================================="""
def plot_camera_pose(R: np.ndarray, t:np.ndarray, ax, color='orange', plane_size: int=10):
    # Calculate the camera center
    C = -np.dot(R.T, t)

    # Calculate the world coordinates of camera axes: X_cam axis, Y_cam axis, Z_cam axis in world coordinates
    x_cam = np.dot(R.T, [1, 0, 0]) + C  
    y_cam = np.dot(R.T, [0, 1, 0]) + C  
    z_cam = np.dot(R.T, [0, 0, 1]) + C 
    
    # Normalize and scale the world coordinates of camera axes:
    x_norm = (x_cam - C) / np.linalg.norm(x_cam - C)
    y_norm = (y_cam - C) / np.linalg.norm(y_cam - C)
    z_norm = (z_cam - C) / np.linalg.norm(z_cam - C)

    # Plot the camera center
    ax.scatter(*C, marker='*', color='yellow')  

    # Plot the axes: X axis(red), Y axis(green), Z axis(blue):
    ax.quiver(*C, x_norm[0], y_norm[1], z_norm[2], length=plane_size+5 ,color='red')    
    ax.quiver(*C, *y_norm, length=plane_size+5 ,color='green')  
    ax.quiver(*C, *z_norm, length=plane_size+5 ,color='blue')   

    # Define the principal plane corners in world coordinates for the outline
    plane_corners = [
                    (C + plane_size * (x_cam - C) + plane_size * (y_cam - C)),
                    (C - plane_size * (x_cam - C) + plane_size * (y_cam - C)),
                    (C - plane_size * (x_cam - C) - plane_size * (y_cam - C)),
                    (C + plane_size * (x_cam - C) - plane_size * (y_cam - C))
                ]

    # Draw the outline of the principal plane as a semi-transparent rectangle
    poly = Poly3DCollection([plane_corners], color=color, alpha=0.1)
    ax.add_collection3d(poly)
    return

# Call the function to plot the 5x4 pattern
def plot_checkerboard_calibration_pattern(ax, rows=9, cols=7, square_size=1):
    for i in range(rows):
        for j in range(cols):
            x, y = i * square_size, j * square_size
            # Define the four corners of each square in 3D
            corners = [
                [x, y, 0],
                [x + square_size, y, 0],
                [x + square_size, y + square_size, 0],
                [x, y + square_size, 0]
            ]
            # Alternate color: start each row with the opposite color from the previous row
            color = 'white' if (i % 2 == 0 and j % 2 == 0) or (i % 2 == 1 and j % 2 == 1) else 'black'
            # Create a square for each grid cell on the Z=0 plane
            poly = Poly3DCollection([corners], color=color, edgecolor='w')
            ax.add_collection3d(poly)
    return

def plot_calibration_pattern(ax, grid_size=[80, 100], square_size=1):
    for i in range(grid_size[0]-10):
        for j in range(grid_size[1]-10):
            x, y = i * square_size, j * square_size
            # Define the four corners of each square in 3D
            corners = [
                [x, y, 0],
                [x + square_size, y, 0],
                [x + square_size, y + square_size, 0],
                [x, y + square_size, 0]
            ]
            # Create a square for each grid cell on the Z=0 plane
            poly = Poly3DCollection([corners], color=(0,0,0), alpha=0.5)
            # poly = Poly3DCollection([corners], color=1, alpha=0.5)
            ax.add_collection3d(poly)
    return

def plot_all_cameras(extrinsic_params):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot each camera using its extrinsic parameters
    cam_num = len(extrinsic_params)
    psize = 20 # Size of the principal plane
    for Rt in extrinsic_params:
        if cam_num == 0:
            break
        cam_num -= 1
        R = Rt[:, :3]
        t = Rt[:, 3]
        plot_camera_pose(R, t, ax, color=np.random.rand(3,), plane_size=psize)  # Assign a random color to each camera
        
    # Add the calibration pattern to the plot
    plot_calibration_pattern(ax)
    # plot_checkerboard_calibration_pattern(ax, rows=5, cols=4, square_size=1)
    
    # Set plot limits and labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_zlim([-300, 0])
    # ax.view_init(-15, -107)
    # ax.view_init(-13, -103)
    # ax.view_init(-1,-103)
    # ax.view_init(-8,-89)
    # ax.view_init(-55,-108)
    plt.show()

import matplotlib.pyplot as plt

# After calculating the initial and optimized reprojection errors

# Plotting mean errors per image before and after optimization
def plot_errors_per_image(inital_errors, optimized_errors):
    mean_errors_before = list()
    mean_errors_after = list()
    variance_errors_before = list()
    variance_errors_after = list()
    for i in range(min(len(inital_errors), len(optimized_errors))):
        mean_errors_before.append(np.mean(inital_errors[i]))
        mean_errors_after.append(np.mean(optimized_errors[i]))
        variance_errors_before.append(np.var(inital_errors[i])) 
        variance_errors_after.append(np.var(optimized_errors[i]))

    num_images = len(mean_errors_before)
    # Image indices:
    x = range(1, num_images + 1)  
    
    # Plot mean reprojection errors
    plt.figure(figsize=(12, 6))
    plt.bar(x, mean_errors_before, color='orange', alpha=0.6, label='Mean Error Before LM')
    plt.bar(x, mean_errors_after, color='green', alpha=0.6, label='Mean Error After LM')
    plt.xlabel('Image Index')
    plt.ylabel('Mean Reprojection Error')
    plt.title('Mean Reprojection Errors Before and After LM Optimization')
    plt.legend()
    ensure_directory("MyResults/ErrorPlots")
    plt.savefig("MyResults/ErrorPlots/Mean_Reprojection_Errors.jpg")
    plt.close()

    # Plot variance of reprojection errors
    plt.figure(figsize=(12, 6))
    plt.bar(x, variance_errors_before, color='orange', alpha=0.6, label='Variance Error Before LM')
    plt.bar(x, variance_errors_after, color='green', alpha=0.6, label='Variance Error After LM')
    plt.xlabel('Image Index')
    plt.ylabel('Variance of Reprojection Error')
    plt.title('Variance of Reprojection Errors Before and After LM Optimization')
    plt.legend()
    plt.savefig("MyResults/ErrorPlots/Variance_Reprojection_Errors.jpg")
    plt.close()
    
    # Save the top 10 best images:
    # with open("MyResults/ErrorPlots/Top_10_Best_Images.txt", "w") as f:
    ensure_directory("MyResults/MyDataset/ErrorPlots/")
    with open("MyResults/MyDataset/ErrorPlots/Top_10_Best_Images.txt", "w") as f:
        f.write("Top 10 Images with the least reprojection errors:\n")
        temp = dict()
        for i, mean in enumerate(mean_errors_after):
            temp[mean] = i+1
        sorted_means = sorted(mean_errors_after)
        for i in range(len(sorted_means)):
            f.write(f"Image {temp[sorted_means[i]]}: {sorted_means[i]}\n")
    
    print("Sorted Initial Mean Errors:")
    print(sorted(mean_errors_before))
    
    print("Sorted Optimized Mean Errors:")
    print(sorted_means)

"""========================================================= END of TASK 3.3 ========================================================="""
@time_decorator
def task1():
    # Task 3.2.1: Detect corners of calibration pattern in images
    temp = process_images(DATASET1_PATH)
    corners_for_all_images = []
    for corners in temp:
        if len(corners) == 80:  # We expect 80 corners
            corners_for_all_images.append(corners)
    cprint(f"Number of images with 80 corners: {len(corners_for_all_images)}", "white")
    
    # Task 3.2.2: Zhang's algo for camera calibration:
    homography_matrices, K, extrinsic_parameters, projection_matrix = zhangs_algo(corners_for_all_images)
    
    # print(homography_matrices)
    # print(K)
    # exit(0)
    # print(extrinsic_parameters)
    # print(projection_matrix)
    
    # Plot the initial homography projections on the image
    plot_homography_projections(corners_for_all_images, projection_matrix, DATASET1_PATH, folder_name="Before_LM_Optimization", reproject_name="Reprojection Before LM Optimization")
    
    # Optimize parameters using Levenberg-Marquardt (LM) optimization
    optimized_K, optimized_extrinsics = lm_optimization(K, extrinsic_parameters, corners_for_all_images, calc_world_cords())
    optimized_projection_matrix = calc_projection_matrix(optimized_K, optimized_extrinsics)
    # print(f"PMAtrix: \n{projection_matrix}")
    # print(f"Optimized PMAtrix: \n{optimized_projection_matrix}")
    
    # Calculate and display the initial reprojection error
    cprint("Calculating reprojection error before LM...", "yellow")
    initial_avg_errors, initial_variance_errors, initial_errors = reprojection_error(corners_for_all_images, projection_matrix)
    # Calculate and display the optimized reprojection error
    cprint("Calculating reprojection error after LM...", "yellow")
    optimized_avg_errors, optimized_variance_errors, optimized_errors = reprojection_error(corners_for_all_images, optimized_projection_matrix)
    
    # Plot the optimized/improved corners from LM optimization
    plot_homography_projections(corners_for_all_images, optimized_projection_matrix, DATASET1_PATH, folder_name="After_LM_Optimization", reproject_name="Reprojection After LM Optimization")
    
    cprint(f"Percentage Improvement after using Levenberg-Marquardt (LM): {((initial_avg_errors - optimized_avg_errors) / initial_avg_errors) * 100:.2f}%", "white")
    

    with open("MyResults/RESULTS.txt", "w") as f:
        f.write(f"K, Intrinsic Matrix:\n{K}\n")
        f.write(f"\n")
        f.write(f"Optimized K, Intrinsic Matrix:\n{optimized_K}\n")
        f.write(f"\n")
        f.write(f"\n")
        f.write(f"\n")
        
        f.write(f"Extrinsic Parameters:\n")
        for i, Rt in enumerate(extrinsic_parameters):
            if i < 10:
                f.write(f"Extrinsic Matrix [R | t] for Image {i + 1}:\n {Rt}\n")
            
        f.write(f"\n")
        f.write(f"\n")
        f.write(f"Optimized Extrinsic Parameters:\n")
        for i, Rt in enumerate(optimized_extrinsics):
            if i < 10:
                f.write(f"Extrinsic Matrix [R | t] for Image {i + 1}:\n {Rt}\n")
        
        
    
    # Plot the mean and variance errors per image:
    # Calling the function to plot errors
    # print(initial_avg_errors, optimized_avg_errors, initial_variance_errors, optimized_variance_errors)
    
    plot_errors_per_image(initial_errors, optimized_errors)
    
    
    
    # Estimating the radial distortion parameters:
    
    # Calculate and display the reprojection error after estimating radial distortion:
    
    # # Estimated pose for the Fixed Image (assume index 0 is the Fixed Image in extrinsic_parameters)
    # R_estimated, t_estimated = optimized_extrinsics[0][:, :3], optimized_extrinsics[0][:, 3]

    # # Calculate pose errors
    # rot_error = rotation_error(R_estimated, R_ground_truth)
    # trans_error = translation_error(t_estimated, t_ground_truth)
    
    # Task 3.3: Plot the camera poses in 3D:
    plot_all_cameras(extrinsic_parameters)
    
    
    
    ########################################################### EXTRA CREDIT: #############################################################
    # Plot initial reprojection without radial distortion
    print()
    print()
    print()
    cprint(f"STARTING EXTRA CREDIT: CALCULATING RADIAL DISTORTION PARAMETERS", "white")
    plot_homography_projections(corners_for_all_images, projection_matrix, DATASET1_PATH, folder_name="Before_Optimization", reproject_name="Reprojection Without Radial Distortion")

    # Calculate reprojection error before LM optimization
    initial_avg_errors, initial_variance_errors, initial_errors = reprojection_error(corners_for_all_images, projection_matrix)

    # Run optimization including radial distortion parameters
    optimized_K, optimized_extrinsics, optimized_k = calc_radial_distortion_params(K, extrinsic_parameters, corners_for_all_images, calc_world_cords())
    
    # Recalculate projection matrices with optimized extrinsic and intrinsic parameters
    optimized_projection_matrix = calc_projection_matrix(optimized_K, optimized_extrinsics)

    # Plot the reprojection with radial distortion applied
    plot_homography_projections(corners_for_all_images, optimized_projection_matrix, DATASET1_PATH, folder_name="After_Optimization", reproject_name="Reprojection With Radial Distortion")

    # Calculate reprojection error after LM optimization with radial distortion
    optimized_avg_errors, optimized_variance_errors, optimized_errors = reprojection_error(corners_for_all_images, optimized_projection_matrix)
    # Plot error comparison before and after optimization
    plot_errors_per_image(initial_errors, optimized_errors)

    # Display results
    # print("Optimized Intrinsic Matrix:", optimized_K)
    # print("Optimized Radial Distortion Parameters:", optimized_k)
    print(f"Improvement in reprojection error: {(initial_avg_errors - optimized_avg_errors) / initial_avg_errors * 100:.2f}%")

    # Plot all camera poses
    # plot_all_cameras(optimized_extrinsics)
    
    cprint("Task 1 completed successfully!", "green")

@time_decorator
def task2():
    # Task 3.2.1: Detect corners of calibration pattern in images
    temp = process_images(DATASET2_PATH, ["MyDataset/Edges", "MyDataset/HoughLines", "MyDataset/Corners"],params=[600, 700, 90, 10, np.pi / 90, 10])
    corners_for_all_images = []
    for corners in temp:
        if len(corners) == 80:  # We expect 80 corners
            corners_for_all_images.append(corners)
    cprint(f"Number of images with 80 corners: {len(corners_for_all_images)}", "white")
    
    # Task 3.2.2: Zhang's algo for camera calibration:
    homography_matrices, K, extrinsic_parameters, projection_matrix = zhangs_algo(corners_for_all_images)
    # print(homography_matrices)
    # print(K)
    # exit(0)
    # print(extrinsic_parameters)
    # print(projection_matrix)
    # Plot the initial homography projections on the image
    plot_homography_projections(corners_for_all_images, projection_matrix, DATASET2_PATH, folder_name="MyDataset/Before_LM_Optimization", reproject_name="Reprojection Before LM Optimization")

    # Optimize parameters using Levenberg-Marquardt (LM) optimization
    optimized_K, optimized_extrinsics = lm_optimization(K, extrinsic_parameters, corners_for_all_images, calc_world_cords())

    optimized_projection_matrix = calc_projection_matrix(optimized_K, optimized_extrinsics)
        
    # print(f"PMAtrix: \n{projection_matrix}")
    # print(f"Optimized PMAtrix: \n{optimized_projection_matrix}")
    
    # Plot the improved corners from LM optimization
    plot_homography_projections(corners_for_all_images, optimized_projection_matrix, DATASET2_PATH, folder_name="MyDataset/After_LM_Optimization", reproject_name="Reprojection After LM Optimization")
    
    # Calculate and display the initial reprojection error
    cprint("Calculating reprojection error before LM...", "yellow")
    initial_avg_errors, initial_variance_errors, initial_errors = reprojection_error(corners_for_all_images, projection_matrix)
    # Calculate and display the optimized reprojection error
    cprint("Calculating reprojection error after LM...", "yellow")
    optimized_avg_errors, optimized_variance_errors, optimized_errors = reprojection_error(corners_for_all_images, optimized_projection_matrix)
    
    # Plot the optimized/improved corners from LM optimization
    plot_homography_projections(corners_for_all_images, optimized_projection_matrix, DATASET2_PATH, folder_name="After_LM_Optimization", reproject_name="Reprojection After LM Optimization")
    
    cprint(f"Percentage Improvement after using Levenberg-Marquardt (LM): {((initial_avg_errors - optimized_avg_errors) / initial_avg_errors) * 100:.2f}%", "white")
    
    plot_errors_per_image(initial_errors, optimized_errors)
    
    
    plot_all_cameras(optimized_extrinsics)    
    cprint("Task 1 completed successfully!", "green")

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
