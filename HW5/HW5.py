import numpy as np
import cv2
from termcolor import cprint
from skimage import io
import matplotlib.pyplot as plt 
from typing import List, Tuple
from scipy.ndimage import convolve
from scipy.optimize import least_squares
import os
import random
from tqdm import tqdm
from time import sleep

''' START of Util Functions:  '''
# Save the image to the file:
def save_to_file(img, file_to_path: str):
    cv2.write(file_to_path, img)

# Create a folder if it does not exist
def ensure_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def print_matrix(matrix: np.ndarray):
    for row in matrix:
        cprint(f" ".join(f"{value: .3f}" for value in row),"white")
        
def verify_homography(src_pts: np.ndarray, dest_pts: np.ndarray)-> np.ndarray:
    return np.array((cv2.find_homography(src_pts, dest_pts))).flatten()

def read_homographies(file_path: str):
    # Read the data from the file
    with open(file_path, 'r') as file:
        data = file.read()

    # Convert the data to a numpy array
    data_array = np.fromstring(data, sep=' ')

    # Reshape the array into a 3x3 matrix
    matrix = data_array.reshape((3, 3))
    print(matrix)
    return matrix

''' END of Util Functions:  '''

 
'''START OF THE SIFT SURF USING OPENCV (TAKEN FROM HW4):'''
def OpenCV_SIFT_SURF(image_pair, pair_name: str):
    # Input images
    img1, img2 = image_pair
    # Convert the images to grayscale
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # Create a SIFT detector object
    sift = cv2.SIFT_create()
    # Detect keypoints and compute descriptors for both images
    keypoints_img1, descriptor_img1 = sift.detectAndCompute(img1_gray, None) 
    keypoints_img2, descriptor_img2 = sift.detectAndCompute(img2_gray, None) 
    # Use BFMatcher to find the best matches between the descriptors
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True) # USE A RATIO TEST !!! CHECK TUTORIAL FOR BF matches!!!
    matches = bf.match(descriptor_img1, descriptor_img2)
    # Sort matches by distance (best matches first)
    matches = sorted(matches, key=lambda x: x.distance)
    # Draw the matches (or fewer if there are not that many matches)
    combined_image = cv2.drawMatches(img1, keypoints_img1, img2, keypoints_img2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Store the points for later use:
    # Extract the corresponding points (keypoints) for both images:
    points_img1 = []
    points_img2 = []

    for match in matches:
        # QueryIdx is the index of the descriptor in img1, and TrainIdx is the index in img2
        img1_idx = match.queryIdx
        img2_idx = match.trainIdx

        # Get the coordinates of the keypoints for each match
        (x1, y1) = keypoints_img1[img1_idx].pt
        (x2, y2) = keypoints_img2[img2_idx].pt

        # Append the corresponding points
        points_img1.append((x1, y1))
        points_img2.append((x2, y2))

    # Create a results folder if it doesn'img_num exist
    save_image_path = f"MyResults/SIFT/"
    ensure_directory(save_image_path)
    # Save the resulting image with correspondences
    cv2.imwrite(f"{save_image_path}{pair_name}.jpg", combined_image)
    cprint(f"Saved SIFT Correspondences for pair {pair_name}", "green")
    
    # Return the corresponding points (keypoints)
    return points_img1, points_img2
'''END OF THE SIFT SURF USING OPENCV (TAKEN FROM HW4):'''

'''START OF THE RANSAC USING OPENCV:'''
'''Compute the Homography H using the concept Ax = b -> x = A^-1*b :'''    
def compute_homography(src_pts: np.ndarray, dest_pts: np.ndarray) -> np.ndarray:
    valid = True
    # Adjust size of A and b for N points
    num_points = src_pts.shape[0]
    A = np.zeros((2 * num_points, 8))
    b = np.zeros((2 * num_points,))

    for i in range(num_points):
        if len(src_pts[i]) == 2 and len(dest_pts[i]) == 2:
            x, y = src_pts[i]
            x_prime, y_prime = dest_pts[i]
        else:
            x, y, _ = src_pts[i]
            x_prime, y_prime, _ = dest_pts[i]

        A[2 * i] = [x, y, 1, 0, 0, 0, -x * x_prime, -y * x_prime]
        A[2 * i + 1] = [0, 0, 0, x, y, 1, -x * y_prime, -y * y_prime]
        b[2 * i] = x_prime
        b[2 * i + 1] = y_prime
        
    # print_matrix(A)  # DEBUG STATEMENT!!!    
    try:
        # Use pseudo-inverse to avoid singular matrix issues
        H_temp = np.dot(np.linalg.pinv(A), b)   # np.dot(np.linalg.inv(np.dot(A.T, A)), np.dot(A.T, b)) # np.dot(np.linalg.pinv(A), b)  #
        H = np.append(H_temp, 1).reshape((3, 3))
    except np.linalg.LinAlgError:
        cprint("Error: Matrix inversion failed. The matrix A might be singular.", "red")
        valid = False
        H = np.eye(3)  # Return identity matrix as fallback
    except Exception as e:
        cprint("Error: {e}", "red")
        valid = False
        H = np.eye(3)  # Return identity matrix as fallback
    return H, valid


def cost_function(H, pts1, pts2):
    '''
    # Reshape H to (3, 3) homography matrix, as H is a 1D array
    # Convert pts1 to homogeneous coordinates by adding a column of ones
    # Project points in image 1 (pts1) using homography matrix H
    # Normalize the projected points to convert back from homogeneous coordinates to 2D
    # Compute residuals (squared error) between projected points and actual points in image 2 (pts2)
    # Return the flattened residuals array for least squares optimization, as 
    # least_squares expects a 1D array, so we flatten the Nx2 array
    '''
    H = H.reshape(3, 3)
    pts1_hc = np.hstack([pts1, np.ones((pts1.shape[0], 1))])  # Nx3 matrix
    projected_pts = (H @ pts1_hc.T).T  # Project the points (Nx3)
    projected_pts /= projected_pts[:, 2][:, np.newaxis]  # Normalize the third column to be 1 (Nx3 -> Nx2)
    res = (projected_pts[:, :2] - pts2)**2  # Residual is the squared difference between the points
    return res.flatten() 

def RANSAC(img1_matches, img2_matches, n_total: int, p: int = 0.99, n: int = 4, sigma: int = 0.5, e: int = 0.9, img_num: int = 0):
    if img1_matches is None or img2_matches is None:
        cprint(f"Matching correspondences using SIFT are missing!!!", "red")
        exit(0)

    # Convert points to homogeneous coordinates
    img1_matches_hc = np.hstack([np.array(img1_matches), np.ones((len(img1_matches), 1))])
    img2_matches_hc = np.hstack([np.array(img2_matches), np.ones((len(img2_matches), 1))])
    
    # Initial parameters
    delta = 3 * sigma
    N = int(np.ceil(np.log(1 - p) / np.log(1 - (1 - e)**n)))
    M = (1-e) * n_total
    
    # Print the parameters:
    cprint(f"Params: p: {p}, n: {n}, e: {e}, N: {N}, delta: {delta}, n_total: {n_total}, M: {M}", "light_blue")
    best_inliers = []
    best_H = None
    num_inliers = 0
    cost_per_iteration = []  # Track the cost for each RANSAC iteration

    for _ in range(N):
        # Step 1: Randomly select 'n' correspondences
        valid = False
        while not valid:
            sample_idx = np.random.choice(len(img2_matches_hc), size=n, replace=False)
            hcpts1 = img1_matches_hc[sample_idx]
            hcpts2 = img2_matches_hc[sample_idx]
            
            # Step 2: Estimate homography using the sample points via Linear Least Squares
            H, valid = compute_homography(hcpts1, hcpts2) 
            if not valid:
                print("Retry!!!")
            
        # Step 3: Reproject img1 points to img2 using H
        reprojected_pts = (H @ img1_matches_hc.T).T  # Homogeneous coordinates
        reprojected_pts /= reprojected_pts[:, 2][:, np.newaxis]  # Normalize
        
        # Step 4: Calculate the error
        error = np.linalg.norm(img2_matches_hc[:, :2] - reprojected_pts[:, :2], axis=1)
            
        # Step 5: Identify inliers
        inliers_indices = np.where(error < delta)[0]
        
        # Step 6: Track the cost for this iteration
        iteration_cost = np.sum(error ** 2)
        cost_per_iteration.append(iteration_cost)  # Store the cost for this iteration
        
        # Step 7: Update best homography if this one has more inliers
        if len(inliers_indices) > num_inliers:
            num_inliers = len(inliers_indices)
            best_inliers = inliers_indices
            best_H = H
            cprint(f"The current number of inliers: {len(best_inliers)}", "white")
            
        if len(best_inliers) >= M:
            cprint(f"REACHED minimum number of inlier points: {M}", "light_magenta")
            break

    # Step 8: Recalculate the final cost using the best homography and inliers
    reprojected_pts_best = (best_H @ img1_matches_hc.T).T  # Homogeneous coordinates
    reprojected_pts_best /= reprojected_pts_best[:, 2][:, np.newaxis]  # Normalize
    final_error = np.linalg.norm(img2_matches_hc[:, :2] - reprojected_pts_best[:, :2], axis=1)
    final_cost = np.sum(final_error ** 2)
    cost_per_iteration.append(final_cost)  # Add the final cost for the best homography

    # Return the best homography, inliers, and the cost tracking list
    return best_H, best_inliers, cost_per_iteration, img1_matches_hc, img2_matches_hc

'''END OF THE RANSAC USING OPENCV:'''

''' PLOT THE INLIERS AND THE OUTLIERS:'''
# Visualize correspondences between images
def visualize_correspondences(image_pair, inliers_pts, correspondences, outliers_pts, save_image_path):    
    img1_color, img2_color = image_pair[0].copy(), image_pair[1].copy()
    # Resize images to have the same height before concatenation:
    height1, width1, _ = img1_color.shape
    height2, width2, _ = img2_color.shape
    if height1 != height2:
        # Resize img2 to match the height of img1 
        # (This is a little dumb but works, note to self, make this shrink the larger image to the smaller image size) 
        # Here i just assume the second image needs to be resized
        img2_color = cv2.resize(img2_color, (width2, height1))

    img_combined = np.hstack((img1_color, img2_color))
    num_out = 40
    for (pt1, pt2) in correspondences:
        if pt1 is not None and pt2 is not None:
            # Offset for the second image points
            pt2_offset = (int(pt2[0] + img1_color.shape[1]), int(pt2[1]))
            
            # Convert points to integers
            pt1_int = (int(pt1[0]), int(pt1[1]))
            
            circle_thickness = -2  # Filled circles
            line_thickness = 1
            if (pt1, pt2) in inliers_pts:
                # Draw red circles around points
                cv2.circle(img_combined, pt1_int, 4, (0, 255, 0), thickness=circle_thickness)  # For img1
                cv2.circle(img_combined, pt2_offset, 4, (0, 255, 0), thickness=circle_thickness)  # For img2
                cv2.line(img_combined, pt1_int, pt2_offset, (0, 255, 0), thickness=line_thickness)
            else:   
                if num_out > 0:
                    # Draw red circles around points
                    cv2.circle(img_combined, pt1_int, 4, (0, 0, 255), thickness=circle_thickness)  # For img1
                    cv2.circle(img_combined, pt2_offset, 4, (0, 0, 255), thickness=circle_thickness)  # For img2
                    cv2.line(img_combined, pt1_int, pt2_offset, (0, 0, 255), thickness=line_thickness)
                    num_out -= 1
            
    ensure_directory(f"MyResults/InOutLiers/")
    cv2.imwrite(f"{save_image_path}.jpg", img_combined)

''' ================================================= START OF PANORAMA: ================================================='''
'''
# Find the Homographies between (img1, img2), (img2, img3)...(img4, img5) i.e H12, H23...H45:
# Determine the middle image is 3:, Always use an odd pair of images!!! 
# So find the homgraphies H13, H35 using H12, H23 and H45 and H34
# Then x` = Hx -> x = (H^-1) x` => x is the inlier points? no its the corner points of the images??? use the corner points (a,b,c,d) -> (a`,b`,c`,d`)
# Find the min and max (x,y) pixel out of each image. This decides the image box for our resultant panorama image.
# Now that we know the box is bounded min_x, max_x,  min_y, max_y; we need to go pixel by pixel, 
# and checking if the (x,y) we're at is in any of the image by using x' = (H^-1)x then we add that pixel value to the resultant panorama, 
# however if that pixel is shared by other images, then we will take the median or average of those pixels and add that pixel value to the resultant panorama,
# And so on...
'''

''' START of Panorama Util Functions: '''
'''Compute the bounding box of the panorama using the corner points of all images.'''
def get_bounding_box(images, homographies):
    '''Map the corners of an image to the panorama frame using the homography H.'''
    def get_corners_of_panorama(image, H):
        '''
        Get the corner points in homogeneous coordinates
        Apply the homography to each corner
        Normalize (homogeneous to Cartesian)
        Return the (x, y) coordinates
        '''
        # Get image dimensions (height and width)
        h, w = image.shape[:2]  
        corners = np.array([[0, 0, 1],       # Top-left corner
                            [w, 0, 1],       # Top-right corner
                            [w, h, 1],       # Bottom-right corner
                            [0, h, 1]])      # Bottom-left corner
        mapped_corners = (H @ corners.T).T  
        mapped_corners /= mapped_corners[:, 2][:, np.newaxis] 
         
        return mapped_corners[:, :2]  

    all_corners = []
    # Map the corners of each image to the panorama frame
    for i, img in enumerate(images):
        mapped_corners = get_corners_of_panorama(img, homographies[i])
        all_corners.append(mapped_corners)
        
    # Stack all the corner points from all images
    all_corners = np.vstack(all_corners)
    # Find the minimum and maximum x, y coordinates
    min_x, min_y = np.min(all_corners, axis=0)
    max_x, max_y = np.max(all_corners, axis=0)
    
    return min_x, max_x, min_y, max_y

'''Warp an image onto the panorama canvas using the homography.'''
def map_image_to_panorama(image, H, panorama, min_x, min_y):
    # Get inverse homography to map the panorama points back to the image
    H_inv = np.linalg.inv(H)
    # Dimensions of blank canvas of panorama image to map to:
    h_pano, w_pano = panorama.shape[:2]
    for y_pano in tqdm(range(h_pano)):
        for x_pano in range(w_pano):
            # Map (x_pano, y_pano) in panorama to the original image
            pano_pt = np.array([x_pano + min_x, y_pano + min_y, 1])
            img_pt = H_inv @ pano_pt.T
            img_pt /= img_pt[2]  # Normalize homogeneous coordinates
            img_x, img_y = int(img_pt[0]), int(img_pt[1])
            # Check if the point is within the image bounds
            if 0 <= img_x < image.shape[1] and 0 <= img_y < image.shape[0]:
                panorama[y_pano, x_pano] = image[img_y, img_x]
    return panorama

def efficient_map_image_to_panorama(image, H, panorama, min_x, min_y):
    # Get inverse homography to map the panorama points back to the image
    H_inv = np.linalg.inv(H)
    
    # Dimensions of blank canvas of panorama image to map to:
    h_pano, w_pano = panorama.shape[:2]

    # Generate all pixel coordinates in the panorama in a single step
    x_pano, y_pano = np.meshgrid(np.arange(w_pano), np.arange(h_pano))

    # Flatten the grid so we can apply transformations to all points at once
    pano_pts = np.stack([x_pano.ravel() + min_x, y_pano.ravel() + min_y, np.ones_like(x_pano).ravel()], axis=0)

    # Apply the inverse homography to all points at once
    img_pts = H_inv @ pano_pts
    img_pts /= img_pts[2, :]  # Normalize homogeneous coordinates

    # Extract integer x and y coordinates for the image
    img_x = img_pts[0, :].astype(int)
    img_y = img_pts[1, :].astype(int)

    # Filter valid points (points that fall within the image bounds)
    valid_mask = (0 <= img_x) & (img_x < image.shape[1]) & (0 <= img_y) & (img_y < image.shape[0])

    # Only update the valid pixels in the panorama
    panorama[y_pano.ravel()[valid_mask], x_pano.ravel()[valid_mask]] = image[img_y[valid_mask], img_x[valid_mask]]

    return panorama

''' END of Panorama Util Functions: '''

def create_panorama(images, pair_homographies, test = 0):
    # Homographies between image pairs
    H12 = pair_homographies[0]
    H23 = pair_homographies[1]
    H34 = pair_homographies[2]
    H45 = pair_homographies[3]
    # Compute H13 and H35
    H13 = H23 @ H12
    H35 = H45 @ H34
        
    # Final homographies relative to img3
    final_homographies = [H13, H23, np.identity(3), np.linalg.inv(H34), np.linalg.inv(H35)]
    # Compute the bounding box of the panorama
    min_x, max_x, min_y, max_y = get_bounding_box(given_images, final_homographies)
    
    panorama_height = int(np.ceil(max_y - min_y))
    panorama_width  = int(np.ceil(max_x - min_x))
    # cprint(f"MIN X: {min_x}", "cyan")
    # cprint(f"MAX X: {max_x}", "cyan")
    # cprint(f"MIN Y: {min_y}", "cyan") 
    # cprint(f"MAX Y: {max_y}", "cyan")  
    # Create the panorama canvas
    width, height = 0, 0
    for i, img, in enumerate(images):
        width += img.shape[1]
        height += img.shape[0]
    
    # panorama_width, panorama_height = width, height
    panorama = np.zeros((panorama_height, panorama_width, 3), dtype=np.uint8)

    # for i in range(2):
    #     panorama = efficient_map_image_to_panorama(given_images[i], final_homographies[i], panorama, min_x, min_y)
    #     cv2.imwrite('MyResults/Panorama/final_panorama.jpg', panorama)

    # for i in range(3,5):
    #     panorama = efficient_map_image_to_panorama(given_images[i], final_homographies[i], panorama, min_x, min_y)
    #     cv2.imwrite('MyResults/Panorama/final_panorama.jpg', panorama)
    
    # panorama = efficient_map_image_to_panorama(given_images[i], final_homographies[i], panorama, min_x, min_y)
    # cv2.imwrite('MyResults/Panorama/final_panorama.jpg', panorama)
    
    # Warp each image onto the panorama canvas
    for i in tqdm(range(len(final_homographies))):  
        panorama = efficient_map_image_to_panorama(given_images[i], final_homographies[i], panorama, min_x, min_y)
        cv2.imwrite('MyResults/Panorama/final_panorama.jpg', panorama)

    return panorama

'''--------------------------------------------------------------------------'''


''' ================================================= END OF PANORAMA: ================================================='''
''' ================================================= START OF LM : ================================================='''
def Jacobian(pts, H):
    J = np.zeros((len(pts) * 2, 9))  # Initialize the Jacobian matrix
    for i, pt in enumerate(pts):
        x, y = pt[0], pt[1]
        f = H @ np.array([x, y, 1])
        J[2 * i] = [x / f[2], y / f[2], 1 / f[2], 0, 0, 0,  -x * f[0] / f[2]**2, -y * f[0] / f[2]**2, -f[0] / f[2]**2]
        J[2 * i + 1] = [0, 0, 0, x / f[2], y / f[2], 1 / f[2],  -x * f[1] / f[2]**2, -y * f[1] / f[2]**2, -f[1] / f[2]**2]
    return J

# Levenberg-Marquardt algorithm implementation
def levenberg_marquardt(H, pts1, pts2):
    I = np.identity(9)
    J = Jacobian(pts1, H)
    A = J.T @ J
    tau = 0.01
    mu = tau * np.max(np.diag(A))
    # List to store the cost at each iteration
    cost_per_iteration = []  
    H = H.flatten()  
    curr_iteration_num = 0
    max_iterations = 100
    while curr_iteration_num < max_iterations:
        error = cost_function_for_LM(H, pts1, pts2)
        start_cost = np.linalg.norm(error) ** 2
        refined_J = Jacobian(pts1, H.reshape(3, 3)) 
        delta_p = np.matmul((np.linalg.inv(np.matmul(refined_J.T,  refined_J) + mu * I) @ refined_J.T), error)
        delta_H = H + delta_p  
        cek = cost_function_for_LM(delta_H.reshape(3, 3), pts1, pts2)
        
        refined_cost = np.linalg.norm(cek) ** 2
        cost_per_iteration.append(refined_cost)  

        # cprint(f"Iteration {curr_iteration_num+1}: Cost = {refined_cost}","cyan")
        rho_num = start_cost - refined_cost
        rho_den = delta_p.T @ ((mu * I) @ delta_p + refined_J.T @ error)
        rho = rho_num / rho_den
        if rho > 0:
            H = delta_H  
            mu = mu * max(1 / 3, 1 - (2 * rho - 1) ** 3)
        else:
            mu *= 2

        # Convergence threshold
        if np.linalg.norm(delta_p) < 1e-6:  
            break

        curr_iteration_num += 1
    return H.reshape(3, 3), cost_per_iteration  

# Cost function used for LM optimization
def cost_function_for_LM(h, pts1, pts2):
    h = h.reshape(3, 3)
    n_total = len(pts1)
    pts1_hc = np.ones((n_total, 3))
    pts1_hc[:, :-1] = pts1
    pts2_hc = np.ones((n_total, 3))
    pts2_hc[:, :-1] = pts2
    est_cpts2 = h @ pts1_hc.T
    est_cpts2 = est_cpts2 / est_cpts2[2]  
    est_cpts2 = est_cpts2.T[:, :2]
    error = pts2 - est_cpts2
    return error.flatten()  

def plot_and_print_costs(costs_without_LM, costs_with_LM, iteration_costs):
    for i, (cost_before, cost_after) in enumerate(zip(costs_without_LM, costs_with_LM), 1):
        # reduction = 100 * (cost_before - cost_after) / cost_before
        # print(f"Pair {i} & {i+1:<10}: {cost_before:<20.4f} {cost_after:<20.4f} {reduction:<20.2f}")

        # Plotting the costs for the current image pair in log scale
        plt.figure(figsize=(8, 6))
        plt.plot([1, 2], [cost_before, cost_after], 'o-', color="orange")
        plt.yscale('log')  
        plt.xticks([1, 2], ['Without LM', 'With LM'])
        plt.xlabel("Optimization Method")
        plt.ylabel("Cost")
        plt.title(f"Cost Comparison for Pair {i} & {i+1}")
        plt.grid(True)

        # Save the plot with the desired filename format
        filename = f"MyResults/Costs/Task1_Cost_{i}_{i+1}.jpg"
        ensure_directory('MyResults/Costs')  
        plt.savefig(filename)
        plt.close()

        # Plot the cost per iteration for LM optimization
        plt.figure(figsize=(8, 6))
        plt.plot(iteration_costs[i-1], label=f'Pair {i} & {i+1}', color='blue', marker='o')
        plt.yscale('log')
        plt.xlabel("Iterations")
        plt.ylabel("Cost")
        plt.title(f"Cost per Iteration for Pair {i} & {i+1}")
        plt.grid(True)
        plt.legend()

        # Save the iteration cost plot
        iter_filename = f"MyResults/Costs/Task1_Cost_{i}_{i+1}.jpg"
        plt.savefig(iter_filename)
        plt.close()

        cprint(f"Iteration cost plot saved as {iter_filename}","green")

    for i, (cost_before, cost_after) in enumerate(zip(costs_without_LM, costs_with_LM), 1):
        reduction = 100 * (cost_before - cost_after) / cost_before
        print(f"Pair {i} & {i+1:<10}: {cost_before:<20.4f} {cost_after:<20.4f} {reduction:<20.2f}")


''' ================================================= END OF LM: ================================================='''

if __name__ == "__main__":
    cprint("Make sure you are running from HW5 directory!!!\n", "red")
    
    # Load in the images:
    given_images = list()
    for i in range(1, 6):
        given_images.append(cv2.imread(f"HW5_images/pics/{i}.jpg"))
    
    # Create a list of every homography pairs (1, 2), (2, 3)...(4, 5):
    pair_homographies = list()
    refined_H = list()

    costs_with_LM = []  # Store costs with LM
    costs_without_LM = []  # Store costs without LM
    avg_inliers = 0
    iteration_costs = list()
    for n in range(1, 5):
        # Get Correspondences using SIFT for consecutive images, n-1 and n-2:
        img1_sift_points, img2_sift_points = OpenCV_SIFT_SURF((given_images[n-1], given_images[n]), f"Task1_SIFT_pairs_{n}_{n+1}")

        # Implement the RANSAC Algo using the SIFT Correspondences:
        best_H, best_inliers, cost, img1_matches_hc, img2_matches_hc = RANSAC(img1_sift_points, img2_sift_points, len(img1_sift_points), img_num=n)
        
        # Calculate the geometric error before LM refinement and store it
        cost_before_LM = np.sum(cost_function(best_H, img1_matches_hc[best_inliers][:, :2], img2_matches_hc[best_inliers][:, :2]))
        costs_without_LM.append(cost_before_LM)  # Collect total cost geometric error for RANSAC 
        
        # Step 8: Refine homography using LM optimization
        temp = best_H.copy()
        
        
        # Levenberg-Marquardt implementation called here:
        temp_H, cost = levenberg_marquardt(best_H,img1_matches_hc[best_inliers][:, :2], img2_matches_hc[best_inliers][:, :2] )
        refined_H.append(temp_H)  # Simply append the refined homography matrix
        iteration_costs.append(cost)
        # Calculate the geometric error after LM refinement and store it
        cost_with_LM = np.sum(cost_function(temp_H, img1_matches_hc[best_inliers][:, :2], img2_matches_hc[best_inliers][:, :2]))
        costs_with_LM.append(cost_with_LM)  # Track costs with LM
        
        cprint(f"Number of best inliers: {len(best_inliers)}", "light_magenta")
        
        #============================================= Inlier and Outlier Points: =============================================
        # Visualize the inlier points:
        visualize_correspondences( (given_images[n-1], given_images[n]),  [(img1_sift_points[i], img2_sift_points[i]) for i in best_inliers], [(img1_sift_points[i], img2_sift_points[i]) for i in range(len(img1_sift_points))],  [], f"MyResults/InOutLiers/Task1_InOutLiers_{n}_{n+1}")

        # Save the pairwise homographies: 
        pair_homographies.append(best_H)
        avg_inliers += len(best_inliers)
        
        cprint(f"For image {n}.jpg and {n+1}.jpg ^^^^") # Formattting purposes for the prints
        cprint("===================================================================================================") # Formattting purposes for the prints
        print() # Formattting purposes for the prints 

        
    cprint(f"Completed RANSAC Algo", "green")
    cprint(f"Average number of inliers: {avg_inliers/4}", "light_magenta")
    
    #============================================= Plotting the Costs with and without LM: =============================================
    # Plot and print the costs comparison before and after LM refinement
    plot_and_print_costs(costs_without_LM, costs_with_LM, iteration_costs)
    
    #============================================= Panorama: =============================================
    # Read from homography files(best found homographies saved here), for DEBUGING PURPOSES !!!
    # store = pair_homographies[-1]
    # pair_homographies = list()
    # for i in range(1,5):
    #     pair_homographies.append(read_homographies(f"MyResults/best_homography_pairs_{i}_{i+1}.txt"))
    # pair_homographies.append(store)
    
    cprint(f"NORMAL HOMOGRAPHIES: \n{pair_homographies}", "cyan")
    cprint(f"REFINED HOMOGRAPHIES:\n {refined_H}", "cyan")
    
    
    # WITHOUT LM REFINEMENT:
    cprint("Starting Panorama: WITHOUT LM REFINEMENT: ")
    # Create the panorama
    panorama = create_panorama(given_images, pair_homographies)
    ensure_directory('MyResults/Panorama')
    cv2.imwrite('MyResults/Panorama/Task1_without_panorama.jpg', panorama)
    cprint("Panorama created and saved as without_panorama.jpg", "green")

    cprint("Starting Panorama: WITH LM REFINEMENT: ")
    refined_panorama = create_panorama(given_images, refined_H)    
    ensure_directory('MyResults/Panorama')
    cv2.imwrite('MyResults/Panorama/Task1_with_panorama.jpg', refined_panorama)
    cprint("Panorama created and saved as with_panorama.jpg", "green")
    
    
    ################################################################################################################################################
    # TASK 2:
    ################################################################################################################################################
    print()
    print()
    print()
    cprint(f"------------------------STARTING TASK 2:------------------------", "red")
    cprint("Make sure you are running from HW5 directory!!!\n", "red")
    given_images = list()
    for i in range(1,6):
        given_images.append(cv2.imread(f"MyImages/{i}.jpg"))
    pair_homographies = list()
    refined_H = list()
    costs_with_LM = []  # Store costs with LM
    costs_without_LM = []  # Store costs without LM
    avg_inliers = 0
    for n in range(1,5):
        # Get Correspondences using SIFT for consecutive images, n-1 and n-2:
        img1_sift_points, img2_sift_points =  OpenCV_SIFT_SURF((given_images[n-1], given_images[n]), f"Task2_SIFT_pairs_{n}_{n+1}")

        # Implement the RANSAC Algo using the SIFT Correspondences:
        best_H, best_inliers, cost, img1_matches_hc, img2_matches_hc =  RANSAC(img1_sift_points, img2_sift_points,len(img1_sift_points), img_num=n)
        costs_without_LM.extend(cost)  # Collect all costs from RANSAC
        
        # Step 8: Refine homography using LM optimization
        temp = best_H.copy()
        temp_H = least_squares(cost_function, temp.flatten(), method='lm', args=(img1_matches_hc[best_inliers][:, :2], img2_matches_hc[best_inliers][:, :2]), verbose=2)
        refined_H.append(temp_H.x.reshape(3, 3))  # Reshape to 3x3 matrix
        
        cost_with_LM = np.sum(cost_function(temp_H.x, img1_matches_hc[best_inliers][:, :2], img2_matches_hc[best_inliers][:, :2])**2)
        costs_with_LM.append(cost_with_LM)  # Track costs with LM
        # if n == 4:  np.savetxt(f'MyResults/best_homography_pairs_{n}_{n+1}.txt', best_H)
        # np.savetxt(f'MyResults/inliers_pairs_{n}_{n+1}.txt', best_inliers)
        cprint(f"Number of best inliers: {len(best_inliers)}", "light_magenta")
        
        #============================================= Inlier and Outlier Points: =============================================
        # Visualize the inlier points:
        visualize_correspondences((given_images[n-1],given_images[n]), [(img1_sift_points[i], img2_sift_points[i]) for i in best_inliers],[(img1_sift_points[i], img2_sift_points[i]) for i in range(len(img1_sift_points))], [] , f"MyResults/InOutLiers/Task2_InOutLiers_{n}_{n+1}")

        # Save the pair wise homographies: 
        pair_homographies.append(best_H)  

        avg_inliers += len(best_inliers)
        
        cprint(f"For image {n}.jpg and {n+1}.jpg ^^^^") # Formattting purposes for the prints
        cprint("===================================================================================================") # Formattting purposes for the prints
        print() # Formattting purposes for the prints 
    cprint(f"Completed RANSAC Algo", "green")
    cprint(f"Average number of inliers: {avg_inliers/4}", "light_magenta")
    
    
    #============================================= Panorama: =============================================
    # WITHOUT LM REFINEMENT:
    cprint("Starting Panorama: WITHOUT LM REFINEMENT: ")
    # Create the panorama
    panorama = create_panorama(given_images, pair_homographies)
    ensure_directory('MyResults/Panorama')
    cv2.imwrite('MyResults/Panorama/Task2_without_panorama.jpg', panorama)
    cprint("Panorama created and saved as without_panorama.jpg", "green")

    panorama = create_panorama(given_images, refined_H)    
    ensure_directory('MyResults/Panorama')
    cv2.imwrite('MyResults/Panorama/Task2_with_panorama.jpg', panorama)
    cprint("Panorama created and saved as with_panorama.jpg", "green")