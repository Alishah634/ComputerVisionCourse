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
    keypoints_img1, descriptor_img1 = sift.detectAndCompute(img1_gray, None) # Can ask for num points here!!! 
    keypoints_img2, descriptor_img2 = sift.detectAndCompute(img2_gray, None) 
    
    cprint(f"The number of keypoints before BF: {len(keypoints_img1)}", "cyan")
    cprint(f"The number of keypoints before BF: {len(keypoints_img1)}", "cyan")
    
    # Use BFMatcher to find the best matches between the descriptors
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True) # USE A RATIO TEST !!! CHECK TUTORIAL FOR BF matches!!!
    matches = bf.match(descriptor_img1, descriptor_img2)

    # Sort matches by distance (best matches first)
    matches = sorted(matches, key=lambda x: x.distance)

    # Draw the matches (or fewer if there are not that many matches)
    combined_image = cv2.drawMatches(img1, keypoints_img1, img2, keypoints_img2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Store the points for later use:
    # Extract the corresponding points (keypoints) for both images
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
    
    # Create a results folder if it doesn't exist
    folder_path = f"MyResults/SIFT/"
    ensure_directory(folder_path)
    # Save the resulting image with correspondences
    cv2.imwrite(f"{folder_path}{pair_name}.jpg", combined_image)
    cprint(f"Saved SIFT Correspondences for pair {pair_name}", "green")
    
    # Return the corresponding points (keypoints)
    return sorted(points_img1), sorted(points_img2)
'''END OF THE SIFT SURF USING OPENCV (TAKEN FROM HW4):'''

'''START OF THE RANSAC USING OPENCV:'''
'''Compute the Homography H using the concept Ax = b -> x = A^-1*b :'''    
def compute_homography(src_pts: np.ndarray, dest_pts: np.ndarray) -> np.ndarray:
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
        H_temp = np.dot(np.linalg.inv(np.dot(A.T, A)), np.dot(A.T, b)) #np.dot(np.linalg.pinv(A), b)
        H = np.append(H_temp, 1).reshape((3, 3))
    except np.linalg.LinAlgError:
        cprint("Error: Matrix inversion failed. The matrix A might be singular.", "red")
        H = np.eye(3)  # Return identity matrix as fallback
    except Exception as e:
        cprint("Error: {e}", "red")
        H = np.eye(3)  # Return identity matrix as fallback
    # cprint(f"Matrix A:{A}", "cyan") 
    # cprint(f"Matrix b:{b}", "yellow") 
    # cprint(f"Matrix H:{H}", "yellow") 
    return H


def residuals(H, pts1, pts2):
    """Calculate residuals for optimization."""
    # Reshape H to (3, 3)
    H = H.reshape(3, 3)
    # Project points using H
    pts1_hc = np.hstack([pts1, np.ones((pts1.shape[0], 1))])  # Homogeneous coordinates
    projected_pts = (H @ pts1_hc.T).T  # Project points
    projected_pts /= projected_pts[:, 2][:, np.newaxis]  # Normalize
    # Compute residuals (error)
    res = (projected_pts[:, :2] - pts2)**2  # Squared error
    return res.flatten()  # Flatten for least_squares

def RANSAC(img1_matches, img2_matches, p: int = 0.99, n: int = 4, sigma: int = 4, e: int = 0.6):
    if img1_matches is None or img2_matches is None:
        cprint(f"Matching correspondences using SIFT are missing!!!", "red")
        exit(0)

    # Convert points to homogeneous coordinates
    img1_matches_hc = np.hstack([np.array(img1_matches), np.ones((len(img1_matches), 1))])
    img2_matches_hc = np.hstack([np.array(img2_matches), np.ones((len(img2_matches), 1))])
    
    # Initial parameters
    delta = 3 * sigma
    N = int(np.ceil(np.log(1 - p) / np.log(1 - (1 - e)**n)))
    best_inliers = list()
    best_H = None
    num_inliers = 0
    cost = list()
    
    cprint(f"Number of Trials: {N}","white")
    for _ in range(N):
        # Step 1: Randomly select 'n' correspondences
        sample_idx = np.random.choice(len(img1_matches_hc), size=n, replace=False)
        hcpts1 = img1_matches_hc[sample_idx]
        hcpts2 = img2_matches_hc[sample_idx]
        
        # Step 2: Estimate homography using the sample points via Linear Least Squares
        H = compute_homography(hcpts1, hcpts2) 
        
        # Step 3: Reproject img1 points to img2 using H
        reprojected_pts = (H @ img1_matches_hc.T).T  # Homogeneous coordinates
        reprojected_pts /= reprojected_pts[:, 2][:, np.newaxis]  # Normalize
        
        # Step 4: Calculate the error
        error = np.linalg.norm(img2_matches_hc[:, :2] - reprojected_pts[:, :2], axis=1)
            
        # Step 5: Identify inliers
        inliers_indices = np.where(error <= delta)[0]
        
        # Step 6: Update best homography if this one has more inliers
        if len(inliers_indices) > num_inliers:
            cprint(f"The current number of inliers: {len(best_inliers)}","white")
            cost.append(error)
            num_inliers = len(inliers_indices)
            best_inliers = inliers_indices
            best_H = H
        
    # Step 7: Find homography using Linear Least Squares, using set of all inliers:
    best_H = compute_homography(img1_matches_hc[inliers_indices][:,:2], img2_matches_hc[inliers_indices][:,:2])

    # Step 8: Refine homography using LM optimization
    # NEED TO USE SCIPY LM VALUE!!!
    refined_H = least_squares(residuals, best_H.flatten(), method='lm', args=(img1_matches_hc[best_inliers][:, :2], img2_matches_hc[best_inliers][:, :2]))
    best_H = refined_H.x.reshape(3, 3)  # Reshape to 3x3 matrix
    
    # Return the best homography and inliers found
    return best_H, best_inliers, cost
 
'''END OF THE RANSAC USING OPENCV:'''

if __name__ == "__main__":
    cprint("Make sure you are running from HW5 directory!!!", "white")
    # Load in the images:
    given_images = list()
    for i in range(1,6):
        given_images.append(cv2.imread(f"HW5_images/pics/{i}.jpg"))
    
    avg_inliers = 0
    for n in range(1,5):
        # Get Correspondences using SIFT for consecutive images, n-1 and n-2:
        img1_sift_points, img2_sift_points =  OpenCV_SIFT_SURF((given_images[n-1], given_images[n]), f"Task1_SIFT_pairs_{n}_{n+1}")
        # [print(f"The Set is Equal") if (img1_sift_points) == (img2_sift_points) else print(f"The set is not equal!!!")]
        cprint(f"Number of SIFT points for the input image {n}.jpg and {n+1}.jpg: {len(img1_sift_points)}", "cyan")
       
        # Implement the RANSAC Algo using the SIFT Correspondences:
        best_H, best_inliers, cost =  RANSAC(img1_sift_points, img2_sift_points)
        np.savetxt(f'MyResults/best_homography_pairs_{n}_{n+1}.txt', best_H)
        np.savetxt(f'MyResults/inliers_pairs_{n}_{n+1}.txt', best_inliers)
        cprint(f"Number of best inliers: {len(best_inliers)}", "light_magenta")
        # cprint(f"Best homography: {best_H}","white")
        # print()
        # print()
        # cprint(f"Best Inliers: {best_inliers}","white")
        # print()
        # print()
        # cprint(f"Cost: {cost}","white")
        
        avg_inliers += len(best_inliers)
        print()
        print()
        
    cprint(f"Completed RANSAC Algo", "green")
    cprint(f"Average number of inliers: {avg_inliers/4}", "light_magenta")
        
        
        
        
        
        
