import numpy as np
import cv2
from termcolor import cprint
from skimage import io
import matplotlib.pyplot as plt 
from typing import List, Tuple
from scipy.ndimage import convolve
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
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptor_img1, descriptor_img2)

    # Sort matches by distance (best matches first)
    matches = sorted(matches, key=lambda x: x.distance)

    # Draw the first 100 matches (or fewer if there are not that many matches)
    combined_image = cv2.drawMatches(img1, keypoints_img1, img2, keypoints_img2, matches[:100], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

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
    cprint(f"Saved SIFT Correspondences for pair {pair_name}\n", "green")
    
    # Return the corresponding points (keypoints)
    return sorted(points_img1), sorted(points_img2)
'''END OF THE SIFT SURF USING OPENCV (TAKEN FROM HW4):'''

'''START OF THE RANSAC USING OPENCV:'''
def RANSAC(img1_matches, img2_matches, p: int = 0.99, n: int = 4, sigma: int = 4, e: int = 0.1):
    if img1_matches is None or img2_matches is None:
        cprint(f"Matching correspondences using SIFT are missing!!!", "red")
        exit(0)
    # First Convert all the points to HC representation:
    img1_matches_hc = np.hstack([np.array(img1_matches), np.ones((len(img1_matches), 1))])
    img2_matches_hc = np.hstack([np.array(img2_matches), np.ones((len(img2_matches), 1))])
    # print(img1_matches_hc) # DEBUG STATMENT!!!
    
    # Initial Parameters
    delta = 3 * sigma
    N = int(np.ceil(np.log(1 - p) / np.log(1 - (1 - e)**n)))
    best_inliers = []
    best_H = None
    num_inliers = 0
    
    
    for _ in range(N):
        # Step 1: Randomly select 'n' correspondences:
        sample_idx = np.random.choice(len(img1_matches_hc), size=n, replace=False)
        cpts1 = img1_matches_hc[sample_idx]
        cpts2 = img2_matches_hc[sample_idx]
        
        # Step 2: Estimate Homography using the sample points
        
        # Step 3: Reproject img1 points to img2 using H and calculate error
        # Step 4: Identify inliers
        continue
    
    pass    
    
    
    
    
    
    
    
    
     
'''END OF THE RANSAC USING OPENCV:'''

if __name__ == "__main__":
    cprint("Make sure you are running from HW5 directory!!!", "white")
    # Load in the images:
    given_images = list()
    for i in range(1,6):
        given_images.append(cv2.imread(f"HW5_images/pics/{i}.jpg"))

    for n in range(1,5):
        # Get Correspondences using SIFT for consecutive images, n-1 and n-2:
        img1_sift_points, img2_sift_points =  OpenCV_SIFT_SURF((given_images[n-1], given_images[n]), f"Task1_SIFT_pairs_{n}_{n+1}")
        # [print(f"The Set is Equal") if (img1_sift_points) == (img2_sift_points) else print(f"The set is not equal!!!")]
        
        # Implement the RANSAC Algo using the SIFT Correspondences:
        RANSAC(img1_sift_points, img2_sift_points)
        cprint(f"Completed RANSAC Algo", "green")
        
        
        
        
        
        