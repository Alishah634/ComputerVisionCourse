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
    combined_image = cv2.drawMatches(img1, keypoints_img1, img2, keypoints_img2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

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
        H_temp =  np.dot(np.linalg.inv(np.dot(A.T, A)), np.dot(A.T, b)) # np.dot(np.linalg.pinv(A), b)  #
        H = np.append(H_temp, 1).reshape((3, 3))
    except np.linalg.LinAlgError:
        cprint("Error: Matrix inversion failed. The matrix A might be singular.", "red")
        valid = False
        H = np.eye(3)  # Return identity matrix as fallback
    except Exception as e:
        cprint("Error: {e}", "red")
        valid = False
        H = np.eye(3)  # Return identity matrix as fallback
    # cprint(f"Matrix A:{A}", "cyan") 
    # cprint(f"Matrix b:{b}", "yellow") 
    # cprint(f"Matrix H:{H}", "yellow") 
    return H, valid


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

def RANSAC(img1_matches, img2_matches, p: int = 0.99, n: int = 4, sigma: int = 0.5, e: int = 0.1, img_num: int = 0):
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
    # for _ in range(N):
    for _ in range(200):
        # Step 1: Randomly select 'n' correspondences
        valid = False
        
        # while not valid:
        sample_idx = np.random.choice(len(img1_matches_hc), size=n, replace=False)
        # with open(f"MyResults/samples_chosen{img_num}_{img_num+1}.txt", 'r') as file:
        #     temp = list()
        #     data = file.read()
        #     temp.append(np.fromstring(data, sep=' '))
        #     sample_idx = np.array(sample_idx)
            
        if img_num != 0:
            np.savetxt(f'MyResults/samples_chosen{img_num}_{img_num+1}.txt', sample_idx)
        # print(sample_idx)
        hcpts1 = img1_matches_hc[sample_idx]
        hcpts2 = img2_matches_hc[sample_idx]
        
        # Step 2: Estimate homography using the sample points via Linear Least Squares
        H, valid = compute_homography(hcpts1, hcpts2) 
        if not valid:
            print("retry`")
            
        # Step 3: Reproject img1 points to img2 using H
        reprojected_pts = (H @ img1_matches_hc.T).T  # Homogeneous coordinates
        reprojected_pts /= reprojected_pts[:, 2][:, np.newaxis]  # Normalize
        
        # Step 4: Calculate the error
        error = np.linalg.norm(img2_matches_hc[:, :2] - reprojected_pts[:, :2], axis=1)
            
        # Step 5: Identify inliers
        inliers_indices = np.where(error < delta)[0]
        
        # Step 6: Update best homography if this one has more inliers
        if len(inliers_indices) > num_inliers:
            cost.append(error)
            num_inliers = len(inliers_indices)
            best_inliers = inliers_indices
            cprint(f"The current number of inliers: {len(best_inliers)}","white")
            best_H = H
    
    
    
    # Step 7: Find homography using Linear Least Squares, using set of all inliers:
    best_H, _ = compute_homography(img1_matches_hc[inliers_indices][:,:2], img2_matches_hc[inliers_indices][:,:2])

    # Step 8: Refine homography using LM optimization
    # NEED TO USE SCIPY LM VALUE!!!
    refined_H = least_squares(residuals, best_H.flatten(), method='lm', args=(img1_matches_hc[best_inliers][:, :2], img2_matches_hc[best_inliers][:, :2]))
    best_H = refined_H.x.reshape(3, 3)  # Reshape to 3x3 matrix
    
    # Return the best homography and inliers found
    return best_H, best_inliers, cost

'''END OF THE RANSAC USING OPENCV:'''

''' PLOT THE INLIERS AND THE OUTLIERS:'''
# Visualize correspondences between images
def visualize_correspondences(image_pair, correspondences, save_image_path):
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

    for (pt1, pt2) in correspondences:
        if pt1 is not None and pt2 is not None:
            # Offset for the second image points
            pt2_offset = (int(pt2[0] + img1_color.shape[1]), int(pt2[1]))
            
            # Convert points to integers
            pt1_int = (int(pt1[0]), int(pt1[1]))
            
            # Draw red circles around points
            circle_thickness = -2  # Filled circles
            cv2.circle(img_combined, pt1_int, 4, (0, 0, 255), thickness=circle_thickness)  # For img1
            cv2.circle(img_combined, pt2_offset, 4, (0, 0, 255), thickness=circle_thickness)  # For img2
            
            # Draw random color lines between each of the image points
            line_thickness = 1
            random_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            cv2.line(img_combined, pt1_int, pt2_offset, random_color, thickness=line_thickness)
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
# This is a pixel mapping function that applies a homography to an image and maps its pixels onto the panorama.
def pixel_mapping_forward(panorama, new_img, H):
    h_new, w_new, _ = new_img.shape
    for x_new in range(w_new):
        for y_new in range(h_new):
            point = np.array([x_new, y_new, 1])
            new_point = np.dot(H, point)
            new_point /= new_point[2]  # Normalize to homogeneous coordinates
            x_pano, y_pano = int(new_point[0]), int(new_point[1])
            # Ensure the mapped point is within the panorama bounds
            if 0 <= x_pano < panorama.shape[1] and 0 <= y_pano < panorama.shape[0]:
                panorama[y_pano, x_pano] = new_img[y_new, x_new]

    return panorama

def compute_panorama_size(H_list, images):
    min_x, min_y = 0, 0
    max_x, max_y = 0, 0

    for H, img in zip(H_list, images):
        h, w = img.shape[:2]
        corners = np.array([[0, 0, w, w],
                            [0, h, 0, h],
                            [1, 1, 1, 1]])

        warped_corners = H @ corners
        warped_corners /= warped_corners[2, :]

        min_x = min(min_x, np.min(warped_corners[0, :]))
        max_x = max(max_x, np.max(warped_corners[0, :]))
        min_y = min(min_y, np.min(warped_corners[1, :]))
        max_y = max(max_y, np.max(warped_corners[1, :]))

    return int(np.ceil(max_x - min_x)), int(np.ceil(max_y - min_y)), int(min_x), int(min_y)


def create_panorama(H_list, images, img_dir='HW5_images/pics/', output_file='MyResults/Panorama/panorama.jpg'):
    # Homographies between image pairs
    H12 = H_list[0]
    H23 = H_list[1]
    H34 = H_list[2]
    H45 = H_list[3]
    # Compute H13 and H35
    # H13 = np.matmul(np.linalg.inv(H23), H12)
    # H35 = np.matmul(np.linalg.inv(H34), H45)
    H13 = np.matmul(H23, H12)
    H35 = np.matmul(H45, H34)
    
    # Final homographies relative to img3
    final_homographies = [H13, H23, np.identity(3), np.linalg.inv(H34), np.linalg.inv(H35)]

    mid_img = len(H_list) // 2

    H_list = final_homographies
    # Initialize translation transformation
    tx = 0
    for i in range(mid_img):
        img = images[i]
        tx += img.shape[1]
    H_t = np.array([[1, 0, tx], [0, 1, 0], [0, 0, 1]], dtype=float)
    # H_t = np.eye(3)
    # Calculate height and width for the panorama
    height = 0
    width = 0
    for i in range(len(H_list)):
        img = images[i]
        height = max(height, img.shape[0])
        width += img.shape[1]

    # Create a blank canvas to hold the panorama
    panorama = np.zeros((height, width, 3), dtype=np.uint8)

    # Map each image to the panorama canvas using the forward pixel mapping
    for i in range(5):#len(H_list)):
        img = images[i]
        H = np.matmul(H_t, H_list[i])
        panorama = pixel_mapping_forward(panorama, img, H)  # Use forward mapping here
        cv2.imwrite('MyResults/Panorama/final_panorama.jpg', panorama)
    # Save the final panorama image
    panorama = cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB)
    return panorama

''' END of Panorama Util Functions: '''
''' ================================================= END OF PANORAMA: ================================================='''




if __name__ == "__main__":
    cprint("Make sure you are running from HW5 directory!!!", "white")
    # Load in the images:
    given_images = list()
    for i in range(1,6):
        given_images.append(cv2.imread(f"HW5_images/pics/{i}.jpg"))
    
    # Create a list of every homography pairs (1,2),(2,3)...(4,5):
    pair_homographies = list()
       
    avg_inliers = 0
    for n in range(1,5):
        # Get Correspondences using SIFT for consecutive images, n-1 and n-2:
        img1_sift_points, img2_sift_points =  OpenCV_SIFT_SURF((given_images[n-1], given_images[n]), f"Task1_SIFT_pairs_{n}_{n+1}")
        cprint(f"Number of SIFT points for the input image {n}.jpg and {n+1}.jpg: {len(img1_sift_points)}", "cyan")
       
        # Implement the RANSAC Algo using the SIFT Correspondences:
        best_H, best_inliers, cost =  RANSAC(img1_sift_points, img2_sift_points, img_num=n)
        # np.savetxt(f'MyResults/best_homography_pairs_{n}_{n+1}.txt', best_H)
        np.savetxt(f'MyResults/inliers_pairs_{n}_{n+1}.txt', best_inliers)
        cprint(f"Number of best inliers: {len(best_inliers)}", "light_magenta")
    
        # cprint(f"Best homography: {best_H}","white") # DEBUG STATEMENTS !!!
        # cprint(f"Best Inliers: {best_inliers}","white") # DEBUG STATEMENTS !!!
        # cprint(f"Cost: {cost}","white") # DEBUG STATEMENTS !!!
        
        #============================================= Inlier Points: =============================================
        # Visualize the inlier points:
        visualize_correspondences((given_images[n-1],given_images[n]), [(img1_sift_points[i], img2_sift_points[i]) for i in best_inliers], f"MyResults/InOutLiers/Task1_InOutLiers_{n}_{n+1}")

        pair_homographies.append(best_H)  # Useful for panorama !!! 

        avg_inliers += len(best_inliers)
        cprint(f"For image {n}.jpg and {n+1}.jpg ^^^^") # Formattting purposes for the prints
        cprint("===================================================================================================") # Formattting purposes for the prints
        print() # Formattting purposes for the prints
        
    cprint(f"Completed RANSAC Algo", "green")
    cprint(f"Average number of inliers: {avg_inliers/4}", "light_magenta")
    
    #============================================= Panorama: =============================================
    pair_homographies = list()
    for i in range(1,5):
        pair_homographies.append(read_homographies(f"MyResults/best_homography_pairs_{i}_{i+1}.txt"))
    
    final_homographies = []  # Store homographies to map all images to the reference (img3)
    # Homographies between image pairs
    H12 = pair_homographies[0]
    H23 = pair_homographies[1]
    H34 = np.linalg.inv(pair_homographies[2])
    H45 = pair_homographies[3]
    
    # Compute H13 and H35
    H13 = np.matmul(H23, H12)
    H35 = H34 @ H45
    H35 = np.matmul(np.linalg.inv(pair_homographies[2]), np.linalg.inv(pair_homographies[3]))
    
    
    # Final homographies relative to img3
    final_homographies = [H13, H23, np.identity(3), H34, H35]
    
   # Create the panorama
    panorama = create_panorama(pair_homographies, given_images)
    ensure_directory('MyResults/Panorama')
    cv2.imwrite('MyResults/Panorama/final_panorama.jpg', panorama)
    print("Panorama created and saved as final_panorama.jpg")

    
    cprint("Panorama created and saved as final_panorama.jpg", "green")