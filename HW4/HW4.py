import numpy as np
import cv2
from termcolor import cprint
from skimage import io
import matplotlib.pyplot as plt 
from typing import List, Tuple
from scipy.ndimage import convolve

'''
Implement an automated approach for interest point detection and correspondence search for a given pair of images of the
same scene.

For the detection part, you will do the following:
1. Implement your own Harris Corner Detection algorithm.
2. Test the SIFT or SURF implementation that are available in OpenCV.
3. Test the CNN-based SuperPoint [1] interest point detector.

And to establish the point-to-point correspondences between the two views, you will do the following:
1. Implement your own functions for computing the SSD (Sum of Squared Differences) and the NCC
(Normalized Cross Correlation) as the feature similarity measures.

2. Test the GNN-based SuperGlue [2] feature matching network.

'''
def calc_NCC(img1, img2, harris_corner_img1, harris_corner_img2, sigma, pair_name):
    # Implement the calculation for Normalized Cross-Correlation (NCC)
    pass

import numpy as np
import cv2
from termcolor import cprint
from scipy.ndimage import convolve
import matplotlib.pyplot as plt

# Custom Sobel function with correct padding and handling
def custom_sobel(img: np.ndarray, axis: str) -> np.ndarray:
    """Manually apply Sobel filter in the specified axis (x or y)."""
    if axis == 'x':
        sobel_x = np.array([[-1, 0, 1], 
                            [-2, 0, 2], 
                            [-1, 0, 1]])
        return apply_convolution(img, sobel_x)
    
    elif axis == 'y':
        sobel_y = np.array([[-1, -2, -1], 
                            [0,  0,  0], 
                            [1,  2,  1]])
        return apply_convolution(img, sobel_y)
    
    else:
        raise ValueError("Invalid axis. Use either 'x' or 'y'.")

# Helper function to apply convolution manually
def apply_convolution(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Applies convolution manually to avoid the clustering issue."""
    # Get dimensions of the kernel and the image
    k_height, k_width = kernel.shape
    img_height, img_width = img.shape
    
    # Calculate padding size
    pad_height = k_height // 2
    pad_width = k_width // 2
    
    # Pad the image to handle boundaries
    padded_img = np.pad(img, ((pad_height, pad_height), (pad_width, pad_width)), mode='reflect')
    
    # Create an output image to store the result
    output_img = np.zeros_like(img, dtype=np.float64)
    
    # Convolution process
    for i in range(img_height):
        for j in range(img_width):
            # Extract the region of interest (ROI)
            roi = padded_img[i:i+k_height, j:j+k_width]
            # Perform element-wise multiplication and sum the result
            output_img[i, j] = np.sum(roi * kernel)
    
    return output_img

# Harris Corner Detection function
def Harris_Corner_Detection(image_pair, sigma: int, pair_name: str):
    def Corner_Detection(img_gray, img_color, sigma, pair_num: str, k=0.05):
        # Compute gradients using the custom Sobel function
        dx = custom_sobel(img_gray, axis='x')
        dy = custom_sobel(img_gray, axis='y')

        # Compute gradient products
        dx2 = dx ** 2
        dy2 = dy ** 2
        dxdy = dx * dy

        # Gaussian window for summing the products
        kernel_size = int(np.ceil(5 * sigma))
        if kernel_size % 2 == 1:
            kernel_size += 1  # Ensure kernel size is even

        kernel = np.ones((kernel_size, kernel_size))
        sum11 = convolve(dx2, kernel, mode='reflect')
        sum22 = convolve(dy2, kernel, mode='reflect')
        sum12 = convolve(dxdy, kernel, mode='reflect')

        # Harris corner response calculation (trace, determinant)
        trace = sum11 + sum22
        det = sum11 * sum22 - sum12 ** 2
        R = det - k * trace ** 2  # Harris response function

        # Thresholding and non-maximal suppression
        R_thresh = 0.01 * np.max(R)
        mask = np.ones(R.shape)
        mask[R < 0] = 0
        corners = []
        win_size = 15
        half_win = int(win_size / 2)
        for x in range(half_win, img_gray.shape[1] - half_win):
            for y in range(half_win, img_gray.shape[0] - half_win):
                if mask[y, x] > 0:
                    local_max = np.amax(R[y - half_win:y + half_win + 1, x - half_win:x + half_win + 1])
                    if R[y, x] == local_max and R[y, x] > R_thresh:
                        corners.append([x, y])

        # Visualize the result on the original color image
        for point in corners:  # Only visualize top 100 corners
            cv2.circle(img_color, tuple(point), radius=4, color=(0, 0, 255), thickness=-1)

        # Save image with detected corners on the original colored image
        cv2.imwrite(f"MyResults/Harris_Corner_{pair_name}_{pair_num}_with_sigma_{sigma}.jpg", img_color)

        return corners

    # Convert the images to grayscale for corner detection
    img1_gray = cv2.cvtColor(image_pair[0], cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(image_pair[1], cv2.COLOR_BGR2GRAY)

    # Keep the original color images for visualization
    img1_color = image_pair[0].copy()
    img2_color = image_pair[1].copy()

    # Apply Harris Corner Detection on both images
    cprint(f"---Harris Corners using a sigma of : {sigma} ---", "white")
    corners_img1 = Corner_Detection(img1_gray, img1_color, sigma, "1")
    cprint(f"Corners of first image in pair {pair_name} found!", "green")
    
    corners_img2 = Corner_Detection(img2_gray, img2_color, sigma, "2")
    cprint(f"Corners of second image in pair {pair_name} found!", "green")
    
    return corners_img1, corners_img2

# Main code to run Harris Corner Detection on image pairs
if __name__ == '__main__':
    # Preprocessing and loading of variables:
    # Individual Images
    img_hovde_2 = cv2.imread('HW4_images/hovde_2.jpg')
    img_hovde_3 = cv2.imread('HW4_images/hovde_3.jpg')
    img_temple_1 = cv2.imread('HW4_images/temple_1.jpg')
    img_temple_2 = cv2.imread('HW4_images/temple_2.jpg')

    # Image pairs:
    hovde_pair = (img_hovde_2, img_hovde_3)
    temple_pair = (img_temple_1, img_temple_2)

    # Pair the images with the names:
    given_img_pairs = [hovde_pair, temple_pair]
    given_img_names = ["hovde", "temple"]

    # TASK 1; Harris Corner Detection:
    sigmas = [0.8, 1.2, 1.6, 2.0]
    for sigma in sigmas:
        for image_pair, img_name in zip(given_img_pairs, given_img_names):
            corners1, corners2 = Harris_Corner_Detection(image_pair, sigma, img_name)
            print(f"Corners in first image: {len(corners1)}, Corners in second image: {len(corners2)}")
