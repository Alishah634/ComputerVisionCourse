import os
import random
from tqdm import tqdm
from time import sleep
from typing import List, Tuple
from argparse import ArgumentParser
import numpy as np
import cv2
from termcolor import cprint
import matplotlib.pyplot as plt 

from skimage import io
from scipy.ndimage import convolve
from scipy.optimize import least_squares

'''START of Util Functions: '''
# Create a folder if it does not exist
def ensure_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
# Plots an image for debugging purposes with optional title and colormap.
# Takes in the image to be plotted, Title to display above the image, and the Colormap to use, default is 'gray', 
def debug_plot_image(image, title=None, cmap='gray'):
    plt.imshow(image, cmap=cmap)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()
    plt.clf()
'''END of Util Functions: '''

# Plot histogram with threshold line and save to file:
def plot_histogram_with_threshold(hist, threshold, image_name, channel_name):
    ensure_directory("MyResults/Histogram")  # Ensure the directory exists
    plt.figure()
    plt.title(f"Histogram and Otsu's Threshold - {image_name} - {channel_name}")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.plot(hist, color='blue')
    plt.axvline(x=threshold, color='red', linestyle='--', label=f'Threshold: {threshold}')
    plt.legend()
    # Save the plot:
    plt.savefig(f"MyResults/Histogram/{image_name}_{channel_name}_otsu_histogram.png")
    plt.show()
    plt.close()

'''TASK 1.1: RGB SEGMENTATION: '''
# Determine whether to use inverse thresholding based on histogram analysis
def determine_inverse_per_channel(img_channel):
    # Generate the histogram:
    hist, bins = np.histogram(img_channel, bins=256, range=(0, 256))
    # Find the peak locations (maximum values) in the histogram
    # Lower intensity range for background:
    background_peak = np.argmax(hist[:128])   
    # Higher intensity range for foreground:
    foreground_peak = np.argmax(hist[128:]) + 128  
    if background_peak is None or foreground_peak is None:
        raise ValueError(f"\033[31m'Background or Foreground not Calculated correctly!!!\033[37m")
    # If foreground peak is higher intensity than background peak, then we should invert:
    return foreground_peak > background_peak

# Otsu's method for grayscale images
def otsu_threshold_grayscale(image_channel, channel_name: str = None, file_name: str = None, channel_bit: int = 1):    
    # Determine whether to use inverse for this channel
    inverse = determine_inverse_per_channel(image_channel)
    
    # Initialize variables for Otsu's method
    hist, bins = np.histogram(image_channel.flatten(), 256, [0,256])
    total_pixels = image_channel.shape[0] * image_channel.shape[1]

    current_max, optimal_threshold = 0, 0
    # Total sum of pixel intensities:
    sum_total = np.dot(np.arange(256), hist)  
    # Initialize cumulative sum and weight for background class:
    sum_0, w_0 = 0, 0  
    
    # Iterate through all possible thresholds
    for t in range(256):
        # Weight of the background (class 0)
        w_0 += hist[t]  
        if w_0 == 0: 
            continue
        # Weight of the foreground (class 1)
        w_1 = total_pixels - w_0  
        if w_1 == 0: 
            break

        sum_0 += t * hist[t]  # Cumulative sum for background
        mu_0 = sum_0 / w_0  # Mean intensity for background
        mu_1 = (sum_total - sum_0) / w_1  # Mean intensity for foreground

        # Calculate between-class variance
        between_class_variance = w_0 * w_1 * (mu_1 - mu_0) ** 2
        if between_class_variance > current_max:
            current_max = between_class_variance
            optimal_threshold = t

    # Apply the threshold to create a binary mask:
    # Set the mask to be a blank white bg image, overwrite with black pixel accodingly:
    if file_name == "flower":
        binary_mask = np.ones(image_channel.shape,dtype=bool)
        if inverse:
            binary_mask[image_channel > optimal_threshold] = 0
        else:
            binary_mask[image_channel <= optimal_threshold] = 0
    else:
        binary_mask = np.zeros(image_channel.shape,dtype=bool)
        if inverse:
            binary_mask[image_channel > optimal_threshold] = 1
        else:
            binary_mask[image_channel <= optimal_threshold] = 1
            
            
    # Plot histogram and threshold
    if channel_name != None:
        plot_histogram_with_threshold(hist, optimal_threshold, file_name, channel_name)

    return binary_mask.astype(np.uint8)

# Otsu's algorithm for each RGB channel, added iterations to refine it:
def otsu_rgb(img, file_name: str = None, iterations=2):
    # Split image into RGB channels
    channels = cv2.split(img)
    binary_masks = []
    # Apply Otsu's algorithm to each channel
    color_channels = ['Blue', 'Green', 'Red']  
    for i, channel in enumerate(channels):
        binary_mask = otsu_threshold_grayscale(channel, file_name)
        for _ in range(iterations):
            # Refine mask using iterative Otsu on the foreground
            foreground = channel * binary_mask
            binary_mask = otsu_threshold_grayscale(foreground, color_channels[i], file_name)
        binary_masks.append(binary_mask)
    # Combine binary masks from all channels using AND operation
    final_mask = binary_masks[0] & binary_masks[1] & binary_masks[2]
    cprint(f"The final mask found is: {final_mask}", "cyan")
    
    return final_mask, binary_masks


'''TASK 1.2: TEXTURE SEGMENTATION: '''
def get_texture(img, win_sizes=[3, 5, 7], num_iterations=[1, 1, 1]):
    # Convert image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Initialize containers for texture images and masks for each window size
    texture_layers = np.zeros((gray_img.shape[0], gray_img.shape[1], len(win_sizes)), dtype=np.uint8)
    combined_masks = np.zeros((gray_img.shape[0], gray_img.shape[1], len(win_sizes)), dtype=np.uint8)

    # Iterate over different window sizes
    for idx, win_size in enumerate(win_sizes):
        # Apply padding around the image for the window size
        pad_size = int((win_size - 1) / 2)
        padded_img = np.pad(gray_img, pad_size, mode='constant', constant_values=np.mean(gray_img))

        # Compute variance in the window for each pixel
        variance_img = np.zeros(gray_img.shape, dtype=np.uint8)
        for i in tqdm(range(gray_img.shape[0])):
            for j in range(gray_img.shape[1]):
                # Extract the window region
                window = padded_img[i:i + win_size, j:j + win_size]
                # Calculate variance in the window and store it
                variance_img[i, j] = np.var(window)
        
        # Normalize the variance image to [0, 255]
        variance_img = cv2.normalize(variance_img, None, 0, 255, cv2.NORM_MINMAX)

        # Apply Otsu’s algorithm to the texture feature map (variance image)
        binary_mask = otsu_threshold_grayscale(variance_img,channel_name=win_sizes[idx], channel_bit=num_iterations[idx])
        combined_masks[:, :, idx] = binary_mask
        texture_layers[:, :, idx] = variance_img

    # Combine masks from all window sizes using AND operation
    final_mask = np.all(combined_masks, axis=2).astype(np.uint8)

    # Return both the combined mask and individual texture masks
    return final_mask, [combined_masks[:, :, i] for i in range(len(win_sizes))]


'''TASK 1.3: Obtain the contour: '''
def detect_contours(binary_mask, kernel_size, file_name: str = None):
    pixel_num = 0 
    # if file_name == "flower":
        # pixel_num = 255
    output = np.zeros(binary_mask.shape, dtype=np.uint8)
    for i in range(kernel_size, binary_mask.shape[0]-kernel_size):
        for j in range(kernel_size, binary_mask.shape[1]-kernel_size):
            if np.min(binary_mask[i - kernel_size:i+kernel_size+1, j-kernel_size:j+kernel_size+1])==0:
                output[i, j] = 255
    output[binary_mask == 0] = pixel_num
    return output

def erode_image(binary_mask, errosion_size, _iterations):
    erroion_kernel = np.ones((errosion_size, errosion_size), dtype=np.uint8)
    output = cv2.erode(binary_mask, erroion_kernel, iterations= _iterations)
    return output
def dilate_image(binary_mask, dilattion_size, _iterations):
    dilation_kernel = np.ones((dilattion_size, dilattion_size), dtype=np.uint8)
    output = cv2.dilate(binary_mask, dilation_kernel, iterations= _iterations)
    return output
    

def task1():
    cprint("Running Task 1", "white")
    ''' INITALIZATION:   '''
    # Load the images
    dog_image = cv2.imread('HW6_images/pics/dog_small.jpg')
    flower_image = cv2.imread('HW6_images/pics/flower_small.jpg')
    # Convert images to RGB (OpenCV loads images in BGR by default)
    dog_image_rgb = cv2.cvtColor(dog_image, cv2.COLOR_BGR2RGB)
    flower_image_rgb = cv2.cvtColor(flower_image, cv2.COLOR_BGR2RGB)

    ''' TASK 1.1: BGR SEGMENTATION: '''
    # Methodology:
    # The Otsu algorithm on RGB images means treat each channel separately,
    # i.e., Obtain per-channel segmentation first 
    # Combine these results using logical ‘AND’ operation.
    # Then parameter tuning, run the Otsu algorithm in an iterative fashion. 
    # Use the foreground segmentation returned by the previous iteration to refine it further by
    # running the Otsu algorithm only on the foreground portion of the image.
    
    # Apply Otsu's algorithm to both images
    dog_combined_mask, dog_bin_masks = otsu_rgb(dog_image_rgb)
    flower_combined_mask, flower_bin_masks = otsu_rgb(flower_image_rgb, "flower")
    color_channels = ["Blue", "Green", "Red"]
    
    ensure_directory(f"MyResults/BGR_SEG")
    for image_name, bin_masks in [("dog", dog_bin_masks), ("flower", flower_bin_masks)]:
        for color_channel, bin_mask in zip(color_channels, bin_masks):
            cv2.imwrite(f"MyResults/BGR_SEG/{image_name}_{color_channel}_otsu_mask.jpg", bin_mask * 255)

    # Save the combined binary masks:
    cv2.imwrite("MyResults/BGR_SEG/dog_combined_otsu_mask.jpg", dog_combined_mask * 255)
    cv2.imwrite("MyResults/BGR_SEG/flower_combined_otsu_mask.jpg", flower_combined_mask * 255)
    
    ''' TASK 1.2: TEXTURE SEGMENTATION: '''
    # Apply texture-based segmentation
    dog_texture_mask, dog_texture_layers = get_texture(dog_image)
    flower_texture_mask, flower_texture_layers = get_texture(flower_image)
    # debug_plot_image(dog_texture_mask, title="Dog TEXTURE COMBINED Image") # DEBUG STATEMENT!!!
    # debug_plot_image(flower_texture_mask, title="Flower TEXTURE COMBINED Image") # DEBUG STATEMENT!!!

    # Save texture masks:
    ensure_directory(f"MyResults/Texture_SEG")
    # Define the window sizes
    win_sizes = [3, 5, 7]
    # Save individual masks for dog and flower:
    for idx, win_size in enumerate(win_sizes):
        cv2.imwrite(f"MyResults/Texture_SEG/dog_N_{win_size}_otsu_mask.jpg", dog_texture_layers[idx] * 255)
        cv2.imwrite(f"MyResults/Texture_SEG/flower_N_{win_size}_otsu_mask.jpg", flower_texture_layers[idx] * 255)

    # Save the combined texture mask for both images
    cv2.imwrite(f"MyResults/Texture_SEG/dog_combined_texture_mask.jpg", dog_texture_mask * 255)
    cv2.imwrite(f"MyResults/Texture_SEG/flower_combined_texture_mask.jpg", flower_texture_mask * 255)

    ''' TASK 1.3: Contour Extraction:   '''
    ''' RGB CONTOURS: '''
    # Extract contours
    dog_rgb_contour_img = detect_contours(dog_combined_mask, 1)
    flower_rgb_contour_img = detect_contours(flower_combined_mask, 1, "flower")
    dog_texture_contour_img = detect_contours(dog_texture_mask, 5)
    flower_texture_contour_img = detect_contours(flower_texture_mask, 1, "flower")
    # debug_plot_image(dog_rgb_contour_img, title="Dog RGB Contour Image") # DEBUG STATEMENT!!!
    # debug_plot_image(flower_rgb_contour_img, title="Flower RGB Contour Image") # DEBUG STATEMENT!!!
    # debug_plot_image(dog_texture_contour_img, title="Dog RGB Contour Image") # DEBUG STATEMENT!!!
    # debug_plot_image(flower_texture_contour_img, title="Flower RGB Contour Image") # DEBUG STATEMENT!!!

    # Save the results
    ensure_directory("MyResults/Contours")
    cv2.imwrite("MyResults/Contours/dog_with_RGB_contours.jpg", cv2.cvtColor(dog_rgb_contour_img, cv2.COLOR_GRAY2BGR))
    cv2.imwrite("MyResults/Contours/flower_with_RGB_contours.jpg", cv2.cvtColor(flower_rgb_contour_img, cv2.COLOR_GRAY2BGR))
    cv2.imwrite("MyResults/Contours/dog_with_TEXTURE_contours.jpg", cv2.cvtColor(dog_texture_contour_img, cv2.COLOR_GRAY2BGR))
    cv2.imwrite("MyResults/Contours/flower_with_TEXTURE_contours.jpg", cv2.cvtColor(flower_texture_contour_img, cv2.COLOR_GRAY2BGR))
   
   
    ''' OPEN AND CLOSING USING EROSION and DILATION: '''
    ''' RGB:'''
    # ERODE and Dilate the masks:
    ensure_directory("MyResults/ErodeDilate/")
    erode_dilate_size = 2
    # ERODE
    temp = erode_image(dog_combined_mask, erode_dilate_size, 1)
    erode_mask = detect_contours(temp, 1)
    cv2.imwrite("MyResults/ErodeDilate/dog_rgb_erode_mask.jpg", cv2.cvtColor(erode_mask, cv2.COLOR_GRAY2BGR))
    #  DILATE:
    temp = dilate_image(dog_combined_mask, erode_dilate_size, 1)
    dilate_mask = detect_contours(temp, 1)
    cv2.imwrite("MyResults/ErodeDilate/dog_rgb_dilate_mask.jpg", cv2.cvtColor(dilate_mask, cv2.COLOR_GRAY2BGR))

    # ERODE -> DILATE:
    temp = erode_image(dog_combined_mask, erode_dilate_size, 1)
    erode_dilate_mask = detect_contours(temp, 1)
    cv2.imwrite("MyResults/ErodeDilate/dog_rgb_opening_mask.jpg", cv2.cvtColor(erode_dilate_mask, cv2.COLOR_GRAY2BGR))
    cv2.imwrite("MyResults/ErodeDilate/dog_rgb_opening_mask.jpg", cv2.cvtColor(erode_dilate_mask, cv2.COLOR_GRAY2BGR))

    # DILATE -> ERODE:
    temp = dilate_image(dog_combined_mask, erode_dilate_size, 1)
    temp = erode_image(temp, erode_dilate_size, 1)
    dilate_erode_mask = detect_contours(temp, 1)
    # debug_plot_image(erode_dilate_mask, "DOG RGB ERODE DILATE CONTOUR IMAGE") # DEBUG STATEMENT!!!
    # debug_plot_image(dilate_erode_mask, "DOG RGB DILATE ERODE  CONTOUR IMAGE") # DEBUG STATEMENT!!!
    cv2.imwrite("MyResults/ErodeDilate/dog_rgb_closing_mask.jpg", cv2.cvtColor(dilate_erode_mask, cv2.COLOR_GRAY2BGR))
    
    '''FLOWER: '''
    ensure_directory("MyResults/ErodeDilate/")
    erode_dilate_size = 2
    # ERODE
    temp = erode_image(flower_combined_mask, erode_dilate_size, 1)
    erode_mask = detect_contours(temp, 1)
    cv2.imwrite("MyResults/ErodeDilate/flower_rgb_erode_mask.jpg", cv2.cvtColor(erode_mask, cv2.COLOR_GRAY2BGR))
    #  DILATE:
    temp = dilate_image(flower_combined_mask, erode_dilate_size, 1)
    dilate_mask = detect_contours(temp, 1)
    cv2.imwrite("MyResults/ErodeDilate/flower_rgb_dilate_mask.jpg", cv2.cvtColor(dilate_mask, cv2.COLOR_GRAY2BGR))

    
    erode_dilate_size = 2
    # ERODE -> DILATE:
    temp = erode_image(flower_combined_mask, erode_dilate_size, 1)
    temp = dilate_image(temp, erode_dilate_size, 1)
    erode_dilate_mask = detect_contours(temp, 1)
    cv2.imwrite("MyResults/ErodeDilate/flower_rgb_opening_mask.jpg", cv2.cvtColor(erode_dilate_mask, cv2.COLOR_GRAY2BGR))
    
   
    # DILATE -> ERODE:
    temp = dilate_image(flower_combined_mask, erode_dilate_size, 1)
    temp = erode_image(temp, erode_dilate_size, 1)
    dilate_erode_mask = detect_contours(temp, 1)
    # debug_plot_image(erode_dilate_mask, "FLOWER RGB DILATE ERODE CONTOUR IMAGE") # DEBUG STATEMENT!!!
    # debug_plot_image(dilate_erode_mask, "FLOWER RGB ERODE DILATE CONTOUR IMAGE") # DEBUG STATEMENT!!!
    cv2.imwrite("MyResults/ErodeDilate/flower_rgb_closing_mask.jpg", cv2.cvtColor(dilate_erode_mask, cv2.COLOR_GRAY2BGR))

    '''TEXTURE CONTOURS: '''
    pass

def task2():
    cprint("Running Task 2", "white")
    
    ''' INITALIZATION:   '''
    # Load the images
    first_image = cv2.imread('MyImages/first_image.jpg')
    second_image = cv2.imread('MyImages/second_image.jpg')
    # Convert images to RGB (OpenCV loads images in BGR by default)
    first_image_rgb = cv2.cvtColor(first_image, cv2.COLOR_BGR2RGB)
    second_image_rgb = cv2.cvtColor(second_image, cv2.COLOR_BGR2RGB)
    
    ''' TASK 2.1: BGR SEGMENTATION: '''
    first_combined_mask, first_bin_masks = otsu_rgb(first_image_rgb)
    second_combined_mask, second_bin_masks = otsu_rgb(second_image_rgb, "second")
    color_channels = ["Blue", "Green", "Red"]
    
    ensure_directory(f"MyImageResults/BGR_SEG")
    for image_name, bin_masks in [("first", first_bin_masks), ("second", second_bin_masks)]:
        for color_channel, bin_mask in zip(color_channels, bin_masks):
            cv2.imwrite(f"MyImageResults/BGR_SEG/{image_name}_{color_channel}_otsu_mask.jpg", bin_mask * 255)

    # Save the combined binary masks:
    cv2.imwrite("MyImageResults/BGR_SEG/first_combined_otsu_mask.jpg", first_combined_mask * 255)
    cv2.imwrite("MyImageResults/BGR_SEG/second_combined_otsu_mask.jpg", second_combined_mask * 255)
    
    ''' TASK 2.2: TEXTURE SEGMENTATION: '''
    # Apply texture-based segmentation
    first_texture_mask, first_texture_layers = get_texture(first_image)
    second_texture_mask, second_texture_layers = get_texture(second_image)
    # debug_plot_image(first_texture_mask, title="first TEXTURE COMBINED Image") # DEBUG STATEMENT!!!
    # debug_plot_image(second_texture_mask, title="second TEXTURE COMBINED Image") # DEBUG STATEMENT!!!

    # Save texture masks:
    ensure_directory(f"MyImageResults/Texture_SEG")
    # Define the window sizes
    win_sizes = [3, 5, 7]
    # Save individual masks for first and second:
    for idx, win_size in enumerate(win_sizes):
        cv2.imwrite(f"MyImageResults/Texture_SEG/first_N_{win_size}_otsu_mask.jpg", first_texture_layers[idx] * 255)
        cv2.imwrite(f"MyImageResults/Texture_SEG/second_N_{win_size}_otsu_mask.jpg", second_texture_layers[idx] * 255)

    # Save the combined texture mask for both images
    cv2.imwrite(f"MyImageResults/Texture_SEG/first_combined_texture_mask.jpg", first_texture_mask * 255)
    cv2.imwrite(f"MyImageResults/Texture_SEG/second_combined_texture_mask.jpg", second_texture_mask * 255)

    ''' TASK 2.3: Contour Extraction:   '''
    ''' RGB CONTOURS: '''
    # Extract contours
    first_rgb_contour_img = detect_contours(first_combined_mask, 1)
    second_rgb_contour_img = detect_contours(second_combined_mask, 1, "second")
    first_texture_contour_img = detect_contours(first_texture_mask, 5)
    second_texture_contour_img = detect_contours(second_texture_mask, 1, "second")
    # debug_plot_image(first_rgb_contour_img, title="first RGB Contour Image") # DEBUG STATEMENT!!!
    # debug_plot_image(second_rgb_contour_img, title="second RGB Contour Image") # DEBUG STATEMENT!!!
    # debug_plot_image(first_texture_contour_img, title="first RGB Contour Image") # DEBUG STATEMENT!!!
    # debug_plot_image(second_texture_contour_img, title="second RGB Contour Image") # DEBUG STATEMENT!!!

    # Save the results
    ensure_directory("MyImageResults/Contours")
    cv2.imwrite("MyImageResults/Contours/first_with_RGB_contours.jpg", cv2.cvtColor(first_rgb_contour_img, cv2.COLOR_GRAY2BGR))
    cv2.imwrite("MyImageResults/Contours/second_with_RGB_contours.jpg", cv2.cvtColor(second_rgb_contour_img, cv2.COLOR_GRAY2BGR))
    cv2.imwrite("MyImageResults/Contours/first_with_TEXTURE_contours.jpg", cv2.cvtColor(first_texture_contour_img, cv2.COLOR_GRAY2BGR))
    cv2.imwrite("MyImageResults/Contours/second_with_TEXTURE_contours.jpg", cv2.cvtColor(second_texture_contour_img, cv2.COLOR_GRAY2BGR))
   
   
    ''' OPEN AND CLOSING USING EROSION and DILATION: '''
    ''' RGB:'''
    # ERODE and Dilate the masks:
    ensure_directory("MyImageResults/ErodeDilate/")
    erode_dilate_size = 5
    # ERODE
    temp = erode_image(first_combined_mask, erode_dilate_size, 1)
    erode_mask = detect_contours(temp, 1)
    cv2.imwrite("MyImageResults/ErodeDilate/first_rgb_erode_mask.jpg", cv2.cvtColor(erode_mask, cv2.COLOR_GRAY2BGR))
    #  DILATE:
    temp = dilate_image(first_combined_mask, erode_dilate_size, 1)
    dilate_mask = detect_contours(temp, 1)
    cv2.imwrite("MyImageResults/ErodeDilate/first_rgb_dilate_mask.jpg", cv2.cvtColor(dilate_mask, cv2.COLOR_GRAY2BGR))

    # ERODE -> DILATE:
    temp = erode_image(first_combined_mask, erode_dilate_size, 1)
    erode_dilate_mask = detect_contours(temp, 1)
    cv2.imwrite("MyImageResults/ErodeDilate/first_rgb_opening_mask.jpg", cv2.cvtColor(erode_dilate_mask, cv2.COLOR_GRAY2BGR))
    cv2.imwrite("MyImageResults/ErodeDilate/first_rgb_opening_mask.jpg", cv2.cvtColor(erode_dilate_mask, cv2.COLOR_GRAY2BGR))

    # DILATE -> ERODE:
    temp = dilate_image(first_combined_mask, erode_dilate_size, 1)
    temp = erode_image(temp, erode_dilate_size, 1)
    dilate_erode_mask = detect_contours(temp, 1)
    # debug_plot_image(erode_dilate_mask, "first RGB ERODE DILATE CONTOUR IMAGE") # DEBUG STATEMENT!!!
    # debug_plot_image(dilate_erode_mask, "first RGB DILATE ERODE  CONTOUR IMAGE") # DEBUG STATEMENT!!!
    cv2.imwrite("MyImageResults/ErodeDilate/first_rgb_closing_mask.jpg", cv2.cvtColor(dilate_erode_mask, cv2.COLOR_GRAY2BGR))
    
    '''second: '''
    ensure_directory("MyImageResults/ErodeDilate/")
    erode_dilate_size = 2
    # ERODE
    temp = erode_image(second_combined_mask, erode_dilate_size, 1)
    erode_mask = detect_contours(temp, 1)
    cv2.imwrite("MyImageResults/ErodeDilate/second_rgb_erode_mask.jpg", cv2.cvtColor(erode_mask, cv2.COLOR_GRAY2BGR))
    #  DILATE:
    temp = dilate_image(second_combined_mask, erode_dilate_size, 1)
    dilate_mask = detect_contours(temp, 1)
    cv2.imwrite("MyImageResults/ErodeDilate/second_rgb_dilate_mask.jpg", cv2.cvtColor(dilate_mask, cv2.COLOR_GRAY2BGR))

    
    erode_dilate_size = 2
    # ERODE -> DILATE:
    temp = erode_image(second_combined_mask, erode_dilate_size, 1)
    temp = dilate_image(temp, erode_dilate_size, 1)
    erode_dilate_mask = detect_contours(temp, 1)
    cv2.imwrite("MyImageResults/ErodeDilate/second_rgb_opening_mask.jpg", cv2.cvtColor(erode_dilate_mask, cv2.COLOR_GRAY2BGR))
    
   
    # DILATE -> ERODE:
    temp = dilate_image(second_combined_mask, erode_dilate_size, 1)
    temp = erode_image(temp, erode_dilate_size, 1)
    dilate_erode_mask = detect_contours(temp, 1)
    # debug_plot_image(erode_dilate_mask, "second RGB DILATE ERODE CONTOUR IMAGE") # DEBUG STATEMENT!!!
    # debug_plot_image(dilate_erode_mask, "second RGB ERODE DILATE CONTOUR IMAGE") # DEBUG STATEMENT!!!
    cv2.imwrite("MyImageResults/ErodeDilate/second_rgb_closing_mask.jpg", cv2.cvtColor(dilate_erode_mask, cv2.COLOR_GRAY2BGR))

    '''TEXTURE CONTOURS: '''
    pass

if __name__ == "__main__":
    # Create argument parser
    parser = ArgumentParser(
        description="Script to run Task 1 or Task 2. Use the arguments below to select the task.",
        epilog="Example usage: python HW6.py --task 1"
    )
    # Add arguments to select task
    parser.add_argument(
        '--task', '-t', type=int, required=True, choices=[1, 2],
        help="Select which task to run: 1 for Task 1, 2 for Task 2"
    )

    args = parser.parse_args()
    # Ensure you're in the correct directory
    cprint("Make sure you are running from HW6 directory!!!\n", "red")
    ensure_directory(f"MyResults")
    # Execute the task based on the argument provided
    if args.task == 1:
        task1()
    elif args.task == 2:
        task2()
        
