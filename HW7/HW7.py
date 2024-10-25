import os
import time
import logging # For logging and decorators
import functools # For logging and decorators
from tqdm import tqdm # Progress bar
from typing import List, Tuple # Format typing
from argparse import ArgumentParser # Parsing arguments
from termcolor import cprint # Formatting prints
import pickle # For saving intermediate values to avoid recomputing

# Other Computer vision, Data science, Math, plotting imports:
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt 
from skimage import io
from scipy.ndimage import convolve
from scipy.optimize import least_squares

'''START of Util Functions: '''
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def log_decorator(enabled=True):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if enabled:
                logger.info(f"Calling function {func.__name__} with arguments {args} and keyword arguments {kwargs}")
            result = func(*args, **kwargs)
            if enabled:
                logger.info(f"{func.__name__} returned {result}")
            return result
        return wrapper
    return decorator

def time_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"{func.__name__} took {elapsed_time:.4f} seconds to execute")
        return result
    return wrapper

def error_handling_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in function {func.__name__}: {e}")
            raise
    return wrapper

# Create a folder if it does not exist
def ensure_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
                
def verify_dir(path: str):
    try:
        if not isinstance(path, str) or not os.path.isdir(path.strip()):
            cprint(f"Error in directory path!!!","red")
            cprint(f"Check if the directory path is indeed a directory!!!","red")
            cprint(f"Directory path given: {path}", "yellow")
            return list()
        os.listdir(path)  # Check accessibility
    except (PermissionError, OSError, ValueError) as e:
        cprint(f"Exception Raised for Directory path!!!","red")
        cprint(f"Exception was: {e}","red")
        return list()       
    
     
# Plots an image for debugging purposes with optional title and colormap.
# Takes in the image to be plotted, Title to display above the image, and the Colormap to use, default is 'gray', 
def debug_plot_image(image, title=None, cmap='gray'):
    plt.imshow(image, cmap=cmap)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()
    plt.clf()

@time_decorator
def save_to_pickle(data, dir_path: str, file_name: str) -> None:
    """
    Save data to a pickle file in the specified directory.
    Args:
        data: The data to be saved (any Python object).
        dir_path (str): The directory where the file will be saved.
        file_name (str): The name of the pickle file (without extension).
    """
    # Ensure the directory exists
    ensure_directory(dir_path)
    # Create the full file path
    file_path = os.path.join(dir_path, f"{file_name}.pkl")
    # Save the data to the pickle file
    with open(file_path, 'wb') as file:
        try: 
            pickle.dump(data, file)
        except Exception as e:
            cprint(f"Failure saving file: {file_path}!!!","red")
            cprint(f"Exception raised: {e}","red")
    cprint(f"Data saved to {file_path}", "green")    

@time_decorator
def load_from_pickle(dir_path: str, file_name: str):
    """
    Load data from a pickle file in the specified directory.

    Args:
        dir_path (str): The directory where the file is located.
        file_name (str): The name of the pickle file (without extension).
    
    Returns:
        The loaded data, or None if the file does not exist.
    """
    # Create the full file path
    file_path = os.path.join(dir_path, f"{file_name}.pkl")
    
    # Check if the file exists before loading
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None
    
    # Load and return the data from the pickle file
    with open(file_path, 'rb') as file:
        try: 
            data = pickle.load(file)
        except Exception as e:
            cprint(f"Failure loading file: {file_path}!!!","red")
            cprint(f"Exception raised: {e}","red")
    cprint(f"Data loaded from {file_path}", "green")
    return data

'''END of Util Functions: '''

''' Read in all the images, assumes, you are one directory outside of the FIRST Auxillary Folder, the images are inside the HW7-Auxilliary/HW7-Auxilliary/data/'''
def read_images(specified_desired_dir: (None | str) = None):
    def read_images_helper(dir_path: (None | str) = None):
        image_list_with_classes = list()
        # Check for VALID image file extensions (pretty sure .DS_Store is the only invalid file but just to be safe):
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}  
        for image_name in sorted(os.listdir(dir_path)):
            # Skip non-image files like .DS_Store:
            if not any(image_name.lower().endswith(ext) for ext in valid_extensions):
                continue  
            image = io.imread(os.path.join(dir_path+image_name))
            
            # cprint(f"Loading image {image_name} with shape {image.shape}", "cyan")
            # Handle multi-frame images (if any)
            if image.ndim > 3:
                cprint(f"Warning: Image {image_name} has {image.ndim} dimensions. Using the first frame.", "yellow")
                image = image[0]
            
            # Convert to RGB if needed
            if image.ndim == 2:  # Grayscale image
                try:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                    cprint(f"Converted grayscale image {image_name} to RGB", "yellow")
                except cv2.error as e:
                    cprint(f"Failed to convert grayscale image {image_name} to RGB: {e}", "red")
                    continue
            elif image.shape[-1] == 4:  # RGBA image
                try:
                    image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
                    cprint(f"Converted RGBA image {image_name} to RGB", "yellow")
                except cv2.error as e:
                    cprint(f"Failed to convert RGBA image {image_name} to RGB: {e}", "red")
                    continue
            elif image.ndim == 3 and image.shape[-1] == 3:
                pass  # Already RGB
            else:
                cprint(f"Warning: Image {image_name} has unexpected shape {image.shape}. Skipping.", "red")
                continue

            # Ensure consistency in data type (e.g., uint8)
            if image.dtype != np.uint8:
                image = image.astype(np.uint8)
                cprint(f"Converted image {image_name} to uint8", "yellow")
    
            # Ensure consistency in data type (e.g., uint8)
            image = image.astype(np.uint8)
            
            if (class_name := next((weather for weather in ["cloudy", "rain", "shine", "sunrise"] if weather in image_name), None)):
                    if class_name is not None:
                        image_list_with_classes.append((image, class_name, image_name))
        return image_list_with_classes
    
    # Assumes filepath is one directory outside of the Auxillary folder:
    desired_dir = "HW7-Auxilliary/HW7-Auxilliary/data/" 
    if specified_desired_dir:
        desired_dir = specified_desired_dir 
    verify_dir(desired_dir)
    
    train_images_with_classes, test_images_with_classes = list(), list() 
    
    # Loop over the training images (training/):
    dir_path = desired_dir+"training/"
    cprint(f"Directory: {dir_path}", "cyan")
    train_images_with_classes = read_images_helper(dir_path)

    # Loop over the testing images (testing/):
    dir_path = desired_dir+"testing/"
    test_images_with_classes = read_images_helper(dir_path)
    cprint(f"Number of training images: {len(train_images_with_classes)}", "cyan")
    cprint(f"Number of testing images: {len(test_images_with_classes)}", "cyan")
    return train_images_with_classes, test_images_with_classes

'''START of function for LBP:'''
def convert_rgb_to_hsi(image: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to HSI.
    
    Args:
        image (np.ndarray): Input RGB image with values in [0, 255].
    
    Returns:
        np.ndarray: HSI image with H in degrees [0, 360), S in [0, 1], and I in [0, 1].
    """
    # Print image shape for debugging
    # print(f"Image shape: {image.shape}")

    # Ensure the image has 3 channels (convert grayscale to RGB if needed)
    if len(image.shape) == 2 or image.shape[-1] == 1:  # Grayscale image check
        cprint("Converting grayscale image to RGB", "yellow")
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[-1] != 3:
        raise ValueError(f"Unexpected number of channels in the image: {image.shape[-1]}")

    # Normalize the RGB values to [0, 1] range
    img = image.astype(np.float32) / 255.0
    b_channels, g_channels, r_channels = cv2.split(img)

    # Initialize the H, S, and I channels
    H = np.zeros_like(r_channels.copy(), dtype=np.float32)
    S = np.zeros_like(r_channels.copy(), dtype=np.float32)
    I = np.zeros_like(r_channels.copy(), dtype=np.float32)

    # Compute I (intensity)
    I = (r_channels + g_channels + b_channels) / 3.0

    # Compute the min and max per pixel across R, G, B
    M = np.maximum(np.maximum(r_channels, g_channels), b_channels)
    m = np.minimum(np.minimum(r_channels, g_channels), b_channels)
    c = M - m

    # Compute H (hue) in degrees
    mask_r = (M == r_channels) & (c != 0)
    mask_g = (M == g_channels) & (c != 0)
    mask_b = (M == b_channels) & (c != 0)

    H[mask_r] = ((60 * ((g_channels[mask_r] - b_channels[mask_r]) / c[mask_r])) % 360)
    H[mask_g] = (60 * ((b_channels[mask_g] - r_channels[mask_g]) / c[mask_g]) + 120)
    H[mask_b] = (60 * ((r_channels[mask_b] - g_channels[mask_b]) / c[mask_b]) + 240)
    
    # Set H to 0 where c == 0 (grayscale pixels)
    H[c == 0] = 0.0

    # Compute S (saturation):
    S[c == 0] = 0.0
    S[c != 0] = 1 - (m[c != 0] /(M[c != 0] - m[c!=0])) # M - 2*m/M-m

    # Stack the H, S, and I channels to create the HSI image:
    HSI = np.stack((H, S, I), axis=-1)
    return HSI

def extract_hue(hsi_iamge: np.ndarray) -> np.ndarray:
    return hsi_iamge[:, :, 0]

'''END of function for LBP:'''


'''
# To summarize, the programming tasks in this homework include:
1. Implement your own routines for extracting LBP histograms as texture descriptors.
2. Given a pretrained CNN encoder, implement your own routines for extracting the Gram Matrix
    based texture descriptor.
3. For extra credits, also implement the channel normalization parameter based texture descriptor.
4. For each type of texture descriptor you implement, train a weather classifier and quantitatively
    demonstrate its performance on a test set.
'''
def task1():
    '''
    # Read in the files if needed, use pickles as needed for saving intermediate values, to prevent recomputations between runs:
    '''
    start = time.time()
    cprint(f"_________________________ Original RGB Images: _________________________","white")
    if (not os.path.exists("Pickle_Files/Original_Images/testing/")) or (not os.path.exists("Pickle_Files/Original_Images/training/")):
        rgb_train_images_with_classes, rgb_test_images_with_classes = read_images()
        save_to_pickle(rgb_train_images_with_classes, "Pickle_Files/Original_Images/training/", f"rgb_train_images_with_classes")        
        save_to_pickle(rgb_test_images_with_classes, "Pickle_Files/Original_Images/testing/", f"rgb_test_images_with_classes")
        cprint(f"Saved from pickle for Original RGB Image", "green")
    else:
        cprint(f"Loading from pickle for Original RGB Image", "white")
        rgb_train_images_with_classes = load_from_pickle(f"Pickle_Files/Original_Images/training/", "rgb_train_images_with_classes" )
        rgb_test_images_with_classes =  load_from_pickle(f"Pickle_Files/Original_Images/testing/", "rgb_test_images_with_classes" )

    cprint(f"TIME TAKEN FOR THE ORIGINAL IMAGES: {time.time()-start}","light_yellow") # DEBUG STATEMENT!!!

    cprint(f"_________________________ HSI and extracting HUE IMAGES: _________________________","white")
    hsi_train_images_with_classes, hsi_test_images_with_classes, hue_train, hue_test = list(), list(), list(), list() 
    
    start = time.time() # DEBUG STATEMENT!!!

    cprint(f"_________________________ RGB -> HSI for TRAINING: _________________________","white")
    # Training Dataset: RGB to HSI, extracing hue and saving to a file!!!:
    if (not os.path.exists("Pickle_Files/HSI_Images/training/hue_train_with_classes.pkl")):
        for i in tqdm(range(len(rgb_train_images_with_classes))):
            temp = convert_rgb_to_hsi(np.array(rgb_train_images_with_classes[i][0]))
            hsi_train_images_with_classes.append((temp,rgb_train_images_with_classes[i][1], rgb_train_images_with_classes[i][2]))
            temp = extract_hue(hsi_train_images_with_classes[i][0])
            hue_train.append((temp,rgb_train_images_with_classes[i][1], rgb_train_images_with_classes[i][2]))   
        save_to_pickle(hue_train, "Pickle_Files/HSI_Images/training/", f"hue_train_with_classes")     
    else: 
        hue_train = load_from_pickle("Pickle_Files/HSI_Images/training/", f"hue_train_with_classes")

    mid = time.time() # DEBUG STATEMENT!!!
    cprint(f"TIME TAKEN for GENERATING RGB to HSI for TRAINING: {time.time()-start}","light_yellow")
    
    cprint(f"_________________________ RGB -> HSI for TESTING: _________________________","white")
    # Testing Dataset: RGB to HSI, extracing hue and saving to a file!!!:
    if (not os.path.exists("Pickle_Files/HSI_Images/training/hue_test_with_classes.pkl")):
        for i in tqdm(range(len(rgb_test_images_with_classes))):
            temp = convert_rgb_to_hsi(np.array(rgb_test_images_with_classes[i][0]))
            hsi_test_images_with_classes.append((temp,rgb_test_images_with_classes[i][1], rgb_test_images_with_classes[i][2]))
            temp = extract_hue(hsi_test_images_with_classes[i][0])
            hue_test.append((temp,rgb_test_images_with_classes[i][1], rgb_test_images_with_classes[i][2])) 
        save_to_pickle(hue_test, "Pickle_Files/HSI_Images/testing/", f"hue_test_with_classes")
    else:
        hue_test = load_from_pickle("Pickle_Files/HSI_Images/testing/", f"hue_test_with_classes")
        
    cprint(f"TIME TAKEN for GENERATING RGB to HSI TESTING: {time.time()-mid}","light_yellow") # DEBUG STATEMENT!!!
    print()
    print()
    cprint(f"TIME TAKEN for GENERATING RGB to HSI TOTAL: {time.time()-start}","light_yellow") # DEBUG STATEMENT!!!
    cprint(f"{len(hue_train)}", "cyan")
    cprint(f"{len(hue_test)}", "cyan")
    cprint(f"YEAH THIS WORKED!!!","light_magenta")
   
    
    # NOTE FROM OFFICE HOURS: !!!
    # Save the intermediate results in pickle 
    # Einops library, or use eisum from numpy for Gbij = sum->  Abi(hw) -> Abj(hw) => Abij  

    
    
    
    
    
    
    
    # DEBUG STATEMENT !!! SECTION !!!    
    # Example: Save one Hue image for verification
    # Note: Hue values range from 0 to 360. To save as an image, scale to 0-255.
    if hue_train:
        sample_hue = hue_train[0][0]  # Get the first Hue channel
        # Scale Hue to [0, 255] for saving
        sample_hue_scaled = (sample_hue / 360.0 * 255).astype(np.uint8)
        cv2.imwrite("MyResults/test_scaled_hue_image.jpg", sample_hue_scaled)
        cv2.imwrite("MyResults/test_hue_image.jpg", hue_train[0][0])
        # cv2.imwrite("MyResults/test_HSI_image.jpg", sample_hue_scaled[0][0])
        cv2.imwrite("MyResults/test_hue_image.jpg", hsi_train_images_with_classes[0][0])
        cprint(f"Saved sample Hue image to MyResults/test_hue_image.jpg", "green")
    # DEBUG STATEMENT !!! SECTION !!! ^^^^^^^^^^^^^^^^    
    
    '''
    # Set up functions for LBP
        You have to use the Hue channel of the HSI images for this task. For visualization, you should plot the LBP histogram feature vector of at least one image,
        from each class.
    '''
     
    
    '''
    # Set up Gram Matrix based texture descriptors.
        Using the output feature maps of the three configurations (VGG19, ResNet-50 Coarse, and ResNet-50 Fine) you need to implement your own Gram matrix
        based descriptor extraction routines. For visualization, you should plot the 2D Gram Matrix for at least one image from each class for the three configurations.
        Comment on your Gram matrix visualizations. Are all feature channels strongly correlated?
    '''
    pass

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
    cprint("Make sure you are running from HW7 directory!!!\n", "red")
    ensure_directory(f"MyResults")
    # Execute the task based on the argument provided
    if args.task == 1:
        task1()
    elif args.task == 2:
        task2()