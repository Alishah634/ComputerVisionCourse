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

def time_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"\033[33m{func.__name__} took {elapsed_time:.4f} seconds to execute\033[0m")
        return result
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
    # data: The data to be saved (any Python object).
    # dir_path (str): The directory where the file will be saved.
    # file_name (str): The name of the pickle file (without extension).
    
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
    # dir_path (str): The directory where the file is located.
    # file_name (str): The name of the pickle file (without extension).
    # The loaded data, or None if the file does not exist.

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
        invalid_images = {"rain141.jpg", "shine131.jpg"}
        
        for image_name in sorted(os.listdir(dir_path)):
            # Skip non-image files like .DS_Store:
            if not any(image_name.lower().endswith(ext) for ext in valid_extensions):
                continue 
            if image_name in invalid_images:
                continue
            # image = io.imread(os.path.join(dir_path+image_name))
            image = cv2.imread(os.path.join(dir_path+image_name))
            # Add the classes, and image name to each respective image:
            class_name = next((weather for weather in ["cloudy", "rain", "shine", "sunrise"] if weather in image_name.lower()), None)
            if class_name:
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

'''START of function for Converting RGB -> HSI:'''
def convert_rgb_to_hsi(image: np.ndarray) -> np.ndarray: 
    # Ensure the image has 3 channels (convert grayscale to RGB if needed)
    if len(image.shape) == 2 or image.shape[-1] == 1:  # Grayscale image check
        cprint("Converting grayscale image to RGB", "yellow")
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[-1] != 3:
        raise ValueError(f"Unexpected number of channels in the image: {image.shape[-1]}")

    # Normalize the RGB values to [0, 1] range
    img = image.astype(np.float64) / 255.0
    b_channels, g_channels, r_channels = cv2.split(img)

    # Initialize the H, S, and I channels
    H = np.zeros_like(r_channels.copy(), dtype=np.float64)
    S = np.zeros_like(r_channels.copy(), dtype=np.float64)
    I = np.zeros_like(r_channels.copy(), dtype=np.float64)

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
    
    H[mask_r] = 60 * (((g_channels[mask_r] - b_channels[mask_r])/c[mask_r]) %6 )
    H[mask_g] = 60 * (((b_channels[mask_g] - r_channels[mask_g])/c[mask_g]) + 2)
    H[mask_b] = 60 * (((r_channels[mask_b] - g_channels[mask_b])/c[mask_b]) + 4)
    # Set H to 0 where c == 0 (grayscale pixels)
    H[c == 0] = 0.0

    # Ensure Hue is within [0, 360)
    H = np.mod(H, 360)
    # Compute S (Saturation)
    S = np.zeros_like(c, dtype=np.float64)
    # Where chroma is not zero, compute saturation
    S[c != 0] = 1 - (m[c != 0] / c[c != 0])
    
    # Stack the H, S, and I channels to create the HSI image:
    HSI = np.stack((H, S, I), axis=-1)
    # Returns: np.ndarray: HSI image with H in degrees [0, 360), S in [0, 1], and I in [0, 1].
    return HSI 

def extract_hue(hsi_iamge: np.ndarray) -> np.ndarray:
    return hsi_iamge[:, :, 0]

'''END of function for Converting to HSI:'''


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
    """
    Task 1: Convert RGB images to HSI, extract Hue channel, and save the data.
    """
    start_total = time.time()

    # Paths for RGB pickles
    rgb_training_pickle = "Pickle_Files/Original_Images/training/rgb_train_images_with_classes.pkl"
    rgb_testing_pickle = "Pickle_Files/Original_Images/testing/rgb_test_images_with_classes.pkl"
    # Check and load/save RGB images
    if not (os.path.exists(rgb_training_pickle) and os.path.exists(rgb_testing_pickle)):
        cprint("Pickle files for RGB images not found. Reading and saving images.", "yellow")
        rgb_train, rgb_test = read_images()
        save_to_pickle(rgb_train, "Pickle_Files/Original_Images/training/", "rgb_train_images_with_classes")        
        save_to_pickle(rgb_test, "Pickle_Files/Original_Images/testing/", "rgb_test_images_with_classes")
        cprint("Saved RGB images to pickle", "green")
    else:
        cprint("Loading RGB images from pickle", "white")
        rgb_train = load_from_pickle("Pickle_Files/Original_Images/training/", "rgb_train_images_with_classes")
        rgb_test = load_from_pickle("Pickle_Files/Original_Images/testing/", "rgb_test_images_with_classes")
        cprint("Loaded RGB images from pickle", "green")
    
    cprint(f"TIME TAKEN FOR THE ORIGINAL IMAGES: {time.time()-start_total:.2f} seconds", "light_yellow")
    
    # Function to convert and extract hue
    def process_images(image_list, desc, save_path):
        hue_data = []
        for img, class_name, name in tqdm(image_list, desc=desc):
            hsi_image = convert_rgb_to_hsi(img)
            hue_data.append((hsi_image, class_name, name))
        save_to_pickle(hue_data, save_path[0], save_path[1])
        return hue_data
    
    # Convert and save Hue channel for training
    # training_hue_pickle = "Pickle_Files/HSI_Images/training/hue_train_with_classes.pkl"
    training_hue_pickle = "Pickle_Files/HSI_Images/training/hsi_train_WORKING_images_with_classes.pkl"
    if not os.path.exists(training_hue_pickle):
        cprint("\nConverting Training Images to Hue...", "white")
        hue_train = process_images(rgb_train, "Converting Training Images to Hue", ("Pickle_Files/HSI_Images/training/", "hue_train_with_classes"))
    else:
        cprint("Loading Hue training images from pickle", "white")
        hue_train = load_from_pickle("Pickle_Files/HSI_Images/training/", "hue_train_with_classes")
    
    # Convert and save Hue channel for testing
    # testing_hue_pickle = "Pickle_Files/HSI_Images/testing/hue_test_with_classes.pkl"
    testing_hue_pickle = "Pickle_Files/HSI_Images/testing/hsi_test_WORKING_images_with_classes.pkl"
    if not os.path.exists(testing_hue_pickle):
        cprint("\nConverting Testing Images to Hue...", "white")
        hue_test = process_images(rgb_test, "Converting Testing Images to Hue", ("Pickle_Files/HSI_Images/testing/", "hue_test_with_classes"))
    else:
        cprint("Loading Hue testing images from pickle", "white")
        hue_test = load_from_pickle("Pickle_Files/HSI_Images/testing/", "hue_test_with_classes")
    cprint(f"TIME TAKEN for GENERATING RGB to HSI TOTAL: {time.time()-start_total:.2f} seconds", "light_yellow")
    # Debugging: Print lengths
    cprint(f"Number of Hue training images: {len(hue_train)}", "cyan")
    cprint(f"Number of Hue testing images: {len(hue_test)}", "cyan")
    cprint("YEAH THIS WORKED!!!", "light_magenta")
    
    '''
        you have to use the hue channel of the hsi images for this task. for visualization, you should plot the lbp histogram feature vector of at least one image,
        FROM EACH CLASS.
    '''
    
    '''
    # SET UP GRAM MATRIX BASED TEXTURE DESCRIPTORS.
        USING THE OUTPUT FEATURE MAPS OF THE THREE CONFIGURATIONS (VGG19, RESNET-50 COARSE, AND RESNET-50 FINE) YOU NEED TO IMPLEMENT YOUR OWN GRAM MATRIX
        based descriptor extraction routines. For visualization, you should plot the 2D Gram Matrix for at least one image from each class for the three configurations.
        Comment on your Gram matrix visualizations. Are all feature channels strongly correlated?
    '''
    # NOTE FROM OFFICE HOURS: !!!
    # Save the intermediate results in pickle 
    # Einops library, or use eisum from numpy for Gbij = sum->  Abi(hw) -> Abj(hw) => Abij  
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