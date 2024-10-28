import os
import time
import logging # For logging and decorators
import functools # For logging and decorators
from tqdm import tqdm # Progress bar
from typing import List, Tuple # Format typing
from argparse import ArgumentParser # Parsing arguments
from termcolor import cprint # Formatting prints
import pickle # For saving intermediate values to avoid recomputing
import math # 
import BitVector
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

def extract_hue(hsi_image: np.ndarray) -> np.ndarray:
    if hsi_image.ndim != 3 or hsi_image.shape[-1] != 3:
        raise ValueError(f"Invalid HSI image shape. Expected 3D HSI image with 3 channels, got shape {hsi_image.shape}")
    # Extract and return the Hue channel
    return hsi_image[:, :, 0]
'''END of function for Converting to HSI:'''

'''START of LBP Function:'''
def compute_lbp(hue_param: np.ndarray, P = 8, R = 1) -> np.ndarray:
    # cprint(f"SIZE OF hue_param: {hue_param.shape}")
    # image = cv2.cvtColor(hue_param, cv2.COLOR_BGR2GRAY)
    IMAGE_SIZE = 64
    image = hue_param[:,:,0]
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    cprint(f"SIZE OF image: {(len(image), len(image[0]))}")
    # image = cv2.resize(hue_param, (64, 64), interpolation=cv2.INTER_AREA)
    rowmax = IMAGE_SIZE - R
    colmax = IMAGE_SIZE - R
    lbp_hist = {t:0 for t in range(P+2)} #(C6)
    for i in range(R,rowmax): #(C7)
        for j in range(R,colmax): #(C8)
            pattern = [] #(C10)
        for p in range(P): #(C11)
            # We use the index k to point straight down and l to point to the
            # right in a circular neighborhood around the point (i,j). And we
            # use (del_k, del_l) as the offset from (i,j) to the point on the
            # R-radius circle as p varies.
            del_k,del_l = R*math.cos(2*math.pi*p/P), R*math.sin(2*math.pi*p/P) #(C12)
            if abs(del_k) < 0.001: del_k = 0.0 #(C13)
            if abs(del_l) < 0.001: del_l = 0.0 #(C14)
            k, l = i + del_k, j + del_l #(C15)
            k_base,l_base = int(k),int(l) #(C16)
            delta_k,delta_l = k-k_base,l-l_base #(C17)
            if (delta_k < 0.001) and (delta_l < 0.001): #(C18)
                image_val_at_p = float(image[k_base][l_base]) #(C19)
            elif (delta_l < 0.001): #(C20)
                image_val_at_p = (1 - delta_k) * image[k_base][l_base] + \
                delta_k * image[k_base+1][l_base] #(C21)
            elif (delta_k < 0.001): #(C22)
                image_val_at_p = (1 - delta_l) * image[k_base][l_base] + \
                delta_l * image[k_base][l_base+1] #(C23)
            else: #(C24)
                image_val_at_p = (1-delta_k)*(1-delta_l)*image[k_base][l_base] + \
                (1-delta_k)*delta_l*image[k_base][l_base+1] + \
                delta_k*delta_l*image[k_base+1][l_base+1] + \
                delta_k*(1-delta_l)*image[k_base+1][l_base] #(C25)
            if image_val_at_p >= image[i][j]: #(C26)
                pattern.append(1) #(C27)
            else: #(C28)
                pattern.append(0) #(C29)
        # print("pattern: %s" % pattern) #(C30)
        bv = BitVector.BitVector( bitlist = pattern ) #(C31)
        intvals_for_circular_shifts = [int(bv << 1) for _ in range(P)] #(C32)
        minbv = BitVector.BitVector( intVal = min(intvals_for_circular_shifts), size = P ) #(C33)
        # print("minbv: %s" % minbv) #(C34)
        bvruns = minbv.runs() #(C35)
        encoding = None
        if len(bvruns) > 2: #(C36)
            lbp_hist[P+1] += 1 #(C37)
            encoding = P+1 #(C38)
        elif len(bvruns) == 1 and bvruns[0][0] == '1': #(C39)
            lbp_hist[P] += 1 #(C40)
            encoding = P #(C41)
        elif len(bvruns) == 1 and bvruns[0][0] == '0': #(C42)
            lbp_hist[0] += 1 #(C43)
            encoding = 0 #(C44)
        else: #(C45)
            lbp_hist[len(bvruns[1])] += 1 #(C46)
            encoding = len(bvruns[1]) #(C47)
        # print("encoding: %s" % encoding) #(C48)
    # print("\nLBP Histogram: %s" % lbp_hist)
    return list(lbp_hist.values())


def lbp_plotter(data, hue_data, class_labels, image_dir, dataset_dir,  num_classes=4):
    """
    Args/Params:
        data (List[List[int]]): List of LBP histograms for images.
        hue_data (List[Tuple[np.ndarray, str, str]]): List of tuples with HSI images, class names, and image filenames.
        class_labels (List[str]): List of class label names (e.g., ["cloudy", "rain", "shine", "sunrise"]).
        image_dir (str): Directory path where image files are stored.
        num_classes (int): Number of classes to visualize (default is 4).
    """
    # Initialize figure with appropriate size
    fig, ax = plt.subplots(2, num_classes, figsize=(6 * num_classes, 12))

    for class_idx, class_name in enumerate(class_labels):
        # Find the first image index for the current class
        class_indices = [i for i, (_, lbl, _) in enumerate(hue_data) if lbl.lower() == class_name.lower()]
        if not class_indices:
            cprint(f"No images found for class '{class_name}'", "yellow")
            # Optionally, hide the subplot if no image is found
            ax[0, class_idx].axis('off')
            ax[1, class_idx].axis('off')
            continue
        img_index = class_indices[0]

        # Plot LBP Histogram
        bins = np.arange(len(data[img_index]))
        ax[0, class_idx].bar(bins, data[img_index], width=1, edgecolor='k')
        ax[0, class_idx].set_title(f"LBP Histogram: {class_name}")
        ax[0, class_idx].set_xlabel("LBP Code")
        ax[0, class_idx].set_ylabel("Frequency")

        # Read and plot the image
        image_filename = hue_data[img_index][2]
        image_path = os.path.join(image_dir, image_filename)

        # Check if the image path exists
        if not os.path.isfile(image_path):
            cprint(f"Warning: Image file '{image_path}' does not exist.", "red")
            ax[1, class_idx].set_title(f"Missing: {image_filename}")
            ax[1, class_idx].axis('off')
            continue

        img = cv2.imread(image_path)
        if img is not None:
            # Convert BGR (OpenCV default) to RGB for correct color display in Matplotlib
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax[1, class_idx].imshow(img_rgb)
            ax[1, class_idx].set_title(image_filename)
        else:
            cprint(f"Warning: Unable to read image '{image_path}'.", "red")
            ax[1, class_idx].set_title(f"Unreadable: {image_filename}")

        # Hide axis for image plots
        ax[1, class_idx].axis('off')

    plt.tight_layout()
    # plt.show()
    ensure_directory("MyResults/LPB_Hist")
    plt.savefig(f"MyResults/LPB_Hist/lbp_{dataset_dir}_histogram.jpg")

    
'''END of LBP Function:'''

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
    """ ========================================================LBP DESCRIPTORS: ========================================================"""
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
        hue_train = load_from_pickle("Pickle_Files/HSI_Images/training/", "hsi_train_WORKING_images_with_classes")
    
    # Convert and save Hue channel for testing
    # testing_hue_pickle = "Pickle_Files/HSI_Images/testing/hue_test_with_classes.pkl"
    testing_hue_pickle = "Pickle_Files/HSI_Images/testing/hsi_test_WORKING_images_with_classes.pkl"
    if not os.path.exists(testing_hue_pickle):
        cprint("\nConverting Testing Images to Hue...", "white")
        hue_test = process_images(rgb_test, "Converting Testing Images to Hue", ("Pickle_Files/HSI_Images/testing/", "hue_test_with_classes"))
    else:
        cprint("Loading Hue testing images from pickle", "white")
        hue_test = load_from_pickle("Pickle_Files/HSI_Images/testing/", "hsi_test_WORKING_images_with_classes")
    
    cprint(f"TIME TAKEN for GENERATING RGB to HSI TOTAL: {time.time()-start_total:.2f} seconds", "light_yellow")
   
    # Debugging: Print lengths
    cprint(f"Number of Hue training images: {len(hue_train)}", "cyan")
    cprint(f"Number of Hue testing images: {len(hue_test)}", "cyan")
    cprint("YEAH THIS WORKED!!!", "light_magenta")
    cprint(f"HUE TEST: {len(hue_test)}","cyan")
    
    '''
        you have to use the hue channel of the hsi images for this task. for visualization, you should plot the lbp histogram feature vector of at least one image,
        from each class.
    '''
    ensure_directory("Pickle_Files/LBP")

    MAJOR_PATH = "HW7-Auxilliary/HW7-Auxilliary/data/"
    ################################### TRAINING ############################################################
    # Define the path to the training images
    training_hue_pickle = "Pickle_Files/LBP/lbp_train_descriptors.pkl"
    training_image_dir = os.path.join(MAJOR_PATH, "training")
    if not os.path.exists(training_hue_pickle):
        cprint("\nCalcualting the TRAIN LBP Descriptors...", "white")
        lbp_descriptors = [compute_lbp(hue_image) for hue_image, _, _ in hue_train]
        # Plot LBP histograms and corresponding images
        lbp_plotter(
            data=lbp_descriptors,
            hue_data=hue_train,
            class_labels=["cloudy", "rain", "shine", "sunrise"],
            image_dir=training_image_dir,
            dataset_dir = "training",
            num_classes=4
        )
        save_to_pickle(lbp_descriptors, "Pickle_Files/LBP/", "lbp_train_descriptors")
        cprint(f"FINISHED LBP TRAINING DESCRIPTORS","green")  
    else:
        cprint("Loading Hue training descriptors from pickle", "white")
        lbp_descriptors= load_from_pickle("Pickle_Files/LBP/", "lbp_train_descriptors")
        lbp_plotter(
            data=lbp_descriptors,
            hue_data=hue_train,
            class_labels=["cloudy", "rain", "shine", "sunrise"],
            image_dir=training_image_dir,
            dataset_dir = "verify_train",
            num_classes=4
        )
    ################################### TESTING############################################################
    # Define the path to the testing images
    testing_hue_pickle = "Pickle_Files/LBP/lbp_test_descriptors.pkl"
    testing_image_dir = os.path.join(MAJOR_PATH, "testing")
    if not os.path.exists(testing_hue_pickle):
        cprint("\nCalcualting the Test LBP Descriptors...", "white")
        save_to_pickle(lbp_descriptors, "Pickle_Files/LBP/","lbp_test_descriptors")
        lbp_descriptors = [compute_lbp(hue_image) for hue_image, _, _ in hue_test]
        save_to_pickle(lbp_descriptors, "Pickle_Files/LBP/","lbp_test_descriptors")
        # Plot LBP histograms and corresponding images
        lbp_plotter(
            data=lbp_descriptors,
            hue_data=hue_test,
            class_labels=["cloudy", "rain", "shine", "sunrise"],
            image_dir=testing_image_dir,
            dataset_dir = "testing",
            num_classes=4
        )
        cprint(f"FINISHED LBP TESTING DESCRIPTORS","green") 
    else:
        cprint("Loading Hue testing descriptors from pickle", "white")
        lbp_descriptors = load_from_pickle("Pickle_Files/LBP", "lbp_test_descriptors")
        lbp_plotter(
            data=lbp_descriptors,
            hue_data=hue_test,
            class_labels=["cloudy", "rain", "shine", "sunrise"],
            image_dir=testing_image_dir,
            dataset_dir = "verify_test",
            num_classes=4
        )
    
    """ ========================================================VGG DESCRIPTORS: ========================================================"""
    


    '''
    # set up gram matrix based texture descriptors.
        using the output feature maps of the three configurations (vgg19, resnet-50 coarse, and resnet-50 fine) you need to implement your own gram matrix
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