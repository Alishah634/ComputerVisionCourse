"""
1 Preprocess all training and testing images by resizing them to (256, 256) for the sake of computational
efficiency. Also you should make sure they all have three channels i.e. RGB.

2. Extract the feature vectors for all images. Together they form a feature matrix of size (S, C), where
S is the number of samples/images and C is the dimension of a feature vector. Do this for all types
of descriptor and both the training and testing images.

3. Train a SVM multi-class classifier to fit the training data. You are free to use open-source imple-
mentations from OpenCV or scikit-learn.

4. Apply your trained classifier on the testing feature matrix. Experiment with different choices of
parameters (e.g. R and P in LBP). Record your best classification accuracy as well as the full
confusion matrix.

5. Finally, for each trained classifier, display at least one correctly classified and one mis-classified image.
Make sure that you indicate the predicted and ground-truth class labels (Cloudy/Rain/Shine/Sun-
rise) for each image


"""

import os
import sys
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
from skimage.feature import local_binary_pattern
from skimage import transform
from scipy.ndimage import convolve
from scipy.optimize import least_squares
from sklearn import svm
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler


# Import the models:
# Add the directory containing the target module to sys.path
sys.path.append(os.path.abspath("HW7-Auxilliary/HW7-Auxilliary"))
from vgg_and_resnet import VGG19, CustomResNet
MODEL_PATH = "HW7-Auxilliary/HW7-Auxilliary/vgg_normalized.pth"

# PATHS FOR THE DATASETS:
MAJOR_PATH = "HW7-Auxilliary/HW7-Auxilliary/data/"
TEST_PATH = MAJOR_PATH + "testing/"
TRAIN_PATH = MAJOR_PATH + "training/"

# Paths for RGB numpy files
rgb_training_numpy = "Numpy_Files/Original_Images/training/rgb_train_images_with_classes.npy"
rgb_testing_numpy = "Numpy_Files/Original_Images/testing/rgb_test_images_with_classes.npy"

accuracy_dict = dict()

def pre_process():
    # Resize the image to (256,256)
    pass

def extract_feature_vectors(images_and_classes, model, isRESNET):
        feature_model = model()
        feature_model.load_weights(MODEL_PATH)    
        combined_features = []
        if not isRESNET:
            for img, class_name, file_name in tqdm(images_and_classes):
                img_resized = transform.resize(img, (256, 256))
                feature = feature_model(img_resized)
                combined_features.append((feature, class_name, file_name))
        return combined_features


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
        logger.info(f"\033[33m{func.__name__} took {elapsed_time:.4f} seconds to execute\n\033[0m")
        return result
    return wrapper

# Create a folder if it does not exist
def ensure_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        

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
def pre_process(image: np.ndarray) -> np.ndarray:
    """
    Preprocess an image by resizing to (256, 256) and ensuring it has three channels (RGB).

    Parameters:
        image (np.ndarray): Input image to preprocess.

    Returns:
        np.ndarray: Preprocessed image of size (256, 256) with 3 channels.
    """
    # Resize the image to (256, 256)
    resized_img = transform.resize(image, (256, 256), mode='reflect', anti_aliasing=True)

    # Ensure image has three channels
    if len(resized_img.shape) == 2:  # Grayscale image
        resized_img = np.stack((resized_img,) * 3, axis=-1)
    elif resized_img.shape[-1] == 1:  # One-channel image
        resized_img = np.concatenate([resized_img] * 3, axis=-1)
    return resized_img

def read_images() -> Tuple[np.ndarray, np.ndarray]:
    """
    Reads, preprocesses, and returns training and testing images with class labels.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Preprocessed training and testing images with class labels.
    """
    def read_images_from_dir(directory: str) -> List[Tuple[np.ndarray, str, str]]:
        image_list_with_classes = []
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
        invalid_images = {"rain141.jpg", "shine131.jpg"}
        for image_name in tqdm(sorted(os.listdir(directory))):
            if not any(image_name.lower().endswith(ext) for ext in valid_extensions) or image_name in invalid_images:
                continue
            image = io.imread(os.path.join(directory, image_name))
            # Ensure the image has three channels, skip if not
            if len(image.shape) < 3 or image.shape[-1] != 3:
                cprint(f"Skipping image '{image_name}' with unexpected number of channels: {image.shape[-1]}", "red")
                continue
            
            image = pre_process(image)
            class_name = next((weather for weather in ["cloudy", "rain", "shine", "sunrise"] if weather in image_name.lower()), None)
            if class_name:
                image_list_with_classes.append((image, class_name, image_name))
        # Returns List[Tuple[np.ndarray, str, str]]: List of preprocessed images with class labels and filenames.
        return image_list_with_classes
    # Ensure the necessary directories exist
    ensure_directory("Numpy_Files/Original_Images/training/")
    ensure_directory("Numpy_Files/Original_Images/testing/")
    # Check and load/save RGB images using NumPy
    if not (os.path.exists(rgb_training_numpy) and os.path.exists(rgb_testing_numpy)):
        cprint("Numpy files for RGB images not found. Reading and saving images.", "yellow")
        train_images_with_classes = read_images_from_dir(TRAIN_PATH)
        test_images_with_classes = read_images_from_dir(TEST_PATH)

        # Convert lists to NumPy arrays for easier saving/loading with NumPy
        np.save(rgb_training_numpy, np.array(train_images_with_classes, dtype=object))
        np.save(rgb_testing_numpy, np.array(test_images_with_classes, dtype=object))
        cprint("SUCCESS! Saved RGB images to numpy files\n", "green")
    else:
        cprint("Loading RGB images from numpy files..", "white")
        train_images_with_classes = np.load(rgb_training_numpy, allow_pickle=True)
        test_images_with_classes = np.load(rgb_testing_numpy, allow_pickle=True)
        cprint("SUCCESS! Loaded RGB images from numpy files\n", "green")
    return train_images_with_classes, test_images_with_classes


''' CONVERT TO RGB to HSI FORMAT:'''
""" ========================================================CONVERTING RGB to HSI Format: ========================================================"""

def save_to_npz_compressed(data, dir_path: str, file_name: str) -> None:
    """
    Save data using NumPy's compressed .npz format.
    Separates images, class names, and filenames into different arrays for saving.

    Parameters:
        data (list): List of data tuples (e.g., images, class names, filenames).
        dir_path (str): Directory path to save the file.
        file_name (str): Name of the file without extension.
    """
    ensure_directory(dir_path)
    file_path = os.path.join(dir_path, f"{file_name}.npz")

    # Extract the separate components from the data list
    images = [item[0] for item in data]
    class_names = [item[1] for item in data]
    file_names = [item[2] for item in data]

    # Save each component separately using np.savez_compressed
    np.savez_compressed(file_path, images=images, class_names=class_names, file_names=file_names)
    cprint(f"Data saved to {file_path}", "green")

def load_from_npz_compressed(dir_path: str, file_name: str):
    """
    Load data from a compressed .npz file and reconstruct the original list of tuples.

    Parameters:
        dir_path (str): Directory path where the file is stored.
        file_name (str): Name of the file without extension.

    Returns:
        List: Reconstructed data list of tuples (histogram, class_name, filename).
    """
    file_path = os.path.join(dir_path, f"{file_name}.npz")
    if not os.path.exists(file_path):
        cprint(f"File not found: {file_path}", "red")
        return None

    # Load the compressed file and reconstruct the data list
    loaded_data = np.load(file_path, allow_pickle=True)
    
    # Check if it's LBP feature loading based on the keys available
    if 'histograms' in loaded_data:
        histograms = loaded_data['histograms']
        class_names = loaded_data['class_names']
        file_names = loaded_data['file_names']
        data_list = list(zip(histograms, class_names, file_names))
    else:
        images = loaded_data['images']
        class_names = loaded_data['class_names']
        file_names = loaded_data['file_names']
        data_list = list(zip(images, class_names, file_names))

    return data_list

def process_images(image_list, desc, save_path):
    """
    Convert and save images to HSI format, extracting and storing the Hue channel.

    Parameters:
        image_list (list): List of images with class names and filenames.
        desc (str): Description for progress tracking.
        save_path (tuple): Tuple containing directory path and file name for saving.
    """
    hue_data = []
    for img, class_name, name in tqdm(image_list, desc=desc):
        hsi_image = convert_rgb_to_hsi(img)
        hue_data.append((hsi_image, class_name, name))
    save_to_npz_compressed(hue_data, save_path[0], save_path[1])
    return hue_data

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
    # THIS IS supposed to be 1 - m/I not m/c
    # S[c != 0] = 1 - (m[c != 0] / c[c != 0])
    S[c != 0] = 1 - (m/I)
    
    # Stack the H, S, and I channels to create the HSI image:
    HSI = np.stack((H, S, I), axis=-1)
    # Returns: np.ndarray: HSI image with H in degrees [0, 360), S in [0, 1], and I in [0, 1].
    return HSI

@time_decorator
def convert_rgb_to_hsi_format(rgb_train, rgb_test):
    """
    Convert and save RGB images to HSI format, extracting the Hue channel.
    """
    # Define file paths for the compressed .npz files
    training_hue_npz = "Numpy_Files/HSI_Images/training/hsi_train_with_classes"
    testing_hue_npz = "Numpy_Files/HSI_Images/testing/hsi_test_with_classes"

    # Ensure directories exist
    ensure_directory("Numpy_Files/HSI_Images/training/")
    ensure_directory("Numpy_Files/HSI_Images/testing/")

    # Convert and save Hue channel for training images
    if not os.path.exists(f"{training_hue_npz}.npz"):
        cprint("\nConverting Training Images to Hue...", "white")
        hue_train = process_images(rgb_train, "Converting Training Images to Hue", ("Numpy_Files/HSI_Images/training", "hsi_train_with_classes"))
        cprint("SUCCESS! Saved HUE training images from numpy files\n", "green")
    else:
        cprint("Loading Hue training images from npz file...", "white")
        hue_train = load_from_npz_compressed("Numpy_Files/HSI_Images/training", "hsi_train_with_classes")
        cprint("SUCCESS! Loaded HUE training images from numpy files\n", "green")

    # Convert and save Hue channel for testing images
    if not os.path.exists(f"{testing_hue_npz}.npz"):
        cprint("\nConverting Testing Images to Hue...", "white")
        hue_test = process_images(rgb_test, "Converting Testing Images to Hue", ("Numpy_Files/HSI_Images/testing", "hsi_test_with_classes"))
        cprint("SUCCESS! Saved HUE testing images from numpy files\n", "green")
    else:
        cprint("Loading Hue testing images from npz file..", "white")
        hue_test = load_from_npz_compressed("Numpy_Files/HSI_Images/testing", "hsi_test_with_classes")
        cprint("SUCCESS! Loaded HUE testing images from numpy files\n", "green")

    cprint(f"Length of HUE training and testing images respectively: {len(hue_train), len(hue_test)}", "cyan")
    return hue_train, hue_test

'''END of function for Converting RGB -> HSI:'''

'''START OF LBP FUNCTIONS'''
def extract_lbp_features(image_list, P=8, R=1, image_size=(64, 64)):
    """
    Extracts LBP histograms as feature vectors from the Hue channel of images.

    Parameters:
        image_list (list): List of tuples containing (HSI image, class name, file name).
        P (int): Number of circularly symmetric neighbor set points.
        R (int): Radius of circle for LBP calculation.
        image_size (tuple): Size to which the images will be resized (default is (64, 64)).

    Returns:
        lbp_feature_list (list): List of tuples containing (LBP histogram, class name, file name).
    """
    lbp_feature_list = []
    for hsi_image, class_name, file_name in tqdm(image_list, desc="Extracting LBP Features"):
        # Extract Hue channel
        hue_channel = hsi_image[:, :, 0]

        # Downsize the Hue channel image to (64, 64)
        resized_hue = transform.resize(hue_channel, image_size, anti_aliasing=True)

        # Calculate LBP histogram using the compute_lbp function
        lbp_histogram = compute_lbp_histogram(resized_hue, P, R)

        # Ensure the histogram is flattened and consistent
        lbp_feature_list.append((lbp_histogram.flatten(), class_name, file_name))
    
    return lbp_feature_list


def compute_lbp(image, P=8, R=1):
    """
    Computes the LBP histogram for a given image based on the specified number of points and radius.

    Parameters:
        image (np.ndarray): Input image, assumed to be in grayscale or a single-channel form.
        P (int): Number of circularly symmetric neighbor set points.
        R (int): Radius of circle for LBP calculation.

    Returns:
        List[int]: LBP histogram counts as a list.
    """
    IMAGE_SIZE = 64
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))

    rowmax = IMAGE_SIZE - R
    colmax = IMAGE_SIZE - R
    lbp_hist = {t: 0 for t in range(P + 2)}

    for i in range(R, rowmax):
        for j in range(R, colmax):
            pattern = []
            for p in range(P):
                del_k, del_l = R * math.cos(2 * math.pi * p / P), R * math.sin(2 * math.pi * p / P)
                k, l = i + del_k, j + del_l
                k_base, l_base = int(k), int(l)
                delta_k, delta_l = k - k_base, l - l_base

                if delta_k < 0.001 and delta_l < 0.001:
                    image_val_at_p = float(image[k_base][l_base])
                elif delta_l < 0.001:
                    image_val_at_p = (1 - delta_k) * image[k_base][l_base] + delta_k * image[k_base + 1][l_base]
                elif delta_k < 0.001:
                    image_val_at_p = (1 - delta_l) * image[k_base][l_base] + delta_l * image[k_base][l_base + 1]
                else:
                    image_val_at_p = (1 - delta_k) * (1 - delta_l) * image[k_base][l_base] + \
                                     (1 - delta_k) * delta_l * image[k_base][l_base + 1] + \
                                     delta_k * delta_l * image[k_base + 1][l_base + 1] + \
                                     delta_k * (1 - delta_l) * image[k_base + 1][l_base]

                if image_val_at_p >= image[i][j]:
                    pattern.append(1)
                else:
                    pattern.append(0)

            bv = BitVector.BitVector(bitlist=pattern)
            intvals_for_circular_shifts = [int(bv << 1) for _ in range(P)]
            minbv = BitVector.BitVector(intVal=min(intvals_for_circular_shifts), size=P)
            bvruns = minbv.runs()

            if len(bvruns) > 2:
                lbp_hist[P + 1] += 1
            elif len(bvruns) == 1 and bvruns[0][0] == '1':
                lbp_hist[P] += 1
            elif len(bvruns) == 1 and bvruns[0][0] == '0':
                lbp_hist[0] += 1
            else:
                lbp_hist[len(bvruns[1])] += 1

    return list(lbp_hist.values())

image_index = 0
def compute_lbp_histogram(image, P, R):
    """
    Computes the LBP histogram for a given grayscale image.

    Parameters:
        image (np.ndarray): Input grayscale image.
        P (int): Number of circularly symmetric neighbor set points.
        R (int): Radius of circle for LBP calculation.

    Returns:
        lbp_hist (np.ndarray): LBP histogram as a feature vector.
    """
    # Ensure image size is (64, 64)
    image = transform.resize(image, (64, 64), anti_aliasing=True)

    # Calculate LBP for the image using skimage's LBP function
    lbp = local_binary_pattern(image, P, R, method="uniform")
    
    # Calculate the histogram of LBP values
    n_bins = int(lbp.max() + 1)
    
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)

    return lbp_hist

def save_lbp_features(lbp_features, dir_path: str, file_name: str):
    """
    Save LBP features using NumPy's compressed .npz format.

    Parameters:
        lbp_features (list): List of tuples containing (LBP histogram, class name, file name).
        dir_path (str): Directory path to save the file.
        file_name (str): Name of the file without extension.
    """
    ensure_directory(dir_path)
    file_path = os.path.join(dir_path, f"{file_name}.npz")

    # Separate the LBP histograms, class names, and file names
    histograms = [feature[0] for feature in lbp_features]
    class_names = [feature[1] for feature in lbp_features]
    file_names = [feature[2] for feature in lbp_features]

    # Save histograms, class names, and file names separately within the .npz file
    np.savez_compressed(file_path, histograms=histograms, class_names=class_names, file_names=file_names)
    cprint(f"LBP features saved to {file_path}", "green")


def load_lbp_features(save_dir, file_name):
    """
    Load LBP features from a compressed .npz file.

    Parameters:
        save_dir (str): Directory path where the file is stored.
        file_name (str): Name of the file without extension.

    Returns:
        List: Loaded LBP feature list.
    """
    file_path = os.path.join(save_dir, f"{file_name}.npz")
    if not os.path.exists(file_path):
        cprint(f"File not found: {file_path}", "red")
        return None

    loaded_data = np.load(file_path, allow_pickle=True)
    return loaded_data['data'].tolist()

@time_decorator
def extract_and_save_lbp(rgb_train, rgb_test):
    """
    Extract, save, and load LBP features for training and testing datasets.

    Parameters:
        rgb_train (list): Training images with classes.
        rgb_test (list): Testing images with classes.

    Returns:
        Tuple: LBP features for training and testing datasets.
    """
    # Define file paths for the LBP features
    training_lbp_npz = "Numpy_Files/LBP_Images/training/lbp_train_features"
    testing_lbp_npz = "Numpy_Files/LBP_Images/testing/lbp_test_features"

    # Ensure the necessary directories exist
    ensure_directory("Numpy_Files/LBP_Images/training/")
    ensure_directory("Numpy_Files/LBP_Images/testing/")

    # Check if the LBP features for training data exist, otherwise extract and save them
    if not os.path.exists(f"{training_lbp_npz}.npz"):
        cprint("Extracting LBP features for training data...", "white")
        lbp_train_features = extract_lbp_features(rgb_train)
        save_lbp_features(lbp_train_features, "Numpy_Files/LBP_Images/training", "lbp_train_features")
    else:
        cprint("Loading LBP features for training data from npz file...", "white")
        lbp_train_features = load_from_npz_compressed("Numpy_Files/LBP_Images/training", "lbp_train_features")

    # Check if the LBP features for testing data exist, otherwise extract and save them
    if not os.path.exists(f"{testing_lbp_npz}.npz"):
        cprint("Extracting LBP features for testing data...", "white")
        lbp_test_features = extract_lbp_features(rgb_test)
        save_lbp_features(lbp_test_features, "Numpy_Files/LBP_Images/testing", "lbp_test_features")
    else:
        cprint("Loading LBP features for testing data from npz file...", "white")
        lbp_test_features = load_from_npz_compressed("Numpy_Files/LBP_Images/testing", "lbp_test_features")

    return lbp_train_features, lbp_test_features


'''END OF LBP FUNCTIONS'''


'''Gram Matrix Functions, Calculation and plotting:'''
def plot_gram_matrix(gram_matrices, labels, classes_to_plot, feature_type="VGG"):
    """
    Plot the 2D Gram Matrix for at least one image from each class.
    
    Parameters:
        gram_matrices (list of np.ndarray): List of flattened Gram matrices.
        labels (list of str): Corresponding class labels for each Gram matrix.
        classes_to_plot (list of str): List of class names to plot.
        feature_type (str): Type of feature descriptor (e.g., "VGG").
    """
    # Ensure the results directory exists
    ensure_directory(f"MyResults/Gram_Matrix/{feature_type}/")
    
    for class_name in classes_to_plot:
        # Find the first instance of a given class
        for idx, (gram_matrix, label) in enumerate(zip(gram_matrices, labels)):
            if label == class_name:
                # Reshape the Gram matrix to its original 2D form
                size = int(np.sqrt(gram_matrix.shape[0] * 2))  # Estimate size of the 2D matrix based on upper triangle size
                full_matrix = np.zeros((size, size))
                upper_indices = np.triu_indices(size)
                full_matrix[upper_indices] = gram_matrix
                full_matrix += full_matrix.T - np.diag(full_matrix.diagonal())

                # Plot the Gram matrix as a heatmap
                plt.figure(figsize=(8, 6))
                # plt.title(f"For {feature_type}: 2D Gram Matrix for class: {class_name}")
                plt.title(f"For {feature_type}:\n2D Plot of Gram Matrix for Class: {class_name}")
                plt.imshow(full_matrix, cmap='viridis')
                plt.colorbar(label="Correlation Intensity")
                plt.savefig(f"MyResults/Gram_Matrix/{feature_type}/{class_name}_gram_matrix.jpg")
                plt.show()
                break
            
def calc_gram_matrix(features):
    features = features.reshape(features.shape[0], -1)
    gram_matrix = np.matmul(features, features.T)
    # Consider the size of the Gram matrix based on feature dimensions
    gram_matrix = gram_matrix[:32, :32]
    # Flatten the upper triangular part of the Gram matrix for consistency
    gram = gram_matrix[np.triu_indices(32)]
    return gram

'''START Of VGG Feature Extraction:'''
""" ========================================================VGG DESCRIPTORS: ========================================================"""
@time_decorator
def get_VGG_features(images_and_classes):
    vgg_model = VGG19()
    vgg_model.load_weights(MODEL_PATH)
    combined_features = []
    for img, class_name, file_name in tqdm(images_and_classes):
        img_resized = transform.resize(img, (256, 256))
        feature = vgg_model(img_resized)
        gram_matrix = calc_gram_matrix(features=feature)  
        combined_features.append((gram_matrix, class_name, file_name))
    return combined_features  

@time_decorator
def calc_VGG_descriptors(rgb_train, rgb_test) -> np.ndarray: 
    ensure_directory("Pickle_Files/VGG/")
    # Process Test Data
    if not os.path.exists("Pickle_Files/VGG/vgg_test_features.pkl") or not os.path.exists("Pickle_Files/VGG/vgg_test_gram_matrix.pkl"):
        vgg_test_features = get_VGG_features(rgb_test)
        vgg_test_gram_matrix = [item[0] for item in vgg_test_features]  # Extract only Gram matrices
        save_to_pickle(vgg_test_features, "Pickle_Files/VGG", "vgg_test_features")
        save_to_pickle(vgg_test_gram_matrix, "Pickle_Files/VGG", "vgg_test_gram_matrix")
    else: 
        vgg_test_features = load_from_pickle("Pickle_Files/VGG", "vgg_test_features")
        vgg_test_gram_matrix = load_from_pickle("Pickle_Files/VGG", "vgg_test_gram_matrix")
    
    # Process Training Data
    if not os.path.exists("Pickle_Files/VGG/vgg_train_features.pkl") or not os.path.exists("Pickle_Files/VGG/vgg_train_gram_matrix.pkl"):
        vgg_train_features = get_VGG_features(rgb_train)
        vgg_train_gram_matrix = [item[0] for item in vgg_train_features]  # Extract only Gram matrices
        save_to_pickle(vgg_train_features, "Pickle_Files/VGG", "vgg_train_features")
        save_to_pickle(vgg_train_gram_matrix, "Pickle_Files/VGG", "vgg_train_gram_matrix")
    else: 
        vgg_train_features = load_from_pickle("Pickle_Files/VGG", "vgg_train_features")
        vgg_train_gram_matrix = load_from_pickle("Pickle_Files/VGG", "vgg_train_gram_matrix")
    
    return vgg_train_gram_matrix, vgg_test_gram_matrix

    
'''END Of VGG Feature Extraction'''

'''START Of RESNET Feature Extraction:'''
def get_RESNET_features(rgb_test):
    resnet_model = CustomResNet(encoder="resnet50")
    combined_coarse = list()
    combined_fine = list()
    combined_coarse_gram = list()
    combined_fine_gram = list()
    for image, class_name, file_name in rgb_test:
        image = transform.resize(image, (256,256))
        coarse_features, fine_features = resnet_model(image)
        gram_matrix_coarse = calc_gram_matrix(coarse_features)
        gram_matrix_fine  = calc_gram_matrix(fine_features)
        # Store the features and gram matricies for the model
        combined_coarse.append(coarse_features)
        combined_fine.append(fine_features)   
        combined_coarse_gram.append(gram_matrix_coarse)
        combined_fine_gram.append(gram_matrix_fine)
    return combined_coarse, combined_coarse_gram, combined_fine, combined_fine_gram

def calc_RESNET_descriptors(rgb_train, rgb_test, isCoarse = True):
    ensure_directory("Pickle_Files/RESNET_Coarse/")
    ensure_directory("Pickle_Files/RESNET_Fine/")
    if not os.path.exists("Pickle_Files/RESNET_Coarse/RESNET_Coarse_test_features.pkl") or not os.path.exists("Pickle_Files/RESNET_Coarse/RESNET_Course_test_gram_matrix.pkl") or \
        not os.path.exists("Pickle_Files/RESNET_Fine/RESNET_Fine_test_features.pkl") or not os.path.exists("Pickle_Files/RESNET_Fine/RESNET_Fine_test_gram_matrix.pkl"):
        resnet_coarse_test_features, resnet_coarse_test_gram_matrix, resnet_fine_test_features, resnet_fine_test_gram_matrix = get_RESNET_features(rgb_test)
        save_to_pickle(resnet_coarse_test_features, "Pickle_Files/RESNET_Coarse/", "RESNET_Coarse_test_features")
        save_to_pickle(resnet_coarse_test_gram_matrix, "Pickle_Files/RESNET_Coarse/", "RESNET_Course_test_gram_matrix")
        save_to_pickle(resnet_fine_test_features, "Pickle_Files/RESNET_Fine/", "RESNET_Fine_test_features")
        save_to_pickle(resnet_fine_test_gram_matrix, "Pickle_Files/RESNET_Fine/", "RESNET_Fine_test_gram_matrix")
    else: 
        resnet_coarse_test_features = load_from_pickle("Pickle_Files/RESNET_Coarse/", "RESNET_Coarse_test_features")
        resnet_coarse_test_gram_matrix = load_from_pickle("Pickle_Files/RESNET_Coarse/", "RESNET_Course_test_gram_matrix")
        resnet_fine_test_features = load_from_pickle("Pickle_Files/RESNET_Fine/", "RESNET_Fine_test_features")
        resnet_fine_test_gram_matrix = load_from_pickle("Pickle_Files/RESNET_Fine/", "RESNET_Fine_test_gram_matrix")
    
    # Training:
    if not os.path.exists("Pickle_Files/RESNET_Coarse/RESNET_Coarse_train_features.pkl") or not os.path.exists("Pickle_Files/RESNET_Coarse/RESNET_Course_train_gram_matrix.pkl") or \
        not os.path.exists("Pickle_Files/RESNET_Fine/RESNET_Fine_train_features.pkl") or not os.path.exists("Pickle_Files/RESNET_Fine/RESNET_Fine_train_gram_matrix.pkl"):
        resnet_coarse_train_features, resnet_coarse_train_gram_matrix, resnet_fine_train_features, resnet_fine_train_gram_matrix = get_RESNET_features(rgb_train)
        save_to_pickle(resnet_coarse_train_features, "Pickle_Files/RESNET_Coarse/", "RESNET_Coarse_train_features")
        save_to_pickle(resnet_coarse_train_gram_matrix, "Pickle_Files/RESNET_Coarse/", "RESNET_Course_train_gram_matrix")
        save_to_pickle(resnet_fine_train_features, "Pickle_Files/RESNET_Fine/", "RESNET_Fine_train_features")
        save_to_pickle(resnet_fine_train_gram_matrix, "Pickle_Files/RESNET_Fine/", "RESNET_Fine_train_gram_matrix")
    else: 
        resnet_coarse_train_features = load_from_pickle("Pickle_Files/RESNET_Coarse/", "RESNET_Coarse_train_features")
        resnet_coarse_train_gram_matrix = load_from_pickle("Pickle_Files/RESNET_Coarse/", "RESNET_Course_train_gram_matrix")
        resnet_fine_train_features = load_from_pickle("Pickle_Files/RESNET_Fine/", "RESNET_Fine_train_features")
        resnet_fine_train_gram_matrix = load_from_pickle("Pickle_Files/RESNET_Fine/", "RESNET_Fine_train_gram_matrix")
    return resnet_coarse_test_gram_matrix, resnet_fine_test_gram_matrix, resnet_coarse_train_gram_matrix, resnet_fine_train_gram_matrix

'''END Of RESNET Feature Extraction'''

'''START of SVM Classifier, with accuracy and classification:'''
def SVM_Classifier(train_data, train_labels, test_data, test_labels, test_files, feature_type="LBP"):
    """
    Trains an SVM classifier and evaluates the accuracy using the given train and test data.
    Displays confusion matrix and sample misclassified images.

    Parameters:
        train_data (np.ndarray): Features for training.
        train_labels (List[str]): Labels corresponding to the training features.
        test_data (np.ndarray): Features for testing.
        test_labels (List[str]): Labels corresponding to the test features.
        test_files (List[str]): List of filenames of the test images for visualization.
        feature_type (str): Feature descriptor type used, e.g., "LBP", "VGG", "ResNet", for labeling purposes.
    """
    # Initialize an SVM classifier with a linear kernel
    cprint(f"Training SVM classifier with {feature_type} descriptors...", "blue")
    svc = svm.SVC(kernel='linear')
    svc.fit(train_data, train_labels)

    # Predict on the test data
    pred = svc.predict(test_data)
    labels = sorted(set(train_labels))  # Extract unique labels for display

    # Compute confusion matrix and accuracy
    conf_mat = confusion_matrix(test_labels, pred)
    accuracy = accuracy_score(test_labels, pred)

    # Display results
    cprint(f"Accuracy for {feature_type} descriptors = {accuracy:.4f}", "green")
    ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=labels).plot()
    plt.title(f"Confusion Matrix for {feature_type} Descriptors")
    ensure_directory(f"MyResults/Confusion_Matrix/{feature_type}/")
    plt.savefig(f"MyResults/Confusion_Matrix/{feature_type}/Confusion_Matrix.jpg")
    plt.show()
    
    accuracy_dict[feature_type] = accuracy
    
    # Show a sample of a misclassified image
    idx_wrong = np.where(test_labels != pred)[0]
    idx_correct = np.where(test_labels == pred)[0]

    ensure_directory(f"MyResults/Classify/{feature_type}/")
    if idx_wrong.size > 0:
        idx_w = idx_wrong[0]
        misclassified_img = cv2.imread(TEST_PATH+test_files[idx_w])
        if misclassified_img is not None:
            plt.imshow(cv2.cvtColor(misclassified_img, cv2.COLOR_BGR2RGB))
            # plt.title(f"Misclassified: Actual = {test_labels[idx_w]}, Predicted = {labels[int(pred[idx_w])]}")    
            plt.title(f"For {feature_type}:\n Misclassified: Actual = {test_labels[idx_w]}, Predicted = {pred[idx_w]}")    
            plt.savefig(f"MyResults/Classify/{feature_type}/Incorrectly_classified.jpg")
            plt.show()
            cprint("SAVED!!!","cyan")
        else:
            cprint(f"Unable to read misclassified image file: {test_files[idx_w]}", "red")
    else:
        cprint("No misclassified images found.", "yellow")

    if idx_correct.size > 0:
        idx_c = idx_correct[0]
        correct_img = cv2.imread(TEST_PATH+test_files[idx_c])
        if correct_img is not None:
            plt.imshow(cv2.cvtColor(correct_img, cv2.COLOR_BGR2RGB))
            # plt.title(f"Correctly Classified: Actual = {test_labels[idx_c]}, Predicted = {labels[int(pred[idx_c])]}")    
            plt.title(f"For {feature_type}:\n Correctly Classified: Actual = {test_labels[idx_c]}, Predicted = {pred[idx_c]}")
            plt.savefig(f"MyResults/Classify/{feature_type}/Correctly_classified.jpg")
            plt.show()
        else:
            cprint(f"Unable to read correctly classified image file: {test_files[idx_c]}", "red")
    else:
        cprint("No correctly classified images found.", "yellow")

@time_decorator
def prepare_and_classify(train_features, train_labels, test_features, test_labels, test_files, feature_type="LBP"):
    """
    A utility function to prepare and classify using the SVM_Classifier function.

    Parameters:
        train_features (List[Tuple]): List of tuples (feature, class_name, file_name).
        train_labels (List[str]): Labels corresponding to the training features.
        test_features (List[Tuple]): List of tuples (feature, class_name, file_name).
        test_labels (List[str]): Labels corresponding to the test features.
        test_files (List[str]): List of filenames of the test images for visualization.
        feature_type (str): Feature descriptor type used, e.g., "LBP", "VGG", "ResNet", for labeling purposes.
    """
    # Extract feature vectors from the list of tuples
    train_data = np.array([feat for feat, label, fname in train_features])
    test_data = np.array([feat for feat, label, fname in test_features])

    # Call the SVM Classifier
    SVM_Classifier(train_data, train_labels, test_data, test_labels, test_files, feature_type)

# Extracts class labels and filenames from an image list.
def extract_labels_and_files(image_list):
    # Returns:
        # labels (List[str]): List of class labels.
        # file_names (List[str]): List of file paths.
    labels = [item[1] for item in image_list]
    file_names = [item[2] for item in image_list]
    return labels, file_names

@time_decorator
def prepare_and_classify_lbp(train_features, train_labels, test_features, test_labels, test_files, feature_type="LBP"):
    """
    Prepares and classifies using the SVM_Classifier function for LBP features.

    Parameters:
        train_features (List[np.ndarray]): List of LBP feature vectors for training.
        train_labels (List[str]): Labels corresponding to the training features.
        test_features (List[np.ndarray]): List of LBP feature vectors for testing.
        test_labels (List[str]): Labels corresponding to the test features.
        test_files (List[str]): List of filenames of the test images for visualization.
        feature_type (str): Feature descriptor type used, e.g., "LBP".
    """
    # Convert lists to NumPy arrays
    train_data = np.array(train_features)
    test_data = np.array(test_features)

    # Feature Scaling
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)

    # Call the SVM Classifier
    SVM_Classifier(train_data, train_labels, test_data, test_labels, test_files, feature_type)


@time_decorator
def prepare_and_classify_vgg(train_features, train_labels, test_features, test_labels, test_files, feature_type="VGG"):
    """
    Prepares and classifies using the SVM_Classifier function for VGG features.

    Parameters:
        train_features (List[np.ndarray]): List of VGG feature vectors for training.
        train_labels (List[str]): Labels corresponding to the training features.
        test_features (List[np.ndarray]): List of VGG feature vectors for testing.
        test_labels (List[str]): Labels corresponding to the test features.
        test_files (List[str]): List of filenames of the test images for visualization.
        feature_type (str): Feature descriptor type used, e.g., "VGG".
    """
    # Extract flattened feature vectors
    train_data = np.array([feat for feat in train_features])
    test_data = np.array([feat for feat in test_features])

    # Feature Scaling
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)

    # Call the SVM Classifier
    SVM_Classifier(train_data, train_labels, test_data, test_labels, test_files, feature_type)


'''END of SVM Classifier, with accuracy and classification:'''


@time_decorator
def task1():
    """
    Task 1: Convert RGB images to HSI, extract Hue channel, and save the data.
    """
    """ ======================================================== READ IN THE ORIGNAL DATASETS: ========================================================"""
    # Check and load/save the original RGB images:
    rgb_train, rgb_test = read_images()    
    """ ======================================================== CONVERT RGB images to HSI Format: ========================================================"""
    hue_train, hue_test = convert_rgb_to_hsi_format(rgb_train, rgb_test)

    """ ======================================================== LBP Feature Descriptors and Histogram plots: ========================================================"""
    # lbp_train_features, lbp_test_features = extract_and_save_lbp(rgb_train, rgb_test)
    lbp_train_features, lbp_test_features = extract_and_save_lbp(hue_train, hue_test)
    # lbp_train_features = load_from_pickle("Pickle_Files/LBP/", "lbp_train_descriptors")
    # lbp_test_features = load_from_pickle("Pickle_Files/LBP/", "lbp_test_descriptors")

    """ ======================================================== Prepare Labels and Files ========================================================"""
    # Extract labels and file names from training and testing data
    train_labels, _ = extract_labels_and_files(rgb_train)
    test_labels, test_files = extract_labels_and_files(rgb_test)
    test_feat = list()
    train_feat = list()

    # Extract only the features from the LBP feature tuples
    # Uncomment depending on whether you are using RGB or HSI format:
    # train_feat = [feat for feat in lbp_train_features] # For RGB format
    # test_feat = [feat for feat in lbp_test_features] # For RGB format
    train_feat = [feat for feat, _, _ in lbp_train_features] # For HSI format
    test_feat = [feat for feat, _, _ in lbp_test_features] # For HSI format

    # Find the minimum number of samples to ensure consistency
    min_train_samples = min(len(train_feat), len(train_labels))
    min_test_samples = min(len(test_feat), len(test_labels))

    # Trim to the minimum number of samples
    train_feat = train_feat[:min_train_samples]
    train_labels = train_labels[:min_train_samples]
    test_feat = test_feat[:min_test_samples]
    test_labels = test_labels[:min_test_samples]
    
    """ ======================================================== SVM Classification for LBP Features ========================================================"""
    cprint("\n===== SVM Classification using LBP Features =====", "magenta")
    SVM_Classifier(train_feat, train_labels, test_feat, test_labels, test_files, "LBP")

    """ ======================================================== VGG19 Feature Descriptors: ========================================================"""
    cprint(f"Starting to generate features using VGG()...", "white")
    vgg_train_gram_matrix, vgg_test_gram_matrix = calc_VGG_descriptors(rgb_train, rgb_test)
    
    """ ======================================================== SVM Classification for VGG Features ========================================================"""
    # Find the minimum number of samples to ensure consistency for VGG
    min_train_samples_vgg = min(len(vgg_train_gram_matrix), len(train_labels))
    min_test_samples_vgg = min(len(vgg_test_gram_matrix), len(test_labels))

    # Trim to the minimum number of samples for VGG
    vgg_train_gram_matrix = vgg_train_gram_matrix[:min_train_samples_vgg]
    train_labels_vgg = train_labels[:min_train_samples_vgg]
    vgg_test_gram_matrix = vgg_test_gram_matrix[:min_test_samples_vgg]
    test_labels_vgg = test_labels[:min_test_samples_vgg]

    # Verifing the shape of the features, gram matrix, expected 528:
    cprint(f"Sample VGG Train Gram Matrix Shape: {vgg_train_gram_matrix[0].shape}", "light_magenta")
    cprint(f"Sample VGG Test Gram Matrix Shape: {vgg_test_gram_matrix[0].shape}", "light_magenta")

    # Perform SVM Classification using VGG Features
    cprint("\n===== SVM Classification using VGG Features =====", "magenta")
    SVM_Classifier(vgg_train_gram_matrix, train_labels_vgg, vgg_test_gram_matrix, test_labels_vgg, test_files, "VGG")
    print("VGG Classification Completed.")

    # Further processing or use of `hue_train` and `hue_test` as needed
    # For example, you could extract features or perform LBP here:
    # feature_vectors = extract_feature_vectors(hue_train, model=VGG19, isRESNET=False)
    # Or save the converted data, perform classifications, etc.
    """ ======================================================== RESNET Feature Descriptors: ========================================================"""
    cprint(f"Staring to generate features using RESNET()...", "white")
    resnet_coarse_test_gram_matrix, resnet_fine_test_gram_matrix, resnet_coarse_train_gram_matrix, resnet_fine_train_gram_matrix = calc_RESNET_descriptors(rgb_train, rgb_test)
    
    # cprint(f"RESNET COARSE TEST GRAM MATRIX: {resnet_coarse_test_gram_matrix[0:2]}")
    cprint(f"Sample RESNET COARSE Train Gram Matrix Shape: {resnet_coarse_train_gram_matrix[0].shape}", "light_magenta")
    cprint(f"Sample RESNET COARSE Test Gram Matrix Shape: {resnet_coarse_test_gram_matrix[0].shape}", "light_magenta")

    SVM_Classifier(resnet_coarse_train_gram_matrix, train_labels_vgg, resnet_coarse_test_gram_matrix, test_labels_vgg, test_files, "RESNET_Coarse")
    SVM_Classifier(resnet_fine_train_gram_matrix, train_labels_vgg, resnet_fine_test_gram_matrix, test_labels_vgg, test_files, "RESNET_Fine")
    
    
    """ ======================================================== Print Accuracy of Weather Classificiation using Feature Descriptors: ========================================================"""
    for key, val in accuracy_dict.items():
        cprint(f"Accuracy of {key} is: {val}","green")
    
    """ ======================================================== Plot the 2D Gram Matrix for each model: ========================================================"""
    # Plot the or visualization, the 2D Gram Matrix for at least one image from each class for the three configurations. 
    # After generating the Gram matrices in task1
    classes_to_plot = ["cloudy", "rain", "shine", "sunrise"]
    cprint("Plotting Gram Matrices for each class...", "white")

    cprint("Plotting Gram Matrices for each VGG:", "white")
    plot_gram_matrix(vgg_train_gram_matrix, train_labels_vgg, classes_to_plot, feature_type="VGG")

    cprint("Plotting Gram Matrices for each RESNET COARSE:", "white")
    plot_gram_matrix(resnet_coarse_train_gram_matrix, train_labels_vgg, classes_to_plot, feature_type="RESNET_Coarse")

    cprint("Plotting Gram Matrices for each RESNET FINE:", "white")
    plot_gram_matrix(resnet_fine_train_gram_matrix, train_labels_vgg, classes_to_plot, feature_type="RESNET_Fine")
    # Dont forget, vgg_train_gram_matrix and vgg_test_gram_matrix are lists of 1D numpy arrays


def task2():
    '''
    Theory Question !!!
    HISTOGRAM CODE FOR LBP!!!
    
    RESNET COARSE AND FINE -> GRAM MATRIX func, PLOT, SVM
    
    Observations of gram matricies -> Comment on your Gram matrix visualizations. Are all feature channels strongly
    correlated?
    
    Record your best classification accuracy as well as the full
    confusion matrix
    
    Observations on accuracy
    
    Finally, for each trained classifier, display at least one correctly classified and one mis-classified image.
    Make sure that you indicate the predicted and ground-truth class labels (Cloudy/Rain/Shine/Sun-
    rise) for each image
    
    Extra Credit: AdaIN
    '''
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
       



