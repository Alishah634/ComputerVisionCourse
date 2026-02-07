import os
import re
import sys
import math
import time
import logging
import functools
import random
from tqdm import tqdm
from typing import List, Tuple
from argparse import ArgumentParser
from termcolor import cprint
# Computer vision, Data science, and plotting imports:
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import scipy as sc
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

'''START of Utility functions '''
# Utility functions
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

def ensure_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
''' END of Utility functions '''


'''======================================================================= START of TASK 1 Functions:======================================================================='''
def process_data(dataset_path: str = None) -> Tuple[List[np.ndarray], List[int]]:
    assert dataset_path != None, "dataset_path should NOT be NONE!!!"
    # human_id and image_id 
    labels = list()

    image_paths = [os.path.join(dataset_path, img) for img in os.listdir(dataset_path)] 
    images = list() 
    labels = list() 

    # Regex pattern for extracting human_id
    pattern = r'(\d{2})_\d{2}\.png'

    for path in tqdm(image_paths, desc=f"Reading images from {dataset_path}"):
        img = cv2.imread(path)

        # Skip if the image cannot be read
        if img is None:
            continue  
        
        img = cv2.resize(img, (64,64), interpolation= cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Extract  human_id using regex
        match = re.search(pattern, os.path.basename(path))
        if not match:
            cprint("FAILED to find the human_id!!!", "red")
            continue

        # Convert human_id to integer:
        human_id = int(match.group(1))  
        labels.append(human_id)
        images.append(img)
    return images, labels

def noramalize_images(dataset_images: List[np.ndarray]) -> List[np.ndarray]:
    # Subtract the mean from all the images:
    # Normalize each image vector to unit magnitude 
    normalized_images = []
    for img in dataset_images:
        # Flatten the image into a vector
        img_vector = img.flatten().astype(np.float32)
        
        # Subtract the mean
        img_mean = np.mean(img_vector)
        img_vector -= img_mean
        
        # Normalize to unit magnitude
        magnitude = np.linalg.norm(img_vector)
        
        # Avoid division by zero
        if magnitude > 0:  
            img_vector /= magnitude
        
        # Reshape back to original image dimensions
        normalized_img = img_vector.reshape(img.shape)
        normalized_images.append(normalized_img)
    
    return normalized_images

def calculate_covariance(images: List[np.ndarray], save_path: str, file_name) -> np.ndarray:
    
    # Calculates the covariance matrix for a list of images and saves it to the specified path.
    cprint("The images must all be normalized to compute covariance!!","yellow")
    
    # Flatten images into vectors and stack them into a matrix
    image_vectors = np.array([img.flatten() for img in images])  # Shape: (num_images, num_features)

    # Compute the mean-centered matrix
    mean_vector = np.mean(image_vectors, axis=0)
    centered_matrix = image_vectors - mean_vector  # Shape: (num_images, num_features)

    # Compute the covariance matrix
    covariance_matrix = np.cov(centered_matrix, rowvar=False)  # Shape: (num_features, num_features)

    # Save the covariance matrix to the specified path
    np.save(save_path+file_name, covariance_matrix)
    cprint(f"Covariance matrix saved to {save_path+file_name}", "green")

    fig = plt.figure()
    plt.matshow(covariance_matrix)
    plt.colorbar()
    plt.title('Covaraince Matrix')
    plt.savefig(f'{save_path+"Convaraince_Plot"}.png')
    return covariance_matrix

@time_decorator
def compute_pca(images: List[np.ndarray], num_components: int, save_dir: str):
    """
    Perform PCA on the given dataset and save the results.
    """
    cprint("Performing PCA...", "yellow")

    # Flatten images into vectors and stack into a matrix
    image_vectors = np.array([img.flatten() for img in images])

    # Compute mean-centered data
    mean_vector = np.mean(image_vectors, axis=0)
    centered_data = image_vectors - mean_vector

    # Compute the covariance matrix trick for PCA
    covariance_compressed = np.dot(centered_data, centered_data.T)

    # Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eig(covariance_compressed)

    # Sort eigenvalues and eigenvectors
    sorted_indices = np.argsort(-eigenvalues)[:num_components]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Compute the full eigenvectors for the original data
    eigenvectors_full = np.dot(centered_data.T, eigenvectors)
    eigenvectors_full = eigenvectors_full / np.linalg.norm(eigenvectors_full, axis=0)

    # Project data
    projections = np.dot(centered_data, eigenvectors)

    # Save results
    np.save(os.path.join(save_dir, "eigenvalues.npy"), eigenvalues)
    np.save(os.path.join(save_dir, "eigenvectors.npy"), eigenvectors_full)
    np.save(os.path.join(save_dir, "projections.npy"), projections)
    cprint(f"PCA completed. Results saved in {save_dir}", "green")

def compute_lda(images: List[np.ndarray], labels: List[int], num_components: int, save_dir: str):
    """
    Perform LDA on the given dataset and save the results.
    """
    cprint("Performing LDA...", "yellow")

    # Flatten images into vectors and stack into a matrix
    image_vectors = np.array([img.flatten() for img in images])
    labels = np.array(labels)

    # Compute mean vectors
    mean_overall = np.mean(image_vectors, axis=0)
    class_labels = np.unique(labels)

    S_W = np.zeros((image_vectors.shape[1], image_vectors.shape[1]))
    S_B = np.zeros((image_vectors.shape[1], image_vectors.shape[1]))

    for label in class_labels:
        class_data = image_vectors[labels == label]
        class_mean = np.mean(class_data, axis=0)

        # Within-class scatter
        centered_class_data = class_data - class_mean
        S_W += np.dot(centered_class_data.T, centered_class_data)

        # Between-class scatter
        n_class = class_data.shape[0]
        mean_diff = (class_mean - mean_overall).reshape(-1, 1)
        S_B += n_class * np.dot(mean_diff, mean_diff.T)

    # Solve generalized eigenvalue problem
    S_W_inv = np.linalg.pinv(S_W)
    eigenvalues, eigenvectors = np.linalg.eig(np.dot(S_W_inv, S_B))

    # Sort eigenvalues and eigenvectors
    sorted_indices = np.argsort(-eigenvalues.real)[:num_components]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Project data
    projections = np.dot(image_vectors, eigenvectors)

    # Save results
    np.save(os.path.join(save_dir, "eigenvalues.npy"), eigenvalues.real)
    np.save(os.path.join(save_dir, "eigenvectors.npy"), eigenvectors.real)
    np.save(os.path.join(save_dir, "projections.npy"), projections)
    cprint(f"LDA completed. Results saved in {save_dir}", "green")
    
def nearest_neighbor_classification(train_projections, train_labels, test_projections, test_labels):
    """
    Perform nearest neighbor classification and compute accuracy.
    """
    train_labels = np.array(train_labels).flatten()
    test_labels = np.array(test_labels).flatten()

    assert train_projections.shape[0] == len(train_labels), (
        f"Mismatch: train_projections rows ({train_projections.shape[0]}) != train_labels size ({len(train_labels)})"
    )
    assert test_projections.shape[0] == len(test_labels), (
        f"Mismatch: test_projections rows ({test_projections.shape[0]}) != test_labels size ({len(test_labels)})"
    )

    correct_count = 0
    for i, test_sample in enumerate(test_projections):
        distances = np.linalg.norm(train_projections - test_sample, axis=1)
        nearest_neighbor_idx = np.argmin(distances)
        if train_labels[nearest_neighbor_idx] == test_labels[i]:
            correct_count += 1

    accuracy = correct_count / len(test_labels)
    return accuracy




def evaluate_subspace_dimensionality(train_images, train_labels, test_images, test_labels,
                                     method: str, save_dir: str, max_components: int):
    """
    Evaluate classification accuracy as a function of subspace dimensionality.
    """
    accuracies = []
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)

    for p in tqdm(range(1, max_components + 1), desc="Evaluating of different P values"):
        if method.upper() == 'PCA':
            projections, eigenvectors = pca_projection(np.array([img.flatten() for img in train_images]).T, p)
            test_projections = np.dot(eigenvectors.T, np.array([img.flatten() for img in test_images]).T)
        elif method.upper() == 'LDA':
            projections, eigenvectors = lda_projection(np.array([img.flatten() for img in train_images]).T, train_labels, p)
            test_projections = np.dot(eigenvectors.T, np.array([img.flatten() for img in test_images]).T)
        else:
            raise ValueError("Invalid method! Use 'PCA' or 'LDA'.")

        # Nearest-neighbor classification
        accuracy = calculate_accuracy(projections, train_labels, k=1)
        accuracies.append((p, accuracy))

    np.save(os.path.join(save_dir, f"{method}_accuracies.npy"), accuracies)
    return accuracies


def plot_accuracy_comparison(pca_accuracies, lda_accuracies, save_path="accuracy_comparison.png"):
    """
    Plot and compare PCA and LDA classification accuracies.
    """
    pca_dimensions, pca_accuracy = zip(*pca_accuracies)
    lda_dimensions, lda_accuracy = zip(*lda_accuracies)
    print("HERE!!")

    plt.figure(figsize=(10, 6))
    plt.plot(pca_dimensions, pca_accuracy, label="PCA", marker='o', linestyle='-')
    plt.plot(lda_dimensions, lda_accuracy, label="LDA", marker='x', linestyle='--')
    plt.xlabel("Number of Components (p)")
    plt.ylabel("Accuracy (%)")
    plt.title("Classification Accuracy vs Subspace Dimensionality")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.show()


def pca_projection(data: np.ndarray, num_components: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform PCA on the data and return projections onto the top `num_components` eigenvectors.
    """
    mean_vector = np.mean(data, axis=1, keepdims=True)
    centered_data = data - mean_vector

    # Covariance trick for PCA
    cov_compressed = np.dot(centered_data.T, centered_data)
    eigenvalues, eigenvectors = np.linalg.eig(cov_compressed)
    sorted_indices = np.argsort(-eigenvalues)
    top_eigenvectors = eigenvectors[:, sorted_indices[:num_components]]
    top_eigenvectors = np.dot(centered_data, top_eigenvectors)  # Map back to original space
    projections = np.dot(top_eigenvectors.T, centered_data)

    return projections, top_eigenvectors


def lda_projection(data: np.ndarray, labels: np.ndarray, num_components: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform LDA and return projections onto the top `num_components` eigenvectors.
    """
    unique_labels = np.unique(labels)
    mean_overall = np.mean(data, axis=1, keepdims=True)

    # Scatter matrices
    S_W = np.zeros((data.shape[0], data.shape[0]))
    S_B = np.zeros((data.shape[0], data.shape[0]))

    for label in unique_labels:
        class_data = data[:, labels == label]
        class_mean = np.mean(class_data, axis=1, keepdims=True)
        S_W += np.dot((class_data - class_mean), (class_data - class_mean).T)
        S_B += class_data.shape[1] * np.dot((class_mean - mean_overall), (class_mean - mean_overall).T)

    # Generalized eigenvalue problem
    eigvals, eigvecs = np.linalg.eig(np.linalg.pinv(S_W).dot(S_B))
    sorted_indices = np.argsort(-eigvals)
    top_eigenvectors = eigvecs[:, sorted_indices[:num_components]]
    projections = np.dot(top_eigenvectors.T, data)

    return projections, top_eigenvectors


def calculate_accuracy(projections: np.ndarray, labels: np.ndarray, k: int = 1) -> float:
    """
    Use k-NN to classify projections and return the accuracy.
    """
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(projections.T, labels)
    predictions = knn.predict(projections.T)
    accuracy = np.mean(predictions == labels) * 100
    return accuracy


'''======================================================================= END of TASK 1 Functions:======================================================================='''

def task1():
    # Set up the directories:
    train_dir = "FaceRecognition/train/"
    test_dir = "FaceRecognition/test/"
    TASK_1_PATH = "Task_1/"
    ensure_directory(TASK_1_PATH)

    # Read in the dataset images, keep a concurrent list of labels:
    train_images, train_labels = process_data(train_dir)
    test_images, test_labels = process_data(test_dir)

    # Subtract the mean and Normalize each image vector to unit magnitude:
    normalized_train_images = noramalize_images(train_images)
    normalized_test_images = noramalize_images(test_images)

    print("normalized!!")
    # Evaluate PCA accuracy
    ensure_directory(TASK_1_PATH + "PCA/")
    pca_accuracies = evaluate_subspace_dimensionality(normalized_train_images, train_labels,
                                                      normalized_test_images, test_labels,
                                                      method='PCA', save_dir=TASK_1_PATH + "PCA/", max_components=50)
    print("PCA START!!")

    # Evaluate LDA accuracy
    ensure_directory(TASK_1_PATH + "LDA/")
    lda_accuracies = evaluate_subspace_dimensionality(normalized_train_images, train_labels,
                                                      normalized_test_images, test_labels,
                                                      method='LDA', save_dir=TASK_1_PATH + "LDA/", max_components=50)
    
    print("PCA DONE!!")
    # Plot comparison
    plot_accuracy_comparison(pca_accuracies, lda_accuracies, save_path=os.path.join(TASK_1_PATH, "accuracy_comparison.png"))


# # Face Recognition using PCA and LDA
# def task1():
#     # Set up the directories:
#     train_dir = "FaceRecognition/train/"
#     test_dir =  "FaceRecognition/test/"
#     TASK_1_PATH = "Task_1/"
#     ensure_directory("Task_1/")
    
#     # Read in the dataset images, keep a concurrent list of labels:
#     train_images, train_labels = process_data(train_dir)
#     test_images, test_labels = process_data(test_dir)

#     # Subtract the mean and Normalize each image vector to unit magnitude:
#     normalized_train_images = noramalize_images(train_images)
#     normalized_test_images = noramalize_images(test_images)

#     # Calculate the covariance matrix.
#     ensure_directory(TASK_1_PATH+"Covariance/")
    
#     normalized_train_images = calculate_covariance(normalized_train_images,TASK_1_PATH+"Covariance/", "train.npy" )
#     normalized_test_images = calculate_covariance(normalized_test_images,TASK_1_PATH+"Covariance/", "test.npy" )


#     # PCA       
#     ensure_directory(TASK_1_PATH+"PCA/")
#     compute_pca(normalized_train_images, num_components=50, save_dir=TASK_1_PATH + "PCA/")
#     evaluate_subspace_dimensionality(normalized_train_images, train_labels,
#                                      normalized_test_images, test_labels,
#                                      method='PCA', save_dir=TASK_1_PATH + "PCA/", max_components=50)

#     #  Plot comparison
#     pca_accuracies = np.load(os.path.join(TASK_1_PATH, "PCA/PCA_accuracies.npy"), allow_pickle=True)
#     lda_accuracies = []
#     plot_accuracy_comparison(pca_accuracies, lda_accuracies, save_path=os.path.join(TASK_1_PATH, "accuracy_comparison.png"))
    
    
    
    # Load PCA and LDA accuracies
    # pca_accuracies = np.load("Task_1/PCA/PCA_accuracies.npy", allow_pickle=True)

    # Extract dimensions and accuracies
    # pca_dimensions, pca_accuracy = zip(*pca_accuracies)

    # # Plot
    # plt.figure(figsize=(10, 6))
    # plt.plot(pca_dimensions, pca_accuracy, label="PCA", marker='o')
    # plt.xlabel("Number of Components (p)")
    # plt.ylabel("Accuracy (%)")
    # plt.title("Classification Accuracy vs Subspace Dimensionality")
    # plt.legend()
    # plt.grid(True)
    # plt.plot(lda_dimensions, lda_accuracy, label="LDA", marker='x')
    # plt.savefig("Task_1/accuracy_comparison_pca_lda.png")
    # plt.show()
    
    
    # lda_accuracies = np.load("Task_1/LDA/LDA_accuracies.npy", allow_pickle=True)
    # lda_accuracies = np.load(os.path.join(TASK_1_PATH, "LDA/LDA_accuracies.npy"), allow_pickle=True)
    # lda_dimensions, lda_accuracy = zip(*lda_accuracies)
    
    
    # LDA
    ensure_directory(TASK_1_PATH + "LDA/")
    compute_lda(normalized_train_images, train_labels, num_components=50, save_dir=TASK_1_PATH + "LDA/")

    pass
def task2():
    cprint("Task 2 completed successfully!", "green")
    pass

@time_decorator
def task3():
    cprint(f"Running Task 3 with window size: {13}", "yellow")

    cprint("Task 3 completed successfully!", "green")
    return

@time_decorator
def task4():
    cprint("Task 4 completed successfully!", "green")
    pass

if __name__ == "__main__":
    parser = ArgumentParser(description="Script to run Task 1 or Task 2.")
    parser.add_argument('--task', '-t', type=int, required=True, choices=[1, 2, 3, 4], help="Select task: 1 for Task 1, 2 for Task 2 and so on")
    args = parser.parse_args()

    cprint("Make sure you are running from HW 10 directory!!!", "red")
    ensure_directory("MyResults")

    if args.task == 1:
        task1()
    elif args.task == 2:
        task2()
    elif args.task == 3:
        task3()
    elif args.task == 4:
        task4()

import os
import numpy as np
from typing import Tuple

def compute_lda(x: np.ndarray, y: np.ndarray, num_components: int, output_path: str = None) -> Tuple[np.ndarray, np.ndarray]:
    # Check if the LDA results are already saved
    lda_output_file = os.path.join(output_path, f'lda_components_{num_components}.npz')
    if os.path.exists(lda_output_file):
        saved_data = np.load(lda_output_file)
        lda_vectors, global_mean = saved_data['lda_vectors'], saved_data['global_mean']
        return lda_vectors, global_mean

    # Determine number of classes and feature dimensions
    unique_classes = np.unique(y)
    num_classes = len(unique_classes)
    feature_dim = x.shape[1]
    
    # Initialize arrays to store scatter matrices and class means
    samples_per_class = {label: 0 for label in unique_classes}
    class_means = {label: np.zeros(feature_dim) for label in unique_classes}

    # Calculate per-class means and accumulate sample counts
    for sample, label in zip(x, y):
        class_means[label] += sample
        samples_per_class[label] += 1

    for label in unique_classes:
        class_means[label] /= samples_per_class[label]

    # Calculate global mean across all classes
    global_mean = np.mean(list(class_means.values()), axis=0, keepdims=True)

    # Calculate between-class scatter matrix (SB)
    sb_matrix = np.zeros((feature_dim, feature_dim))
    for label in unique_classes:
        num_samples = samples_per_class[label]
        mean_diff = (class_means[label] - global_mean).reshape(-1, 1)
        sb_matrix += num_samples * np.dot(mean_diff, mean_diff.T)

    # Calculate within-class scatter matrix (SW)
    sw_matrix = np.zeros((feature_dim, feature_dim))
    for sample, label in zip(x, y):
        diff = (sample - class_means[label]).reshape(-1, 1)
        sw_matrix += np.dot(diff, diff.T)

    # Compute the pseudo-inverse of SW and find eigenvectors of SW^-1 * SB
    sw_inverse = np.linalg.pinv(sw_matrix)
    lda_matrix = np.dot(sw_inverse, sb_matrix)
    _, _, right_singular_vectors = np.linalg.svd(lda_matrix)

    # Extract the top 'num_components' eigenvectors
    lda_vectors = right_singular_vectors.T[:, :num_components]

    # Save the LDA components for future use
    np.savez_compressed(lda_output_file, lda_vectors=lda_vectors, global_mean=global_mean)
    return lda_vectors, global_mean


def calculate_lda(x: np.ndarray, y: np.ndarray, p: int, task_1_path: str = None) -> Tuple[np.ndarray, np.ndarray]:
    # Check if the results are already saved
    lda_file_path = os.path.join(task_1_path, f'lda_p_{p}.npz')
    if os.path.exists(lda_file_path):
        data = np.load(lda_file_path)
        w, m = data['w'], data['m']
        return w, m

    # Calculate LDA
    num_classes = np.max(y)
    x_dim = x.shape[1]
    
    # Create arrays to store within-class scatter and class means
    dat_per_class = np.zeros(num_classes)
    class_means = np.zeros((num_classes, x_dim))

    # Accumulate data per class to compute class means
    for data, lbl in zip(x, y):
        class_means[lbl - 1] += data
        dat_per_class[lbl - 1] += 1

    # Normalize to get the actual class means
    class_means = class_means / dat_per_class[:, np.newaxis]

    # Global mean
    global_mean = np.mean(class_means, axis=0, keepdims=True)

    # Between-class scatter matrix SB
    sb = np.zeros((x_dim, x_dim))
    for i in range(num_classes):
        mean_diff = (class_means[i] - global_mean).reshape(-1, 1)
        sb += dat_per_class[i] * np.dot(mean_diff, mean_diff.T)

    # Within-class scatter matrix SW
    sw = np.zeros((x_dim, x_dim))
    for data, lbl in zip(x, y):
        class_scatter = (data - class_means[lbl - 1]).reshape(-1, 1)
        sw += np.dot(class_scatter, class_scatter.T)

    # Calculate the eigenvectors of SW^-1 * SB
    sw_inv = np.linalg.pinv(sw)  # Use pseudo-inverse in case SW is singular
    s_matrix = np.dot(sw_inv, sb)
    _, _, u_t = np.linalg.svd(s_matrix)

    # Eigenvectors to form the projection matrix W
    w = u_t.T[:, :p]  # Retain first p eigenvectors

    # Save the results
    np.savez_compressed(lda_file_path, w=w, m=global_mean)
    return w, global_mean
