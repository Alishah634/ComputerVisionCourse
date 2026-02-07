import os
import re
import sys
import time
import logging
import functools
import numpy as np
import cv2
from tqdm import tqdm
from typing import List, Tuple
from argparse import ArgumentParser
from termcolor import cprint
import matplotlib.pyplot as plt
import umap
import seaborn as sns


from Task3 import task3_main
from autoencoder import task2_main


'''START of Utility functions'''
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
        logger.info(f"{func.__name__} took {elapsed_time:.4f} seconds to execute\n")
        return result
    return wrapper

def ensure_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
'''END of Utility functions'''

TASK_1_PATH = "Task_1/"
TASK_2_PATH = "Task_2/"
TASK_3_PATH = "Task_3/"
ensure_directory(TASK_1_PATH)
ensure_directory(TASK_2_PATH)
ensure_directory(TASK_3_PATH)
ensure_directory("TASK_2/AutoModels/")

'''======================================================================= START of TASK 1 Functions ======================================================================='''

@time_decorator
def process_data(dataset_path: str = None) -> Tuple[List[np.ndarray], List[int]]:
    assert dataset_path is not None, "dataset_path should NOT be NONE!!!"
    images, labels = list(), list()
    
    # Regex pattern for extracting human_id
    pattern = r'(\d{2})_\d{2}\.png'

    for path in tqdm([os.path.join(dataset_path, img) for img in os.listdir(dataset_path)], desc=f"Reading images from {dataset_path}"):
        img = cv2.imread(path, 0)  # Read in grayscale
        if img is None:
            continue
        img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
        match = re.search(pattern, os.path.basename(path))
        if not match:
            cprint("FAILED to find the human_id!!!", "red")
            continue
        human_id = int(match.group(1))
        labels.append(human_id)
        images.append(img.flatten().astype(np.float32))  # Flatten the image

    return images, labels

@time_decorator
def normalize_images(dataset_images: List[np.ndarray]) -> np.ndarray:
    dataset_images = np.array(dataset_images)
    dataset_images = dataset_images.reshape(len(dataset_images), -1)
    img_mean = np.mean(dataset_images, axis=1, keepdims=True)
    normalized_images = dataset_images - img_mean
    magnitude = np.linalg.norm(normalized_images, axis=1, keepdims=True)
    normalized_images = np.divide(normalized_images, magnitude, out=np.zeros_like(normalized_images), where=(magnitude != 0))
    return normalized_images

def calculate_pca(norm_train_imgs: np.ndarray, p: int = None, task_1_path: str = None) -> Tuple[np.ndarray, np.ndarray]:
    # Check if the results are already saved
    pca_file_path = os.path.join(task_1_path, f'pca_p_{p}.npz')
    if os.path.exists(pca_file_path):
        data = np.load(pca_file_path)
        eigen_vector, train_mean = data['eigen_vector'], data['train_mean']
        return eigen_vector, train_mean
    # Subtract the mean and calculate the covariance matrix:
    train_mean = np.mean(norm_train_imgs.T  , axis=1, keepdims=True)
    norm_train_imgs = norm_train_imgs.T - train_mean
    # eigen_vector = np.dot(norm_train_imgs, np.linalg.svd(np.dot(norm_train_imgs.T, norm_train_imgs))[2])
    eigen_vector = np.dot(norm_train_imgs, np.linalg.eig(np.dot(norm_train_imgs.T, norm_train_imgs))[1])
    eigen_vector = eigen_vector[:, :p]
    # Save the results
    pca_file_path = os.path.join(task_1_path, f'pca_p_{p}.npz')
    np.savez_compressed(pca_file_path, eigen_vector=eigen_vector, train_mean=train_mean)
    return eigen_vector, train_mean


def calculate_lda(x: np.ndarray, y: np.ndarray, p: int, output_path: str = None) -> Tuple[np.ndarray, np.ndarray]:
    # Check if the LDA results are already saved
    lda_file_path = os.path.join(TASK_1_PATH, f'lda_p_{p}.npz')
    if os.path.exists(lda_file_path):
        saved_data = np.load(lda_file_path)
        # lda_vectors, _ = saved_data['lda_vectors'], saved_data['global_mean']
        lda_vectors = saved_data['lda_vectors']
        cprint(f"{lda_vectors} \n\n","cyan")
        return lda_vectors
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
    lda_matrix = np.dot(np.linalg.pinv(sw_matrix), sb_matrix)
    right_singular_vectors = np.linalg.svd(lda_matrix)[2]
    right_singular_vectors = right_singular_vectors.T
    
    # Extract the top 'num_components' eigenvectors
    lda_vectors = right_singular_vectors[:, :p]

    # Save the LDA components for future use
    np.savez_compressed(lda_file_path, lda_vectors=lda_vectors)
    return lda_vectors


def findNearestNeighbor(training_data: np.ndarray, test_point: np.ndarray, labels: List[int]) -> int:
    distances = np.linalg.norm(training_data - test_point[:, np.newaxis], axis=0)
    nearest_index = np.argmin(distances)
    return labels[nearest_index]

def evaluate_pca_lda(pca_weights: np.ndarray, pca_avg: np.ndarray, pca_projected_space: np.ndarray, lda_weights: np.ndarray, lda_projected_space: np.ndarray, test_data: np.ndarray, train_labels: List[int], test_labels: List[int], num_components: int, output_dir: str = TASK_1_PATH) -> Tuple[float, float]:
    num_samples = len(test_labels)
    centered_test_data = test_data.T - pca_avg
    predicted_pca_labels = list()
    predicted_lda_labels = list()

    for sample_idx in range(num_samples):
        pca_projection = np.dot(pca_weights.T, centered_test_data[:, sample_idx])
        lda_projection = np.dot(lda_weights.T, centered_test_data[:, sample_idx])
        nearest_pca_label = findNearestNeighbor(pca_projected_space, pca_projection, train_labels)
        nearest_lda_label = findNearestNeighbor(lda_projected_space, lda_projection, train_labels)
        predicted_pca_labels.append(nearest_pca_label)
        predicted_lda_labels.append(nearest_lda_label)

    pca_accuracy = np.mean(np.equal(np.array(predicted_pca_labels), test_labels)) * 100
    lda_accuracy = np.mean(np.equal(np.array(predicted_lda_labels), test_labels)) * 100
    return pca_accuracy, lda_accuracy


def plot_umap_embeddings(data: np.ndarray, labels: List[int], title: str, output_path: str):
    # Ensure labels are integers from 0 to 29, representing 30 classes
    labels = np.array(labels)
    unique_labels = np.unique(labels)
    if len(unique_labels) != 30:
        cprint(f"Warning: Expected 30 unique classes, but found {len(unique_labels)} unique classes.", "yellow")
        
    reducer = umap.UMAP(n_components=2, random_state=42)
    embedding = reducer.fit_transform(data)
    
    # Plot UMAP Embedding
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='Spectral', s=10, alpha=0.8)
    cbar = plt.colorbar(scatter, ticks=np.arange(30))  # Explicitly define 30 ticks for color bar
    cbar.set_label('Classes')
    cbar.set_ticks(np.arange(30))  # Set ticks for each class label
    
    plt.title(title)
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()
    return

ensure_directory("TASK_2/Autoencoder")
def umap_visualize(X_train, y_train, X_test, y_test, p, save_path="TASK_2/Autoencoder"):
    reducer = umap.UMAP(n_components=2, random_state=42)

    # Perform UMAP dimensionality reduction for training data
    train_emb = reducer.fit_transform(X_train)

    # Plot UMAP embeddings for training data
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=train_emb[:, 0],
        y=train_emb[:, 1],
        hue=y_train,
        palette="coolwarm",
        alpha=0.7
    )
    plt.title(f"UMAP Visualization for Training Data (p={p})")
    plt.legend(title="Classes", loc='upper right', bbox_to_anchor=(1.15, 1.05))
    plt.tight_layout()
    train_save_path  = f"{save_path}/Autoencoder_UMAP_Train_p{p}.png"
    plt.savefig(train_save_path)
    cprint(f"UMAP visualization for training data saved to {train_save_path}", "green")
    plt.close()

    # Perform UMAP dimensionality reduction for testing data
    test_emb = reducer.fit_transform(X_test)

    # Plot UMAP embeddings for testing data
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=test_emb[:, 0],
        y=test_emb[:, 1],
        hue=y_test,
        palette="dark",
        alpha=0.8
    )
    plt.title(f"UMAP Visualization for Testing Data (p={p})")
    plt.legend(title="Classes", loc='upper right', bbox_to_anchor=(1.15, 1.05))
    plt.tight_layout()
    test_save_path = f"{save_path}/Autoencoder_UMAP_Test_p{p}.png"
    plt.savefig(test_save_path)
    cprint(f"UMAP visualization for testing data saved to {test_save_path}", "green")
    plt.close()

'''======================================================================= END of TASK 1 Functions ======================================================================='''
    
@time_decorator
def task1():
    # Set up the directories:
    train_dir = "FaceRecognition/train/"
    test_dir = "FaceRecognition/test/"

    # Step 1: Read and preprocess data
    train_images, train_labels = process_data(train_dir)
    test_images, test_labels = process_data(test_dir)

    # Step 2: Subtract mean and normalize each image vector
    normalized_train_images = normalize_images(train_images)
    normalized_test_images = normalize_images(test_images)

    # Step 3 & 4: Evaluate PCA and LDA classifiers for different subspace dimensions
    PCA_accuracies, LDA_accuracies = list(), list()

    # Evaluate PCA and LDA for different p subspace dimensions:
    subspace_dimensions = list(range(1, 21))
    for p in tqdm(subspace_dimensions, desc="Classifiers for different p subspace dimensions"):
        w_pca, pca_mean,  = calculate_pca(normalized_train_images, p, TASK_1_PATH)
        w_lda = calculate_lda(normalized_train_images, train_labels, p, TASK_1_PATH)
        
        PCA_space = np.dot(w_pca.T, normalized_train_images.T - pca_mean)
        LDA_space = np.dot(w_lda.T, normalized_train_images.T - pca_mean)

        pca_acc, lda_acc = evaluate_pca_lda(w_pca, pca_mean, PCA_space, w_lda, LDA_space, normalized_test_images, train_labels, test_labels, p)

        PCA_accuracies.append(pca_acc)
        LDA_accuracies.append(lda_acc)

    # Log classification accuracy
    for i, p in enumerate(subspace_dimensions):
        cprint(f"Dimension {p}: PCA Accuracy = {PCA_accuracies[i]}%, LDA Accuracy = {LDA_accuracies[i]}%", "cyan")

    # Step 5: Plot UMAP embeddings for PCA and LDA
    for p, w_pca, w_lda in zip(subspace_dimensions, PCA_accuracies, LDA_accuracies):
        w_pca, pca_mean = calculate_pca(normalized_train_images, p, TASK_1_PATH)
        w_lda = calculate_lda(normalized_train_images, train_labels, p, TASK_1_PATH)

        # PCA Embeddings
        pca_train_projection = np.dot(w_pca.T, normalized_train_images.T - pca_mean).T
        pca_test_projection = np.dot(w_pca.T, normalized_test_images.T - pca_mean).T
        plot_umap_embeddings(pca_train_projection, train_labels, f'PCA UMAP Embeddings (Train, p={p})', f"{TASK_1_PATH}PCA_UMAP_Train_p{p}.png")
        plot_umap_embeddings(pca_test_projection, test_labels, f'PCA UMAP Embeddings (Test, p={p})', f"{TASK_1_PATH}PCA_UMAP_Test_p{p}.png")

        # LDA Embeddings
        lda_train_projection = np.dot(w_lda.T, normalized_train_images.T).T
        lda_test_projection = np.dot(w_lda.T, normalized_test_images.T).T
        plot_umap_embeddings(lda_train_projection, train_labels, f'LDA UMAP Embeddings (Train, p={p})', f"{TASK_1_PATH}LDA_UMAP_Train_p{p}.png")
        plot_umap_embeddings(lda_test_projection, test_labels, f'LDA UMAP Embeddings (Test, p={p})', f"{TASK_1_PATH}LDA_UMAP_Test_p{p}.png")

    
    '''
    ======================================================== TASK 2 CODE BELOW: ========================================================
    '''
    
    
    # TASK 2: Autoencoder Features Classification:
    
    # Auto encoder for dimensionality reduction:
    # evaluate autoencoder classifier for different subspace dimensions

    auto_encoder_accuracies = list()
    
    # List of p values for the autoencoder:
    plist_auto_encoder = [3, 5, 10, 15, 19, 20]
    
    # List for Provided pre-trained weights:
    plist_auto_encoder = [3, 8, 16] # Uncomment when using Provided pre-trained weights !!!
    
    index = 0
    for p in tqdm(plist_auto_encoder, desc="Autoencoder for different p subspace dimensions"):
        # Path to Provided pre-trained weights:
        autoencoder_path = f"weights/model_{p}.pt" # Uncomment when using Provided pre-trained weights!!!
        accuracy, feature_data = task2_main(p, autoencoder_path, False) # Uncomment when using Provided pre-trained weights!!!
        auto_encoder_accuracies.append(accuracy) # Uncomment when using Provided pre-trained weights!!!
        X_train, y_train, X_test, y_test, p = feature_data
        umap_visualize(X_train, y_train, X_test, y_test, p)

        # UnComment when NOT using Provided pre-trained weights!!!: 
        # auto_encoder_accuracies.append(task2_main(p, None, True)) 
        cprint(f"Dimension {p}: Autoencoder Accuracy {auto_encoder_accuracies[index]}", "cyan")
        index += 1
    
    # Plot the accuracies for PCA and LDA
    plt.plot(subspace_dimensions, PCA_accuracies, label='PCA Accuracy', marker='x')
    plt.plot(subspace_dimensions, LDA_accuracies, label='LDA Accuracy', marker='o', color='orange')
    plt.plot(plist_auto_encoder, auto_encoder_accuracies, label="Autoencoder" , marker=".", color='green')
    plt.xlabel("Number of Components (p)")
    plt.ylabel("Accuracy (%)")
    plt.title("Classification Accuracy vs Subspace Dimensionality")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{TASK_1_PATH}PCA_vs_LDA_Accuracy.png")
    
    plt.show()
    
    # TODO: UMAP for the autoencoder features:
    umap_visualize(X_train, y_train, X_test, y_test, p)    
    
    """
    ====================================================== TASK 2 CODE ABOVE^^^: ======================================================
    """

def task2():
    task2_main(3, None, True)
    pass


def task3():
    task3_main()
    pass

if __name__ == "__main__":
    parser = ArgumentParser(description="Script to run Task 1.")
    parser.add_argument('--task', '-t', type=int, required=True, choices=[1, 2, 3], help="Select task: 1 for Task 1, 2 for Task 2 and so...")
    args = parser.parse_args()

    ensure_directory("MyResults")

    if args.task == 1:
        task1()

    if args.task == 2:
        task2()

    if args.task == 3:
        task3()
