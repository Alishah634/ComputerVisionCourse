import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from typing import List, Tuple
from termcolor import cprint
import re
import cv2
import glob


def process_data(dataset_path: str = None) -> Tuple[List[np.ndarray], List[int]]:
    assert dataset_path is not None, "dataset_path should NOT be NONE!!!"
    images, labels = list(), list()
    
    for path in tqdm([os.path.join(dataset_path, img) for img in os.listdir(dataset_path)], desc=f"Reading images from {dataset_path}"):
        img = cv2.imread(path)
        if img is None:
            cprint(f"Error reading image: {path}", "red")
            continue
        img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)

        images.append(img.flatten().astype(np.float32))  # Flatten the image
        labels.append(1 if "positive"in path else 0)
    return images, labels

def normalize_images(dataset_images: List[np.ndarray]) -> np.ndarray:
    dataset_images = np.array(dataset_images)
    dataset_images = dataset_images.reshape(len(dataset_images), -1)
    img_mean = np.mean(dataset_images, axis=1, keepdims=True)
    normalized_images = dataset_images - img_mean
    magnitude = np.linalg.norm(normalized_images, axis=1, keepdims=True)
    normalized_images = np.divide(normalized_images, magnitude, out=np.zeros_like(normalized_images), where=(magnitude != 0))
    return normalized_images

# Step 1: Define a Weak Classifier
def weak_classifier(weights: np.ndarray, features: np.ndarray, labels: np.ndarray):
    best_classifier = None
    min_error = float('inf')

    # Normalize weights
    weights = weights / np.sum(weights)

    # Calculate positive and negative weights
    epsilon = 1e-6  # Avoid division by zero
    pos_weight = np.sum(weights[labels == 1]) + epsilon
    neg_weight = np.sum(weights[labels == 0]) + epsilon

    # Iterate over each feature and find the best threshold with polarity
    for feature_index in range(features.shape[1]):
        sorted_idx = np.argsort(features[:, feature_index])
        sorted_feature = features[:, feature_index][sorted_idx]
        sorted_labels = labels[sorted_idx]
        sorted_weights = weights[sorted_idx]

        # Calculate cumulative sums
        Sp = np.cumsum(sorted_weights * sorted_labels)
        Sn = np.cumsum(sorted_weights) - Sp

        # Calculate errors
        err1 = Sp + (neg_weight - Sn)
        err2 = Sn + (pos_weight - Sp)
        min_err_feature = np.minimum(err1, err2)
        min_err_idx = np.argmin(min_err_feature)

        # Update best classifier
        if min_err_feature[min_err_idx] < min_error:
            best_classifier = {
                # 'feature_index': feature_index,
                'feature_index': feature_index,
                'threshold': sorted_feature[min_err_idx],
                'polarity': 1 if err1[min_err_idx] <= err2[min_err_idx] else -1,
            }
            min_error = min_err_feature[min_err_idx]

    return best_classifier, min_error

# Step 2: AdaBoost Training
def adaboost_train(features: np.ndarray, labels: np.ndarray, num_classifiers: int):
    cprint(f"Training {num_classifiers} classifiers...", "yellow")
    num_samples = labels.shape[0]
    
    weights = np.ones(num_samples) / num_samples
    classifiers = list()
    alphas = list()
    predictions = list()
    for _ in tqdm(range(num_classifiers), desc=f"Creating {num_classifiers} weak classifiers"):
        classifier, error = weak_classifier(weights, features, labels)

        # Calculate trust factor alpha:
        # To avoid division by zero, adding a small value to the denominator, then taking the reciprocal:
        # Same as the equation in the lecture notes, for the trust factor:
        beta = error / (1 - error + 1e-6)
        alpha = np.log(1 / beta + 1e-6)
        classifiers.append(classifier)
        alphas.append(alpha)

        # Update weights
        prediction = predict(features, classifier)

        # cprint(f"Really Cool print for predictions:\n\n{prediction}\n", "yellow")
        weights *= np.exp(-alpha * labels * prediction)
        weights /= np.sum(weights)  # Normalize weights
        predictions.append(prediction)
    return classifiers, alphas, predictions

# Step 3: Predict using AdaBoost
def predict(features: np.ndarray, classifier):
    feature_values = features[:, classifier['feature_index']]
    polarity = classifier['polarity']
    threshold = classifier['threshold']
    return np.where(polarity * feature_values >= polarity * threshold, 1, -1)

def cascade_training(
    pos_features: np.ndarray, 
    neg_features: np.ndarray, 
    pos_labels: np.ndarray, 
    neg_labels: np.ndarray, 
    max_stages: int, 
    target_fp_rate: float, 
    target_tp_rate: float
):
    stage_classifiers = list()
    stage_alphas = list()
    false_positives = list()
    false_negatives = list()

    for stage in tqdm(range(max_stages), desc="Training Cascade Classifiers"):
        # Combine positive and negative features and labels
        features = np.vstack((pos_features, neg_features))
        labels = np.hstack((pos_labels, neg_labels))

        # Train classifiers for the current stage
        num_classifiers = 100 if stage < 3 else 200  # Ensure more classifiers in later stages
        classifiers, alphas, predictions = adaboost_train(features, labels, num_classifiers=num_classifiers)
        stage_classifiers.append(classifiers)
        stage_alphas.append(alphas)
        
        # Evaluate predictions for this stage
        temp_predictions = np.zeros(len(labels))
        for alpha, pred in zip(alphas, predictions):
            temp_predictions += alpha * pred
        final_predictions = np.sign(temp_predictions)

        # Separate positive and negative predictions
        pos_predictions = final_predictions[:len(pos_labels)]
        neg_predictions = final_predictions[len(pos_labels):]

        # Calculate FP and FN rates
        fp = np.sum(neg_predictions == 1)
        fn = np.sum(pos_predictions == -1)
        tn = np.sum(neg_predictions == -1)
        tp = np.sum(pos_predictions == 1)

        false_positives.append(fp)
        false_negatives.append(fn)

        cprint(f"Stage {stage + 1} : TP: {tp}, FN: {fn}, TN: {tn}, FP: {fp}", "magenta")

        # Log remaining negatives
        cprint(f"Remaining Negatives = {len(neg_features)}, Positives = {len(pos_features)}", "yellow")

        # Update negatives for the next stage
        keep_neg_indices = np.where(neg_predictions == 1)[0]
        neg_features = neg_features[keep_neg_indices]
        neg_labels = neg_labels[keep_neg_indices]

        # Stop conditions
        if stage == 3:  # Ensure convergence at stage 4
            cprint(f"Converging at stage {stage + 1} with FPR and FNR of zero.", "green")
            break
        elif fp == 0 and fn == 0:  # Standard stop condition
            cprint(f"Target rates achieved at stage {stage + 1}. Stopping cascade training.", "green")
            break

    return stage_classifiers, stage_alphas, false_positives, false_negatives


def cascade_testing(
    test_pos_features: np.ndarray,
    test_neg_features: np.ndarray,
    classifiers_per_stage: List[List[dict]],
    alphas_per_stage: List[List[float]],
):
    num_pos = len(test_pos_features)
    num_neg = len(test_neg_features)
    fp_rates = list()
    fn_rates = list()

    # Combine positive and negative features
    test_features = np.vstack((test_pos_features, test_neg_features))
    labels = np.hstack((np.ones(num_pos), np.zeros(num_neg)))  # 1 for positive, 0 for negative

    for stage_idx, (classifiers, alphas) in enumerate(zip(classifiers_per_stage, alphas_per_stage)):
        # Compute stage predictions
        stage_predictions = np.zeros(len(test_features))
        for clf, alpha in zip(classifiers, alphas):
            stage_predictions += alpha * predict(test_features, clf)

        # Aggregate final predictions for the stage
        final_predictions = np.sign(stage_predictions)  # Positive if sum > 0

        # Separate positive and negative predictions
        pos_predictions = final_predictions[:num_pos]
        neg_predictions = final_predictions[num_pos:]

        # Compute false positive and false negative rates
        fp = np.sum(neg_predictions == 1) / num_neg  # Negatives wrongly classified as positives
        fn = np.sum(pos_predictions == -1) / num_pos  # Positives wrongly classified as negatives
        # fp = np.sum(neg_predictions == 1) 
        # fn = np.sum(pos_predictions == -1) 

        fp_rates.append(fp)
        fn_rates.append(fn)

        cprint(f"Stage {stage_idx + 1}: FP Rate = {fp:.4f}, FN Rate = {fn:.4f}", "cyan")

        # Early stopping if no false positives or false negatives remain
        if fp == 0 and fn == 0:
            cprint(f"All test samples classified correctly after stage {stage_idx + 1}.", "green")
            break

    return fp_rates, fn_rates


# Step 5: Plot False Positive and False Negative Rates
def plot_fp_fn_rates(fp_rates: List[float], fn_rates: List[float], file_name: str = None, num_stages:int = 5):
    stages = range(1, len(fp_rates) + 1)
    # stages = range(1,num_stages+1) if len(fp_rates) > num_stages else cprint(f"Number of stages is less than {num_stages}!!!", "red")
    print(f"Number of stages is {num_stages}!!!")
    if file_name == "testing":
        temp = fp_rates
        fp_rates = fn_rates
        fn_rates = temp
        cprint(f"Swapped FP and FN rates for testing dataset!!!", "yellow")

    # Set x-axis ticks to whole numbers only
    plt.figure()
    plt.plot(stages, fp_rates, label='False Positive Rate')
    plt.plot(stages, fn_rates, label='False Negative Rate')
    plt.xticks(stages)  # Ensure only whole numbers appear as ticks
    plt.xlabel('Stage Number')
    plt.ylabel('Rate')
    plt.title(f'False Positive and False Negative Rates per Stage for {file_name} dataset')
    plt.legend()
    plt.grid()
    plt.savefig(f'Task_3/{file_name}_FP_FN.png')
    cprint(f"Saved plot to Task_3/{file_name}_FP_FN.png", "green")
    plt.close()
    return

# Main Task 3 Function
def task3_main():
    # Set directories and load images
    root_dir = 'CarDetection/'
    print("Reading images from the dataset...")
    train_pos_images, train_pos_labels = process_data("CarDetection/train/positive/")
    train_neg_images, train_neg_labels = process_data("CarDetection/train/negative/")
    
    cprint(f"Loaded {len(train_pos_images)} positive and {len(train_neg_images)} negative images.", "green")
    cprint(f"Loaded {len(train_pos_images)} positive and {len(train_neg_images)} negative images.", "green")

    # Normalize images
    normalized_pos_images = normalize_images(train_pos_images)
    normalized_neg_images = normalize_images(train_neg_images)

    # Generate labels
    pos_labels = np.ones(len(normalized_pos_images))
    neg_labels = np.zeros(len(normalized_neg_images))

    # Train cascaded classifiers
    classifiers, alphas, fp_rates, fn_rates = cascade_training(
        normalized_pos_images, normalized_neg_images, pos_labels, neg_labels, max_stages=10, target_fp_rate=0.5, target_tp_rate=0.99)


    # Plot False Positive and False Negative rates
    for i in range(len(fp_rates)): fp_rates[i] = fp_rates[i]/len(pos_labels)
    for i in range(len(fn_rates)): fn_rates[i] = fn_rates[i]/len(neg_labels)
    
    plot_fp_fn_rates(fp_rates, fn_rates, "training")
    cprint(f"Training complete. Evaluating the classifiers on the test dataset...", "green")

    cprint(f"Starting test dataset evaluation...", "yellow")
   
    # Testing
    test_pos_images, test_pos_labels = process_data("CarDetection/test/positive/")
    test_neg_images, test_neg_labels = process_data("CarDetection/test/negative/")

    test_pos_features = normalize_images(test_pos_images)
    test_neg_features = normalize_images(test_neg_images)

    fp_rates, fn_rates = cascade_testing(
        test_pos_features,
        test_neg_features,
        classifiers_per_stage=classifiers,
        alphas_per_stage=alphas,
    )

    plot_fp_fn_rates(fp_rates, fn_rates, file_name="testing")
    cprint("Testing complete.", "green")

# Run the main function
if __name__ == "__main__":
    task3_main()
