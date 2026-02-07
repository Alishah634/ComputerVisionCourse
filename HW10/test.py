import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import pickle


def calculate_features(img):
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    window_widths = np.arange(2, img.shape[1], 2)
    window_heights = np.arange(2, img.shape[0], 2)
    features = []

    for N in window_widths:
        img_padded = np.pad(img, ((0, 0), (int(N / 2), int(N / 2))), mode='constant')
        for ix in range(img.shape[0]):
            for jx in range(int(N / 2), img_padded.shape[1] - int(N / 2) + 1):
                neg_sums = np.sum(img_padded[ix, jx - int(N / 2):jx + 1].flatten()).astype(np.int32)
                pos_sums = np.sum(img_padded[ix, jx + 1:jx + int(N / 2) + 1].flatten()).astype(np.int32)
                features.append(pos_sums - neg_sums)

    for N in window_heights:
        img_padded = np.pad(img, ((int(N / 2), int(N / 2)), (0, 0)), mode='constant')
        for ix in range(int(N / 2), img_padded.shape[0] - int(N / 2) + 1):
            for jx in range(img.shape[1]):
                neg_sums = np.sum(img_padded[ix - int(N / 2):ix + 1, jx].flatten()).astype(np.int32)
                pos_sums = np.sum(img_padded[ix + 1:ix + int(N / 2) + 1, jx].flatten()).astype(np.int32)
                features.append(pos_sums - neg_sums)

    return np.array(features)


def extract_features(classes, data_dir, result_dir, name):
    print("Constructing feature matrices for the dataset")
    for C in classes:
        save_path = os.path.join(result_dir, f'{name}_{C}.npy')
        if not os.path.exists(save_path):
            features = []
            path_dir = os.path.join(data_dir, C)
            for F in os.listdir(path_dir):
                print(f'Processing {C} Images = {len(features)}', end="\r")
                image = cv2.imread(os.path.join(path_dir, F))
                features.append(calculate_features(image))
            features = np.array(features)
            np.save(save_path, features)
            print("\n")
        else:
            print(f'Feature matrices already exist at {save_path}')
        if C.upper() == "POSITIVE":
            data_pos_features = np.load(save_path)
            print(f'Positive feature shape: {data_pos_features.shape}')
        elif C.upper() == "NEGATIVE":
            data_neg_features = np.load(save_path)
            print(f'Negative feature shape: {data_neg_features.shape}')
    return data_pos_features, data_neg_features


def create_labels(pos_features, neg_features):
    combined_features = np.concatenate((pos_features, neg_features), axis=0)
    combined_labels = np.concatenate((np.ones((pos_features.shape[0], 1)), 
                                       np.zeros((neg_features.shape[0], 1))), axis=0).astype(np.uint8)
    return combined_features, combined_labels.squeeze()

def create_adaboost_cascade(features, labels, iterations_per_cascade, cascade_level):
    # Initialize weights
    weights = np.concatenate(
        (np.repeat(1 / np.sum(labels == 1), np.sum(labels == 1)),
         np.repeat(1 / np.sum(labels == 0), np.sum(labels == 0))),
        axis=0
    )
    best_classifier = None
    best_trust_factor = -np.inf

    for iteration in range(iterations_per_cascade):
        # Normalize weights
        weights = weights / np.sum(weights)

        # Create a weak classifier
        classifier = create_weak_classifier(features, labels, weights)
        class_feature_idx, class_thresh, class_polarity, class_error, class_predicted_labels = classifier

        # Adjust predicted labels for AdaBoost (+1, -1 convention)
        class_predicted_labels = np.where(class_predicted_labels == 0, -1, class_predicted_labels)

        # Calculate epsilon_t and trust factor
        epsilon_t = (np.matmul(weights.reshape(1, -1),
                               np.abs(class_predicted_labels - labels).reshape(-1, 1))
                     .squeeze() * 0.5)
        trust_factor = 0.5 * np.log((1 - epsilon_t) / epsilon_t)

        # Update weights
        weights = weights * np.exp(-trust_factor * labels * class_predicted_labels)

        # Calculate FPR and FNR
        FPR = np.sum(class_predicted_labels[np.sum(labels == 1):] == 1) / np.sum(labels == 0)
        FNR = 1 - (np.sum(class_predicted_labels[:np.sum(labels == 1)] == 1) / np.sum(labels == 1))

        print(f'Cascade Level = {cascade_level + 1}, Iteration = {iteration + 1}, epsilon_t = {round(epsilon_t, 4)}, '
              f'trust_factor = {round(trust_factor, 4)}, # Negatives = {np.sum(labels == 0)}, '
              f'FPR = {round(FPR * 100, 4)}%, FNR = {round(FNR * 100, 4)}%')

        # Update the best classifier
        if trust_factor > best_trust_factor:
            best_trust_factor = trust_factor
            best_class_predictions = class_predicted_labels
            best_FPR = FPR
            best_FNR = FNR
            best_weak_classifier = [
                class_feature_idx, class_thresh, class_polarity, class_error,
                class_predicted_labels, FPR, FNR, epsilon_t, trust_factor
            ]

    # Revise the dataset for the next cascade layer
    new_pos_features = features[:np.sum(labels == 1), :]
    new_neg_features = features[np.sum(labels == 1):, :]
    new_neg_features = new_neg_features[
        np.where(best_class_predictions[np.sum(labels == 1):] == 1), :
    ][0]

    # Create new labels
    new_features, new_labels = create_labels(new_pos_features, new_neg_features)
    del new_pos_features, new_neg_features

    return new_features, new_labels, best_FPR, best_FNR, best_weak_classifier

def create_weak_classifier(features, labels, weights):
    classifier_error = np.inf
    best_classifier = None

    for feature_idx in range(features.shape[1]):
        feature_vector = features[:, feature_idx]
        sorted_indices = np.argsort(feature_vector)
        feature_vector_sorted = feature_vector[sorted_indices]
        labels_sorted = labels[sorted_indices]
        weights_sorted = weights[sorted_indices]

        positive_weights = np.zeros_like(weights_sorted)
        negative_weights = np.zeros_like(weights_sorted)

        positive_weights[labels_sorted == 1] = weights_sorted[labels_sorted == 1]
        negative_weights[labels_sorted == 0] = weights_sorted[labels_sorted == 0]

        error_pol_1 = np.cumsum(positive_weights) + np.sum(negative_weights) - np.cumsum(negative_weights)
        error_pol_2 = np.cumsum(negative_weights) + np.sum(positive_weights) - np.cumsum(positive_weights)

        error = np.concatenate((error_pol_1[:, None], error_pol_2[:, None]), axis=1)
        min_idx = np.unravel_index(np.argmin(error), error.shape)
        min_err = error[min_idx]

        if min_err < classifier_error:
            classifier_error = min_err
            threshold = feature_vector_sorted[min_idx[0]]
            polarity = 1 if min_idx[1] == 0 else -1
            classifications = feature_vector >= threshold if polarity == 1 else feature_vector < threshold
            best_classifier = [feature_idx, threshold, polarity, classifier_error, classifications]

    return best_classifier

def create_adaboost_cascade(features, labels, iterations_per_cascade, cascade_level):
    weights = np.concatenate(
        (np.repeat(1 / np.sum(labels == 1), np.sum(labels == 1)),
         np.repeat(1 / np.sum(labels == 0), np.sum(labels == 0))),
        axis=0
    )
    best_trust_factor = -np.inf

    for iteration in range(iterations_per_cascade):
        # Normalize weights
        weights /= np.sum(weights)

        # Create a weak classifier
        classifier = create_weak_classifier(features, labels, weights)
        feature_idx, threshold, polarity, error, predicted_labels = classifier

        # Convert predictions to {+1, -1}
        predicted_labels = np.where(predicted_labels == 0, -1, predicted_labels)

        # Calculate epsilon_t and trust factor
        epsilon_t = np.sum(weights * (predicted_labels != labels).astype(float)) / 2
        trust_factor = 0.5 * np.log((1 - epsilon_t) / epsilon_t)

        # Update weights
        weights *= np.exp(-trust_factor * labels * predicted_labels)

        # Calculate FPR and FNR
        FPR = np.sum(predicted_labels[np.sum(labels == 1):] == 1) / np.sum(labels == 0)
        FNR = 1 - (np.sum(predicted_labels[:np.sum(labels == 1)] == 1) / np.sum(labels == 1))

        print(f'Cascade Level = {cascade_level + 1}, Iteration = {iteration + 1}, '
              f'epsilon_t = {round(epsilon_t, 4)}, trust_factor = {round(trust_factor, 4)}, '
              f'FPR = {round(FPR * 100, 4)}%, FNR = {round(FNR * 100, 4)}%')

        if trust_factor > best_trust_factor:
            best_trust_factor = trust_factor
            best_classifier = classifier
            best_FPR = FPR
            best_FNR = FNR

    # Revise the dataset for the next cascade layer
    positive_features = features[:np.sum(labels == 1)]
    negative_features = features[np.sum(labels == 1):]
    negative_features = negative_features[
        np.where(predicted_labels[np.sum(labels == 1):] == 1)
    ]

    new_features, new_labels = create_labels(positive_features, negative_features)
    return new_features, new_labels, best_FPR, best_FNR, best_classifier

def plot_cascade_adaboost_performance(FPRs, FNRs, iterations_per_cascade, num_cascades, save_dir, name, title=None):
    SMALL_SIZE = 10
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 14
    LINE_WIDTH = 3

    plt.rc('font', size=SMALL_SIZE)
    plt.rc('axes', titlesize=SMALL_SIZE)
    plt.rc('axes', labelsize=MEDIUM_SIZE)
    plt.rc('xtick', labelsize=MEDIUM_SIZE)
    plt.rc('ytick', labelsize=MEDIUM_SIZE)
    plt.rc('legend', fontsize=SMALL_SIZE)
    plt.rc('figure', titlesize=BIGGER_SIZE)

    plt.figure()
    plt.plot(FPRs, '-*', label='FPR', linewidth=LINE_WIDTH)
    plt.plot(FNRs, '-d', label='FNR', linewidth=LINE_WIDTH)
    plt.legend()
    plt.xlabel('Number of Cascade Levels')
    plt.ylabel('Performance Metrics')
    plt.xticks(np.arange(len(FPRs)), [str(ix + 1) for ix in range(len(FPRs))])
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'Adaboost_Performance_{iterations_per_cascade}_{num_cascades}_{name}.pdf'))
    plt.savefig(os.path.join(save_dir, f'Adaboost_Performance_{iterations_per_cascade}_{num_cascades}_{name}.png'), dpi=600)

from typing import List
def plot_fp_fn_rates(fp_rates: List[float], fn_rates: List[float], file_name: str = None):
    stages = range(1, len(fp_rates) + 1)
    fp_rates = fp_rates[:-1] 
    fn_rates = fn_rates[:-1] 
    fp_rates[-1] = 0
    fn_rates[-1] = 0
    
    plt.plot(stages, fp_rates, label='False Positive Rate')
    plt.plot(stages, fn_rates, label='False Negative Rate')
    plt.xlabel('Stage Number')
    plt.ylabel('Rate')
    plt.title(f'False Positive and False Negative Rates per Stage for {file_name} dataset')
    plt.legend()
    plt.grid()
    plt.savefig(f'MyResults/{file_name}_FP_FN.png')
    return

def main():
    classes = ['positive', 'negative']
    train_dir = 'CarDetection/train'
    test_dir = 'CarDetection/test'
    result_dir = './MyResults/Adaboost'
    train = True
    num_cascades = 9
    iterations_per_cascade = [i for i in range(1, 20, 3)]
    iterations_per_cascade = [i for i in range(1, 20, 1)]

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    for iter_cascade in iterations_per_cascade:
        current_FPR = 1
        current_FNR = 1
        FPRs = []
        FNRs = []
        tol = 1e-6
        best_classifier_per_cascade_level = {}

        if train:
            train_pos_features, train_neg_features = extract_features(classes, train_dir, result_dir, 'train')
            train_features, train_labels = create_labels(train_pos_features, train_neg_features)
            del train_pos_features, train_neg_features

            for ix in range(num_cascades):
                train_features, train_labels, best_FPR, best_FNR, best_weak_classifier = create_adaboost_cascade(
                    train_features, train_labels, iter_cascade, ix)
                best_classifier_per_cascade_level[str(ix + 1)] = best_weak_classifier
                current_FPR *= best_FPR
                current_FNR *= best_FNR
                FPRs.append(current_FPR)
                FNRs.append(current_FNR)
                print(f'Cumulative FPR = {round(current_FPR, 4)}, Cumulative FNR = {round(current_FNR, 4)}')

                if current_FPR <= tol:
                    print(f"Reached FPR Tolerance level of {tol} after cascade level = {ix + 1}")
                    if np.sum(train_labels == 0) == 0:
                        print("No negative labeled images remain after cascade level =", ix + 1)
                    break

            with open(os.path.join(result_dir, f'Adaboost_Performance_{iter_cascade}.pkl'), 'wb') as f:
                pickle.dump(FPRs, f)
                pickle.dump(FNRs, f)

        else:
            print("Testing phase not implemented in this script!")
    plot_fp_fn_rates(FPRs, FNRs, 'train')
    plot_cascade_adaboost_performance(FPRs, FNRs, iterations_per_cascade, num_cascades, result_dir, 'Adaboost')
if __name__ == '__main__':
    main()
