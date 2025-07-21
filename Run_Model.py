# Created by Camron Stout

import os
import json
import gzip
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
import joblib

from sklearn.metrics import classification_report, confusion_matrix
from scipy.stats import skew, kurtosis
from scipy.signal import welch

# ============================
# Configuration Section
# ============================

# Model configuration
MODEL_DIRECTORY = Path(r"C:\Users\stoutcam\Desktop")  # Update this path to your model directory
MODEL_FILENAME = 'odor_classification_model.pkl'
MODEL_PATH = MODEL_DIRECTORY / MODEL_FILENAME

# Top-level directory to search for subfolders
ANALYSIS_DIRECTORY = Path(r"C:\Users\stoutcam\Desktop\Neural_recordings_MM_Test\08_22_2024")

# Define your odor labels
odor_label_mapping = {
    'Air': 0,
    'EtFOSE': 1,
    'FTAcr_6_2': 2,
    'FTOH_6_2': 3,
    'FTOH_8_2': 4,
    'MeFOSA': 5,
    'MeFOSE': 6,
    'PFOSA': 7,
}

statistical_features = ['mean', 'std', 'skewness', 'kurtosis']
spectral_features = ['spectral_centroid', 'spectral_bandwidth']

# Sampling rate (Hz)
SAMPLING_RATE = 20000  # 20,000 Hz

# Time bin duration in seconds
BIN_DURATION = 0.1

# Number of samples per bin
SAMPLES_PER_BIN = int(BIN_DURATION * SAMPLING_RATE)  # 0.1 * 20000 = 2000 samples

# Time window for analysis (in seconds)
START_TIME = 5.5
END_TIME = 6.0

# ============================
# Helper Functions
# ============================

def is_valid_odor_subfolder(folder_name):
    """
    Check if folder_name matches the pattern: 'OdorName(Position#)',
    and that 'OdorName' is in odor_label_mapping.

    Returns (odor_label, odor_name) if valid, otherwise (None, None).
    """
    match = re.match(r'^(?P<odor>.+?)\((?P<position>\d+)\)$', folder_name)
    if match:
        odor_name = match.group('odor').strip()
        odor_label = odor_label_mapping.get(odor_name)
        if odor_label is not None:
            return odor_label, odor_name
    return None, None

def extract_features(amplitude_data, sampling_rate=20000):
    """
    Extracts statistical and spectral features from amplitude data.
    """
    # Convert amplitude data to numpy array if it's not
    amplitude_data = np.array(amplitude_data)

    features = {}
    # Statistical Features
    features['mean'] = np.mean(amplitude_data)
    features['std'] = np.std(amplitude_data)
    features['skewness'] = skew(amplitude_data)
    features['kurtosis'] = kurtosis(amplitude_data)

    # Spectral Features
    freqs, psd = welch(amplitude_data, fs=sampling_rate)
    if np.sum(psd) == 0:
        # Avoid divide-by-zero
        features['spectral_centroid'] = 0
        features['spectral_bandwidth'] = 0
    else:
        features['spectral_centroid'] = np.sum(freqs * psd) / np.sum(psd)
        features['spectral_bandwidth'] = np.sqrt(
            np.sum(((freqs - features['spectral_centroid'])**2) * psd) / np.sum(psd)
        )
    return features

def analyze_file(file_path, odor_label, odor_name, model):
    """
    Analyzes a single .json.gz file using the saved model,
    returning (predicted_label, actual_label).
    """
    # Load the compressed JSON file
    try:
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None

    amplitude_data = data.get('amplifier_data')
    t_amplifier = data.get('t_amplifier')
    if amplitude_data is None or t_amplifier is None:
        print(f"No amplifier_data or t_amplifier in {file_path}")
        return None, None

    # amplitude_data is a list of lists (channels x time)
    amplitude_data = amplitude_data[0]  # first channel
    amplitude_data = np.array(amplitude_data)
    t_amplifier = np.array(t_amplifier)

    # Extract the specified time window
    start_index = int(START_TIME * SAMPLING_RATE)
    end_index = int(END_TIME * SAMPLING_RATE)
    if end_index > len(amplitude_data):
        end_index = len(amplitude_data)
    amplitude_data = amplitude_data[start_index:end_index]
    t_amplifier = t_amplifier[start_index:end_index]

    # Split data into bins
    num_bins = int(len(amplitude_data) / SAMPLES_PER_BIN)
    if num_bins < 1:
        print(f"File {file_path.name} => Not enough data for at least 1 bin in time window.")
        return None, None

    bin_features_list = []
    for i in range(num_bins):
        bin_start = i * SAMPLES_PER_BIN
        bin_end = bin_start + SAMPLES_PER_BIN
        bin_amplitude = amplitude_data[bin_start:bin_end]

        # Extract features from the bin
        features = extract_features(bin_amplitude, sampling_rate=SAMPLING_RATE)
        bin_features_list.append(features)

    bin_features_df = pd.DataFrame(bin_features_list)
    feature_columns = statistical_features + spectral_features
    X_bins = bin_features_df[feature_columns]

    # Predict using the saved model
    from collections import Counter
    y_pred_bins = model.predict(X_bins)

    # Aggregate bin predictions => file-level
    label_counts = Counter(y_pred_bins)
    most_common_label, count = label_counts.most_common(1)[0]
    predicted_label = most_common_label

    # Convert labels to odor names
    label_to_odor = {v: k for k, v in odor_label_mapping.items()}
    predicted_odor_name = label_to_odor.get(predicted_label, "Unknown")

    print(f"\nFile: {file_path}")
    print(f"Predicted odor => {predicted_odor_name}")
    print(f"Actual odor    => {odor_name}")

    # For each bin, compare predicted vs. actual
    y_true_bins = np.full_like(y_pred_bins, odor_label)

    # Generate bin-level classification info
    all_labels = list(odor_label_mapping.values())
    all_odor_names = list(odor_label_mapping.keys())

    print("\nClassification Report (Bins):")
    from sklearn.metrics import classification_report, confusion_matrix
    print(classification_report(
        y_true_bins,
        y_pred_bins,
        labels=all_labels,
        target_names=all_odor_names,
        zero_division=0
    ))

    print("Confusion Matrix (Bins):")
    cm_bins = confusion_matrix(y_true_bins, y_pred_bins, labels=all_labels)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm_bins, annot=True, fmt='d',
                xticklabels=all_odor_names,
                yticklabels=all_odor_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix (Bins)')
    plt.show()

    return predicted_label, odor_label


def main():
    # 1) Load the saved model
    if MODEL_PATH.is_file():
        clf = joblib.load(MODEL_PATH)
        print(f"Loaded the model from '{MODEL_PATH}'")
    else:
        print(f"Model file '{MODEL_PATH}' does not exist.")
        return

    # 2) Recursively search for directories that match OdorName(Position#),
    #    and are in the dictionary.
    all_json_files = []
    # We'll keep track of (file_path, odor_label, odor_name) for each valid subfolder.
    files_to_analyze = []

    for root, dirs, files in os.walk(ANALYSIS_DIRECTORY):
        folder_name = Path(root).name
        # Check if it's a valid odor subfolder
        odor_label, odor_name = is_valid_odor_subfolder(folder_name)
        if odor_label is not None:
            # Found a directory named like 'Air(1)' or 'FTOH_6_2(2)' etc.
            # Let's see if there are any .json.gz in it
            # or in a subfolder named 'Neural_Recording_json'?
            # If .json.gz are directly in `root`, gather them:
            for f in files:
                if f.lower().endswith('.json.gz'):
                    file_path = Path(root) / f
                    files_to_analyze.append((file_path, odor_label, odor_name))

            # If you specifically want them inside 'Neural_Recording_json' under this folder:
            # for subdir in dirs:
            #     if subdir == 'Neural_Recording_json':
            #         neural_dir = Path(root) / subdir
            #         for jf in neural_dir.glob('*.json.gz'):
            #             files_to_analyze.append((jf, odor_label, odor_name))

    if not files_to_analyze:
        print("No subfolders match an odor in the dictionary, or no .json.gz files found.")
        return

    # Limit to 5 if you want
    # files_to_analyze = files_to_analyze[:5]

    # 3) Analyze each valid file
    file_predicted_labels = []
    file_actual_labels = []

    for file_path, odor_label, odor_name in files_to_analyze:
        # Analyze the file with the known odor_label, odor_name
        predicted_label, actual_label = analyze_file(file_path, odor_label, odor_name, clf)
        if predicted_label is not None and actual_label is not None:
            file_predicted_labels.append(predicted_label)
            file_actual_labels.append(actual_label)

    # 4) Summarize at the file level
    if file_predicted_labels and file_actual_labels:
        all_labels = list(odor_label_mapping.values())
        all_odor_names = list(odor_label_mapping.keys())

        print("\nFile-Level Classification Report:")
        from sklearn.metrics import classification_report, confusion_matrix
        print(classification_report(
            file_actual_labels,
            file_predicted_labels,
            labels=all_labels,
            target_names=all_odor_names,
            zero_division=0
        ))

        print("Confusion Matrix (Files):")
        cm_files = confusion_matrix(file_actual_labels, file_predicted_labels, labels=all_labels)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm_files, annot=True, fmt='d',
                    xticklabels=all_odor_names,
                    yticklabels=all_odor_names)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix (Files)')
        plt.show()
    else:
        print("\nNo valid file-level predictions were made.")


def is_valid_odor_subfolder(folder_name):
    """
    Check if folder_name matches the pattern: 'OdorName(Position#)',
    and that 'OdorName' is in odor_label_mapping.

    Returns (odor_label, odor_name) if valid, otherwise (None, None).
    """
    match = re.match(r'^(?P<odor>.+?)\((?P<position>\d+)\)$', folder_name)
    if match:
        odor_name = match.group('odor').strip()
        odor_label = odor_label_mapping.get(odor_name)
        if odor_label is not None:
            return odor_label, odor_name
    return None, None


if __name__ == '__main__':
    main()
