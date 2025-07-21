# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('TkAgg')

from pathlib import Path
import os
import json
import gzip
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from scipy.stats import skew, kurtosis
from scipy.signal import welch
from sklearn.tree import export_graphviz, plot_tree

# Model and data directories
MODEL_DIRECTORY = Path(r"C:\Users\stoutcam\Desktop\Saha Lab\Python")
MODEL_FILENAME = 'rf_model_.joblib'
MODEL_PATH = MODEL_DIRECTORY / MODEL_FILENAME
print("MODEL_PATH =", MODEL_PATH)

ROOT_DIRECTORY = Path(r"C:\Users\stoutcam\Desktop\Saha Lab\PFAS\Pure\NeuralRecordings_Python")
ANALYSIS_DIRECTORY = Path(r"C:\Users\stoutcam\Desktop\Saha Lab\PFAS\Pure\NeuralRecordings_Python")

# Odor mapping and feature configuration
odor_label_mapping = {
    'Air': 0,
    'EtFOSE': 1,
    'FTAcr_6_2': 2,
    'FTOH_6_2': 3,
    'FTOH_8_2': 4,
    'MeFOSE': 5,
    'MeFOSA': 6,
    'PFOSA': 7,
}

# Define the features to be used for training and prediction.
statistical_features = ['mean', 'std', 'skewness']
spectral_features = ['spectral_centroid', 'spectral_bandwidth']
additional_features = ['latency', 'frequency', 'mode']
NUM_FEATURES = len(statistical_features) + len(spectral_features) + len(additional_features)

exclude_folders = {'plots', 'Position_1', 'Position_2', 'Position_3', 'Neural_Recording_json'}

SAMPLING_RATE = 20000
BIN_DURATION = 0.1
trialS_PER_BIN = int(BIN_DURATION * SAMPLING_RATE)
print(f"BIN_DURATION = {BIN_DURATION}s, trialS_PER_BIN (raw) = {trialS_PER_BIN} samples/bin")

START_TIME = 5
END_TIME = 12

def save_plot(plot_name):
    figs_dir = Path(r"C:\Users\stoutcam\Desktop\Figures")
    figs_dir.mkdir(exist_ok=True)
    filename = figs_dir / f"{plot_name}.png"
    plt.savefig(filename, bbox_inches="tight")
    print(f"Saved plot as {filename}")

def prompt_save_model(model, filename='rf_model2.joblib'):
    ans = input(f"Save the trained model as '{filename}'? (y/n): ").strip().lower()
    if ans == 'y':
        joblib.dump(model, filename)
        print(f"Model saved as '{filename}'")
    else:
        print("Model was not saved.")

def extract_odor_label(file_path):
    p = Path(file_path)
    parent_name = p.parent.name
    if parent_name.lower() == "neural_recording_json" and p.parent.parent:
        odor_folder = p.parent.parent.name
    else:
        odor_folder = parent_name
    match = re.match(r'^(?P<odor>.+?)\((?P<position>\d+)\)$', odor_folder)
    if match:
        odor_name = match.group('odor').strip()
        odor_label = odor_label_mapping.get(odor_name)
        if odor_label is not None:
            return odor_label, odor_name
        else:
            print(f"Warning: Odor '{odor_name}' not found in odor_label_mapping.")
            return None, None
    else:
        print(f"Warning: Unable to parse odor from folder '{odor_folder}'.")
        return None, None

def extract_features(amplitude_data, sampling_rate=20000):
    amplitude_data = np.array(amplitude_data)
    features = {}
    features['mean'] = np.mean(amplitude_data)
    features['std'] = np.std(amplitude_data)
    features['skewness'] = skew(amplitude_data)
    
    freqs, psd = welch(amplitude_data, fs=sampling_rate)
    if np.sum(psd) == 0:
        features['spectral_centroid'] = 0
        features['spectral_bandwidth'] = 0
    else:
        features['spectral_centroid'] = np.sum(freqs * psd) / np.sum(psd)
        features['spectral_bandwidth'] = np.sqrt(np.sum(((freqs - features['spectral_centroid'])**2) * psd) / np.sum(psd))
    
    max_index = np.argmax(amplitude_data)
    features['latency'] = max_index / sampling_rate

    if len(psd) > 0:
        dominant_freq = freqs[np.argmax(psd)]
        features['frequency'] = dominant_freq
    else:
        features['frequency'] = 0

    rounded_data = np.round(amplitude_data, 2)
    values, counts = np.unique(rounded_data, return_counts=True)
    mode_value = values[np.argmax(counts)]
    features['mode'] = mode_value

    return features

def load_data(root_directory):
    bin_data_records = []
    trial_ids = []
    trial_file_names = []
    trial_id_counter = 0
    NORMALIZATION_DURATION = 5.0  # seconds

    for root, dirs, files in os.walk(root_directory):
        if 'Neural_Recording_json' in dirs:
            neural_dir = Path(root) / 'Neural_Recording_json'
            json_files = list(neural_dir.glob('*.json.gz'))
            for json_file in json_files:
                odor_label, odor_name = extract_odor_label(json_file)
                if odor_label is None:
                    continue
                try:
                    with gzip.open(json_file, 'rt', encoding='utf-8') as f:
                        data = json.load(f)
                except Exception as e:
                    print(f"Issue loading {json_file}: {e}")
                    continue
                amplitude_data = data.get('amplifier_data')
                t_amplifier = data.get('t_amplifier')
                if amplitude_data is None or t_amplifier is None:
                    print(f"No amplitude/time in {json_file}")
                    continue
                amplitude_data = amplitude_data[0]
                amplitude_data = np.array(amplitude_data)
                t_amplifier = np.array(t_amplifier)
                if len(t_amplifier) != len(amplitude_data):
                    print(f"Length mismatch in {json_file}")
                    continue
                t_amplifier = t_amplifier - t_amplifier[0]
                # Baseline normalization
                baseline_mask = t_amplifier < NORMALIZATION_DURATION
                if np.any(baseline_mask):
                    baseline = np.mean(amplitude_data[baseline_mask])
                    amplitude_data = amplitude_data - baseline
                else:
                    print(f"Warning: Not enough data in first {NORMALIZATION_DURATION}s in {json_file.name}")
                
                mask = (t_amplifier >= START_TIME) & (t_amplifier <= END_TIME)
                if not np.any(mask):
                    print(f"No voltage in time window for {json_file}")
                    continue
                amplitude_data = amplitude_data[mask]
                t_amplifier = t_amplifier[mask]
                print(f"{json_file.name} -> Using {len(amplitude_data)} samples in range {START_TIME}-{END_TIME}s")
                
                if len(t_amplifier) < int(SAMPLING_RATE * BIN_DURATION):
                    trialS_PER_BIN_effective = 100
                    effective_sampling_rate = 1.0 / BIN_DURATION  # 20 Hz for RMS data
                    print(f"Detected RMS data in {json_file.name}: using trialS_PER_BIN_effective = {trialS_PER_BIN_effective}")
                else:
                    trialS_PER_BIN_effective = trialS_PER_BIN
                    effective_sampling_rate = SAMPLING_RATE
                
                if len(amplitude_data) < trialS_PER_BIN_effective:
                    print(f"Not enough data in {json_file}")
                    continue
                num_bins = int(len(amplitude_data) / trialS_PER_BIN_effective)
                for i in range(num_bins):
                    bin_start = i * trialS_PER_BIN_effective
                    bin_end = bin_start + trialS_PER_BIN_effective
                    bin_amplitude = amplitude_data[bin_start:bin_end]
                    feats = extract_features(bin_amplitude, sampling_rate=effective_sampling_rate)
                    feats['label'] = odor_label
                    feats['odor_name'] = odor_name
                    bin_data_records.append(feats)
                    trial_ids.append(trial_id_counter)
                    trial_file_names.append(json_file.name)
                trial_id_counter += 1

    bin_dataset = pd.DataFrame(bin_data_records)
    bin_dataset['trial_id'] = trial_ids
    bin_dataset['file_name'] = trial_file_names
    return bin_dataset

def analyze_specific_file(file_path, model):
    """
    Analyzes a single JSON.GZ file using the saved model.
    Returns (predicted_label, actual_label).
    """
    file_path = Path(file_path)
    odor_label, odor_name = extract_odor_label(file_path)
    if odor_label is None:
        print(f"Skipping file {file_path.name} due to invalid odor folder.")
        return None, None

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
        return None, odor_label

    amplitude_data = amplitude_data[0]
    amplitude_data = np.array(amplitude_data)
    t_amplifier = np.array(t_amplifier)
    t_amplifier = t_amplifier - t_amplifier[0]

    NORMALIZATION_DURATION = 5.0
    baseline_mask = t_amplifier < NORMALIZATION_DURATION
    if np.any(baseline_mask):
        baseline = np.mean(amplitude_data[baseline_mask])
        amplitude_data = amplitude_data - baseline
    else:
        print(f"Warning: Not enough data in first {NORMALIZATION_DURATION}s in {file_path.name}")

    if len(t_amplifier) < int(SAMPLING_RATE * BIN_DURATION):
        trialS_PER_BIN_effective = 100
        effective_sampling_rate = 1.0 / BIN_DURATION  # 20 Hz for RMS data
        print(f"Detected RMS data in {file_path.name}: using trialS_PER_BIN_effective = {trialS_PER_BIN_effective}")
    else:
        trialS_PER_BIN_effective = trialS_PER_BIN
        effective_sampling_rate = SAMPLING_RATE

    mask = (t_amplifier >= START_TIME) & (t_amplifier <= END_TIME)
    if not np.any(mask):
        print(f"No voltage in time window for {file_path}")
        return None, odor_label
    amplitude_data = amplitude_data[mask]
    t_amplifier = t_amplifier[mask]
    
    num_bins = int(len(amplitude_data) / trialS_PER_BIN_effective)
    if num_bins < 1:
        print(f"File {file_path.name} => Not enough data in time window.")
        return None, odor_label

    bin_features_list = []
    for i in range(num_bins):
        bin_start = i * trialS_PER_BIN_effective
        bin_end = bin_start + trialS_PER_BIN_effective
        bin_amplitude = amplitude_data[bin_start:bin_end]
        feats = extract_features(bin_amplitude, sampling_rate=effective_sampling_rate)
        bin_features_list.append(feats)

    bin_features_df = pd.DataFrame(bin_features_list)
    feature_columns = statistical_features + spectral_features + additional_features
    X_bins = bin_features_df[feature_columns]

    y_pred_bins = model.predict(X_bins)
    label_counts = Counter(y_pred_bins)
    most_common_label, _ = label_counts.most_common(1)[0]
    predicted_label = most_common_label

    return predicted_label, odor_label

def main():
    # Load or train the model
    if MODEL_PATH.is_file():
        clf = joblib.load(MODEL_PATH)
        print(f"Loaded the model from '{MODEL_PATH}'")
    else:
        print(f"Model file '{MODEL_PATH}' does not exist. Training a new model.")
        bin_dataset = load_data(ROOT_DIRECTORY)
        if bin_dataset.empty:
            print("No data available for training.")
            return
        feature_columns = statistical_features + spectral_features + additional_features
        X = bin_dataset[feature_columns]
        y = bin_dataset['label']
        trial_ids = bin_dataset['trial_id']
        file_names = bin_dataset['file_name']
        X_train, X_test, y_train, y_test, _, _, _, _ = train_test_split(
            X, y, trial_ids, file_names, test_size=0.2, random_state=42, stratify=y)
        clf = RandomForestClassifier(n_estimators=1000, random_state=42, n_jobs=-1,
                                     max_depth=10, min_samples_leaf=15)
        clf.fit(X_train, y_train)
        print("Training complete.")

    # Recursively find all .json.gz files in ANALYSIS_DIRECTORY
    all_json_files = []
    for root, dirs, files in os.walk(ANALYSIS_DIRECTORY):
        for f in files:
            if f.lower().endswith('.json.gz'):
                all_json_files.append(Path(root) / f)
    if not all_json_files:
        print(f"No .json.gz files found under {ANALYSIS_DIRECTORY}")
        return

    # Filter: keep only files in valid odor subfolders (matching the dictionary)
    files_to_analyze = []
    for file_path in all_json_files:
        odor_label, odor_name = extract_odor_label(file_path)
        if odor_label is not None:
            files_to_analyze.append((file_path, odor_label, odor_name))
    if not files_to_analyze:
        print("No files in directories matching an odor in the dictionary were found.")
        return

    print(f"Found {len(files_to_analyze)} files in valid odor directories. Analyzing all...")

    file_predicted_labels = []
    file_actual_labels = []
    for file_path, odor_label, odor_name in files_to_analyze:
        print(f"\nAnalyzing file: {file_path}")
        predicted_label, actual_label = analyze_specific_file(file_path, clf)
        if predicted_label is not None and actual_label is not None:
            file_predicted_labels.append(predicted_label)
            file_actual_labels.append(actual_label)

    if not file_predicted_labels or not file_actual_labels:
        print("No valid predictions were made for the selected files.")
        return

    all_labels = list(odor_label_mapping.values())
    all_odor_names = list(odor_label_mapping.keys())
    print("\nFile-Level Classification Report:")
    print(classification_report(
        file_actual_labels,
        file_predicted_labels,
        labels=all_labels,
        target_names=all_odor_names,
        zero_division=0
    ))
    cm_files = confusion_matrix(file_actual_labels, file_predicted_labels, labels=all_labels)

    fig, axs = plt.subplots(1, 3, figsize=(24, 6))

    # Subplot 1: Confusion Matrix (Counts)
    sns.heatmap(cm_files, annot=True, fmt='d', cmap='Reds',
                xticklabels=all_odor_names, yticklabels=all_odor_names, ax=axs[0])
    axs[0].set_xlabel('Predicted')
    axs[0].set_ylabel('Actual')
    axs[0].set_title('Confusion Matrix (Counts)')

    # Subplot 2: Normalized Confusion Matrix (%)
    # Normalize each row (true label) so that its sum is 100%
    norm_cm_files = cm_files.astype(float)
    row_sums = norm_cm_files.sum(axis=1, keepdims=True)
    # Avoid division by zero if a row sum is zero.
    norm_cm_files = np.divide(norm_cm_files, row_sums, where=row_sums!=0) * 100
    sns.heatmap(norm_cm_files, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=all_odor_names, yticklabels=all_odor_names, ax=axs[1])
    axs[1].set_xlabel('Predicted')
    axs[1].set_ylabel('Actual')
    axs[1].set_title('Normalized Confusion Matrix (%)')

    # Subplot 3: Feature Importance Chart
    feature_names = statistical_features + spectral_features + additional_features
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    axs[2].bar(range(len(importances)), importances[indices], align="center")
    axs[2].set_xticks(range(len(importances)))
    axs[2].set_xticklabels([feature_names[i] for i in indices], rotation=45)
    axs[2].set_title("Feature Importances")
    axs[2].set_ylabel("Importance")

    plt.tight_layout()
    save_plot("Combined_Classification_Figure")
    plt.show()

    prompt_save_model(clf, filename='rf_model.joblib')

if __name__ == '__main__':
    main()
