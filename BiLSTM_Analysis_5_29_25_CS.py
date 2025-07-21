# -*- coding: utf-8 -*-
"""
Author: stoutcam
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
from scipy.io import loadmat
from scipy.stats import skew
from scipy.signal import welch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.dummy import DummyClassifier
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Masking, Bidirectional, LSTM, Dropout, Dense, concatenate
from tensorflow.keras.utils import to_categorical, plot_model

# Configuration
TARGET_TIME_STEPS = None   # None to infer from training data
N_PERMUTATIONS = 10000    # Permutations for p-value test

# Map MAT variables to numeric labels
base_map = {
    '12Z':           0,
    'H1644_ESC':     1,
    'H1657_EEC':     2,
    'iEc_ESC':       3,
    'Media':         4,
}
odor_label_mapping = {}
for key, idx in base_map.items():
    odor_label_mapping[f"C_{key}_spikes"] = idx
    odor_label_mapping[f"A_{key}_spikes"] = idx

# GUI file picker helper
def select_file(title):
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(title=title, filetypes=[("MAT files","*.mat"),("All files","*")])
    root.destroy()
    return Path(path) if path else None

# Load sequences and compute static features
def load_data(mat_path, odor_map, fs=20):
    mat = loadmat(str(mat_path))
    seqs, feats, labels = [], [], []
    for odor, lbl in odor_map.items():
        if odor not in mat:
            print(f"Warning: '{odor}' missing in {mat_path.name}")
            continue
        data = mat[odor]
        if data.ndim == 2:
            data = data[:, :, np.newaxis]
        for t in range(data.shape[2]):
            arr = np.sum(data[:, :, t], axis=1)
            seqs.append(arr)
            sk_val = skew(arr)
            freqs, psd = welch(arr, fs=fs)
            if psd.sum() == 0:
                cent, bw = 0.0, 0.0
            else:
                cent = np.sum(freqs * psd) / np.sum(psd)
                bw = np.sqrt(np.sum(((freqs - cent)**2) * psd) / np.sum(psd))
            nz = np.where(arr > 0)[0]
            latency = float(nz[0]) if nz.size > 0 else float(len(arr))
            feats.append([sk_val, cent, bw, latency])
            labels.append(lbl)
    X_seq = np.stack(seqs)[..., np.newaxis]
    X_stat = np.array(feats)
    y = np.array(labels)
    return X_seq, X_stat, y

# Pad/truncate sequences to a fixed length
# (or use the length of the training data if not specified)
def adjust_time_steps(X, target):
    n, t, c = X.shape
    if t < target:
        pad = ((0,0), (0, target - t), (0,0))
        return np.pad(X, pad, mode='constant', constant_values=0)
    return X[:, :target, :]

# Build BiLSTM
def build_model(steps, n_classes):
    inp_seq = Input((steps,1), name='seq_input')
    x = Masking(0.0)(inp_seq)
    x = Bidirectional(LSTM(64))(x)
    x = Dropout(0.2)(x)
    inp_stat = Input((4,), name='stat_input')
    x = concatenate([x, inp_stat])
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    out = Dense(n_classes, activation='softmax')(x)
    model = Model([inp_seq, inp_stat], out)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Main function to run the model
def main():
    train_file = select_file("Select training MAT file")
    if not train_file:
        print("No training file selected. Exiting.")
        return
    eval_file = select_file("Select eval MAT file (Cancel to split)")

    print(f"Loading training data from {train_file}")
    X_seq_all, X_stat_all, y_all = load_data(train_file, odor_label_mapping)
    if X_seq_all.size == 0:
        print("No training data found. Exiting.")
        return
    n_classes = int(y_all.max()) + 1
    y_cat = to_categorical(y_all, n_classes)

    steps = TARGET_TIME_STEPS or X_seq_all.shape[1]
    print(f"Using {steps} time bins per sequence.")
    X_seq_all = adjust_time_steps(X_seq_all, steps)

    if eval_file:
        print(f"Loading evaluation data from {eval_file}")
        X_seq_test, X_stat_test, y_test = load_data(eval_file, odor_label_mapping)
        X_seq_test = adjust_time_steps(X_seq_test, steps)
        y_test_cat = to_categorical(y_test, n_classes)
        X_seq_train, X_stat_train, y_train_cat = X_seq_all, X_stat_all, y_cat
    else:
        X_seq_train, X_seq_test, X_stat_train, X_stat_test, y_train_cat, y_test_cat = train_test_split(
            X_seq_all, X_stat_all, y_cat,
            test_size=0.2,
            random_state=42,
            stratify=y_all
        )

    model = build_model(steps, n_classes)
    model.summary()
    tb_cb = tf.keras.callbacks.TensorBoard(log_dir='logs', histogram_freq=1)

    model.fit(
        [X_seq_train, X_stat_train], y_train_cat,
        validation_data=([X_seq_test, X_stat_test], y_test_cat),
        epochs=25, batch_size=1,
        callbacks=[
            tf.keras.callbacks.EarlyStopping('val_loss', patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau('val_loss', factor=0.5, patience=5),
            tb_cb
        ]
    )

    loss, acc = model.evaluate([X_seq_test, X_stat_test], y_test_cat, verbose=0)
    print(f"RNN Test accuracy: {acc:.3f}")
    y_pred = model.predict([X_seq_test, X_stat_test]).argmax(axis=1)
    y_true = y_test_cat.argmax(axis=1)
    names = [k for k,_ in sorted(odor_label_mapping.items(), key=lambda x: x[1])]

    cm = confusion_matrix(y_true, y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=names, yticklabels=names, ax=axes[0])
    axes[0].set(title='RNN Confusion (Count)', xlabel='Predicted', ylabel='Actual')
    sns.heatmap(cm.astype('float')/cm.sum(axis=1)[:, None], annot=True, fmt='.2f', cmap='Reds', xticklabels=names, yticklabels=names, ax=axes[1])
    axes[1].set(title='RNN Confusion (Percent)', xlabel='Predicted', ylabel='Actual')
    plt.show()

    # Baselines
    n_tr, n_te = X_seq_train.shape[0], X_seq_test.shape[0]
    Xf_tr = np.hstack([X_seq_train.reshape(n_tr, -1), X_stat_train])
    Xf_te = np.hstack([X_seq_test.reshape(n_te, -1), X_stat_test])
    y_lbl = y_train_cat.argmax(axis=1)
    d_rand = DummyClassifier(strategy='uniform', random_state=42).fit(Xf_tr, y_lbl)
    d_maj  = DummyClassifier(strategy='most_frequent').fit(Xf_tr, y_lbl)
    for nm, yp in [('Random', d_rand.predict(Xf_te)), ('Majority', d_maj.predict(Xf_te))]:
        cm_b = confusion_matrix(y_true, yp)
        fig, ax = plt.subplots(1, 2, figsize=(16,7))
        sns.heatmap(cm_b, annot=True, fmt='d', cmap='Oranges', xticklabels=names, yticklabels=names, ax=ax[0])
        ax[0].set(title=f'{nm} Baseline (Count)')
        sns.heatmap(cm_b.astype('float')/cm_b.sum(axis=1)[:, None], annot=True, fmt='.2f', cmap='Purples', xticklabels=names, yticklabels=names, ax=ax[1])
        ax[1].set(title=f'{nm} Baseline (Percent)')
        plt.show()
        print(f"{nm} accuracy: {np.mean(yp == y_true):.3f}")
        print(classification_report(y_true, yp, target_names=names))

    actual_acc = np.mean(y_pred == y_true)
    perms = [np.mean(y_pred == np.random.permutation(y_true)) for _ in range(N_PERMUTATIONS)]
    perms = np.array(perms)
    pval = (np.sum(perms >= actual_acc) + 1) / (N_PERMUTATIONS + 1)
    print(f"Permutation test p-value: {pval:.3f}")
    plt.figure(figsize=(8,4))
    sns.histplot(perms, bins=30, kde=False)
    plt.axvline(actual_acc, color='red', linestyle='--', label=f'Actual acc={actual_acc:.3f}')
    plt.title('Permutation Null Distribution')
    plt.xlabel('Accuracy')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
