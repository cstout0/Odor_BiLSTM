#! /bin/env python3
# Adrian Foy September 2023 (Modified by Camron Stout October 2024)

"""
Module to read Intan Technologies RHD2000 data files from all subfolders within
a specified root directory, serialize the amplifier data to compressed JSON,
generate amplitude plots, and produce a peristimulus time histogram (PSTH).
Additionally, for every 5 trials a combined PSTH is produced.
Processes only the first 5 RHD files in each odor subfolder.
"""

import sys
import time
import re
import matplotlib.pyplot as plt
from pathlib import Path
import json
import numpy as np
import gzip

# Define the load_intan_rhd_format folder path to import functions
sys.path.append(r'C:\Users\stoutcam\Desktop\Saha Lab\Python\load_intan_rhd_format_REQUIRED')

from intanutil.header import read_header, header_to_result
from intanutil.data import (
    calculate_data_size,
    read_all_data_blocks,
    check_end_of_file,
    parse_data,
    data_to_result
)
from intanutil.filter import apply_notch_filter

# ============================
# Configuration Section
# ============================

# Define the root directory to search for RHD files
ROOT_DIRECTORY = Path(r"C:\Users\stoutcam\Desktop\12_08_2022")

# Define the base directory where plots will be saved
BASE_PLOTS_DIRECTORY = Path(r"C:\Users\stoutcam\Desktop\12_08_2022\plots")

# Ensure that the base plots directory exists; create it if it doesn't
BASE_PLOTS_DIRECTORY.mkdir(parents=True, exist_ok=True)

# Define folders to exclude from processing
exclude_folders = {
    'plots',
    'Position_1',
    'Position_2',
    'Position_3',
    'Neural_Recording_json'
}

# Define your odor labels
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

PLOT_X_START_TIME = 3.0
PLOT_X_END_TIME = 12.0

PLOT_Y_MIN = -75.0
PLOT_Y_MAX = 75.0  

# Define the stimulus window (odor release) times
BG_X_START_TIME = 5.0
BG_X_END_TIME = 9.0

# Define the background color for the shaded region
BACKGROUND_COLOR = '#141414'
V_TRACE_COLOR = 'darkgreen'

# Smoothing parameter for the PSTH line plot (set to an integer >= 1; 1 means no smoothing)
SMOOTHING_WINDOW = 3

# ----------------------------

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON Encoder that converts NumPy data types to Python data types."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        else:
            return super(NumpyEncoder, self).default(obj)


def get_base_odor_name(folder_name):
    return re.sub(r'\(.*\)', '', folder_name).strip()


def parse_folder_name(folder_name):
    match = re.match(r'^(?P<odor>.+?)\((?P<position>\d+)\)$', folder_name)
    if match:
        odor = match.group('odor').strip()
        position = int(match.group('position'))
        return odor, position
    else:
        return None, None


def read_data(filename):
    """Reads an Intan RHD file and returns the processed data."""
    tic = time.time()
    try:
        with open(filename, 'rb') as fid:
            header = read_header(fid)
            data_present, filesize, num_blocks, num_samples = calculate_data_size(header, filename, fid)
            if data_present:
                data = read_all_data_blocks(header, num_samples, num_blocks, fid)
                check_end_of_file(filesize, fid)
            else:
                data = []

        result = {}
        header_to_result(header, result)
        if data_present:
            parse_data(header, data)
            apply_notch_filter(header, data)
            data_to_result(header, data, result)
        else:
            data = []

        elapsed_time = time.time() - tic
        print(f'Done reading {filename}! Elapsed time: {elapsed_time:0.1f} seconds')
        return result

    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return None


def detect_spikes(t, amp, threshold=35, std_multiplier=2.5, refractory_period=0.001):
    baseline_indices = np.where(t < BG_X_START_TIME)[0]
    if len(baseline_indices) == 0:
        print("Warning: No baseline data available (t < stimulus onset).")
        baseline_mean = 0
        baseline_std = 1
    else:
        baseline = amp[baseline_indices]
        baseline_mean = np.mean(baseline)
        baseline_std = np.std(baseline)
    effective_threshold = max(threshold, baseline_mean + std_multiplier * baseline_std)
    print(f"Baseline mean: {baseline_mean:.2f}, baseline std: {baseline_std:.2f}, effective threshold: {effective_threshold:.2f}")

    spike_times = []
    spike_amplitudes = []
    last_spike_time = -np.inf

    for i in range(1, len(amp)-1):
        if t[i] - last_spike_time < refractory_period:
            continue
        if amp[i] > effective_threshold and amp[i] > amp[i-1] and amp[i] > amp[i+1]:
            spike_times.append(t[i])
            spike_amplitudes.append(amp[i])
            last_spike_time = t[i]

    spike_times = np.array(spike_times)
    spike_amplitudes = np.array(spike_amplitudes)
    spike_count = len(spike_times)
    print(f"Total spikes detected: {spike_count}")
    return spike_times, spike_amplitudes, spike_count, baseline_std


def plot_data(result, plots_directory, filename):
    """Generates and saves a plot of the amplifier data with background shading."""
    fig, ax = plt.subplots(figsize=(10, 5))
    try:
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')

        if 't_amplifier' not in result:
            print(f"Error: 't_amplifier' not found in the result for {filename}. Skipping plot.")
            plt.close(fig)
            return

        t = result['t_amplifier']
        amp = result['amplifier_data'][0, :]

        t_min = min(t)
        t_max = max(t)

        if PLOT_X_START_TIME < t_min or PLOT_X_END_TIME > t_max:
            print(f"Warning: Plot x-axis range ({PLOT_X_START_TIME}, {PLOT_X_END_TIME}) exceeds data range ({t_min}, {t_max}). Adjusting.")
            plot_start = max(PLOT_X_START_TIME, t_min)
            plot_end = min(PLOT_X_END_TIME, t_max)
        else:
            plot_start = PLOT_X_START_TIME
            plot_end = PLOT_X_END_TIME

        if BG_X_START_TIME < t_min or BG_X_END_TIME > t_max:
            print(f"Warning: Background range ({BG_X_START_TIME}, {BG_X_END_TIME}) exceeds data range ({t_min}, {t_max}). Adjusting.")
            bg_start = max(BG_X_START_TIME, t_min)
            bg_end = min(BG_X_END_TIME, t_max)
        else:
            bg_start = BG_X_START_TIME
            bg_end = BG_X_END_TIME

        ax.axvspan(bg_start, bg_end, color=BACKGROUND_COLOR, alpha=0.3, zorder=0)
        print(f"Added background shading from {bg_start} to {bg_end} seconds.")

        ax.plot(t, amp, color=V_TRACE_COLOR, zorder=1)
        print(f"Plotted amplifier data for {filename}.")
        ax.set_xlim(plot_start, plot_end)
        ax.set_ylim(PLOT_Y_MIN, PLOT_Y_MAX)
        ax.margins(x=0, y=0)

    except Exception as e:
        print(f"Error while plotting {filename}: {e}")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_filename = Path(filename).stem + '.png'
    plot_full_path = plots_directory / plot_filename
    try:
        plt.savefig(plot_full_path)
        print(f"Data plot saved to {plot_full_path}")
    except Exception as e:
        print(f"Error saving data plot for {filename}: {e}")
    finally:
        plt.close(fig)


def plot_psth(spike_times, plots_directory, filename):
    """
    Generates and saves a peristimulus time histogram (PSTH) for spikes occurring within the stimulus window.
    Instead of a bar graph, a line plot is produced that shows the mean spike count per bin with a shaded region
    representing the standard deviation (approximated as sqrt(count) for Poisson statistics).
    The plotted line is smoothed using a moving average filter, with the window size defined by SMOOTHING_WINDOW.
    """
    psth_window = (BG_X_START_TIME, BG_X_END_TIME)
    stimulus_spikes = spike_times[(spike_times >= psth_window[0]) & (spike_times <= psth_window[1])]
    if len(stimulus_spikes) == 0:
        print(f"No spikes detected within the stimulus window for {filename}. Producing an empty PSTH.")

    relative_spike_times = stimulus_spikes - psth_window[0]
    bin_edges = np.linspace(0, psth_window[1] - psth_window[0], 41)
    spike_counts, _ = np.histogram(relative_spike_times, bins=bin_edges)
    
    # Calculate bin centers for the line plot
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Calculate standard deviation for each bin assuming Poisson statistics (std = sqrt(count))
    spike_std = np.sqrt(spike_counts)
    
    # Smooth the spike counts and standard deviation using a simple moving average filter
    if SMOOTHING_WINDOW > 1:
        spike_counts_smoothed = np.convolve(spike_counts, np.ones(SMOOTHING_WINDOW)/SMOOTHING_WINDOW, mode='same')
        spike_std_smoothed = np.convolve(spike_std, np.ones(SMOOTHING_WINDOW)/SMOOTHING_WINDOW, mode='same')
    else:
        spike_counts_smoothed = spike_counts
        spike_std_smoothed = spike_std

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(bin_centers, spike_counts_smoothed, color='blue', lw=2, label='Smoothed Mean Spike Count')
    ax.fill_between(bin_centers, spike_counts_smoothed - spike_std_smoothed, spike_counts_smoothed + spike_std_smoothed,
                    color='blue', alpha=0.3, label='Smoothed SD')
    
    ax.set_xlabel('Time relative to stimulus onset (s)')
    ax.set_ylabel('Spike count')
    ax.set_title('Peristimulus Time Histogram (PSTH)')
    ax.set_xlim(0, psth_window[1] - psth_window[0])
    ax.legend()

    psth_filename = Path(filename).stem + '_PSTH.png'
    psth_full_path = plots_directory / psth_filename
    try:
        plt.savefig(psth_full_path)
        print(f"PSTH plot saved to {psth_full_path}")
    except Exception as e:
        print(f"Error saving PSTH plot for {filename}: {e}")
    finally:
        plt.close(fig)


def plot_combined_psth(spike_times_list, plots_directory, combined_filename):
    """
    Generates and saves a combined peristimulus time histogram (PSTH) from a list of trials.
    For each trial, a histogram is computed (using the stimulus window), and then the mean and standard deviation
    across trials are calculated (with optional smoothing). The resulting line plot shows the combined PSTH.
    """
    psth_window = (BG_X_START_TIME, BG_X_END_TIME)
    bin_edges = np.linspace(0, psth_window[1] - psth_window[0], 41)
    all_counts = []
    
    # Compute histogram counts for each trial
    for trial_spike_times in spike_times_list:
        trial_stim_spikes = trial_spike_times[(trial_spike_times >= psth_window[0]) & (trial_spike_times <= psth_window[1])]
        relative_spike_times = trial_stim_spikes - psth_window[0]
        counts, _ = np.histogram(relative_spike_times, bins=bin_edges)
        all_counts.append(counts)
    
    all_counts = np.array(all_counts)  # shape: (n_trials, n_bins)
    mean_counts = np.mean(all_counts, axis=0)
    std_counts = np.std(all_counts, axis=0)
    
    # Optional smoothing with a moving average filter
    if SMOOTHING_WINDOW > 1:
        mean_counts_smoothed = np.convolve(mean_counts, np.ones(SMOOTHING_WINDOW)/SMOOTHING_WINDOW, mode='same')
        std_counts_smoothed = np.convolve(std_counts, np.ones(SMOOTHING_WINDOW)/SMOOTHING_WINDOW, mode='same')
    else:
        mean_counts_smoothed = mean_counts
        std_counts_smoothed = std_counts
    
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(bin_centers, mean_counts_smoothed, color='black', lw=2, label='Smoothed Mean Spike Count')
    ax.fill_between(bin_centers, mean_counts_smoothed - std_counts_smoothed, mean_counts_smoothed + std_counts_smoothed,
                    color='red', alpha=0.3, label='Smoothed SD')
    
    ax.set_xlabel('Time relative to stimulus onset (s)')
    ax.set_ylabel('Spike count')
    ax.set_title(f'Combined PSTH (n = {len(spike_times_list)} trials)')
    ax.set_xlim(0, psth_window[1] - psth_window[0])
    ax.legend()
    
    combined_path = plots_directory / combined_filename
    try:
        plt.savefig(combined_path)
        print(f"Combined PSTH plot saved to {combined_path}")
    except Exception as e:
        print(f"Error saving combined PSTH plot: {e}")
    finally:
        plt.close(fig)


def process_folder(folder_path, base_plots_directory):
    """Processes all .RHD files in the given folder, saving JSON and generating plots including individual and combined PSTHs."""
    folder = Path(folder_path)
    if not folder.is_dir():
        print(f"Warning: {folder_path} is not a valid directory.")
        return

    odor, position = parse_folder_name(folder.name)
    if odor is None or position is None:
        return

    odor_label = odor_label_mapping.get(odor, None)
    if odor_label is None:
        print(f"Warning: Odor '{odor}' has no corresponding label. Skipping folder '{folder.name}'.")
        return

    rhd_files = list(folder.glob('*.[Rr][Hh][Dd]'))
    if not rhd_files:
        print(f"No .RHD files found in {folder_path}.")
        return

    json_subfolder = folder / "Neural_Recording_json"
    json_subfolder.mkdir(exist_ok=True)

    specific_plots_directory = base_plots_directory / f"position_{position}" / odor
    specific_plots_directory.mkdir(parents=True, exist_ok=True)

    # Accumulator for combined PSTH (grouping trials in sets of 5)
    combined_spike_times_list = []
    group_counter = 1

    for rhd_file in rhd_files:
        print(f"\nProcessing file: {rhd_file}")
        result = read_data(rhd_file)
        if result is None:
            print(f"Failed to read {rhd_file}. Skipping.")
            continue

        essential_result = {}
        if 't_amplifier' in result and 'amplifier_data' in result:
            essential_result['t_amplifier'] = result['t_amplifier']
            essential_result['amplifier_data'] = result['amplifier_data']
        else:
            print(f"Warning: Amplifier data missing in {rhd_file}. Skipping serialization and plotting.")
            continue

        t = essential_result['t_amplifier']
        amp = essential_result['amplifier_data'][0, :]

        spike_times, spike_amplitudes, spike_count, baseline_std = detect_spikes(t, amp)
        print(f"Detected {spike_count} spikes in {rhd_file}.")
        essential_result['spike_count'] = spike_count
        essential_result['spike_times'] = spike_times
        essential_result['spike_amplitudes'] = spike_amplitudes
        essential_result['baseline_std'] = baseline_std

        # Serialize individual trial results
        result_filename = Path(rhd_file).stem + '.json.gz'
        result_full_path = json_subfolder / result_filename
        try:
            with gzip.open(result_full_path, 'wt', encoding='utf-8') as f:
                json.dump(essential_result, f, cls=NumpyEncoder, indent=4)
            print(f"Compressed results saved to {result_full_path}")
        except Exception as e:
            print(f"Error during serialization for {rhd_file}: {e}")

        # Produce individual plots
        plot_data(result=essential_result, plots_directory=specific_plots_directory, filename=rhd_file)
        plot_psth(spike_times, plots_directory=specific_plots_directory, filename=rhd_file)

        # Accumulate spike times for combined PSTH
        combined_spike_times_list.append(spike_times)
        if len(combined_spike_times_list) == 5:
            combined_filename = f"Combined_PSTH_Group{group_counter}.png"
            plot_combined_psth(combined_spike_times_list, plots_directory=specific_plots_directory, combined_filename=combined_filename)
            group_counter += 1
            combined_spike_times_list = []

    # For any leftover trials (less than 5), still produce a combined PSTH plot
    if combined_spike_times_list:
        combined_filename = f"Combined_PSTH_Group{group_counter}_partial.png"
        plot_combined_psth(combined_spike_times_list, plots_directory=specific_plots_directory, combined_filename=combined_filename)


def main():
    """Main function to process all '.RHD' files within specified directories."""
    print(f"Starting processing in root directory: {ROOT_DIRECTORY}")
    for subfolder in ROOT_DIRECTORY.rglob('*'):
        if subfolder.is_dir() and subfolder.name not in exclude_folders:
            print(f"\nEntering subfolder: {subfolder}")
            process_folder(subfolder, BASE_PLOTS_DIRECTORY)
    print("\nProcessing completed.")


if __name__ == '__main__':
    main()
