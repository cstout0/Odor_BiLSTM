#! /bin/env python3
# Adrian Foy September 2023 (Modified by Camron Stout October 2024)

"""
Module to read Intan Technologies RHD2000 data files from all subfolders within
a specified root directory, serialize the amplifier data to compressed JSON,
and save amplitude plots in organized subdirectories within a central 'plots' directory.
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
ROOT_DIRECTORY = Path(r"C:\Users\stoutcam\Desktop\Saha Lab\Wireless\NeuralRecordings\01_10_2025")

# Define the base directory where plots will be saved
BASE_PLOTS_DIRECTORY = Path(r"C:\Users\stoutcam\Desktop\Saha Lab\Wireless\NeuralRecordings\01_10_2025\plots")

# Ensure that the base plots directory exists; create it if it doesn't
BASE_PLOTS_DIRECTORY.mkdir(parents=True, exist_ok=True)

# Define folders to exclude from processing
exclude_folders = {
    'plots',
    'Position_1'
    'Position_2'
    'Position_3'
    'Neural_Recording_json'
}

# Define your odor labels
odor_label_mapping = {
    'Hexanal': 0,
    'HexAce': 1,
    'Pent': 2,
    'PropBen': 3,
    'MineralOil': 4,
}

PLOT_X_START_TIME = 3.0
PLOT_X_END_TIME = 12.0

PLOT_Y_MIN = -100.0
PLOT_Y_MAX = 100.0  

# Define the starting and ending times for odor release
BG_X_START_TIME = 5.0
BG_X_END_TIME = 9.0

# Define the background color for the shaded region
BACKGROUND_COLOR = '#141414'

V_TRACE_COLOR = 'darkgreen'

# Define RMS window (in seconds) for computing the RMS of the voltage.
RMS_WINDOW = 0.05  # seconds

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


def compute_rms(t, amplifier_data, window_sec=0.05):
    """
    Computes the root mean square (RMS) of the amplifier data over non-overlapping time windows.
    
    Parameters:
      - t: 1D numpy array of time values.
      - amplifier_data: 2D numpy array of voltage data (n_channels, n_samples).
      - window_sec: Duration of each window in seconds.
      
    Returns:
      - t_rms: 1D numpy array of time points (center of each window).
      - rms_data: 2D numpy array of RMS values for each channel, shape (n_channels, n_windows).
    """
    dt = np.mean(np.diff(t))
    fs = 1.0 / dt
    window_samples = int(round(window_sec * fs))
    if window_samples < 1:
        window_samples = 1
    n_samples = amplifier_data.shape[1]
    n_windows = n_samples // window_samples

    rms_list = []
    t_rms = []
    for i in range(n_windows):
        start_idx = i * window_samples
        end_idx = start_idx + window_samples
        window_data = amplifier_data[:, start_idx:end_idx]
        rms_value = np.sqrt(np.mean(window_data**2, axis=1))
        rms_list.append(rms_value)
        t_rms.append(np.mean(t[start_idx:end_idx]))
    rms_data = np.array(rms_list).T  # shape (n_channels, n_windows)
    t_rms = np.array(t_rms)
    return t_rms, rms_data


def plot_data(result, plots_directory, filename):
    """Generates and saves a plot with customizable x-axis and y-axis limits and background."""
    fig, ax = plt.subplots(figsize=(10, 5))

    try:
        # Plot Configuration
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')

        # Validate that 't_amplifier' exists in the result
        if 't_amplifier' not in result:
            print(f"Error: 't_amplifier' not found in the result for {filename}. Skipping plot.")
            plt.close(fig)
            return

        t = result['t_amplifier']
        amp = result['amplifier_data'][0, :]

        # Debugging statements
        t_min = min(t)
        t_max = max(t)

        # Check if plot range is within data range
        if PLOT_X_START_TIME < t_min or PLOT_X_END_TIME > t_max:
            print(f"Warning: Plot x-axis range ({PLOT_X_START_TIME}, {PLOT_X_END_TIME}) "
                  f"exceeds data range ({t_min}, {t_max}). Adjusting to data limits.")
            plot_start = max(PLOT_X_START_TIME, t_min)
            plot_end = min(PLOT_X_END_TIME, t_max)
        else:
            plot_start = PLOT_X_START_TIME
            plot_end = PLOT_X_END_TIME

        # Check if background range is within data range
        if BG_X_START_TIME < t_min or BG_X_END_TIME > t_max:
            print(f"Warning: Background x-axis range ({BG_X_START_TIME}, {BG_X_END_TIME}) "
                  f"exceeds data range ({t_min}, {t_max}). Adjusting to data limits.")
            bg_start = max(BG_X_START_TIME, t_min)
            bg_end = min(BG_X_END_TIME, t_max)
        else:
            bg_start = BG_X_START_TIME
            bg_end = BG_X_END_TIME

        # Add a light gray background spanning the designated background x-axis range
        ax.axvspan(bg_start, bg_end, color=BACKGROUND_COLOR, alpha=0.3, zorder=0)
        print(f"Added background shading from {bg_start} to {bg_end} seconds.")

        # Plot Amplifier Data
        ax.plot(t, amp, color=V_TRACE_COLOR, zorder=1)
        print(f"Plotted amplifier data for {filename}.")

        # Set x-axis limits to designated start and end times
        ax.set_xlim(plot_start, plot_end)
        print(f"Set plot x-axis limits to {plot_start} - {plot_end} seconds.")

        # Set y-axis limits to designated start and end times
        ax.set_ylim(PLOT_Y_MIN, PLOT_Y_MAX)

        # Adjust margins
        ax.margins(x=0, y=0)

    except KeyError as e:
        print(f"Missing data for plotting {filename}: {e}")
    except IndexError as e:
        print(f"Index error while plotting {filename}: {e}")
    except Exception as e:
        print(f"Unexpected error while plotting {filename}: {e}")

    # Adjust layout to make room for the title
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save the plot
    plot_filename = Path(filename).stem + '.png'
    plot_full_path = plots_directory / plot_filename
    try:
        plt.savefig(plot_full_path)
        print(f"Plot saved to {plot_full_path}")
    except Exception as e:
        print(f"Error saving plot for {filename}: {e}")
    finally:
        plt.close(fig)


def process_folder(folder_path, base_plots_directory):
    """Processes all .RHD files in the given folder, saves amplifier data as compressed JSON, and generates plots."""
    folder = Path(folder_path)
    if not folder.is_dir():
        print(f"Warning: {folder_path} is not a valid directory.")
        return

    # Parse the folder name to get odor and position
    odor, position = parse_folder_name(folder.name)
    if odor is None or position is None:
        return

    # Retrieve the odor label using the base odor name
    odor_label = odor_label_mapping.get(odor, None)
    if odor_label is None:
        print(f"Warning: Odor '{odor}' does not have a corresponding odor label. Skipping folder '{folder.name}'.")
        return

    # Search for .RHD files (case-insensitive) using a single glob pattern to prevent duplicate processing
    rhd_files = list(folder.glob('*.[Rr][Hh][Dd]'))

    if not rhd_files:
        print(f"No .RHD files found in {folder_path}.")
        return

    # Define the subfolder for JSON files within the current odor folder
    json_subfolder = folder / "Neural_Recording_json"
    json_subfolder.mkdir(exist_ok=True)  # Create the subfolder if it doesn't exist

    # Define the specific plots directory: plots/position_X/Odor_Name
    specific_plots_directory = base_plots_directory / f"position_{position}" / odor
    specific_plots_directory.mkdir(parents=True, exist_ok=True)  # Create directories as needed

    for rhd_file in rhd_files:
        print(f"\nProcessing file: {rhd_file}")
        result = read_data(rhd_file)
        if result is None:
            print(f"Failed to read {rhd_file}. Skipping.")
            continue

        # Extract only amplifier data and related time points
        essential_result = {}
        if 't_amplifier' in result and 'amplifier_data' in result:
            essential_result['t_amplifier'] = result['t_amplifier']
            essential_result['amplifier_data'] = result['amplifier_data']
        else:
            print(f"Warning: Amplifier data missing in {rhd_file}. Skipping serialization and plotting.")
            continue

        # Compute RMS of amplifier voltage over the specified window
        t_rms, rms_data = compute_rms(essential_result['t_amplifier'],
                                      essential_result['amplifier_data'],
                                      window_sec=RMS_WINDOW)
        essential_result['t_amplifier'] = t_rms
        essential_result['amplifier_data'] = rms_data
        print(f"Computed RMS voltage with a window of {RMS_WINDOW} seconds.")

        # Serialize the essential_result using the custom encoder and compress with gzip
        # Save JSON file within the "Neural_Recording_json" subfolder
        result_filename = Path(rhd_file).stem + '.json.gz'
        result_full_path = json_subfolder / result_filename
        try:
            with gzip.open(result_full_path, 'wt', encoding='utf-8') as f:
                json.dump(essential_result, f, cls=NumpyEncoder, indent=4)
            print(f"Compressed results saved to {result_full_path}")
        except TypeError as e:
            print(f"Serialization error for {rhd_file}: {e}")
        except Exception as e:
            print(f"Unexpected error during serialization for {rhd_file}: {e}")

        # Generate and save the plot in the specific plots directory
        plot_data(result=essential_result, plots_directory=specific_plots_directory, filename=rhd_file)


def main():
    """
    Main function to process all '.RHD' files within specified directories.
    """
    print(f"Starting processing in root directory: {ROOT_DIRECTORY}")
    for subfolder in ROOT_DIRECTORY.rglob('*'):
        if subfolder.is_dir() and subfolder.name not in exclude_folders:
            print(f"\nEntering subfolder: {subfolder}")
            process_folder(subfolder, BASE_PLOTS_DIRECTORY)
    print("\nProcessing completed.")


if __name__ == '__main__':
    main()
