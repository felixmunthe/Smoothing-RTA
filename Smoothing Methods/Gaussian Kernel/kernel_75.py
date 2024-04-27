# -*- coding: utf-8 -*-
"""
Gaussian Kernel Smoothing Methods

Author: Munthe, Felix A.
Created on Wednesday, 1 November 2023
"""

import os
import numpy as np
from scipy.ndimage import gaussian_filter

# Import True RNP Data
def read_true_RNP(file_path):

    # Step 1: Open the text file
    with open(file_path, "r", encoding = "utf-8") as file:
        lines = file.readlines()

    # Step 2: Process the data and create an array
    true_time = []
    true_RNP = []
    
    # Process each subsequent line and append the values
    for line in lines[1:]:
        values = line.split()
        if len(values) >= 2:
            true_time.append(float(values[0]))
            true_RNP.append(float(values[1]))
    
    return true_time, true_RNP

# ----- Import Noisy RNP Data -----
def read_noisy_RNP (file_path):

    # Step 1: Open the text file
    with open (file_path, "r", encoding = "utf-8") as file:
        lines = file.readlines()

    # Step 2: Process the data and create an array
    noisy_time = []
    noisy_RNP = []
    
    # Step 3: Process each subsequent line and append the values
    for line in lines[1:]:
        values = line.split()
        if len(values) >= 2:
            noisy_time.append(float(values[0]))
            noisy_RNP.append(float(values[1]))
    
    return noisy_time, noisy_RNP

# ----- Main Execution -----
def main():
    # True RNP data file path
    true_RNP_file_path = r"C:\Users\ASUS\Documents\Kuliah\2. Master's Texas A&M University\Publications\Conference Paper\Smoothing Paper\Smoothing Simulator\true_RNP.txt"
    
    # Define time and RNP true data
    true_time, true_RNP = read_true_RNP(true_RNP_file_path) # days, psia2/cp-d/Mscf
    true_time = np.array(true_time) # days
    true_RNP = np.array(true_RNP) # psia2/cp-d/Mscf

    # Specify the folder path and file name for result file
    folder_path = "C:/Users/ASUS/Documents/Kuliah/2. Master's Texas A&M University/Publications/Conference Paper/Smoothing Paper/Smoothing Simulator/Smoothing Methods/Gaussian Kernel/75%/"

    # Create the directory if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    # Define table headers
    headers = {
        "Data_Set": "Data_Set",
        "Noise_Level": "Noise_Level",
        "SSE": "SSE"
    }

    # Process the Gaussian Kernel smoothing over the range of noisy data
    number_data_set = 10
    initial_data_set_number = 200
    data_set = []
    sse_store = []
    for i in range (1, number_data_set + 1):
        data_set_number = initial_data_set_number + i
        print("----- Noisy Data:", data_set_number, "-----")
        # Read the noisy RNP data
        noisy_RNP_file_path = f"C:/Users/ASUS/Documents/Kuliah/2. Master's Texas A&M University/Publications/Conference Paper/Outlier Detection Paper/Outlier Detection Simulator/Noisy RNP Model/All/75%/noisy_data_{i}.txt"
        noisy_time, noisy_RNP = read_noisy_RNP(noisy_RNP_file_path) # days, psia2/cp-d/Mscf
        noisy_time = np.array(noisy_time) # days
        noisy_RNP = np.array(noisy_RNP) # psia2/cp-d/Mscf
        
        # Specify the sub-folder path and file name for result file
        subfolder_path = f"C:/Users/ASUS/Documents/Kuliah/2. Master's Texas A&M University/Publications/Conference Paper/Smoothing Paper/Smoothing Simulator/Smoothing Methods/Gaussian Kernel/75%/Data Set {data_set_number}/"

        # Create the directory if it doesn't exist
        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)

        # Check dimensions of the data
        if noisy_time.shape != noisy_RNP.shape:
            raise ValueError("Dimensions of time and RNP not match!")

        # Check for NaN or infinite values
        if np.isnan(noisy_time).any() or np.isnan(noisy_RNP).any() or np.isinf(noisy_time).any() or np.isinf(noisy_RNP).any():
            raise ValueError("Data contains NaN or infinite values!")

        # Perform Gaussian kernel smoothing
        smoothed_RNP = gaussian_filter(np.log(noisy_RNP), sigma = 5)

        # Transform back the smoothed RNP
        smoothed_RNP = np.exp(smoothed_RNP)

        # Export smoothed results
        smoothed_file_path = f"C:/Users/ASUS/Documents/Kuliah/2. Master's Texas A&M University/Publications/Conference Paper/Smoothing Paper/Smoothing Simulator/Smoothing Methods/Gaussian Kernel/75%/Data Set {data_set_number}/smoothed_RNP_{data_set_number}.txt"
        with open(smoothed_file_path, "w") as file:
            # Write the header
            file.write("t(days)\tsmoothed_RNP(psia2/cp-d/Mscf)\n")
            
            # Write the data to the file
            for x, y in zip(noisy_time, smoothed_RNP):
                file.write(f"{x}\t{y}\n")

        sse = 0
        relative_residual = np.zeros(len(true_RNP), dtype = float)
        for j in range(len(smoothed_RNP)):
            if smoothed_RNP[j] < true_RNP[j]:
                relative_residual[j] = (smoothed_RNP[j] - true_RNP[j]) / (0.3413 * true_RNP[j])
            elif smoothed_RNP[j] >= true_RNP[j]:
                relative_residual[j] = (smoothed_RNP[j] - true_RNP[j]) / (0.3413 * true_RNP[j])
            sse += relative_residual[j] ** 2
        
        # Export residual results
        residual_file_path = f"C:/Users/ASUS/Documents/Kuliah/2. Master's Texas A&M University/Publications/Conference Paper/Smoothing Paper/Smoothing Simulator/Smoothing Methods/Gaussian Kernel/75%/Data Set {data_set_number}/residual_{data_set_number}.txt"
        with open(residual_file_path, "w") as file:
            # Write the header
            file.write("t(days)\tresidual_RNP(psia2/cp-d/Mscf)\n")
            
            # Write the data to the file
            for x, y in zip(noisy_time, relative_residual):
                file.write(f"{x}\t{y}\n")

        # Export SSE results
        data_set.append(data_set_number)
        sse_store.append(sse)

        sse_file_path = f"C:/Users/ASUS/Documents/Kuliah/2. Master's Texas A&M University/Publications/Conference Paper/Smoothing Paper/Smoothing Simulator/Smoothing Methods/Gaussian Kernel/75%/SSE_results_75%.txt"
        with open(sse_file_path, "w") as file:
            # Write the header
            file.write("Data_Set\tSSE\n")
            
            # Write the data to the file
            for x, y in zip(data_set, sse_store):
                file.write(f"{x}\t{y}\n")
        
        print("-> Success")

    print("--- Outlier Detection Simulator Complete ---")
    return

main()