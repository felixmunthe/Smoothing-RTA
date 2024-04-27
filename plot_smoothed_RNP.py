# -*- coding: utf-8 -*-
"""
Smoothed RNP Plot

Author: Munthe, Felix A.
Created on Wednesday, 1 November 2023
"""

import numpy as np
import matplotlib.pyplot as plt

# Import Noisy RNP Data
def read_noisy_RNP(file_path):

    # Step 1: Open the text file
    with open(file_path, "r", encoding = "utf-8") as file:
        lines = file.readlines()

    # Step 2: Process the data and create an array
    noisy_time = []
    noisy_RNP = []
    
    # Process each subsequent line and append the values
    for line in lines[1:]:
        values = line.split()
        if len(values) >= 2:
            noisy_time.append(float(values[0]))
            noisy_RNP.append(float(values[1]))
    
    return noisy_time, noisy_RNP

# ----- Import Smoothed RNP Data -----
def read_smoothed_RNP (file_path):

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

# ----- Import Residual of Smoothed RNP Data -----
def read_residual_RNP (file_path):

    # Step 1: Open the text file
    with open (file_path, "r", encoding = "utf-8") as file:
        lines = file.readlines()

    # Step 2: Process the data and create an array
    residual_time = []
    residual_RNP = []
    
    # Step 3: Process each subsequent line and append the values
    for line in lines[1:]:
        values = line.split()
        if len(values) >= 2:
            residual_time.append(float(values[0]))
            residual_RNP.append(float(values[1]))
    
    return residual_time, residual_RNP

# ----- Main Execution -----
def main():
    # Noisy RNP data file path
    noisy_RNP_file_path = r"C:\Users\ASUS\Documents\Kuliah\2. Master's Texas A&M University\Publications\Conference Paper\Outlier Detection Paper\Outlier Detection Simulator\Noisy RNP Model\All\50%\noisy_data_9.txt"
    
    # Define time and RNP noisy data
    noisy_time, noisy_RNP = read_noisy_RNP(noisy_RNP_file_path) # days, psia2/cp-d/Mscf
    noisy_time = np.array(noisy_time) # days
    noisy_RNP = np.array(noisy_RNP) # psia2/cp-d/Mscf

    # Smoothed RNP data file path
    smoothed_RNP_file_path = r"C:\Users\ASUS\Documents\Kuliah\2. Master's Texas A&M University\Publications\Conference Paper\Smoothing Paper\Smoothing Simulator\Smoothing Methods\Lowess\50%\Data Set 199\smoothed_RNP_199.txt"
    smoothed_time, smoothed_RNP = read_smoothed_RNP(smoothed_RNP_file_path) # days, psia2/cp-d/Mscf
    smoothed_time = np.array(smoothed_time) # days
    smoothed_RNP = np.array(smoothed_RNP) # psia2/cp-d/Mscf

    # Residual data file path
    residual_RNP_file_path = r"C:\Users\ASUS\Documents\Kuliah\2. Master's Texas A&M University\Publications\Conference Paper\Smoothing Paper\Smoothing Simulator\Smoothing Methods\Lowess\50%\Data Set 199\residual_199.txt"
    residual_time, residual_RNP = read_smoothed_RNP(residual_RNP_file_path) # days, psia2/cp-d/Mscf
    residual_time = np.array(residual_time) # days
    residual_RNP = np.array(residual_RNP) # psia2/cp-d/Mscf

    # Plot noisy RNP vs time
    plt.figure()
    plt.plot(noisy_time, noisy_RNP, 'o', label = 'noisy RNP')
    plt.plot(smoothed_time, smoothed_RNP, 'x', label = 'smoothed RNP')
    plt.xlabel("t, days", fontsize = 20)
    plt.ylabel("$\Delta$m(p)/$q_{g}$, $psia^{2}$/cp$\cdot$d/Mscf", fontsize = 20)
    plt.xscale("log")
    plt.yscale("log")
    # plt.title("Rate-normalized Pseudo-pressure (RNP) vs. Time", fontsize = 24)
    plt.minorticks_on()
    plt.tick_params(axis = 'both', which = 'major', labelsize = 18)
    plt.grid(True, which = 'major', linestyle = '-', linewidth = 1)
    plt.grid(True, which = 'minor', linestyle = ':', linewidth = 0.5)
    plt.legend(fontsize = 20)
    plt.show()

    # Plot residual RNP vs time
    plt.figure()
    plt.plot(residual_time, residual_RNP, 'o')
    plt.xlabel("t, days", fontsize = 20)
    plt.ylabel("relative residual", fontsize = 20)
    plt.xscale("log")
    # plt.title("Residual vs. Time", fontsize = 24)
    plt.minorticks_on()
    plt.tick_params(axis = 'both', which = 'major', labelsize = 18)
    plt.grid(True, which = 'major', linestyle = '-', linewidth = 1)
    plt.grid(True, which = 'minor', linestyle = ':', linewidth = 0.5)
    # plt.legend(fontsize = 20)
    plt.show()
    return

main()