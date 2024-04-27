# -*- coding: utf-8 -*-
"""
Smoothed RNP Plot

Author: Munthe, Felix A.
Created on Wednesday, 1 November 2023
"""

import numpy as np
import matplotlib.pyplot as plt
import Pseudopressure_Conversion
from scipy.stats import linregress

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

# ----- Function to calculate the slope within a window -----
def calculate_slope(x, y):
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    return slope, intercept, r_value

# ----- Main Execution -----
def main():
    # True RNP data file path
    true_RNP_file_path = r"C:\Users\ASUS\Documents\Kuliah\2. Master's Texas A&M University\Publications\Conference Paper\Smoothing Paper\Smoothing Simulator\true_RNP.txt"
    true_time, true_RNP = read_true_RNP(true_RNP_file_path) # days, psia2/cp-d/Mscf
    true_time = np.array(true_time) # days
    true_RNP = np.array(true_RNP) # days

    # Noisy RNP data file path
    noisy_RNP_file_path = r"C:\Users\ASUS\Documents\Kuliah\2. Master's Texas A&M University\Publications\Conference Paper\Outlier Detection Paper\Outlier Detection Simulator\Noisy RNP Model\All\50%\noisy_data_9.txt"
    noisy_time, noisy_RNP = read_noisy_RNP(noisy_RNP_file_path) # days, psia2/cp-d/Mscf
    noisy_time = np.array(noisy_time) # days
    noisy_RNP = np.array(noisy_RNP) # psia2/cp-d/Mscf

    # Smoothed RNP data file path
    smoothed_RNP_file_path = r"C:\Users\ASUS\Documents\Kuliah\2. Master's Texas A&M University\Publications\Conference Paper\Smoothing Paper\Smoothing Simulator\Smoothing Methods\Lowess\50%\Data Set 199\smoothed_RNP_199.txt"
    smoothed_time, smoothed_RNP = read_smoothed_RNP(smoothed_RNP_file_path) # days, psia2/cp-d/Mscf
    smoothed_time = np.array(smoothed_time) # days
    smoothed_RNP = np.array(smoothed_RNP) # psia2/cp-d/Mscf

    # Filter RNP data
    filtered_time = true_time <= 5000
    filtered_true_time = true_time[filtered_time]
    filtered_true_RNP = true_RNP[filtered_time]
    filtered_noisy_time = noisy_time[filtered_time]
    filtered_noisy_RNP = noisy_RNP[filtered_time]
    filtered_smootherd_time = smoothed_time[filtered_time]
    filtered_smoothed_RNP = smoothed_RNP[filtered_time]

    # Find the interval where the slope is close to 0.5
    desired_slope = 0.5
    slope_tolerance = 0.01
    r_value_tolerance = 0.7

    # Find interval of noisy time with slope 0.5
    #print("Noisy RNP results:")
    for window_size in range(2, len(filtered_noisy_time) + 1):
        for start_index in range(len(filtered_noisy_time) - window_size + 1):
            log_x = np.log(filtered_noisy_time[start_index:start_index + window_size])
            log_y = np.log(filtered_noisy_RNP[start_index:start_index + window_size])
            slope, intercept, r_value = calculate_slope(log_x, log_y)
            if abs(slope - desired_slope) < slope_tolerance and r_value > r_value_tolerance:
                end_index = start_index + window_size - 1
                #print(f"Interval with a slope:", slope, "and Rsquared:", r_value, "and intercept:", intercept)
                #print("Start transient time:", filtered_noisy_time[start_index])
                #print("End transient time:", filtered_noisy_time[end_index])
                #print("")

    fitted_noisy_time = noisy_time[(noisy_time > 0.3 - 1) & (noisy_time < 1405.0 + 1)]
    fitted_noisy_RNP = np.exp([15.5146989788083 + 0.490097495690205 * np.log(fitted_noisy_time[i]) for i in range(len(fitted_noisy_time))])
    
    # Find interval of smoothed time with slope 0.5
    #print("Smoothed RNP results:")
    for window_size in range(2, len(filtered_smootherd_time) + 1):
        for start_index in range(len(filtered_smootherd_time) - window_size + 1):
            log_x = np.log(filtered_smootherd_time[start_index:start_index + window_size])
            log_y = np.log(filtered_smoothed_RNP[start_index:start_index + window_size])
            slope, intercept, r_value = calculate_slope(log_x, log_y)
            if abs(slope - desired_slope) < slope_tolerance and r_value > r_value_tolerance:
                end_index = start_index + window_size - 1
                #print(f"Interval with a slope:", slope, "and Rsquared:", r_value, "and intercept:", intercept)
                #print("Start transient time:", filtered_noisy_time[start_index])
                #print("End transient time:", filtered_noisy_time[end_index])
                #print("")
    
    fitted_smoothed_time = smoothed_time[(smoothed_time > 125.0 - 1) & (smoothed_time < 4402.0 + 1)]
    fitted_smoothed_RNP = np.exp([15.2748526972262 + 0.49906299342644 * np.log(fitted_smoothed_time[i]) for i in range(len(fitted_smoothed_time))])
    
    # Plot RNP vs time
    plt.figure()
    plt.plot(noisy_time, noisy_RNP, 'o', label = 'noisy RNP')
    plt.plot(smoothed_time, smoothed_RNP, 'x', label = 'smoothed RNP')
    plt.plot(fitted_noisy_time, fitted_noisy_RNP, '-', label = 'fitted noisy RNP', linewidth = '3')
    plt.plot(fitted_smoothed_time, fitted_smoothed_RNP, '-', label = 'fitted smoothed RNP', linewidth = '3')
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

    # Define linear flow time data
    pseudotime = np.array(Pseudopressure_Conversion.pseudotime)
    pseudotime = pseudotime[1:]

    true_linear_pseudotime = np.array(pseudotime[true_time < 4000])
    true_sqrt_pseudotime = np.sqrt(true_linear_pseudotime)
    true_linear_RNP = true_RNP[(true_time < 4000)]
    
    noisy_start_index = np.where(noisy_time == 0.25)[0][0]
    noisy_linear_pseudotime = [pseudotime[i] for i in range(noisy_start_index, noisy_start_index + len(fitted_noisy_time))]
    noisy_linear_pseudotime = np.array(noisy_linear_pseudotime)
    noisy_sqrt_pseudotime = np.sqrt(noisy_linear_pseudotime)
    #noisy_linear_time = np.array(fitted_noisy_time)
    #noisy_sqrt_time = np.sqrt(noisy_linear_time)
    noisy_linear_RNP = noisy_RNP[(noisy_time > 0.3 - 1) & (noisy_time < 1405.0 + 1)]
    
    smoothed_start_index = np.where(smoothed_time == 124.9920833)[0][0]
    smoothed_linear_pseudotime = [pseudotime[i] for i in range(smoothed_start_index, smoothed_start_index + len(fitted_smoothed_time))]
    smoothed_linear_pseudotime = np.array(smoothed_linear_pseudotime)
    smoothed_sqrt_pseudotime = np.sqrt(smoothed_linear_pseudotime)
    #smoothed_linear_time = np.array(fitted_smoothed_time)
    #smoothed_sqrt_time = np.sqrt(smoothed_linear_time)
    smoothed_linear_RNP = smoothed_RNP[(smoothed_time > 125.0 - 1) & (smoothed_time < 4402.0 + 1)]

    # Calculate linear slope for square-root time
    true_linear_slope, true_linear_intercept, true_linear_r_value = calculate_slope(true_sqrt_pseudotime, true_linear_RNP)
    fitted_true_linear_RNP = [true_linear_intercept + true_linear_slope * true_sqrt_pseudotime[i] for i in range(len(true_sqrt_pseudotime))]
    print("True linear slope:", true_linear_slope)
    print("True R-squared:", true_linear_r_value)

    noisy_linear_slope, noisy_linear_intercept, noisy_linear_r_value = calculate_slope(noisy_sqrt_pseudotime, noisy_linear_RNP)
    fitted_noisy_linear_RNP = [noisy_linear_intercept + noisy_linear_slope * noisy_sqrt_pseudotime[i] for i in range(len(noisy_sqrt_pseudotime))]
    print("Noisy linear slope:", noisy_linear_slope)
    print("Noisy R-squared:", noisy_linear_r_value)

    smoothed_linear_slope, smoothed_linear_intercept, smoothed_linear_r_value = calculate_slope(smoothed_sqrt_pseudotime, smoothed_linear_RNP)
    fitted_smoothed_linear_RNP = [smoothed_linear_intercept + smoothed_linear_slope * smoothed_sqrt_pseudotime[i] for i in range(len(smoothed_sqrt_pseudotime))]
    print("Smoothed linear slope:", smoothed_linear_slope)
    print("Smoothed R-squared:", smoothed_linear_r_value)

    # Reservoir properties
    T = 212 # deg.Fahrenheit
    h = 100 # ft
    phi = 0.05 # fraction
    miugi = 0.026527595 # cp
    cti = Pseudopressure_Conversion.cti
    
    # Extracted properties
    true_kxf = ((40.93 * T) / (true_linear_slope * h * np.sqrt(phi * miugi * cti))) ** 2
    print("True k-xf square: ", true_kxf * 100000)
    noisy_kxf = ((40.93 * T) / (noisy_linear_slope * h * np.sqrt(phi * miugi * cti))) ** 2
    print("Noisy k-xf square: ", noisy_kxf * 100000)
    smoothed_kxf = ((40.93 * T) / (smoothed_linear_slope * h * np.sqrt(phi * miugi * cti))) ** 2
    print("Smoothed k-xf square: ", smoothed_kxf * 100000)
    
    # Plot RNP vs square-root time
    newfiltered_noisy_time = np.sqrt(pseudotime[noisy_time < 5000])
    newfiltered_noisy_RNP = noisy_RNP[noisy_time < 5000]
    newfiltered_smoothed_time = np.sqrt(pseudotime[smoothed_time < 5000])
    newfiltered_smoothed_RNP = smoothed_RNP[smoothed_time < 5000]

    plt.figure()
    plt.plot(newfiltered_noisy_time, newfiltered_noisy_RNP, 'o', label = 'noisy RNP')
    plt.plot(newfiltered_smoothed_time, newfiltered_smoothed_RNP, 'x', label = 'smoothed RNP')
    #plt.plot(true_sqrt_pseudotime, fitted_true_linear_RNP, '-', label = 'fitted noisy RNP', linewidth = '3')
    plt.plot(noisy_sqrt_pseudotime, fitted_noisy_linear_RNP, '-', label = 'fitted noisy RNP', linewidth = '3')
    plt.plot(smoothed_sqrt_pseudotime, fitted_smoothed_linear_RNP, '-', label = 'fitted smoothed RNP', linewidth = '3')
    plt.xlabel("$\sqrt{t_{a}}$, $days^{1/2}$", fontsize = 20)
    plt.ylabel("$\Delta$m(p)/$q_{g}$, $psia^{2}$/cp$\cdot$d/Mscf", fontsize = 20)
    # plt.title("Rate-normalized Pseudo-pressure (RNP) vs. Square-Root Time", fontsize = 24)
    plt.minorticks_on()
    plt.tick_params(axis = 'both', which = 'major', labelsize = 18)
    plt.grid(True, which = 'major', linestyle = '-', linewidth = 1)
    plt.grid(True, which = 'minor', linestyle = ':', linewidth = 0.5)
    plt.legend(fontsize = 20)
    plt.show()
    return

main()