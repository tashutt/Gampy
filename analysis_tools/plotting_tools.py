import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os, sys


"""
A collection of plotting tools for visualizing data.
"""

plt.rcParams.update({'font.size': 18, 
                     'axes.titlesize': 18, 
                     'ytick.labelsize': 14, 
                     'legend.fontsize': 18,
                     'legend.title_fontsize': 18,
                     'axes.labelsize': 18,
                     'axes.labelpad': 10,})

import numpy as np
import matplotlib.pyplot as plt

# Plot angular resolution vs energy (both individual angles and mean)
def plot_angular_resolution_vs_energy(mono_ene_results):
    # Plot for individual angles
    fig, ax = plt.subplots(figsize=(8, 6))
    for angle in sorted(mono_ene_results.Angle.unique()):
        df_angle = mono_ene_results.query('Energy>300 & Angle==@angle')
        ax.plot(df_angle['Energy'], df_angle['FWHM'], 
                label=f'{np.degrees(np.arccos(angle)):0.2f}°', 
                marker='o', linestyle='--', linewidth=1.2)

    ax.set_xlabel('Energy (keV)', fontsize=14)
    ax.set_ylabel('FWHM (°)', fontsize=14)
    ax.set_title('Angular Resolution at Different Angles', fontsize=16)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xscale('log')
    ax.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('plots/angular_resolution_vs_energy.png', dpi=300)
    plt.close()

    # Plot for mean values
    fig, ax = plt.subplots(figsize=(8, 6))
    mean_fwhm = mono_ene_results.query('Energy>300').groupby('Energy')['etrFWHM'].mean()
    ax.plot(mean_fwhm.index, mean_fwhm.values, marker='o', linewidth=2, color='darkblue')
    ax.annotate(f'{mean_fwhm.values[-1]:0.2f}°', (mean_fwhm.index[-1], mean_fwhm.values[-1]), 
                textcoords="offset points", xytext=(0,10), ha='center', fontsize=12)

    ax.set_xlabel('Energy (keV)', fontsize=14)
    ax.set_ylabel('FWHM (°)', fontsize=14)
    ax.set_title('Mean Angular Resolution', fontsize=16)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xscale('log')
    plt.tight_layout()
    plt.savefig('plots/angular_resolution_vs_energy_mean.png', dpi=300)
    plt.close()

# Plot angular resolution (etrFWHM) vs energy
def plot_angular_resolution_etrFWHM(mono_ene_results):
    fig, ax = plt.subplots(figsize=(8, 6))
    for angle in sorted(mono_ene_results.Angle.unique()):
        df_angle = mono_ene_results.query('Energy>300 & Angle==@angle')
        ax.plot(df_angle['Energy'], df_angle['etrFWHM'], 
                label=f'{np.degrees(np.arccos(angle)):0.2f}°', marker='o')

    ax.set_xlabel('Energy (keV)', fontsize=14)
    ax.set_ylabel('etrFWHM (°)', fontsize=14)
    ax.set_title('Angular Resolution (etrFWHM) vs Energy', fontsize=16)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xscale('log')
    ax.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('plots/angular_resolution_etrFWHM.png', dpi=300)
    plt.close()

    # Plot for mean values
    fig, ax = plt.subplots(figsize=(8, 6))
    mean_fwhm = mono_ene_results.query('Energy>300').groupby('Energy')['etrFWHM'].mean()
    ax.plot(mean_fwhm.index, mean_fwhm.values, marker='o', linewidth=2, color='darkblue')

    ax.set_xlabel('Energy (keV)', fontsize=14)
    ax.set_ylabel('etrFWHM (°)', fontsize=14)
    ax.set_title('Mean Angular Resolution (etrFWHM)', fontsize=16)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xscale('log')
    plt.tight_layout()
    plt.savefig('plots/angular_resolution_etrFWHM_mean.png', dpi=300)
    plt.close()

# Plot energy resolution vs energy
def plot_energy_resolution_vs_energy(mono_ene_results):
    fig, ax = plt.subplots(figsize=(8, 6))
    for angle in sorted(mono_ene_results.Angle.unique()):
        df_angle = mono_ene_results.query('Angle==@angle')
        ax.plot(df_angle['Energy'], df_angle['Sigma_E'], 
                label=f'{np.degrees(np.arccos(angle)):0.2f}°', marker='o')

    ax.set_xlabel('Energy (keV)', fontsize=14)
    ax.set_ylabel('Sigma E (keV)', fontsize=14)
    ax.set_title('Energy Resolution vs Energy', fontsize=16)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xscale('log')
    ax.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('plots/energy_resolution_vs_energy.png', dpi=300)
    plt.close()

    # Plot for mean values
    fig, ax = plt.subplots(figsize=(8, 6))
    mean_sigma_e = mono_ene_results.groupby('Energy')['Sigma_E'].mean()
    ax.plot(mean_sigma_e.index, mean_sigma_e.values, marker='o', linewidth=2, color='darkblue')

    ax.set_xlabel('Energy (keV)', fontsize=14)
    ax.set_ylabel('Sigma E (keV)', fontsize=14)
    ax.set_title('Mean Energy Resolution vs Energy', fontsize=16)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xscale('log')
    plt.tight_layout()
    plt.savefig('plots/energy_resolution_vs_energy_mean.png', dpi=300)
    plt.close()

# Plot energy resolution percent vs energy
def plot_energy_resolution_percent(mono_ene_results):
    fig, ax = plt.subplots(figsize=(8, 6))
    for angle in sorted(mono_ene_results.Angle.unique()):
        df_angle = mono_ene_results.query('Angle==@angle')
        ax.plot(df_angle['Energy'], df_angle['Sigma_E'] / df_angle['Energy'] * 100, 
                label=f'{np.degrees(np.arccos(angle)):0.2f}°', marker='o')

    ax.set_xlabel('Energy (keV)', fontsize=14)
    ax.set_ylabel('Energy Resolution (%)', fontsize=14)
    ax.set_title('Energy Resolution Percentage vs Energy', fontsize=16)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xscale('log')
    ax.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('plots/energy_resolution_percent.png', dpi=300)
    plt.close()

    # Plot for mean values
    fig, ax = plt.subplots(figsize=(8, 6))
    mean_res_percent = (mono_ene_results.groupby('Energy')['Sigma_E'].mean() / 
                        mono_ene_results.groupby('Energy')['Energy'].mean()) * 100
    ax.plot(mean_res_percent.index, mean_res_percent.values, marker='o', linewidth=2, color='darkblue')

    ax.set_xlabel('Energy (keV)', fontsize=14)
    ax.set_ylabel('Energy Resolution (%)', fontsize=14)
    ax.set_title('Mean Energy Resolution Percentage vs Energy', fontsize=16)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xscale('log')
    plt.tight_layout()
    plt.savefig('plots/energy_resolution_percent_mean.png', dpi=300)
    plt.close()

# Plot efficiency and acceptance vs energy
def plot_efficiency_and_acceptance(mono_ene_results):
    fig, ax = plt.subplots(figsize=(8, 6))
    for angle in sorted(mono_ene_results.Angle.unique()):
        df_angle = mono_ene_results.query('Angle==@angle')
        ax.plot(df_angle['Energy'], df_angle['Efficiency'], 
                label=f'{np.degrees(np.arccos(angle)):0.2f}° Efficiency', marker='o')
        ax.plot(df_angle['Energy'], df_angle['Accepted_line'], 
                label=f'{np.degrees(np.arccos(angle)):0.2f}° Line Acceptance', marker='o')
        ax.plot(df_angle['Energy'], df_angle['Accepted_cont'], 
                label=f'{np.degrees(np.arccos(angle)):0.2f}° Cont. Acceptance', marker='o')

    ax.set_xlabel('Energy (keV)', fontsize=14)
    ax.set_ylabel('Efficiency and Acceptance', fontsize=14)
    ax.set_title('Efficiency and Acceptance vs Energy', fontsize=16)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xscale('log')
    ax.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('plots/efficiency_and_acceptance.png', dpi=300)
    plt.close()

    # Plot for mean values
    fig, ax = plt.subplots(figsize=(8, 6))
    mean_efficiency = mono_ene_results.groupby('Energy')['Efficiency'].mean()
    mean_acc_line = mono_ene_results.groupby('Energy')['Accepted_line'].mean()
    mean_acc_cont = mono_ene_results.groupby('Energy')['Accepted_cont'].mean()

    ax.plot(mean_efficiency.index, mean_efficiency.values, marker='o', linewidth=2, color='darkblue', label='Mean Efficiency')
    ax.plot(mean_acc_line.index, mean_acc_line.values, marker='o', linewidth=2, color='darkgreen', label='Mean Line Acceptance')
    ax.plot(mean_acc_cont.index, mean_acc_cont.values, marker='o', linewidth=2, color='darkred', label='Mean Cont. Acceptance')

    ax.set_xlabel('Energy (keV)', fontsize=14)
    ax.set_ylabel('Efficiency and Acceptance', fontsize=14)
    ax.set_title('Mean Efficiency and Acceptance vs Energy', fontsize=16)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xscale('log')
    ax.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('plots/efficiency_and_acceptance_mean.png', dpi=300)
    plt.close()




def add_decay_lines(min_decay_rate = 10):
    decay_data = [
    {"isotope": "39Cl", "energy": [1517.49, 1267.2, 250], "probability": [0.392, 0.356, 0.46], "rate": 0.266},
    {"isotope": "38Cl", "energy": [1642.68, 2167.4], "probability": [0.329, 0.44], "rate": 0.251},
    {"isotope": "41Ar", "energy": [1293], "probability": [0.99], "rate": 0.182},
    {"isotope": "37S", "energy": [3103], "probability": [0.939], "rate": 0.146},
    {"isotope": "38S", "energy": [1940, 1746, 2751], "probability": [0.865, 0.025, 0.014], "rate": 0.124},
    {"isotope": "40Cl", "energy": [5880, 3919, 3101, 2880, 2622, 2524, 2458, 2220, 1797, 1747, 1460, 1432, 1394, 1063, 881, 660, 644], 
     "probability": [0.045, 0.039, 0.113, 0.27, 0.15, 0.02, 0.047, 0.07, 0.022, 0.027, 0.81, 0.016, 0.012, 0.023, 0.026, 0.025, 0.067], "rate": 0.09},
    {"isotope": "34P", "energy": [2127], "probability": [0.15], "rate": 0.014},
    {"isotope": "35P", "energy": [1572.29], "probability": [0.988], "rate": 0.064},
    {"isotope": "28Al", "energy": [1778.987], "probability": [1.0], "rate": 0.052},
    {"isotope": "36P", "energy": [3290, 2539, 2020, 1960, 1638, 1284, 1256, 1058, 901.8, 828.8, 757, 185], 
     "probability": [1.0, 0.174, 0.051, 0.135, 0.353, 0.041, 0.045, 0.053, 0.704, 0.161, 0.017, 0.025], "rate": 0.029},
    {"isotope": "38K", "energy": [2167.5], "probability": [1.0], "rate": 0.029},
    {"isotope": "29Al", "energy": [2425, 2028, 1273], "probability": [0.052, 0.035, 0.913], "rate": 0.028},
    {"isotope": "34Cl", "energy": [3304, 2127, 1176], "probability": [0.122, 0.428, 0.141], "rate": 0.009},
    {"isotope": "38P", "energy": [3698, 4713, 3516, 2224, 1292], "probability": [0.09, 0.08, 0.10, 0.18, 0.78], "rate": 0.014},
    {"isotope": "37P", "energy": [646, 1582, 2100, 2254], "probability": [1.0, 0.74, 0.06, 0.08], "rate": 0.02},
    {"isotope": "39S", "energy": [1696, 1300, 904, 874, 485, 397], "probability": [0.44, 0.519, 0.035, 0.13, 0.11, 0.40], "rate": 0.002}
    ]

    for isotope_data in decay_data:
        for i, energy in enumerate(isotope_data['energy']):
            line_rate = isotope_data['rate'] * isotope_data['probability'][i]  # Calculate rate for this energy line
            if line_rate > min_decay_rate:
                plt.axvline(x=energy, color='red', linestyle='--', linewidth=1)
                plt.text(energy, plt.ylim()[1]*isotope_data['probability'][i]/500, f'* {isotope_data['isotope']}  |   Cell Rate: {line_rate:.3g} Bq', 
                         rotation=75, verticalalignment='center', horizontalalignment='left', color='blue', 
                         fontsize=8, alpha=min(1, 10*line_rate))


def plot_activation_energy_distribution(energy_axis, activation_rate, activation_rate_c):
    plt.figure(figsize=(10, 8))
    plt.step(energy_axis, activation_rate, where='mid', color='grey', linewidth=0.7, label='Total Activation')
    plt.step(energy_axis, activation_rate_c, where='mid', color='royalblue', linewidth=1.5, label='Classified Activation')
    add_decay_lines(min_decay_rate=0.11)
    plt.xlim(0, 3_500)
    plt.xlabel('Energy (keV)')
    plt.ylabel('Counts')
    plt.title('Energy Distribution')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig('plots/background_energy_hist_activation.png')


def plot_cosmic_background_energy_distribution(energy_axis, background_rate, background_rate_c):
    plt.figure(figsize=(10, 8))
    plt.step(energy_axis, background_rate, where='mid', color='grey', linewidth=0.7, label='Total Cosmic Background')
    plt.step(energy_axis, background_rate_c, where='mid', color='royalblue', linewidth=1.5, label='Classified Cosmic Background')

    plt.xlabel('Energy (keV)')
    plt.ylabel('Counts')
    plt.title('Energy Distribution')
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig('plots/background_energy_hist_cosmic.png')



def plot_combined_energy_distribution(energy_axis, total_rate, total_rate_c, activation_rate, activation_rate_c):
    plt.figure(figsize=(10, 8))
    plt.step(energy_axis, total_rate, where='mid', color='grey', linewidth=0.7, label='Raw Background')
    plt.step(energy_axis, total_rate_c, where='mid', color='royalblue', linewidth=1.5, label='Classified Background')

    plt.step(energy_axis, activation_rate, where='mid', color='gray', 
             linewidth=0.7, linestyle='--', label='Raw Activation', alpha=0.7)
    plt.step(energy_axis, activation_rate_c, where='mid', color='red', 
             linewidth=0.7, linestyle='--', label='Classified Activation')

    plt.xlabel('Energy (keV)')
    plt.ylabel('Counts')
    plt.title('Energy Distribution')
    plt.yscale('log')
    plt.xscale('log')
    plt.xlim(100, None)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig('plots/background_energy_hist_combined.png')


def plot_feature_hist_cosmic_vs_activ(background_data, features, energy_axis, activation_rate, activation_rate_c):
    for feature in features:
        plt.figure(figsize=(10, 8))
        background_data_grouped = background_data.groupby(['source', feature]).size().reset_index(name='counts')
        sns.histplot(data=background_data, x=feature, bins=100, alpha=0.5, stat='density', hue='source')
        plt.title(f'{feature} distribution')
        plt.yscale('log')
        plt.savefig(f'plots/features_{feature}_hist.png')
        plt.close()
        plt.clf()

    plt.figure(figsize=(10, 8))
    sns.histplot(data=background_data, x='calc_to_sum_ene', hue='source', bins=100, alpha=0.5, stat='density', binrange=(0, 15))
    plt.yscale('log')
    plt.xlabel('Ratio of calculated to summed energy $\\frac{E_{calc}}{E_{sum}}$', fontsize=18)
    plt.savefig(f'plots/a_ratio_cosmic_activation.png')
    plt.close()
    plt.clf()

    plt.figure(figsize=(10, 8))
    plt.step(energy_axis, activation_rate_c/activation_rate, where='mid', color='grey', linewidth=0.7, label='Ratio')
    
    plt.xlabel('Energy (keV)')
    plt.ylabel('Ratio')
    plt.title('Ratio of classified to raw activation')
    
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig('plots/ratio_class_raw_activation.png')
    plt.close()
    plt.clf()




### fitting background


def plot_background_rate_fit(x, y, popt, filename, background_func, log_scale=False):
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, label='Normalized Data')
    plt.plot(x, np.exp(background_func(x, *popt)), label='Fitted Function', color='red')
    plt.xlabel('Energy (keV)')
    plt.ylabel('Normalized Counts')
    plt.title('Background Rate Fit')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.yscale('log')
    if log_scale:
        plt.xscale('log')
    plt.savefig(filename)
    plt.close()


def plot_arm_distribution(background_data):
    fig, axes = plt.subplots(2, 1, figsize=(10, 15))

    # Plot 1: ARM Distribution for all and good data
    axes[0].hist(background_data['ARM'], bins=100, histtype='step', color='grey', linewidth=1.5, label='All Data')
    axes[0].hist(background_data.query('good==1')['ARM'], bins=100, histtype='step', color='royalblue', linewidth=1.5, label='Good Data')
    axes[0].hist(background_data.query('good==1 and beta_minus_Dbeta < 0')['ARM'], 
                 bins=100, histtype='step', color='red', linewidth=1.5, label='Good & ETR')
    axes[0].set(xlabel='ARM', ylabel='Counts', title='ARM Distribution')
    axes[0].legend()
    axes[0].grid(True, which='both', linestyle='--', linewidth=0.5)

    # Plot 2: ARM Distribution with limited range
    axes[1].hist(background_data['ARM'], bins=100, range=(-10, 10), histtype='step', color='grey', linewidth=1.5, label='All Data')
    axes[1].hist(background_data.query('good==1')['ARM'], bins=100, range=(-10, 10), histtype='step', color='royalblue', linewidth=1.5, label='Good Data')
    axes[1].hist(background_data.query('good==1 and beta_minus_Dbeta < 0')['ARM'], 
                 bins=100, range=(-10, 10), histtype='step', color='red', linewidth=1.5, label='Good & ETR')
    axes[1].set(xlabel='ARM', ylabel='Counts', title='ARM Distribution (limited range)')
    axes[1].legend()
    axes[1].grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.savefig('plots/aa_arm_distribution_combined.png')
    plt.clf()


def plot_azimuthal_angle_difference(background_data, ACCEPTANCE_ANGLE):
    plt.figure(figsize=(10, 8))
    sns.histplot(data=background_data.query('good == 1'), x='beta_minus_Dbeta', 
                 bins=90, color='grey', hue='precise_e_readout')
    plt.axvline(ACCEPTANCE_ANGLE, color='red', linestyle='--')
    plt.xlabel('Azimuthal Angle Difference ($ \\beta $ - $\\Delta \\beta$) [deg]')
    plt.ylabel('Counts')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig('plots/aa_dif_distribution_good_data.png')
    plt.clf()


def plot_energy_distribution_fit(x1, y1, popt_base, background_func, binwidth1, ARM_MAX, EMIN):
    plt.figure(figsize=(10, 8))
    plt.plot(x1, y1, 'ro', label='data', alpha=0.5)
    plt.plot(x1, np.exp(background_func(x1, *popt_base)), 'r-', label='fit')
    # annotate the function and the parameters
    plt.text(0.05, 0.1, f"Fit function: a*log(x) + b*sqrt(x) + c\n", 
                ha='left', va='center', fontsize=10, transform=plt.gca().transAxes)
    plt.text(0.05, 0.05, f"Fit parameters: a={popt_base[0]:.2f}, b={popt_base[1]:.2f}, c={popt_base[2]:.2f}",
                ha='left', va='center', fontsize=10, transform=plt.gca().transAxes)
    plt.yscale('log')
    plt.xlabel('Energy (keV)')
    plt.ylabel('Counts')
    plt.title(f'Energy Distribution (|ARM|<{ARM_MAX}) E>{EMIN} keV')
    plt.annotate(f'Bin width: {binwidth1:.2f}', (0.5, 0.5), xycoords='axes fraction')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig('plots/aa_energy_distribution_fit.png')
    plt.clf()


def plot_energy_distribution_fit_loglog(x1, y1, popt_base, background_func, binwidth1, ARM_MAX, EMIN):
    plt.figure(figsize=(10, 8))
    plt.plot(x1, y1, 'ro', label='data', alpha=0.5)
    plt.plot(x1, np.exp(background_func(x1, *popt_base)), 'r-', label='fit')
    # annotate the function and the parameters
    plt.text(0.05, 0.1, f"Fit function: a*log(x) + b*sqrt(x) + c\n", 
                ha='left', va='center', fontsize=10, transform=plt.gca().transAxes)
    plt.text(0.05, 0.05, f"Fit parameters: a={popt_base[0]:.2f}, b={popt_base[1]:.2f}, c={popt_base[2]:.2f}",
                ha='left', va='center', fontsize=10, transform=plt.gca().transAxes)
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('Energy (keV)')
    plt.ylabel('Counts')
    plt.title(f'Energy Distribution (|ARM|<{ARM_MAX}) E>{EMIN} keV')
    plt.annotate(f'Bin width: {binwidth1:.2f}', (0.5, 0.5), xycoords='axes fraction')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig('plots/aa_energy_distribution_fit_loglog.png')
    plt.clf()


def plot_background_rate_energy_fitted(E, base_background, etr_background):
    plt.figure(figsize=(10, 8))
    plt.plot(E, base_background, 'b-', label='Background Rate')
    plt.plot(E, etr_background, 'g-', label='Background Rate with ETR', linewidth=2)
    plt.plot([1e-1, 1e1], [4, 2e-3], 'r--', label='GRAMS background rate')
    plt.xlabel('Energy (MeV)')
    plt.ylabel('Background Rate (ph/cm2/MeV/sr/s)')
    plt.title('Background Rate as a function of Energy')
    plt.grid(True, which='both', linestyle='--', linewidth=0.7)
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/aa_background_rate_energy_fitted.png')
    plt.clf()


def plot_background_rate_ratio(Y1, background_ratio, electron_track_rejection, etr_popt):
    plt.figure(figsize=(10, 8))
    plt.plot(Y1[:-1], background_ratio, 'o', label='Data')
    plt.plot(Y1[:-1], electron_track_rejection(Y1[:-1], *etr_popt), 'r--', label=f'Fit')
    plt.text(0.05, 0.06, f"Fit function: 1 - a/(1+exp(b*(x^0.25-c))) + d*x\n"
                        f"a={etr_popt[0]:.2f}, b={etr_popt[1]:.2f}, c={etr_popt[2]:.2f}, d={etr_popt[3]:.2e}",
             ha='left', va='center', fontsize=10, transform=plt.gca().transAxes)
    plt.axvline(etr_popt[2]**4, color='silver', linestyle='--', label='Sigmoid midpoint')
    plt.xlabel('Energy (keV)')
    plt.ylabel('Background Rate Ratio')
    plt.title('Background Rate Ratio with and without ETR')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.xscale('log')
    plt.tight_layout()
    plt.savefig('plots/aa_background_rate_ratio.png')
    plt.clf()


### Sensitivity stuff 

def plot_line_sensitivity_vs_energy(LSF, DIAMETER, AREA_cm):
    plt.figure(figsize=(10, 8))
    for angle in LSF['Angle'].unique():
        df_angle = LSF[LSF['Angle'] == angle]
        plt.plot(df_angle['Energy'], df_angle['line_sens_base'], 
                 label=f'base Angle {np.degrees(np.arccos(angle)):0.2f}°', marker='o', alpha=0.7)
        plt.plot(df_angle['Energy'], df_angle['line_sens_etr'],
                    label=f'etr Angle {np.degrees(np.arccos(angle)):0.2f}°', marker='o')

    plt.xlabel('Energy (keV)')
    plt.ylabel('Line Sensitivity [ph/cm2/s]')
    plt.title('Line Sensitivity vs Energy')
    plt.annotate(f'Diameter: {DIAMETER:.0f} cm', (0.5, 0.55), xycoords='axes fraction')
    plt.annotate(f'Area: {AREA_cm:.0f} $cm^2$', (0.5, 0.5), xycoords='axes fraction')
    plt.annotate(f'Time: $10^6$ s', (0.5, 0.45), xycoords='axes fraction')
    plt.grid(True, which='both', linestyle='--', linewidth=0.7)
    plt.legend()
    plt.xscale('log')
    plt.clf()

def plot_mean_line_sensitivity_vs_energy(LSF, DIAMETER, AREA_cm):
    df_grouped = LSF.groupby('Energy').mean()
    plt.figure(figsize=(10, 8))
    plt.plot(df_grouped.index, df_grouped['line_sens_base'], 'o-', 
             label='Base Sensitivity', color='r', linewidth=1, alpha=0.7)
    plt.plot(df_grouped.index, df_grouped['line_sens_etr'], 'o-', 
             label='With Electron Track Reconstruction', color='r', linewidth=2)
    plt.xlabel('Energy (keV)')
    plt.ylabel('Line Sensitivity [ph/$cm^2$/s]')
    plt.title('Line Sensitivity vs Energy')
    plt.grid(True, which='both', linestyle='--', linewidth=0.7)
    plt.annotate(f'Diameter: {DIAMETER:.0f} cm', (0.05, 0.11), xycoords='axes fraction')
    plt.annotate(f'Area: {AREA_cm:.0f} $cm^2$', (0.05, 0.06), xycoords='axes fraction')
    plt.annotate(f'Time: $10^6$ s', (0.05, 0.02), xycoords='axes fraction')
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('plots/line_sensitivity_mean.png')
    plt.clf()

def plot_line_sensitivity_comparison(LSF):
    energy_levels_previous = [511, 666, 847, 1157, 1173, 1333, 1809, 2223, 4438]
    sensitivity_satellite = [6.3e-7, 4.2e-7, 2.7e-7, 1.9e-7, 1.9e-7, 1.7e-7, 1.3e-7, 1.1e-7, 7.3e-8]
    sensitivity_spi_integral = [5.0e-5, 2.0e-5, 2.0e-5, 2.0e-5, 2.0e-5, 2.0e-5, 2.5e-5, 2.0e-5, 1.0e-5]
    COSIx = 1e3*np.array([0.147571, 0.2, 0.367354, 0.508666, 0.85134, 1.01441, 1.95191, 3.45926, 5.39, 6.77638])
    COSIy = [4.03346e-05, 1.25254e-05, 4.66412e-06, 3.22023e-06, 2.19126e-06, 1.96506e-06, 1.77505e-06, 3.01644e-06, 9.92763e-06, 3.26735e-05]

    plt.figure(figsize=(10, 8))
    plt.plot(energy_levels_previous, sensitivity_spi_integral, 'o--', label='SPI Integral $10^6$s')
    plt.plot(COSIx, COSIy, 'o--', label='COSI 2y')
    plt.plot(energy_levels_previous, sensitivity_satellite, 'o--', label='GRAMS Satellite $10^6$s')

    df_grouped = LSF.groupby('Energy').mean()
    plt.plot(df_grouped.index, df_grouped['line_sens_base'], 'o--',color='red',
             label='GammaTPC $10^6$s', linewidth=1, alpha=0.7)
    plt.plot(df_grouped.index, df_grouped['line_sens_etr'], 'o-',color='red',  
                label='GammaTPC $10^6$s with ETR', linewidth=1.9, alpha=0.7)

    plt.xlabel('Energy (keV)')
    plt.ylabel('Line Sensitivity [ph/$cm^2$/s]')
    plt.title('Line Sensitivity vs Energy')
    plt.grid(True, which='both', linestyle='--', linewidth=0.7)
    plt.legend(fontsize=14) 
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('plots/line_sensitivity_comparison.png')
    plt.clf()

def plot_effective_area_angular_resolution_energy_resolution(LSF, AREA_cm):
    fig, axs = plt.subplots(3, 1, figsize=(9, 20), sharex='all')
    fig.subplots_adjust(hspace=0.12)

    titles = ['Effective Area vs Energy', 
              'Angular Resolution vs Energy', 
              'Energy Resolution vs Energy']

    y_labels = ['Effective Area [cm$^2$]', 
                'Angular Resolution [°]', 
                'Energy Resolution [%]']

    for i, (ax, title, y_label) in enumerate(zip(axs, titles, y_labels)):
        for angle in LSF['Angle'].unique():
            df_angle = LSF[LSF['Angle'] == angle]
            if i == 0:  
                y_data = df_angle['Accepted_line'] * AREA_cm
            elif i == 1:  
                y_data = df_angle['etrFWHM']
            else:  
                y_data = df_angle['Sigma_E']/df_angle['Energy'] * 100 

            ax.plot(df_angle['Energy'], y_data, 
                    label=f'{np.degrees(np.arccos(angle)):0.1f}°', 
                    marker='o', linestyle='-')

        ax.set_ylabel(y_labels[i])
        ax.set_title(titles[i])
        ax.grid(True, which='both', linestyle='--', linewidth=0.6)
        ax.set_ylim(0, None)
        ax.legend(title='Angle of Incidence')

    plt.xlabel('Energy (keV)')
    axs[-1].set_xscale('log')

    plt.show()


def plot_continuum_sensitivity_vs_energy(LSF, mev_to_erg):
    plt.figure(figsize=(10, 8))
    for angle in LSF['Angle'].unique():
        df_angle = LSF[LSF['Angle'] == angle]
        Yax = mev_to_erg * df_angle['cont_sensitivity_base'] * df_angle['Energy_MeV'] ** 2
        plt.plot(df_angle['Energy_MeV'], Yax, 
                 label=f'Angle {np.degrees(np.arccos(angle)):0.2f}°', marker='o', alpha=0.7)
        Yax = mev_to_erg * df_angle['cont_sensitivity_etr'] * df_angle['Energy_MeV'] ** 2
        plt.plot(df_angle['Energy_MeV'], Yax,
                 label=f'Angle {np.degrees(np.arccos(angle)):0.2f}° with ETR', marker='o')

    plt.xlabel('Energy (MeV)')
    plt.ylabel('Continuum Sensitivity x $E^2$ [erg/cm2/s]')
    plt.title('Continuum Sensitivity x $E^2$ vs Energy')
    plt.grid(True, which='both', linestyle='--', linewidth=0.7)
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('plots/continuum_sensitivity_vs_energy.png')
    plt.clf()





continuum_data = {
    'NuSTAR': {
        "X": [0.04936630896057163, 0.07026116207448584, 0.08037287229717947, 0.08037287229717947, 0.08312047745422195],
        "Y": [6.088113756264881e-14, 1.8139306939110706e-13, 7.573739175895024e-13, 2.1261123338996643e-12, 8.70272323809911e-12]
    },
    'AMEGO-X': {
        "X": [0.09193982010218539, 0.1522235170295561, 0.213042943204976, 0.3836556566345653, 0.8312047745422203, 1.991911656141189, 3.772613684737265, 4.936630896057167, 9.833332623922983],
        "Y": [3.706512910922168e-10, 1.1490656115802539e-10, 3.162277660168392e-11, 1.244020640728236e-11, 7.424883695230986e-12, 6.3346720767083094e-12, 1.061358418412884e-11, 1.9638280019297738e-11, 1.7090651588023986e-11]
    },
    'SPI': {
        "X": [0.0477344713046953, 0.10695401201561898, 0.16837490832180593, 0.2478338274108304, 0.36479073171466564, 0.6352134833012268, 0.9349812888309114, 1.1633045604912686, 1.8313599932506708, 3.032157979099799, 4.854353364246509, 6.569287187116032, 7.771609113838906],
        "Y": [1.547587354557891e-11, 2.3018073130224702e-11, 5.0920956367734075e-11, 7.135892121363277e-11, 1.0200481902174008e-10, 3.0391953823132075e-10, 4.2590265248188705e-10, 7.135892121363248e-10, 1.2944013747378925e-09, 1.8502967215037666e-09, 2.541981665446668e-09, 3.706512910922168e-09, 4.520353656360259e-09]
    },
    'COMPTEL': {
        "X": [0.7145203438308527, 0.8452930520422213, 1.087667997271001, 1.6280915869438926, 2.5630620724274378, 5.279924801735741, 7.2663091055471565, 10.000000000000021],
        "Y": [1.1043416410511044e-09, 5.968456995122317e-10, 3.856620421163488e-10, 2.4920211513780647e-10, 2.30180731302247e-10, 2.5929437974046775e-10, 2.920903717032256e-10, 3.2256756042196726e-10]
    }
}


def plot_continuum_sensitivity_comparison(LSF, mev_to_erg, DIAMETER, AREA_cm, Teff_y):
    plt.figure(figsize=(10, 8))

    for instrument, dataset in continuum_data.items():
        plt.plot(dataset['X'], dataset['Y'], label=instrument, marker='o', alpha=0.7, linestyle='--')
        plt.annotate(instrument, xy=(dataset['X'][-2], dataset['Y'][-2]), textcoords="offset points", xytext=(5, 5), ha='right')

    df_grouped = LSF.groupby('Energy').mean()
    gtpcX = df_grouped.index / 1000
    gtpcY = mev_to_erg * df_grouped['cont_sensitivity_base'] * df_grouped['Energy_MeV'] ** 2
    plt.plot(gtpcX, gtpcY, 'o-', label='GammaTPC - 3yrs', linewidth=3)
    plt.annotate("GammaTPC 3y", xy=(list(gtpcX)[-2], list(gtpcY)[-2]), textcoords="offset points", xytext=(5, 5), ha='right')

    gtpcYetr = mev_to_erg * df_grouped['cont_sensitivity_etr'] * df_grouped['Energy_MeV'] ** 2
    plt.plot(gtpcX, gtpcYetr, 'o-', label='GammaTPC - 3yrs with ETR', linewidth=3)
    plt.annotate("GammaTPC 3y ETR", xy=(list(gtpcX)[-4], list(gtpcYetr)[-2]), 
                 textcoords="offset points", xytext=(5, 5), ha='right')

    plt.xlabel('Energy (MeV)')
    plt.ylabel('Continuum Sensitivity x $E^2$ [erg/cm2/s]')
    plt.title('Continuum Sensitivity x $E^2$ vs Energy')
    plt.grid(True, which='both', linestyle='--', linewidth=0.7)
    plt.xscale('log')
    plt.yscale('log')

    plt.annotate(f'Diameter: {DIAMETER:.0f} cm', (0.05, 0.11), xycoords='axes fraction')
    plt.annotate(f'Area: {AREA_cm:.0f} $cm^2$', (0.05, 0.06), xycoords='axes fraction')
    plt.annotate(f'Time: 3y {Teff_y / 1e6:0.1f} $10^6$ s', (0.05, 0.02), xycoords='axes fraction')

    plt.tight_layout()
    plt.savefig('plots/continuum_sensitivity_comparison.png')
    plt.clf()

def plot_efficiency_acceptance_vs_energy(LSF):
    df_grouped = LSF.groupby('Energy').mean()
    plt.figure(figsize=(8, 6))
    plt.plot(df_grouped.index, df_grouped['Efficiency'], 'o-', label='Efficiency')
    plt.plot(df_grouped.index, df_grouped['Accepted_cont'], 'o-', label='Continuum Acceptance')
    plt.plot(df_grouped.index, df_grouped['Accepted_line'], 'o-', label='Line Acceptance')

    plt.xlabel('Energy (keV)')
    plt.ylabel('Efficiency / Acceptance')
    plt.title('Efficiency and Acceptance vs Energy')
    plt.grid(True, which='both', linestyle='--', linewidth=0.7)
    plt.legend()
    plt.xscale('log')

    plt.tight_layout()
    plt.savefig('plots/efficiency_acceptance_vs_energy.png')
    plt.clf()



