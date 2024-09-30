import pandas as pd
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shutil
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy.optimize import curve_fit

"""
CompleteSensitivityAnalysis.py
This script performs a comprehensive sensitivity analysis for gamma-ray detection using machine learning classifiers 
and various data processing techniques. The analysis includes loading and processing monoenergetic line data and 
background data, training classifiers, evaluating their performance, and plotting various results related to 
angular resolution, energy resolution, efficiency, and background rates.

Functions:
    - background_func(x, a, b, c): Logarithmic background function.
    - angle_to_solid(angle): Converts angle in degrees to solid angle in steradians.
    - phi_B(energy, popt, bw, Time, Area, ARMMAX): Calculates the background rate for a given energy.
    - electron_track_rejection(x, a, b, c, d): Electron track rejection function.
    - line_sensitivity(E, FWHM, sigma_e, efficiency, etr_on): Calculates line sensitivity.
    - cont_sensitivity(E, FWHM, sigma_e, efficiency, etr_on): Calculates continuum sensitivity.
Data Loading:
    - Loads monoenergetic line data and background data.
    - Trains classifiers if not already trained and saved.
    - Evaluates classifiers and makes predictions on the data.
Analysis:
    - Analyzes monoenergetic line data and background data.
    - Calculates activation rates, cosmic rates, and combined rates.
    - Fits background rate data to a logarithmic function.
    - Calculates background rates with and without electron track rejection.
Plotting:
    - Generates various plots for angular resolution, energy resolution, efficiency, and background rates.
    - Plots sensitivity analysis results.
Output:
    - Saves the final sensitivity analysis results to a pickle file.
"""


ACCEPTANCE_ANGLE = 1.6
ARM_MAX = 7
BINS = 400
EMIN = 160
SIM_AREA = (0.797885*100)**2 * np.pi

# Make Gampy available to the script
paths = {'root':''}
paths['gampy']       = os.path.abspath(os.path.dirname(__file__))
paths['comp_params'] = os.path.join(paths['gampy'], 'Gampy', 'default_inputs', 'default_computation_inputs.yaml')
sys.path.append(paths['gampy'])

from Gampy.analysis_tools import workflow_tools as wft
from Gampy.analysis_tools import plotting_tools 


base_name  = os.path.basename(os.getcwd())
source_dir = f'/sdf/scratch/kipac/MeV/gammatpc/{base_name}/working_background/'
MEL_folder = 'monoenergetic_line_results'
name_of_background_file = 'background_data.pkl'

if not os.path.exists('plots'):
    os.makedirs('plots')

if not os.path.exists('classifiers'):
    os.makedirs('classifiers')


### LOADING OF DATA ### --------------------------------------------------------
# Load the monoenergetic line data 
if os.path.exists(os.path.join(MEL_folder, 'monoenergetic_line_data.pkl')):
    monoenergetic_line_df = pd.read_pickle(os.path.join(MEL_folder, 'monoenergetic_line_data.pkl'))
    monoenergetic_summary = pd.read_pickle(os.path.join(MEL_folder, 'all_summaries.pickle'))
else:
    monoenergetic_summary, monoenergetic_line_df = wft.process_mono_energetic_results(MEL_folder)
monoenergetic_summary = pd.DataFrame(monoenergetic_summary)

# Load the background data (from the server)
if os.path.exists(name_of_background_file):
    background_df = pd.read_pickle(name_of_background_file)
else:
    background_df = wft.load_background_data_from_server(source_dir, name_of_background_file)


classifiers = {
    #"monoenergetic": "monoenergetic_classifier.pkl",
    #"activation_cosmic": "activation_cosmic_classifier.pkl",
    "all_data": "all_data_classifier.pkl"
}
clfs = {}
features = ['e_out_CKD', 'min_hit_distance', 'kn_probability', 'calc_to_sum_ene', 'num_of_hits', 'compton_angle',"energy_from_sum"]
targets  = ['truth_escaped_energy', 'truth_correct_order', 'truth_calorimeter_first3']

for name, file in classifiers.items():
    path = f'classifiers/{file}'
    if os.path.exists(path):
        clfs[name] = joblib.load(path)
    else:
        df = monoenergetic_line_df if name == "monoenergetic" else background_df if name == "activation_cosmic" else pd.concat([monoenergetic_line_df, background_df])
        df_clean = df.dropna(subset=features)
        print("Training classifier for", name)
        print(f"Dropped {(len(df) - len(df_clean)) * 100 / len(df):.2f}% due to missing features")
        X, y, clf = wft.train_model(df_clean, name, features, targets)
        wft.evaluate_classifier(X, y, clf)
        clfs[name] = clf


############################## Monoenergetic Line Data ############################

line_data        = monoenergetic_line_df.dropna(subset=features).copy()
reduction_factor = (len(monoenergetic_line_df) - len(line_data))  / len(monoenergetic_line_df)
print('Dropped {:.3f}% of the data due to missing features'.format(reduction_factor*100))

line_data.loc[:, 'good'] = clfs['all_data'].predict(line_data[features])

# Ensure that minARM is calculated and added to the dataframe
line_data['minARM'] = line_data.apply(
    lambda row: row['ARM'] if abs(row['ARM']) <= row['delta_e_direction'] * 180 / np.pi
    else (-row['beta_angle'] * 2 if row['ARM'] < 0 else row['beta_angle'] * 2), axis=1)

mono_ene_reults = wft.analyze_monoenergetic_line_data(line_data, monoenergetic_summary, reduction_factor)\
      if not os.path.exists(f'{MEL_folder}.pkl') else pd.read_pickle(f'{MEL_folder}.pkl')


plotting_tools.plot_angular_resolution_vs_energy(mono_ene_reults)
plotting_tools.plot_angular_resolution_etrFWHM(mono_ene_reults)
plotting_tools.plot_energy_resolution_vs_energy(mono_ene_reults)
plotting_tools.plot_energy_resolution_percent(mono_ene_reults)
plotting_tools.plot_efficiency_and_acceptance(mono_ene_reults)

############################## TOTAL BACKGROUND DATA ##############################

min_e = 0
max_e = 12_000
num_of_bins = 1000

background_data = background_df.dropna(subset=features).copy()
reduction_factor_B = (len(background_df) - len(background_data))  / len(background_df)
print('Dropped {:.3f}% of the background data due to missing features'.format(reduction_factor_B*100))
background_data.loc[:, 'good'] = clfs['all_data'].predict(background_data[features])

background_data.loc[:, 'beta_minus_Dbeta'] = background_data['beta_angle'] - background_data['delta_e_direction'] * 90 / np.pi
background_data.loc[:, 'precise_e_readout'] = (background_data['delta_e_direction'] * 180 / np.pi) < 30


# activation
activation_rate, energy_axis     = wft.get_activation_rates(background_data, min_e, max_e, num_of_bins, classified=False)
activation_rate_c, energy_axis_c = wft.get_activation_rates(background_data, min_e, max_e, num_of_bins, classified=True)
plotting_tools.plot_activation_energy_distribution(energy_axis, activation_rate, activation_rate_c)


## cosmic
background_rate, energy_axis     = wft.get_cosmic_rates(background_data, min_e, max_e, num_of_bins, classified=False)
background_rate_c, energy_axis_c = wft.get_cosmic_rates(background_data, min_e, max_e, num_of_bins, classified=True)
plotting_tools.plot_cosmic_background_energy_distribution(energy_axis, background_rate, background_rate_c)

# combined
total_rate   = activation_rate + background_rate
total_rate_c = activation_rate_c + background_rate_c
plotting_tools.plot_combined_energy_distribution(energy_axis, total_rate, total_rate_c, activation_rate, activation_rate_c)


#### compare feature distribution between cosmic and activation data
#plotting_tools.plot_feature_hist_cosmic_vs_activ(background_data, features, energy_axis, activation_rate, activation_rate_c)


# ---------------------------------------------------------------------------------------------------------------------
def background_func(x, a, b, c):
    return a*np.log(x) + b*np.sqrt(x) + c


def angle_to_solid(angle):
    """Takes in degrees and returns the solid angle in sr"""
    return 2 * np.pi * (1 - np.cos(np.radians(angle)))

def phi_B(energy, 
          popt,
          bw,
          Time=1,
          Area=SIM_AREA,
          ARMMAX=ARM_MAX):
    """
    Calculate the background rate for a given energy
    in: energy (keV),
        popt (fit parameters), 
        bw (bin width),
        Time (s),
        Area (cm2),
        ARMMAX (degrees)
    Rate is in ph/cm2/MeV/sr/s
    """
    counts = np.exp(background_func(energy, *popt)) / bw
    solid_angle = ARMMAX / 180 / (4 * np.pi)
    cts_ats = counts / Area / Time / solid_angle
    # the rate is in ph/cm2/MeV/sr/s hence the division by 1e-3
    return cts_ats / (1e-3)


gdf = background_data.query('good==1').query("energy_from_sum<12_000").query("ARM>-@ARM_MAX").query("ARM<@ARM_MAX")

activation_rate_lmtd, energy_axis_lmtd = wft.get_activation_rates(gdf, min_e, max_e, BINS, classified=True)
background_rate_lmtd, energy_axis_lmtd = wft.get_cosmic_rates(gdf, min_e, max_e, BINS, classified=True)
total_rate_lmtd = activation_rate_lmtd + background_rate_lmtd



x1 = energy_axis_lmtd[total_rate_lmtd>0]
y1 = total_rate_lmtd[total_rate_lmtd>0] * (1+reduction_factor_B)
binwidth1 = x1[1] - x1[0]
popt_base, _ = curve_fit(background_func, x1, np.log(y1), p0=[1, 1, 1])


plotting_tools.plot_background_rate_fit(x1, y1, popt_base, 'plots/aa_background_rate_fit.png',background_func)
plotting_tools.plot_background_rate_fit(x1, y1, popt_base, 'plots/aa_background_rate_fit_loglog.png',background_func, log_scale=True)

# determining the ratio of background rate with and without ETR
X1, Y1 = np.histogram(background_data.query('good == 1')['energy_from_sum'], bins=55, range=(EMIN, 10_000))
X2, Y2 = np.histogram(background_data.query('good == 1 and beta_minus_Dbeta < @ACCEPTANCE_ANGLE')['energy_from_sum'], 
                      bins=55, range=(EMIN, 10_000))
background_ratio = X2/X1

def electron_track_rejection(x, a, b, c, d):
    return 1 - a/(1+np.exp(b * (x**0.27-c) )) + d*x
    
etr_popt, _ = curve_fit(electron_track_rejection, 
                        Y1[:-1], background_ratio,
                          p0=[1, 0.001, 500,1e-5])

E = np.logspace(-1, 1, 100)
base_background = phi_B(E*1000, popt_base, binwidth1)
etr_background  = base_background * electron_track_rejection(E*1000, *etr_popt)



############################## DIAGNOSIS PLOTS ####################################

plotting_tools.plot_arm_distribution(background_data)
plotting_tools.plot_azimuthal_angle_difference(background_data, ACCEPTANCE_ANGLE)
plotting_tools.plot_energy_distribution_fit(x1, y1, popt_base, background_func, binwidth1, ARM_MAX, EMIN)
plotting_tools.plot_energy_distribution_fit_loglog(x1, y1, popt_base, background_func, binwidth1, ARM_MAX, EMIN)
plotting_tools.plot_background_rate_energy_fitted(E, base_background, etr_background)
plotting_tools.plot_background_rate_ratio(Y1, background_ratio, electron_track_rejection, etr_popt)


############################## SENSITIVITY CALCULATION ############################

LSF = pd.read_pickle(f'{MEL_folder}.pkl')

# make a Energy_MeV column
LSF['Energy_MeV'] = LSF['Energy']/1000
mev_to_erg = 1.60218e-6


DIAMETER  = 300 #cm
AREA_cm   = np.pi*(DIAMETER/2)**2
Teff_y = 365*24*3600*3*0.40

def line_sensitivity(E, FWHM, sigma_e, efficiency, etr_on=True):
    Aeff = AREA_cm * efficiency
    if etr_on:
        return  3*np.sqrt(phi_B(E, popt=popt_base, bw=binwidth1)
                 *angle_to_solid(FWHM/2)
                 *sigma_e/Aeff/Teff_y
                 *electron_track_rejection(E, *etr_popt))

    return 3*np.sqrt(phi_B(E, popt=popt_base, bw=binwidth1)
                    *angle_to_solid(FWHM/2)
                    *sigma_e/Aeff/Teff_y)

LSF['line_sens_base'] = line_sensitivity(LSF['Energy'], 
                                        LSF['FWHM'], 
                                        LSF['Sigma_E']/1000,
                                        LSF['Accepted_line'],
                                        etr_on=False)

LSF['line_sens_etr'] = line_sensitivity(LSF['Energy'],
                                        LSF['etrFWHM'],
                                        LSF['Sigma_E']/1000,
                                        LSF['Accepted_line'],
                                        etr_on=True)



def cont_sensitivity(E, FWHM, sigma_e, efficiency, etr_on=True):
    """ The supplied E assumes keV"""
    Aeff = AREA_cm * efficiency
    E_MeV = E / 1000
    delta_e = E_MeV / 2
    background = phi_B(E, popt=popt_base, bw=binwidth1) 
    if etr_on:
        background *= electron_track_rejection(E, *etr_popt)
    omega = angle_to_solid(FWHM / 2)
    sensitivity = 3 * np.sqrt(background * omega / Aeff / Teff_y / delta_e)
    return sensitivity

LSF['cont_sensitivity_base'] = cont_sensitivity(LSF['Energy'], 
                                                    LSF['FWHM'], 
                                                    LSF['Sigma_E'],
                                                    LSF['Accepted_cont'],
                                                    etr_on=False)

LSF['cont_sensitivity_etr'] = cont_sensitivity(LSF['Energy'],
                                                    LSF['etrFWHM'],
                                                    LSF['Sigma_E'],
                                                    LSF['Accepted_cont'],
                                                    etr_on=True)



# Call the plotting functions
plotting_tools.plot_line_sensitivity_vs_energy(LSF, DIAMETER, AREA_cm)
plotting_tools.plot_mean_line_sensitivity_vs_energy(LSF, DIAMETER, AREA_cm)
plotting_tools.plot_line_sensitivity_comparison(LSF)
plotting_tools.plot_effective_area_angular_resolution_energy_resolution(LSF, AREA_cm)


plotting_tools.plot_continuum_sensitivity_vs_energy(LSF, mev_to_erg)
plotting_tools.plot_continuum_sensitivity_comparison(LSF, mev_to_erg, DIAMETER, AREA_cm, Teff_y)
plotting_tools.plot_efficiency_acceptance_vs_energy(LSF)

print(LSF)
LSF.to_pickle(f'final_sensitivity_{base_name}.pkl')
