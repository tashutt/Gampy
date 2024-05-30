#! python 
# -*- coding: utf-8 -*-

# this script will read in all the recon_*.pkl files and the summary files
# and combine them into a single dataframe. It will then train a classifier
# to determine if a photon is good or bad. It will then plot the angular
# resolution as a function of energy and angle. It will also plot the energy
# resolution as a function of energy and angle. Finally, it will plot the
# efficiency and acceptance as a function of energy and angle.

# THIS OUTPUTS:
#   - a_combined_classifier.pkl: the trained classifier
#   - analysis_combined.pkl: the combined dataframe
#   - a_line_response_results.pkl: <---- the results of the line response analysis
#   ------------------------------------------------------------------------
#   - a_confusion_matrix.png:    confusion matrix of the classifier
#   - a_hist_grid.png:           grid of histograms of the ARM and minARM
#   - angular_resolution.png:    the angular resolution as a function of energy
#   - angular_resolution_e.png:  the angular resolution as a function of energy and angle
#   - an_energy_resolution.png:  the energy resolution as a function of energy
#   - an_energy_resolution%.png: the energy resolution as a function of energy in percent
#   - an_efficiency.png:         the efficiency and acceptance as a function of energy and angle
#   - energy_hist_*.png:         the energy histogram for each energy and angle


#%%
import pandas as pd
import os, glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy.optimize import curve_fit
import pickle

TRAIN_MODEL = False

plt.rcParams.update({'font.size': 18, 
                     'axes.titlesize': 18, 
                     'ytick.labelsize': 14, 
                     'legend.fontsize': 18})

dataframes = []
for file in os.listdir('.'):
    if file.endswith('.pkl') and 'recon_' in file:
        if 'MeV' and 'Cos' in file:
            in_energy = float(file.split('_')[2].strip('MeV')) * 1000
            in_angle = float(file.split('_')[3].strip('Cos').replace('.inc1.id1.pkl', ''))
        else:
            in_energy = 1000.0
            in_angle  = 1.0
            in_time   = float(file.split('_')[2].replace('.pkl', ''))
            print(f"Energy and angle not found for {file}. Using time={in_time} instead.")

        df = pd.read_pickle(file)
        df['in_energy'] = in_energy
        df['in_angle'] = in_angle
        dataframes.append(df)

# crate a plots directory if it doesn't exist
if not os.path.exists('plots'):
    os.makedirs('plots')
# below, save all the plots to the plots directory

if os.path.exists('all_summaries.pickle'):
    with open('all_summaries.pickle', 'rb') as f:
        all_summaries = pickle.load(f)
else:
    summary_files = glob.glob("*_summary.pickle")
    all_summaries = [pickle.load(open(file, 'rb')) for file in summary_files]

    for file in summary_files:
        os.remove(file)
    
    with open('all_summaries.pickle', 'wb') as f:
        pickle.dump(all_summaries, f)

summary_df = pd.DataFrame(all_summaries)

combined_df = pd.concat(dataframes, ignore_index=True)
combined_df.to_pickle('analysis_combined.pkl')

##########################################
# TRAIN AND EVALUATE CLASSIFIER FUNCTION #
##########################################
def train_model():
    df = pd.read_pickle('analysis_combined.pkl')
    X = df[['e_out_CKD', 
            'min_hit_distance', 
            'kn_probability', 
            'calc_to_sum_ene', 
            'num_of_hits', 
            'compton_angle',
            'energy_from_sum',
            ]]

    y = df[['truth_escaped_energy', 'truth_correct_order', 'truth_calorimeter_first3']]

    y_new = y.copy()
    y_new['good'] = ( (y['truth_escaped_energy'] == False) # no escaped energy)
                    & (y['truth_correct_order']  == True) # good order
                    & (y['truth_calorimeter_first3'] == False)).astype(int) # no calorimeter in first 3

    y = y_new[['good']]

    X = X.dropna(subset=['compton_angle'])
    y = y.loc[X.index]  # Ensure y is aligned with the modified X

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35)

    clf = RandomForestClassifier(n_estimators=60)
    clf.fit(X_train, y_train.values.ravel())

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    print(classification_report(y_test, y_pred))
    joblib.dump(clf, 'a_combined_classifier.pkl')
    return X, y


def evaluate_classifier(X, y):
    clf = joblib.load('a_combined_classifier.pkl')

    y_pred_all = clf.predict(X)
    conf_matrix = confusion_matrix(y, y_pred_all)
    print(conf_matrix)

    plt.figure(figsize=(6, 4))
    plt.clf()

    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)

    plt.title("Confusion Matrix")
    plt.xlabel("Prediction")
    plt.ylabel("Truth")
    plt.xticks(ticks=[0.5, 1.5], labels=["Bad (0)", "Good (1)"])
    plt.yticks(ticks=[0.5, 1.5], labels=["Bad (0)", "Good (1)"], rotation=0)

    plt.text(2.5, 1.5, "'Good':\nNo Escaped Energy\nand Correct Order", 
            ha="center", va="center", fontsize=9, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.text(2.5, 0.5, "'Bad':\nOpposite of Good", 
            ha="center", va="center", fontsize=9, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    plt.savefig('a_confusion_matrix.png')
    plt.clf()

    importances = clf.feature_importances_
    feature_names = X.columns
    feature_importances_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_importances_df = feature_importances_df.sort_values(by='Importance', ascending=False)
    print(feature_importances_df)

#######################

if TRAIN_MODEL:
    X, y = train_model()
    evaluate_classifier(X, y)


classifier = joblib.load('a_combined_classifier.pkl')
features = ['e_out_CKD', 'min_hit_distance', 'kn_probability', 
            'calc_to_sum_ene', 'num_of_hits', 'compton_angle',
            "energy_from_sum", ]

def lorentzian(x, x0, gamma, A, r):
    return A * (gamma**2 / ((x - x0)**2 + gamma**2)) + r

def gaussian(x, x0, sigma, A, k, n):
    return A * np.exp(-(x - x0)**2 / (2 * sigma**2)) + k + n*x

fwhm_results = []
groups =combined_df.groupby(['in_energy', 'in_angle'])

unique_energies = sorted(combined_df['in_energy'].unique())
unique_angles   = sorted(combined_df['in_angle'].unique())


fig, axs = plt.subplots(len(unique_energies), len(unique_angles), 
                        figsize=(20, 25), sharex=True)
if len(unique_energies) == 1 or len(unique_angles) == 1:
    axs = axs.reshape(len(unique_energies), len(unique_angles))

for (energy, angle), group in groups:
    group_clean = group.dropna(subset=features).copy()
    
    if not group_clean.empty:
        X_group = group_clean[features]
        group_clean.loc[:, 'use'] = classifier.predict(X_group)
        
        # beta angle is multiplied by 2 to get the full angle
        if group_clean['use'].any():
            useful_data = group_clean.query('use == 1').copy()
            useful_data.loc[:, 'minARM'] = useful_data.apply(
                lambda row: row['ARM'] 
                if abs(row['ARM']) <= row['delta_e_direction'] * 180 / np.pi
                else (
                    -row['beta_angle'] * 2 
                    if row['ARM'] < 0 
                    else row['beta_angle'] * 2
                ), 
                axis=1
            )


            # ARM binning
            PLOT_BINS = 100
            hist, bins = np.histogram(useful_data.ARM, 
                                      bins=PLOT_BINS, range=(-10,10))
            x = (bins[1:] + bins[:-1]) / 2
            y = hist

            # min ARM binning
            hist_min, bins_min = np.histogram(useful_data.minARM, 
                                              bins=PLOT_BINS, range=(-10,10))
            x_min = (bins_min[1:] + bins_min[:-1]) / 2
            y_min = hist_min

            try:
                popt, _ = curve_fit(lorentzian, x, y, 
                                    p0=[0, 1, max(y),0])
                fwhm = 2 * popt[1]
                
                # min ARM fit
                popt_min, _ = curve_fit(lorentzian, x_min, y_min, 
                                        p0=[0, 1, max(y_min),0])
                fwhm_min = abs(2 * popt_min[1])

                en = useful_data.energy_from_sum
                # only include between 0.95 and 1.05 of the mean
                en_cut = en[(en > 0.97*energy - 10) & (en < 1.05*energy)]
                # get a new histogram of the energy cut and fit to a gaussian
                hist2, bins2 = np.histogram(en_cut, bins=PLOT_BINS)
                x2 = (bins2[1:] + bins2[:-1]) / 2
                y2 = hist2
                popt2, _ = curve_fit(gaussian, x2, y2, 
                                     p0=[energy, en_cut.std(), max(y2), 0, 0])
                sigma_e = abs(popt2[1])

                ################ PLOTTING ################
                fig2, ax2 = plt.subplots()
                ax2.hist(en_cut, bins=80)
                ax2.plot(x2, gaussian(x2, *popt2), 'r-', linewidth=2, label='Gaussian Fit')
                ax2.set_title(f'E={energy}, A={np.degrees(np.arccos(angle)):0.2f}°')
                ax2.text(0.5, 0.9, f'Sigma E={sigma_e:0.2f} keV', ha='center', va='center', transform=ax2.transAxes)
                plt.savefig(f'plots/energy_hist_{energy}_{angle}.png')
                plt.close()

                
                try:
                    num_incoming_photons = summary_df.query(f'energy=={energy} and angle=={angle}').number_of_events_after_cuts.values[0]
                    num_detected_photons = useful_data.shape[0]
                    # accept 2 sigma in energy and - fwhm to fwhm in angle (line)
                    num_accepted_photons = (useful_data
                                    .query(f'energy_from_sum > {popt2[0] - sigma_e}')
                                    .query(f'energy_from_sum < {popt2[0] + sigma_e}')
                                    .query(f'minARM > {-fwhm_min} and minARM < {fwhm_min}')
                                    .shape[0]
                                            )
                    # accept 2 sigma in energy and - fwhm to fwhm in angle (continuous)
                    num_accepted_photons_cont = (useful_data
                                    .query(f'minARM > {-fwhm_min} and minARM < {fwhm_min}')
                                    .shape[0]
                                            )

                    efficiency      = num_detected_photons / num_incoming_photons
                    acceptence_line = num_accepted_photons / num_incoming_photons
                    acceptance_cont = num_accepted_photons_cont / num_incoming_photons
                except:
                    efficiency      = np.nan
                    acceptence_line = np.nan
                    acceptance_cont = np.nan

                fwhm_results.append((energy, 
                                     angle, 
                                     fwhm, 
                                     fwhm_min, 
                                     sigma_e, 
                                     efficiency, 
                                     acceptence_line,
                                     acceptance_cont))
                
                i = unique_energies.index(energy)
                j = unique_angles.index(angle)
                axs[i, j].plot(x, lorentzian(x, *popt), 'r-', linewidth=2, 
                               label='Lorentzian ARM Fit')
                axs[i, j].plot(x_min, lorentzian(x_min, *popt_min), 'g-', linewidth=1.6, 
                               label='Lorentzian minARM Fit')
                axs[i, j].hist(useful_data.ARM, bins=bins, alpha=0.6, label='Histogram')
                axs[i, j].hist(useful_data.minARM, bins=bins_min, alpha=0.4, label='eARM Histogram')

                axs[i, j].set_title(f'E={energy}, A={np.degrees(np.arccos(angle)):0.2f}°')
                axs[i, j].text(0.5, 0.9, f'FWHM={fwhm:0.2f}°', ha='center', va='center', 
                               transform=axs[i, j].transAxes)
                axs[i, j].text(0.5, 0.8, f'etrFWHM={fwhm_min:0.2f}°', ha='center', va='center', 
                               transform=axs[i, j].transAxes)
            except RuntimeError:
                print(f"Fit could not converge for Energy={energy} keV, Angle={np.degrees(np.arccos(angle)):0.2f}°")
    else:
        print(f"No data Energy={energy} keV, Angle={np.degrees(np.arccos(angle)):0.2f}°")



plt.tight_layout()
fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.06), ncol=1)
plt.savefig('plots/a_hist_grid.png')
plt.close()
plt.clf()

fwhm_df = pd.DataFrame(fwhm_results, columns=['Energy', 
                                              'Angle', 
                                              'FWHM', 
                                              'etrFWHM', 
                                              'Sigma_E', 
                                              'Efficiency', 
                                              'Accepted_line',
                                              'Accepted_cont'])
print(fwhm_df)
fwhm_df.to_pickle('a_line_response_results.pkl')







##########################
#### PLOTS FOR REPORT ####
##########################

fig, axs = plt.subplots(len(unique_energies), 1, figsize=(10, 44), sharex=True)
axs = axs.reshape(len(unique_energies), 1)

for (energy, angle), group in groups:
    if angle > 0.9:
        group_clean = group.dropna(subset=features).copy()
        X_group = group_clean[features]
        group_clean.loc[:, 'use'] = classifier.predict(X_group)
        
        i = unique_energies.index(energy)
        axs[i, 0].scatter(group_clean.xsi_angle,
                          group_clean.compton_angle,
                          c=group_clean['use'],
                          s=0.4,
                          alpha=0.6)
        
plt.tight_layout()
plt.savefig('plots/a_angle_vs_xsi.png')

plt.close()
plt.clf()


fig, ax = plt.subplots(figsize=(8, 6))
unique_angles.sort()

for idx, angle in enumerate(unique_angles):
    fwhm_df_f = fwhm_df.query('Energy>300')
    df_angle = fwhm_df_f[fwhm_df_f['Angle'] == angle]
    ax.plot(
        df_angle['Energy'], df_angle['FWHM'], 
        label=f'Angle {np.degrees(np.arccos(angle)):0.2f}° FWHM', 
        marker='o', linestyle='--', linewidth=1)
    
# also plot the mean of etrFWHM across all angles
mean_fwhm = fwhm_df.query('Energy>300').groupby('Energy')['etrFWHM'].mean()
ax.plot(mean_fwhm.index, mean_fwhm.values, label='Mean etrFWHM', marker='o', linewidth=2)

# annotate the last point of the mean etrFWHM line with the value with arrow
ax.annotate(f'{mean_fwhm.values[-1]:0.2f}°', 
            (mean_fwhm.index[-1], mean_fwhm.values[-1]), 
            textcoords="offset points", xytext=(0,10), ha='center')

ax.set_xlabel('Energy (keV)')
ax.set_ylabel('FWHM (°)')
ax.set_title('Angular Resolution vs Energy')
ax.grid(True)
ax.legend()
ax.set_xscale('log')
ax.set_ylim(0, None)
plt.tight_layout()
plt.savefig('plots/angular_resolution.png')
plt.close()
plt.clf()

# plot the angular resolution etrFWHM as a function of energy and angle
fig, ax = plt.subplots(figsize=(8, 6))
for angle in unique_angles:
    df_angle = fwhm_df.query('Energy>300')[fwhm_df.query('Energy>300')['Angle'] == angle]
    ax.plot(df_angle['Energy'], 
            df_angle['etrFWHM'], 
            label=f'Angle {np.degrees(np.arccos(angle)):0.0f}°', marker='o')

ax.set_xlabel('Energy (keV)')
ax.set_ylabel('FWHM (°)')
ax.set_title('Angular Resolution vs Energy')
ax.grid(True)
ax.legend()
ax.set_ylim(0, None)
ax.set_xscale('log')
plt.tight_layout()
plt.savefig('plots/angular_resolution_e.png')



fig, ax = plt.subplots(figsize=(8, 6))
for angle in unique_angles:
    df_angle = fwhm_df[fwhm_df['Angle'] == angle]
    ax.plot(df_angle['Energy'], df_angle['Sigma_E'], label=f'Angle {np.degrees(np.arccos(angle)):0.2f}°', marker='o')

ax.set_xlabel('Energy (keV)')
ax.set_ylabel('Sigma E (keV)')
ax.set_title('Energy Resolution vs Energy')
ax.grid(True)
ax.legend()
ax.set_ylim(0, None)
ax.set_xscale('log')
plt.tight_layout()
plt.savefig('plots/an_energy_resolution.png')
plt.close()
plt.clf()

# also plot the Sigma E/E [%]
fig, ax = plt.subplots(figsize=(8, 6))
for angle in unique_angles:
    df_angle = fwhm_df[fwhm_df['Angle'] == angle]
    ax.plot(df_angle['Energy'], df_angle['Sigma_E'] / df_angle['Energy'] * 100, label=f'Angle {np.degrees(np.arccos(angle)):0.2f}°', marker='o')

ax.set_xlabel('Energy (keV)')
ax.set_ylabel('Sigma E/E (%)')
ax.set_title('Energy Resolution vs Energy')
ax.grid(True)
ax.legend()
ax.set_xscale('log')
plt.tight_layout()
plt.savefig('plots/an_energy_resolution_percent.png')
plt.close()
plt.clf()


# also plot efficiency and acceptance as a function of energy
fig, ax = plt.subplots(figsize=(8, 6))
for angle in unique_angles:
    df_angle = fwhm_df[fwhm_df['Angle'] == angle]
    ax.plot(df_angle['Energy'], df_angle['Efficiency'], label=f'Angle {np.degrees(np.arccos(angle)):0.2f}° Efficiency', marker='o')
    ax.plot(df_angle['Energy'], df_angle['Accepted_line'], label=f'Angle {np.degrees(np.arccos(angle)):0.2f}° Acceptance_line', marker='o')
    ax.plot(df_angle['Energy'], df_angle['Accepted_cont'], label=f'Angle {np.degrees(np.arccos(angle)):0.2f}° Acceptance_cont', marker='o')

# annotate the three plots using the first point of each with the name of the plot
ax.annotate('Efficiency', (df_angle['Energy'].values[2], df_angle['Efficiency'].values[1]), 
            textcoords="offset points", xytext=(3,10), ha='right')
ax.annotate('Acceptance Line', (df_angle['Energy'].values[2], df_angle['Accepted_line'].values[3]),
            textcoords="offset points", xytext=(3,10), ha='right')
ax.annotate('Acceptance Cont', (df_angle['Energy'].values[2], df_angle['Accepted_cont'].values[2]),
            textcoords="offset points", xytext=(3,10), ha='right')

ax.set_xlabel('Energy (keV)')
ax.set_ylabel('Efficiency and Acceptance')
ax.set_title('Efficiency vs Energy')
ax.grid(True)
ax.set_xscale('log')

plt.tight_layout()
plt.savefig('plots/an_efficiency.png')
plt.close()

# %%
