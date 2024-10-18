import pandas as pd
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy.optimize import curve_fit
import pickle




"""
Here we load the background data from the server and plot the energy spectrum.
Activation and cosmic data are plotted separately and combined.
First, they are plotted raw
After that we 
"""


def process_mono_energetic_results(mono_energetic_results_folder="monoenergetic_line_results"):
    import glob
    import pickle
    dataframes = []
    for file in os.listdir(mono_energetic_results_folder):
        if file.endswith('.pkl') and 'recon_' in file:
            if 'MeV' and 'Cos' in file:
                in_energy = float(file.split('_')[2].strip('MeV')) * 1000
                in_angle = float(file.split('_')[3].strip('Cos').replace('.inc1.id1.pkl', ''))
            else:
                in_energy = 1000.0
                in_angle  = 1.0
                in_time   = float(file.split('_')[2].replace('.pkl', ''))
                print(f"Energy and angle not found for {file}. Using time={in_time} instead.")

            df = pd.read_pickle(os.path.join(mono_energetic_results_folder, file))
            df['in_energy'] = in_energy
            df['in_angle'] = in_angle
            dataframes.append(df)


    all_summaries_path = os.path.join(mono_energetic_results_folder, 'all_summaries.pickle')

    if os.path.exists(all_summaries_path):
        with open(all_summaries_path, 'rb') as f:
            all_summaries = pickle.load(f)
    else:
        os.chdir(mono_energetic_results_folder)
        summary_files = glob.glob("*_summary.pickle")
        all_summaries = [pickle.load(open(file, 'rb')) for file in summary_files]

        for file in summary_files:
            os.remove(file)
        
        with open('all_summaries.pickle', 'wb') as f:
            pickle.dump(all_summaries, f)
        os.chdir('..')

    summary_df = pd.DataFrame(all_summaries)
    combined_df = pd.concat(dataframes, ignore_index=True)
    combined_df.to_pickle(os.path.join(mono_energetic_results_folder, 'monoenergetic_line_data.pkl'))
    return summary_df, combined_df


def analyze_monoenergetic_line_data(line_data, monoenergetic_summary, reduction_factor):
    fwhm_results = []
    groups = line_data.groupby(['in_energy', 'in_angle'])

    unique_energies = sorted(line_data['in_energy'].unique())
    unique_angles   = sorted(line_data['in_angle'].unique())

    fig, axs = plt.subplots(len(unique_energies), 
                            len(unique_angles), 
                            figsize=(20, 34), sharex=True, sharey=True)

    if len(unique_energies) == 1 or len(unique_angles) == 1:
        axs = axs.reshape(len(unique_energies), len(unique_angles))

    def lorentzian(x, x0, gamma, A, r):
        return A * (gamma**2 / ((x - x0)**2 + gamma**2)) + r

    def gaussian(x, x0, sigma, A, k, n):
        return A * np.exp(-(x - x0)**2 / (2 * sigma**2)) + k + n*x

    for (energy, angle), group in groups:
        group_good = group.query("good == 1")
        
        # ARM and minARM histograms
        y, bins = np.histogram(group_good.ARM, bins=70, range=(-7, 7))
        x = (bins[1:] + bins[:-1]) / 2
        
        y_min, bins_min = np.histogram(group_good.minARM, bins=70, range=(-7, 7))
        x_min = (bins_min[1:] + bins_min[:-1]) / 2

        try:
            # ARM fit
            popt, _ = curve_fit(lorentzian, x, y, p0=[0, 1, max(y), 0])
            fwhm = 2 * popt[1]

            # minARM fit
            popt_min, _ = curve_fit(lorentzian, x_min, y_min, p0=[0, 1, max(y_min), 0])
            fwhm_min = abs(2 * popt_min[1])

            # Energy distribution and Gaussian fit
            en_cut = group_good.energy_from_sum[(group_good.energy_from_sum > 0.95 * energy) & 
                                                (group_good.energy_from_sum < 1.05 * energy)]
            hist2, bins2 = np.histogram(en_cut, bins=70)
            x2 = (bins2[1:] + bins2[:-1]) / 2
            popt2, _ = curve_fit(gaussian, x2, hist2, p0=[energy, en_cut.std(), max(hist2), 0, 0])
            sigma_e = abs(popt2[1])

            # Efficiency and acceptance calculations
            num_incoming_photons = monoenergetic_summary.query(f'energy=={energy} and angle=={angle}').number_of_events_after_cuts.values[0]
            num_detected_photons = group_good.shape[0]

            # Correct for the reduction factor
            efficiency = num_detected_photons / num_incoming_photons * (1 + reduction_factor)

            num_accepted_photons = group_good.query(f'energy_from_sum > {popt2[0] - sigma_e}')
            num_accepted_photons = num_accepted_photons.query(f'energy_from_sum < {popt2[0] + sigma_e}')
            num_accepted_photons = num_accepted_photons.query(f'minARM > {-fwhm_min} and minARM < {fwhm_min}')
            num_accepted_photons = num_accepted_photons.shape[0]

            acceptance_line = num_accepted_photons / num_incoming_photons * (1 + reduction_factor)

            num_accepted_photons_cont = group_good.query(f'minARM > {-fwhm_min} and minARM < {fwhm_min}').shape[0]
            acceptance_cont = num_accepted_photons_cont / num_incoming_photons * (1 + reduction_factor)

            fwhm_results.append((energy, angle, fwhm, fwhm_min, sigma_e, efficiency, acceptance_line, acceptance_cont))

            # Plotting
            i = unique_energies.index(energy)
            j = unique_angles.index(angle)
            axs[i, j].plot(x, lorentzian(x, *popt), 'r-', linewidth=2, label='Lorentzian ARM Fit')
            axs[i, j].plot(x_min, lorentzian(x_min, *popt_min), 'g-', linewidth=1.6, label='Lorentzian minARM Fit')
            axs[i, j].hist(group_good.ARM, bins=bins, alpha=0.6, label='ARM Histogram')
            axs[i, j].hist(group_good.minARM, bins=bins_min, alpha=0.4, label='minARM Histogram')
            axs[i, j].set_title(f'E={energy}, A={np.degrees(np.arccos(angle)):0.2f}°')
            axs[i, j].text(0.5, 0.9, f'FWHM={fwhm:0.2f}°', ha='center', va='center', transform=axs[i, j].transAxes)
            axs[i, j].text(0.5, 0.8, f'etrFWHM={fwhm_min:0.2f}°', ha='center', va='center', transform=axs[i, j].transAxes)
            axs[i, j].legend()

        except RuntimeError:
            print(f"Fit could not converge for Energy={energy} keV, Angle={np.degrees(np.arccos(angle)):0.2f}°")

    plt.tight_layout()
    plt.savefig('plots/arm_hist_grid.png')
    plt.close()
    plt.clf()

    # Save results to DataFrame
    fwhm_df = pd.DataFrame(fwhm_results, columns=['Energy', 'Angle', 'FWHM', 'etrFWHM', 'Sigma_E', 'Efficiency', 'Accepted_line', 'Accepted_cont'])
    fwhm_df.to_pickle('monoenergetic_line_results.pkl')

    return fwhm_df



############################## background_data ##############################
def load_background_data_from_server(source_dir, name_of_background_file='background_data.pkl'):
    tdata = {'activation':0, 'cosmic':0}
    bdata = {'activation':[], 'cosmic':[]}
    for folder in os.listdir(source_dir):
        run_folder = os.path.join(source_dir, folder)
        suffix     = 'activation' if 'activation_run' in folder else 'cosmic'
        if "run" not in folder:
            continue
        
        in_energy = None
        temp_DF   = {"recon": None, "summary": None}
        for file_name in os.listdir(run_folder):
            if "recon_" in file_name:
                df = pd.read_pickle(os.path.join(run_folder, file_name))
                temp_DF["recon"] = df
            if "summary" in file_name:
                de = pd.read_pickle(os.path.join(run_folder, file_name))
                temp_DF["summary"] = de
                if "SimulationStep" in file_name:
                    in_energy = int(file_name.split("_")[1].split("to")[0])

        if temp_DF["recon"] is None or temp_DF["summary"] is None:
            print(f"Skipping {folder}")
            continue

        bdata[suffix].append(temp_DF["recon"])
        de = temp_DF["summary"]
        bdata[suffix][-1]['in_time'] = de['time']
        tdata[suffix] += de['time']
        bdata[suffix][-1]['in_energy'] = in_energy
    
    bdata['activation'][-1]['in_time'] = tdata['activation'] 

    dataframes = []
    for key in bdata:
        if len(bdata[key]) == 0:
            continue
        df = pd.concat(bdata[key], ignore_index=True)
        df['source'] = key
        dataframes.append(df)

    background_df = pd.concat(dataframes, ignore_index=True)
    background_df.to_pickle(name_of_background_file)
    return background_df




##############################
######### Activation #########
##############################

def get_activation_rates(background_df, min_e, max_e, num_of_bins, classified=False):
    classified_activation_events = background_df.query('source=="activation"').query('good==1')
    activation_events = background_df.query('source=="activation"').query("energy_from_sum>@min_e and energy_from_sum < @max_e")
    aR, aE = np.histogram(activation_events['energy_from_sum'], bins=num_of_bins, range=(min_e, max_e))
    aRc, aEc = np.histogram(classified_activation_events['energy_from_sum'], bins=num_of_bins, range=(min_e, max_e))
    activation_rate   = aR/activation_events.in_time.max()
    activation_rate_c = aRc/activation_events.in_time.max()
    if classified:
        return activation_rate_c, aEc[:-1]
    return activation_rate, aE[:-1]

# add_decay_lines(min_decay_rate=0.01)


##############################
######### Cosmic #############
##############################

def get_cosmic_rates(background_df, min_e, max_e, num_of_bins, classified=False):
    assert 'good' in background_df.columns, "The data is not classified. Please classify the data first."
    unique_in_energy = background_df.query('source=="cosmic"')['in_energy'].unique()
    classified_cosmic_background = background_df.query('source=="cosmic"').query('good==1')

    rates   = []
    rates_c = []
    for in_energy in unique_in_energy:
        limited_run  = background_df.query(f'source=="cosmic" and in_energy=={in_energy}')
        limited_run_c = classified_cosmic_background.query(f'in_energy=={in_energy}')
        total_lmtd_T = limited_run['in_time'].unique().sum()

        N,E = np.histogram(limited_run['energy_from_sum'], bins=num_of_bins, range=(min_e, max_e))
        Nc,Ec = np.histogram(limited_run_c['energy_from_sum'], bins=num_of_bins, range=(min_e, max_e))
        R   = N/total_lmtd_T
        Rc  = Nc/total_lmtd_T
        R[E[:-1]<10**in_energy]  = np.nan
        Rc[E[:-1]<10**in_energy] = np.nan
        rates.append(R)
        rates_c.append(Rc)

    rates    = np.array(rates)
    rates_c  = np.array(rates_c)

    if classified:
        return np.nanmean(rates_c,axis=0), Ec[:-1]
    return np.nanmean(rates,axis=0), E[:-1]




def train_model(df, classifier_name, features, target):
    X = df[features]
    y = df[target]
    y_new = y.copy()
    y_new['good'] = ( (y['truth_escaped_energy'] == False) # no escaped energy)
                    & (y['truth_correct_order']  == True) # good order
                    & (y['truth_calorimeter_first3'] == False)).astype(int) # no calorimeter in first 3

    y = y_new[['good']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    clf = RandomForestClassifier(n_estimators=60)
    clf.fit(X_train, y_train.values.ravel())

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    print(classification_report(y_test, y_pred))
    clf_dest = os.path.join('classifiers', classifier_name) + '_classifier.pkl'
    joblib.dump(clf, clf_dest)
    return X, y, clf


def evaluate_classifier(X, y, clf):
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

    # Annotations for "good" and "bad" meanings
    plt.text(2.5, 1.5, "'Good':\nNo Escaped Energy\nand Correct Order", 
            ha="center", va="center", fontsize=9, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.text(2.5, 0.5, "'Bad':\nOpposite of Good", 
            ha="center", va="center", fontsize=9, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    plt.savefig('plots/a_confusion_matrix.png')
    plt.clf()

    importances = clf.feature_importances_
    feature_names = X.columns
    feature_importances_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_importances_df = feature_importances_df.sort_values(by='Importance', ascending=False)
    print(feature_importances_df)


