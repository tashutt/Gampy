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

TRAIN_MODEL = True

dataframes = []
for file in os.listdir('.'):
    if file.endswith('.pkl') and 'recon_' in file:
        try:
            in_energy = float(file.split('_')[2].strip('MeV')) * 1000
            in_angle = float(file.split('_')[3].strip('Cos').replace('.inc1.id1.pkl', ''))
        except:
            in_energy = 1000.0
            in_angle  = 1.0
            in_time   = float(file.split('_')[2].replace('.pkl', ''))
            print(f"Energy and angle not found for {file}. Using time={in_time} instead.")

        df = pd.read_pickle(file)
        df['in_energy'] = in_energy
        df['in_angle'] = in_angle

        dataframes.append(df)


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

    #%% Train new clasifier
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

    # Annotations for "good" and "bad" meanings
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

def lorentzian(x, x0, gamma, A):
    return A * (gamma**2 / ((x - x0)**2 + gamma**2))

fwhm_results = []
groups =combined_df.groupby(['in_energy', 'in_angle'])

unique_energies = sorted(combined_df['in_energy'].unique())
unique_angles   = sorted(combined_df['in_angle'].unique())

plt.rcParams.update({'font.size': 18, 'axes.titlesize': 18, 'ytick.labelsize': 14, 'legend.fontsize': 18})
fig, axs = plt.subplots(len(unique_energies), len(unique_angles), figsize=(20, 20), sharex=True, sharey=True)

if len(unique_energies) == 1 or len(unique_angles) == 1:
    axs = axs.reshape(len(unique_energies), len(unique_angles))

for (energy, angle), group in groups:
    group_clean = group.dropna(subset=features).copy()
    
    if not group_clean.empty:
        X_group = group_clean[features]
        group_clean.loc[:, 'use'] = classifier.predict(X_group)
        
        if group_clean['use'].any():
            hist, bins = np.histogram(group_clean.query("use==1").ARM, bins=70, range=(-10,10))
            x = (bins[1:] + bins[:-1]) / 2
            y = hist

            try:
                popt, _ = curve_fit(lorentzian, x, y, p0=[0, 1, max(y)])
                fwhm = 2 * popt[1]

                en = group_clean.query("use==1").energy_from_sum
                # only include between 0.95 and 1.05 of the mean
                en_cut = en[(en > 0.95*energy) & (en < 1.05*energy)]
                sigma_e = en_cut.std()
                
                try:
                    num_incoming_photons = summary_df.query(f'energy=={energy} and angle=={angle}').number_of_events_after_cuts.values[0]
                    num_detected_photons = group_clean.query("use==1").shape[0]
                    efficiency = num_detected_photons / num_incoming_photons
                except:
                    efficiency = np.nan

                fwhm_results.append((energy, angle, fwhm, sigma_e, efficiency))
                i = unique_energies.index(energy)
                j = unique_angles.index(angle)
                axs[i, j].plot(x, lorentzian(x, *popt), 'r-', linewidth=2, label='Lorentzian Fit')
                axs[i, j].hist(group_clean.query("use==1").ARM, bins=bins, alpha=0.6, label='Histogram')
                axs[i, j].set_title(f'E={energy}, A={np.degrees(np.arccos(angle)):0.2f}°')
                axs[i, j].text(0.5, 0.9, f'FWHM={fwhm:0.2f}°', ha='center', va='center', transform=axs[i, j].transAxes)
                axs[i, j].legend()
            except RuntimeError:
                print(f"Fit could not converge for Energy={energy} keV, Angle={np.degrees(np.arccos(angle)):0.2f}°")
    else:
        print(f"No data Energy={energy} keV, Angle={np.degrees(np.arccos(angle)):0.2f}°")

plt.tight_layout()
plt.savefig('a_hist_grid.png')
plt.close()
plt.clf()

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
plt.savefig('a_angle_vs_xsi.png')

plt.close()
plt.clf()

fwhm_df = pd.DataFrame(fwhm_results, columns=['Energy', 'Angle', 'FWHM','Sigma_E', 'Efficiency'])

fig, ax = plt.subplots(figsize=(10, 6))
for angle in unique_angles:
    df_angle = fwhm_df[fwhm_df['Angle'] == angle]
    ax.plot(df_angle['Energy'], df_angle['FWHM'], label=f'Angle {np.degrees(np.arccos(angle)):0.2f}°', marker='o')

ax.set_xlabel('Energy (keV)')
ax.set_ylabel('FWHM (degrees)')
ax.set_title('Angular Resolution vs Energy')
ax.grid(True)
ax.legend()
ax.set_xscale('log')
ax.set_ylim(0, 10)
ax.set_xlim(100, None)
plt.tight_layout()
plt.savefig('angular_resolution.png')
plt.close()
plt.clf()

#also plot sigma E as a function of energy
fig, ax = plt.subplots(figsize=(10, 6))
for angle in unique_angles:
    df_angle = fwhm_df[fwhm_df['Angle'] == angle]
    ax.plot(df_angle['Energy'], df_angle['Sigma_E'], label=f'Angle {np.degrees(np.arccos(angle)):0.2f}°', marker='o')

ax.set_xlabel('Energy (keV)')
ax.set_ylabel('Sigma E (keV)')
ax.set_title('Energy Resolution vs Energy')
ax.grid(True)
ax.legend()

plt.tight_layout()
plt.savefig('an_energy_resolution.png')
plt.close()
plt.clf()

# also plot efficiency as a function of energy
fig, ax = plt.subplots(figsize=(10, 6))
for angle in unique_angles:
    df_angle = fwhm_df[fwhm_df['Angle'] == angle]
    ax.plot(df_angle['Energy'], df_angle['Efficiency'], label=f'Angle {np.degrees(np.arccos(angle)):0.2f}°', marker='o')

ax.set_xlabel('Energy (keV)')
ax.set_ylabel('Efficiency')
ax.set_title('Efficiency vs Energy')
ax.grid(True)

plt.tight_layout()
plt.savefig('an_efficiency.png')
plt.close()

print(fwhm_df)

# Save the fwhm results to a pickle file
fwhm_df.to_pickle('a_line_response_results.pkl')
