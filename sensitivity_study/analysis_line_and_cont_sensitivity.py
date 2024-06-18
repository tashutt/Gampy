#%% 
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy.optimize import curve_fit

### SETTINGS ###

TRAIN_MODEL = False
ACCEPTANCE_ANGLE = 1
ARM_MAX = 7
EMIN = 220
BINS = 33
SIM_AREA = (0.797885*100)**2 * np.pi


plt.rcParams.update({'font.size': 18, 
                     'axes.titlesize': 18, 
                     'ytick.labelsize': 14, 
                     'legend.fontsize': 18,
                     'legend.title_fontsize': 18,
                     'axes.labelsize': 18,
                     'axes.labelpad': 10,})


# create folder for plots if it does not exist
if not os.path.exists('plots'):
    os.makedirs('plots')

# Load all the dataframes and combine them into one called 'combined_df'
# and save it as a pickle file called 'analysis_background_combined.pkl'
# the time is also saved in the file name
dataframes = []
ttime = 0
for file in os.listdir('.'):
    if file.endswith('.pkl') and 'recon_' in file:
        try:
            in_energy = float(file.split('_')[2].strip('MeV')) * 1000
            in_angle = float(file.split('_')[3].strip('Cos').replace('.inc1.id1.pkl', ''))
        except:
            in_energy = None
            in_angle  = 1.0
            in_time   = float(file.split('_')[2].replace('.pkl', ''))
            ttime += in_time
            print(f"Energy and angle not found for {file}. Using time={in_time} instead.")

        df = pd.read_pickle(file)
        df['in_energy'] = in_energy
        df['in_angle']  = in_angle
        df['in_time']   = ttime

        dataframes.append(df)

combined_df = pd.concat(dataframes, ignore_index=True)
combined_df.to_pickle('analysis_background_combined.pkl')
print("total time: ", ttime)
tttime = np.max(combined_df.in_time)

###################
# Train the model #
###################
def train_model():
    df = pd.read_pickle('analysis_background_combined.pkl')
    X = df[['e_out_CKD', 
            'min_hit_distance', 
            'kn_probability', 
            'calc_to_sum_ene', 
            'num_of_hits', 
            'compton_angle',
            'energy_from_sum'
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

    plt.figure(figsize=(10, 8))
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
    # save into plots folder
    plt.savefig('plots/a_background_confusion_matrix.png')
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
            "energy_from_sum"]
#%%

# Predict the good data and clean it up
background_data = combined_df.dropna(subset=features)
background_data['good'] = classifier.predict(background_data[features])
background_data = background_data.dropna(subset=['ARM'])
background_data['beta_minus_Dbeta']  = background_data.beta_angle - background_data.delta_e_direction*90/np.pi
background_data['precise_e_readout'] = (background_data.delta_e_direction * 180 / np.pi) < 30


### Background Rate Calculation ###
#ETR means with electron track rejection

def background_func(x, a, b, c):
    return a*np.log(x) + b*np.sqrt(x) + c


def angle_to_solid(angle):
    """Takes in degrees and returns the solid angle in sr"""
    return 2 * np.pi * (1 - np.cos(np.radians(angle)))

def phi_B(energy, 
          popt,
          bw,
          Area=SIM_AREA,
          Time=tttime,
          ARMMAX=ARM_MAX):
    """
    Calculate the background rate for a given energy
    in: energy (keV),
        popt (fit parameters), 
        bw (bin width),
        Area (cm2),
        Time (s),
        ARMMAX (degrees)
    Rate is in ph/cm2/MeV/sr/s
    """
    counts = np.exp(background_func(energy, *popt)) / bw
    solid_angle = ARMMAX / 180 / (4 * np.pi)
    cts_ats = counts / Area / Time / solid_angle
    # the rate is in ph/cm2/MeV/sr/s hence the division by 1e-3
    return cts_ats / (1e-3)


# Fit the data for the background rate (passed reconstruction selection)
gdf = background_data.query('good==1').query("energy_from_sum<10_000")
Y1 = np.array(gdf.query('ARM>-@ARM_MAX').query('ARM<@ARM_MAX')
             .query('energy_from_sum>@EMIN')
             .energy_from_sum)

y1, x1 = np.histogram(Y1, bins=BINS)
x1 = x1[:-1] + (x1[1] - x1[0]) / 2
x1,y1 = x1[y1 > 0], y1[y1 > 0]
binwidth1 = x1[1] - x1[0]
popt_base, _ = curve_fit(background_func, x1, np.log(y1), p0=[1, 1, 1])

# determining the ratio of background rate with and without ETR
bins_here = 44
X1, Y1 = np.histogram(background_data.query('good == 1')['energy_from_sum'], bins=bins_here, range=(210, 10_000))
X2, Y2 = np.histogram(background_data.query('good == 1 and beta_minus_Dbeta < @ACCEPTANCE_ANGLE')['energy_from_sum'], 
                      bins=bins_here, range=(210, 10_000))
background_ratio = X2/X1


def electron_track_rejection(x, a, b, c, d):
    return 1 - a/(1+np.exp(b * (x**0.25-c) )) + d*x
    
etr_popt, _ = curve_fit(electron_track_rejection, 
                        Y1[:-1], background_ratio,
                          p0=[1, 0.001, 500,1e-5])



############################## DIAGNOSIS PLOTS ##############################
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
plt.savefig('plots/arm_distribution_combined.png')
plt.clf()

# Subset of good data
gdf = background_data.query('good==1').query("energy_from_sum<10_000")

# Plot 4: ARM Distribution |ARM|<1 all energies
plt.figure(figsize=(10, 8))
sns.histplot(gdf.query('ARM>-1').query('ARM<1')['ARM'], bins=20, color='royalblue')
plt.xlabel('ARM')
plt.ylabel('Counts')
plt.title('ARM Distribution (|ARM|<1) all energies')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig('plots/arm_distribution_all_energies.png')
plt.clf()

# Plot 5: ARM Distribution |ARM|<5, 495<E<505
plt.figure(figsize=(10, 8))
sns.histplot(gdf.query('energy_from_sum>495').query('energy_from_sum<505')
             .query('ARM>-5').query('ARM<5')['ARM'], bins=5*20, color='royalblue')
plt.xlabel('ARM')
plt.ylabel('Counts')
plt.title('ARM Distribution (|ARM|<5) 495<E<505')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig('plots/arm_distribution_495_505.png')
plt.clf()

# Plot 6: Energy Distribution (|ARM|<5) E>150 keV
plt.figure(figsize=(10, 8))
sns.histplot(gdf.query('ARM>-@ARM_MAX')
             .query('ARM<@ARM_MAX')
             .query('energy_from_sum>@EMIN')['energy_from_sum'], 
             bins=50, color='royalblue')
plt.xlabel('Energy (keV)')
plt.ylabel('Counts')
plt.title(f'Energy Distribution (|ARM|<{ARM_MAX}) E>{EMIN} keV')
plt.yscale('log')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig('plots/energy_distribution_greater_150.png')
plt.clf()



plt.figure(figsize=(10, 8))
sns.histplot(data=background_data.query('good == 1'), x='beta_minus_Dbeta', 
             bins=90, color='grey', hue='precise_e_readout')
plt.axvline(ACCEPTANCE_ANGLE, color='red', linestyle='--')
plt.xlabel('Azimuthal Angle Difference ($ \\beta $ - $\\Delta \\beta$) [deg]')
plt.ylabel('Counts')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig('plots/dif_distribution_good_data.png')
plt.clf()

plt.figure(figsize=(10, 8))
n, bins, patches = plt.hist(background_data['energy_from_sum'], bins=BINS, range=(110, 10_000),
                            histtype='step', color='grey', linewidth=1.5, label='All Data')
plt.hist(background_data.query('good == 1')['energy_from_sum'], bins=bins, histtype='step',
         color='royalblue', linewidth=1.5, label='Good Reconstruction')
plt.hist(background_data.query('good == 1 and beta_minus_Dbeta < @ACCEPTANCE_ANGLE')['energy_from_sum'], bins=bins, 
         histtype='step', color='red', linewidth=1.5, label=f'Good With $\\beta - \\Delta \\beta < {ACCEPTANCE_ANGLE}$')

plt.text(0.05, 0.05, "$\\beta - \\Delta \\beta < 0$ means the azimuthal angle allows for the background to be from the same direction as the signal",
         ha='left', va='center', fontsize=8, transform=plt.gca().transAxes)
plt.xlabel('Energy from Sum')
plt.ylabel('Counts')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.yscale('log')
plt.tight_layout()
plt.savefig('plots/energy_from_sum_distribution.png')
plt.clf()


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
plt.savefig('plots/energy_distribution_fit.png')
plt.clf()


# Background Rate as a Function of Energy
E = np.logspace(-1, 1, 100)
base_background = phi_B(E*1000, popt_base, binwidth1) 
etr_background  = base_background * electron_track_rejection(E*1000, *etr_popt)

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
plt.savefig('plots/background_rate_energy.png')
plt.clf()


plt.figure(figsize=(10, 8))
plt.plot(Y1[:-1], background_ratio, 'o', label='Data')
plt.plot(Y1[:-1],   electron_track_rejection(Y1[:-1], *etr_popt), 'r--', label=f'Fit')
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
plt.savefig('plots/background_rate_ratio.png')
plt.clf()

################################################################################


#%%

#load the FWHM data a_line_response_results.pkl

LSF = pd.read_pickle('a_line_response_results.pkl')
LSF = LSF.query('Energy > 200')

DIAMETER  = 300 #cm
AREA_cm   = np.pi*(DIAMETER/2)**2
Teff = 1e6 #s

def line_sensitivity(E, FWHM, sigma_e, efficiency,etr_on=True):
    Aeff = AREA_cm * efficiency
    if etr_on:
        return  3*np.sqrt(phi_B(E, popt=popt_base, bw=binwidth1)
                 *angle_to_solid(FWHM/2)
                 *sigma_e/Aeff/Teff
                 *electron_track_rejection(E, *etr_popt))

    return 3*np.sqrt(phi_B(E, popt=popt_base, bw=binwidth1)
                    *angle_to_solid(FWHM/2)
                    *sigma_e/Aeff/Teff)



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
# properly annotate the area and time
plt.annotate(f'Diameter: {DIAMETER:.0f} cm', (0.5, 0.55), xycoords='axes fraction')
plt.annotate(f'Area: {AREA_cm:.0f} $cm^2$', (0.5, 0.5), xycoords='axes fraction')
plt.annotate(f'Time: $10^6$ s', (0.5, 0.45), xycoords='axes fraction')
plt.grid(True, which='both', linestyle='--', linewidth=0.7)
plt.legend()
plt.xscale('log')
plt.clf()

# do the same but only plot mean values for each energy
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

# %%

energy_levels_previous = [511, 666, 847, 1157, 1173, 1333, 1809, 2223, 4438]
sensitivity_satellite = [6.3e-7, 4.2e-7, 2.7e-7, 1.9e-7, 1.9e-7, 1.7e-7, 1.3e-7, 1.1e-7, 7.3e-8]
sensitivity_spi_integral = [5.0e-5, 2.0e-5, 2.0e-5, 2.0e-5, 2.0e-5, 2.0e-5, 2.5e-5, 2.0e-5, 1.0e-5]
COSIx = 1e3*np.array([0.147571, 0.2, 0.367354, 0.508666, 0.85134, 1.01441, 1.95191, 3.45926, 5.39, 6.77638])
COSIy = [4.03346e-05, 1.25254e-05, 4.66412e-06, 3.22023e-06, 2.19126e-06, 1.96506e-06, 1.77505e-06, 3.01644e-06, 9.92763e-06, 3.26735e-05]

plt.figure(figsize=(10, 8))
plt.plot(energy_levels_previous, sensitivity_spi_integral, 'o--', label='SPI Integral $10^6$s')
plt.plot(COSIx, COSIy, 'o--', label='COSI 2y ($10^6$s eff)')
plt.plot(energy_levels_previous, sensitivity_satellite, 'o--', label='GRAMS Satellite $10^6$s')

#make the line a bit thicker
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



#%%
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
    #ax.set_yscale('log') 
    ax.legend(title='Angle of Incidence')

plt.xlabel('Energy (keV)')
axs[-1].set_xscale('log')

plt.show()


#%%

DIAMETER  = 300 #cm
AREA_cm   = np.pi*(DIAMETER/2)**2
Teff_y = 365*24*3600*3*0.40

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




# make a Energy_MeV column
LSF['Energy_MeV'] = LSF['Energy']/1000
mev_to_erg = 1.60218e-6

# plot the log log of Sensitivity x $E^2$ (erg/(cm$^2$ * s)) vs E (MeV)
plt.figure(figsize=(10, 8))
for angle in LSF['Angle'].unique():
    df_angle = LSF[LSF['Angle'] == angle]
    Yax = mev_to_erg * df_angle['cont_sensitivity_base']*df_angle['Energy_MeV']**2
    plt.plot(df_angle['Energy_MeV'], Yax, 
             label=f'Angle {np.degrees(np.arccos(angle)):0.2f}°', marker='o', alpha=0.7)
    Yax = mev_to_erg * df_angle['cont_sensitivity_etr']*df_angle['Energy_MeV']**2
    plt.plot(df_angle['Energy_MeV'], Yax,
                label=f'Angle {np.degrees(np.arccos(angle)):0.2f}° with ETR', marker='o')


    
plt.xlabel('Energy (MeV)')
plt.ylabel('Continuum Sensitivity x $E^2$ [erg/cm2/s]')
plt.title('Continuum Sensitivity x $E^2$ vs Energy')
plt.grid(True, which='both', linestyle='--', linewidth=0.7)
plt.legend()
plt.xscale('log')
plt.yscale('log')

#%%

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


plt.figure(figsize=(10, 8))

for instrument, dataset in continuum_data.items():
    plt.plot(dataset['X'], dataset['Y'], label=instrument, marker='o',alpha=0.7, linestyle='--')
    plt.annotate(instrument, xy=(dataset['X'][-2], dataset['Y'][-2]), textcoords="offset points", xytext=(5,5), ha='right')

df_grouped = LSF.groupby('Energy').mean()
gtpcX = df_grouped.index/1000
gtpcY = mev_to_erg * df_grouped['cont_sensitivity_base']*df_grouped['Energy_MeV']**2
plt.plot(gtpcX, gtpcY, 'o-', label='GammaTPC - 3yrs', linewidth=3)
plt.annotate("GammaTPC 3y", xy=(list(gtpcX)[-2], list(gtpcY)[-2]), textcoords="offset points", xytext=(5,5), ha='right')

gtpcYetr = mev_to_erg * df_grouped['cont_sensitivity_etr']*df_grouped['Energy_MeV']**2
plt.plot(gtpcX, gtpcYetr, 'o-', label='GammaTPC - 3yrs with ETR', linewidth=3)
plt.annotate("GammaTPC 3y ETR", xy=(list(gtpcX)[-4], list(gtpcYetr)[-2]), 
             textcoords="offset points", xytext=(5,5), ha='right')


plt.xlabel('Energy (MeV)')
plt.ylabel('Continuum Sensitivity x $E^2$ [erg/cm2/s]')
plt.title('Continuum Sensitivity x $E^2$ vs Energy')
plt.grid(True, which='both', linestyle='--', linewidth=0.7)
plt.xscale('log')
plt.yscale('log')


plt.tight_layout()
plt.savefig('plots/continuum_sensitivity_comparison.png')
plt.clf()




#%%

# group by energy and average over angle, plot efficiency and acceptance vs energy
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
#%%

# look up the working folder name 
# save LSF to a pickle file with the folder name
folder_name = os.path.basename(os.getcwd())
LSF.to_pickle(f'{folder_name}_final_df.pkl')

# %%
