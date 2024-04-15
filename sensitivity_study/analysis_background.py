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

TRAIN_MODEL = True

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
combined_df.to_pickle('analysis_combined.pkl')
print("total time: ", ttime)

def train_model():
    df = pd.read_pickle('analysis_combined.pkl')
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
            "energy_from_sum"]





combined_df_clean = combined_df.dropna(subset=features)
combined_df_clean['good'] = classifier.predict(combined_df_clean[features])
combined_df_clean = combined_df_clean.dropna(subset=['ARM'])

plt.rcParams.update({'font.size': 18, 'axes.titlesize': 18, 'ytick.labelsize': 14, 'legend.fontsize': 18})

plt.figure(figsize=(10, 8))
n, bins, patches = plt.hist(combined_df_clean['ARM'], bins=100, histtype='step', color='grey', linewidth=1.5, label='All Data')
plt.hist(combined_df_clean.query('good==1')['ARM'], bins=bins, histtype='step', color='royalblue', linewidth=1.5, label='Good Data')

plt.xlabel('ARM')
plt.ylabel('Counts')
plt.title('ARM Distribution')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig('a_bckgrnd_arm_hist.png')
plt.clf()

plt.figure(figsize=(10, 8))
n, bins, patches = plt.hist(combined_df_clean['ARM'], bins=100,range=(-10,10), histtype='step', color='grey', linewidth=1.5, label='All Data')
plt.hist(combined_df_clean.query('good==1')['ARM'], bins=bins, histtype='step', color='royalblue', linewidth=1.5, label='Good Data')

plt.xlabel('ARM')
plt.ylabel('Counts')
plt.title('ARM Distribution')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig('a_bckgrnd_arm_hist_lmtd.png')
plt.clf()

