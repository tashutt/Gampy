from itertools import permutations
import numpy as np
import awkward as ak
from tqdm import tqdm
import pandas as pd

# events with [1, 2] hits will have a seperate treatement
# events with [3, 4, 5, 6, 7] hits will be treated with CKD, 
# events with more than 8 hits will be treated with probabilistic methods

# first we will calculate the CKD for the events with [3, 4, 5, 6] hits
# the result of the reconstruction will have parameters inherited from 
# the truth, such as the total energy, incoming angle, missing ennergy stats
# ID and so on
# the result of the reconstructed events will be total energy, 
# reconstructed order, correct_order flag, and the CKD value
# and a bunch of other parameters that will be used to evaluate the 
# reconstruction quality (min_arm, min_compton_angle, min_geometric_angle,..)


def klein_nishina(energy_keV, theta_rad):
    r0 = 2.8179403227e-15  
    mc2 = 8.187105649650028e-14 
    keV_to_Joules = 1.60218e-16  

    E = energy_keV * keV_to_Joules
    E_prime = E / (1 + (E / mc2) * (1 - np.cos(theta_rad)))
    differential_cross_section = 0.5 * r0**2 * ((E_prime / E)**2) * ((E / E_prime) + (E_prime / E) - np.sin(theta_rad)**2)
    E_prime_at_0 = E / (1 + (E /mc2) * (1 - np.cos(0)))
    differential_cross_section_at_0 = 0.5 * r0**2 * ((E_prime_at_0 / E)**2) * ((E / E_prime_at_0) + (E_prime_at_0 / E))
    clipped = np.clip(differential_cross_section / differential_cross_section_at_0, 0, 1)
    return clipped

def geometric_cos(v1, v2):
    dot_product = (np.sum(v1*v2,axis=1) 
                   / (np.linalg.norm(v1,axis=1) 
                      * np.linalg.norm(v2,axis=1))
                    )
    return dot_product

def eout_scatter_error(E_in, E_out, geometric_cos):
    m_e_c2 = 511.0 
    E_out_calc = E_in / (1 + (E_in / m_e_c2) * (1 - geometric_cos)) 
    error = (E_out_calc - E_out)**2    
    return error

def ein_scatter_error(E_in, E_out, geometric_cos):
    m_e_c2 = 511.0 
    E_in_calc = 1 / (1/E_out - (1 - geometric_cos)/m_e_c2)  
    error = (E_in_calc - E_in)**2    
    return error

def e_3_formula(e2, cos_ang2):
    m_e_c2 = 511.0 
    SQ = e2**2/4 + m_e_c2*e2/(1-cos_ang2)
    return SQ**0.5 - e2/2

def calculate_compton_angle(E_incoming, E_outgoing):
    m_e_c2 = 511.0
    cos_theta = 1 - m_e_c2*(1/E_outgoing - 1/E_incoming) 
    return np.arccos(cos_theta)

def surplus_cal_comparison(sur, cal, calorimeter_tolerance=0.1):
    return (sur > cal*(1-calorimeter_tolerance)) & (sur < cal*(1+calorimeter_tolerance))


def reconstruct(events, 
                LEN_OF_CKD_HITS,
                IN_VECTOR = None, 
                use_truth_hits=False, 
                outside_mask=None, 
                MIN_ENERGY=0.1,
                filename=''):

    if use_truth_hits:
        events_to_reconstruct = events.truth_hits
        events.truth_hits['calorimeter_energy'] = events.measured_hits['calorimeter_energy']
    else:
        events_to_reconstruct = events.measured_hits

    if outside_mask is None:
        outside_mask = ((np.sum(events_to_reconstruct['energy'], axis=1) 
                         + events_to_reconstruct['calorimeter_energy']) > MIN_ENERGY)  

    data = {}
    CKD_DEPTH = 2
    for hit_len in LEN_OF_CKD_HITS:
        data[hit_len] = {}
        hit_len_mask = ak.num(events_to_reconstruct['energy']) == hit_len
        energies = events_to_reconstruct['energy'][hit_len_mask&outside_mask]
        calorimeter_energy = events_to_reconstruct['calorimeter_energy'][hit_len_mask&outside_mask]
        positions = events_to_reconstruct['r'][hit_len_mask&outside_mask]
        #recoil = events_to_reconstruct['s_secondary'][hit_len_mask&outside_mask]

        hit_len_prime = hit_len
        PERMUTATIONS = list(permutations(range(hit_len)))
        if hit_len > CKD_DEPTH + 2:
            hit_len_prime = CKD_DEPTH + 3
            PERMUTATIONS = list(permutations(range(hit_len),
                                            hit_len_prime))
            
        e_out_error        = np.zeros((len(energies), len(PERMUTATIONS)))
        e_out_error_prime  = np.zeros((len(energies), len(PERMUTATIONS)))
        total_energy = (np.sum(energies, axis=1) 
                        + calorimeter_energy)

        for p,permutation in tqdm(enumerate(PERMUTATIONS), desc=f"Scatter {hit_len}"):
            cumulative_energies = np.cumsum(energies[:,list(permutation)],axis=1)
            for i in range(hit_len_prime - 2):
                r1_r2 = positions[:,:,permutation[i+1]] - positions[:,:,permutation[i]]
                r2_r3 = positions[:,:,permutation[i+2]] - positions[:,:,permutation[i+1]]
                geometric_cosine = geometric_cos(r1_r2, r2_r3)

                E_in = total_energy - cumulative_energies[:,i]
                E_ou = E_in - energies[:,permutation[i+1]]

                kn = klein_nishina(E_in, np.arccos(geometric_cosine))

                eouterr = eout_scatter_error(E_in, E_ou, geometric_cosine)
                e_out_error[:,p] += np.array(eouterr) - np.array(np.log(kn))
                
                # calculate the expected energy and make inferences about
                # the order of the hits (calorimeter first?)
                if i==0:
                    e2 = energies[:,permutation[1]]
                    e3 = e_3_formula(e2, geometric_cosine)
                    e_surplus = e3 + e2 + energies[:,permutation[0]] - total_energy
                    iem = (e_surplus > 0) & (e_surplus < 0.3*total_energy)
                    # check if the surplus is close to the calorimeter energy
                    sec = surplus_cal_comparison(-e_surplus, calorimeter_energy, 0.05)
                
                if hit_len > 4: # here I suscpect that some hits are lost
                    eouterr_prime = eout_scatter_error(E_in[iem] + e_surplus[iem], 
                                                    E_ou[iem] + e_surplus[iem], 
                                                            geometric_cosine[iem])
                    e_out_error_prime[:,p][iem] += np.array(eouterr_prime)

                # Here I suspect that calorimeter came in first or second
                eouterr_second = eout_scatter_error(E_in[sec] - calorimeter_energy[sec], 
                                                    E_ou[sec] - calorimeter_energy[sec], 
                                                        geometric_cosine[sec]) 
                                                        
                
                e_out_error_prime[:,p][sec] += np.array(eouterr_second)
        
        #fix the zeros
        e_out_error_prime[e_out_error_prime==0] = 1e12
        prime_better = e_out_error > e_out_error_prime
        e_out_error[prime_better] = e_out_error_prime[prime_better]

        best_e_out_order = np.argmin(e_out_error, axis=1)
        best_e_out       = np.min(e_out_error, axis=1)**0.5 

        data[hit_len]['best_e_out_order'] = best_e_out_order
        data[hit_len]['e_out_CKD']        = best_e_out
        data[hit_len]['energy_from_sum']  = total_energy  

        klein_nishina_sums = np.ones(len(best_e_out_order))
        min_distances      = np.ones(len(best_e_out_order))
        calculated_energy  = np.zeros(len(best_e_out_order))
        first_hit_compton  = np.zeros(len(best_e_out_order))
        first_hit_dot_ang  = np.zeros(len(best_e_out_order))
        truth_angle12      = np.zeros(len(best_e_out_order))


        # now calculate the relevant stats given the order
        for p,permutation in enumerate(PERMUTATIONS):
            u_mask = p == best_e_out_order
            if sum(u_mask) == 0:
                continue
            positions_m = positions[u_mask]
            energies_m  = energies[u_mask]
            cumulative_energies = np.cumsum(energies_m[:,list(permutation)],axis=1)


        
            r1_r2 = positions_m[:,:,permutation[1]] - positions_m[:,:,permutation[0]]
            r2_r3 = positions_m[:,:,permutation[2]] - positions_m[:,:,permutation[1]]
            geometric_cosine = geometric_cos(r1_r2, r2_r3)
            
            E_in = total_energy[u_mask] - cumulative_energies[:,0]

            kn = klein_nishina(E_in, np.arccos(geometric_cosine))
            klein_nishina_sums[u_mask] *= np.array(kn)

            e2 = energies_m[:,permutation[1]]
            e3 = e_3_formula(e2, geometric_cosine)
            calculated_in = e3+ e2 + energies_m[:,permutation[0]]
            calculated_energy[u_mask] = np.array(calculated_in)

            min_arm_len = np.minimum(np.linalg.norm(r1_r2, axis=1), 
                                        np.linalg.norm(r2_r3, axis=1))
            min_distances[u_mask] = min_arm_len

            theta = calculate_compton_angle(total_energy[u_mask], E_in)
            first_hit_compton[u_mask] = theta
            first_hit_dot_ang[u_mask] = np.arccos(geometric_cos(r1_r2, 
                                                                np.array([IN_VECTOR])))
            R1R2 = events.truth_hits.s_primary[hit_len_mask&outside_mask][u_mask][:,:,permutation[0]] 
            truth_angle12[u_mask] = np.arccos(geometric_cos(r1_r2, R1R2))*180/np.pi
                
        first3_good = np.array(PERMUTATIONS)[best_e_out_order,:3]==[0,1,2]

        data[hit_len]['min_hit_distance']  = min_distances
        data[hit_len]['kn_probability']    = klein_nishina_sums
        data[hit_len]['calculated_energy'] = calculated_energy/total_energy - 1
        data[hit_len]['num_of_hits']       = hit_len*np.ones(len(best_e_out_order))
        data[hit_len]['first_scatter_angle'] = first_hit_compton

        data[hit_len]['ARM'] = (first_hit_compton - first_hit_dot_ang) * 180/np.pi

        data[hit_len]['truth_lost_in_passive'] = events.truth.passive_energy[hit_len_mask&outside_mask] > 15
        data[hit_len]['truth_escaped_energy']  = events.truth.escaped_energy[hit_len_mask&outside_mask] > 15
        data[hit_len]['truth_correct_order']   = first3_good.all(axis=1)
        data[hit_len]['truth_angle12']         = truth_angle12
        data[hit_len]['truth_calorimeter_first3'] = events.truth.calorimeter_in_first_3[hit_len_mask&outside_mask]

        this = data[hit_len]
        bad_events = (this['truth_escaped_energy'] 
                    | this['truth_calorimeter_first3'])
        
        correct_order = this['truth_correct_order']

        print(f"-> {hit_len} hits:",end=' | ')
        #print percentage of good events
        print(f"Good events: {100*sum(~bad_events)/len(bad_events):.2f} %", end=' | ')
        print(f"Correct order: {100*sum(correct_order)/len(correct_order):.2f} %")


    dfs = [pd.DataFrame(data[k]) for k in sorted(data.keys())]
    df = pd.concat(dfs, axis=0, ignore_index=True)

    if len(filename) > 0:
        df.to_pickle(f'{filename}_reconstructed.pkl')

    return df
    

def train_classifier(df, filename='classifier.pkl', plot_confusion_matrix=True):
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    import joblib

    X = df[['e_out_CKD', 
            'min_hit_distance', 
            'kn_probability', 
            'calculated_energy', 
            'num_of_hits', 
            'first_scatter_angle',
            ]]

    y = df[['truth_escaped_energy', 'truth_correct_order', 'truth_calorimeter_first3']]

    y_new = y.copy()
    y_new['good'] = ((y['truth_escaped_energy'] == False) # no escaped energy
                   & (y['truth_correct_order']  == True) # good order
                   & (y['truth_calorimeter_first3'] == False)).astype(int) # no calorimeter in first 3
    y = y_new[['good']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train.values.ravel())

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    print(classification_report(y_test, y_pred))
    joblib.dump(clf, filename)

    importances = clf.feature_importances_
    feature_names = X.columns
    feature_importances_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_importances_df = feature_importances_df.sort_values(by='Importance', ascending=False)
    print(feature_importances_df)

    if plot_confusion_matrix:
        import matplotlib.pyplot as plt
        import seaborn as sns

        y_pred_all = clf.predict(X)
        conf_matrix = confusion_matrix(y, y_pred_all)

        plt.cla()
        plt.clf()
        plt.figure(figsize=(6, 4))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
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

    return clf







