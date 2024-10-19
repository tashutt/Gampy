import os
import sys

sys.path.append('tools')

import awkward as ak
import events_tools
import numpy as np
import params_tools


paths = {}
paths['root'] = 'cosima_run2'
paths['data'] = os.path.join(paths['root'], 'data')

tools_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(tools_dir)
import pickle


geo_file_name = "GammaTPC_GeoT01v03.geo.setup"
sim_file_name = 'SimulationStep1For550km_2to8keV.inc1.id1'
activation_sim = "ActivationStep3For550km_2to8keV.inc1.id1"


num_events = 100
activation_events_number = 0#4000

# Function to clean and process activation file
def process_activation_file(file_name):
    # Remove lines starting with "CC"
    with open(file_name + ".sim", "r") as file:
        modified_lines = [line for line in file if not line.startswith("CC")]

    with open(file_name + ".sim", "w") as file:
        file.writelines(modified_lines)

    # Read the cleaned file and find number of activation events
    with open(file_name + ".sim", "r") as file:
        lines = file.readlines()
        for line in reversed(lines):
            if line.startswith("ID"):
                return int(line.split(" ")[1]), lines

# Function to combine activation file with sim file
def combine_files(activation_lines, sim_file_name):
    with open(os.path.join(paths['root'], sim_file_name) + ".sim", "r") as file:
        sim_lines = file.readlines()

    en_index = sim_lines.index("EN\n")
    se_ai = activation_lines.index("SE\n")
    en_ai = activation_lines.index("EN\n")

    combined_lines = sim_lines[:en_index-1] + activation_lines[se_ai:en_ai] + sim_lines[en_index:]

    with open(os.path.join(paths['root'], "combined_file.sim"), "w") as file:
        file.writelines(combined_lines)

    return "combined_file"

# Process activation file
activation_file1 = os.path.join(paths['root'], activation_sim)
activation_events_number, activation_lines = process_activation_file(activation_file1)
print("Number of events in activation:", activation_events_number)

# Combine files
sim_file_name = combine_files(activation_lines, sim_file_name)

# Rename .geo.pickle files
for filename in os.listdir(paths['root']):
    if filename.endswith(".geo.pickle"):
        new_filename = filename.replace(".geo.pickle", ".geo.setup.pickle")
        os.rename(os.path.join(paths['root'], filename), os.path.join(paths['root'], new_filename))
        print(f"Renamed: {filename} to {new_filename}")

# Create Events object
events = events_tools.Events(os.path.join(paths['root'], sim_file_name),
                             os.path.join(paths['root'], geo_file_name),
                             num_events,# + activation_events_number,
                             read_sim_file=True,
                             write_events_files=False)


with open(os.path.join(paths['root'], geo_file_name) + '.pickle', 'rb') as f:
    geo_params = pickle.load(f)

params = params_tools.ResponseParams(geo_params=geo_params)

events.apply_detector_response()
events.pileup_analysis()


#%%
import awkward as ak
import numpy as np

MIN_DETECTABLE_ENERGY = 2
MIN_NEEDED_HITS       = 2
ACD_THRESHOLD = 15


def print_results(events,decision_tree_results, pileup_stats):
    TIME = events.truth['time'][-1]
    AREA = ak.max(events.truth_hits['cell']) * events.read_params.cells['area'] / 2
    print(f"Sim duration {TIME:2.2f} s| Detector Area {AREA:2.2f} m^2")
    print(f"{'Stage':<20}{'Count':<8}{'Rate (s^-1 m^-2)':<17}{'Percentage':<10}")
    print("-" * 55)

    initial_count = len(decision_tree_results['Incoming Particles'])+1
    results_dict = {}



    for name, lst in decision_tree_results.items():
        if isinstance(lst, list):
            survived_count = len(lst)
            rate = round(survived_count / TIME / AREA, 2)
            survived_percentage = (survived_count / initial_count) * 100
            print(f"{name + ':':<25} {survived_count:<5} {rate:<15} ({survived_percentage:.2f}%)")

            results_dict[name] = {
                "count": survived_count,
                "percentage": f"{survived_percentage:.2f}%",
                "rate": rate
            }

    # Adding pileup statistics to the results
    print("-"*25)
    for key, value in pileup_stats.items():
        print(f"{key + ':':<25} {value:2.2f}")
        results_dict[key] = value

    return results_dict


def acceptance_rejection(events,
                         particle_type = "photon_cos",
                           good_gamma_min_ene_kev=100,
                           good_gamma_max_ene_kev=10_000):

    acd_energy = events.truth['front_acd_energy'] + events.truth['back_acd_energy']
    acd_activ = acd_energy > ACD_THRESHOLD

    cosmic_photons = []
    good_energy_range = []
    all_energy_contained = []
    detectable_gammas = []
    det_with_enough_hits = []
    all_requirements_met = []
    seen_gammas = []

    for event_id in range(len(events.truth['incident_energy'])):
        event_has_overlap = False

        p_energy   = events.truth['incident_energy'][event_id]
        p_name = events.truth['particle_name'][event_id]

        if p_name == particle_type:
            cosmic_photons.append(p_energy)
        elif particle_type == "decay_activity" and any(char.isdigit() for char in p_name):
            cosmic_photons.append(p_energy)
        else:
            continue

        # check if the gamma ray has good incident energy
        if not (p_energy < good_gamma_min_ene_kev
         or p_energy > good_gamma_max_ene_kev):
            good_energy_range.append(p_energy)

        if not events.truth['missing_energy'][event_id]:
            all_energy_contained.append(p_energy)

        hit_energies = events.truth_hits["energy"][event_id]
        hit_energies = hit_energies[hit_energies > 0.01]

        detectable = hit_energies > MIN_DETECTABLE_ENERGY
        lost_to_threshold_ene = ak.sum(hit_energies[~detectable])
        if lost_to_threshold_ene < 0.02 * np.sum(hit_energies):
            detectable_gammas.append(p_energy)

        # has to have scatters to detect it
        det = hit_energies[detectable]
        num_of_hits_in_this_event = len(ak.mask(det, det>1))
        if num_of_hits_in_this_event > MIN_NEEDED_HITS:
            det_with_enough_hits.append(p_energy)

        # has the acd been activated?
        if acd_activ[event_id]:
            continue
        all_requirements_met.append(p_energy)

        # only detectable gammas survive to this point
        # now we find if they coincide with any of the cell Light-ups
        # Iterate through each cell hit in the current event

        event_has_overlap = False
        in_good_coordinates = sum(events.measured_hits['_good_mask'][:event_id])
        event_has_overlap = in_good_coordinates in events.measured_hits['pileup_detected']

        if not event_has_overlap:
            seen_gammas.append(p_energy)
            # print(det)
            # print(f"Seen a {p_energy:2.1f} keV gamma! - {num_of_hits_in_this_event} hits\n")

    return {
        "Incoming Particles": cosmic_photons,
        "Good Energy Range": good_energy_range,
        "All Energy Contained": all_energy_contained,
        "Detectable Gammas": detectable_gammas,
        "With Enough Hits": det_with_enough_hits,
        "Met All Req.(ACD=0)": all_requirements_met,
        "Seen Gammas": seen_gammas
    }

def pileup_statistics_function(events,
                               decision_tree_results,
                               ):

    velocity = events.read_params.charge_drift['velocity']
    SIM_TIME = max(events.truth['time'])
    NUM_CELLS = ak.max(events.truth_hits['cell'])
    CELL_H = 0.175

    clearing_time = CELL_H / velocity
    good_ones = len(decision_tree_results["Met All Req.(ACD=0)"])
    if good_ones:
        pileup = (good_ones - len(decision_tree_results["Seen Gammas"]))/good_ones
    else:
        pileup = 0

    pileup_stats = {
        "clearing_time_ms": clearing_time * 1000,
        "pileup":pileup * 100
    }

    return pileup_stats

print("Rates of particles")
for particle_name in ['photon_cos','photon_atm','positron','decay_activity']:
    print(particle_name)
    decision_tree_results = acceptance_rejection(events, particle_name)
    pileup_stats = pileup_statistics_function(events, decision_tree_results)
    print_results(events, decision_tree_results, pileup_stats)
print()

#%%
ACD_THRESHOLD = 15
MIN_NEEDED_HITS = 3

def calculate_detection_rates_measured_hits(events, particle="photon_cos"):
    TIME = events.truth['time'][-1]
    AREA = ak.max(events.truth_hits['cell']) * events.read_params.cells['area'] / 2

    # Applying the good mask to filter out the relevant hits
    good_mask = events.measured_hits["_good_mask"]
    if particle == "decay_activity":
        name_filter = np.array([any(char.isdigit() for char in name) for name in events.truth["particle_name"]])
    else:
        name_filter = events.truth["particle_name"] == particle
    name_mask = name_filter[good_mask]

    filtered_measured_hits = events.measured_hits

    # Rate of particles that are detected
    total_detected_particles = len(filtered_measured_hits['energy'][name_mask])
    raw_rate = total_detected_particles / TIME / AREA

    # Rate after ACD veto
    acd_energy = events.truth['front_acd_energy'] + events.truth['back_acd_energy']
    acd_activ = acd_energy > ACD_THRESHOLD
    rate_acd_veto = np.sum(~acd_activ[good_mask][name_mask]) / TIME / AREA

    # Rate after ACD veto and not piled up
    pileup_detected = set(events.measured_hits['pileup_detected'])
    not_piled_up = ~np.isin(np.arange(len(filtered_measured_hits['energy'])), pileup_detected)
    rate_acd_not_piled_up = np.sum(~acd_activ[good_mask][name_mask] & not_piled_up[name_mask]) / TIME / AREA

    # Rate after ACD veto, not piled up, and enough hits
    enough_hits = ak.sum(filtered_measured_hits['cell'] > MIN_NEEDED_HITS, axis=1) > 0
    rate_acd_not_piled_up_enough_hits = np.sum(~acd_activ[good_mask][name_mask]
                                               & not_piled_up[name_mask] & enough_hits[name_mask]) / TIME / AREA

    # Rate after ACD veto, not piled up, enough hits, and full energy contained
    full_energy_contained = ~events.truth['missing_energy'][good_mask][name_mask]
    rate_acd_not_piled_up_enough_hits_full_energy = np.sum(~acd_activ[good_mask][name_mask]
                                                           & not_piled_up[name_mask] & enough_hits[name_mask] & full_energy_contained) / TIME / AREA

    detection_rates = {
        "Raw Rate": raw_rate,
        " - and no ACD veto": rate_acd_veto,
        "  - and not piled up": rate_acd_not_piled_up,
        "   - and enough hits": rate_acd_not_piled_up_enough_hits,
        "    - and full energy contained": rate_acd_not_piled_up_enough_hits_full_energy
    }

    return detection_rates


def format_detection_rates(detection_rates):
    formatted_rates = {key: f"{value:.0f} particles/s/m²" for key, value in detection_rates.items()}
    return formatted_rates

def display_detection_rates_as_table(detection_rates):
    # Formatting the header
    header = "Detection Criteria | Rate (particles/s/m²)"
    print(header)
    print("-" * len(header))

    # Formatting and displaying each row
    for criteria, rate in detection_rates.items():
        print(f"{criteria:<35} | {rate}")



print("Rates of detected/measured particles")
for particle_name in ['photon_cos','photon_atm','positron','decay_activity']:
    print(particle_name)
    detection_rates = calculate_detection_rates_measured_hits(events, particle=particle_name)
    formatted_rates = format_detection_rates(detection_rates)
    display_detection_rates_as_table(formatted_rates)
    print()


