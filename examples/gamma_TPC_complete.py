#%%
import os
import sys
import pickle
import awkward as ak
import numpy as np
import subprocess
import argparse

tools_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(tools_dir)
sys.path.append('tools')

sim_file_name = "FarFieldPointSourceFixed_1.000MeV_Cos1.0.inc1.id1.sim"
sim_file_name = sim_file_name.strip('.sim')

from gampy.tools import events_tools
from gampy.tools import params_tools

# Define paths
paths = {}
paths['root'] = '.'
paths['data'] = os.path.join(paths['root'], 'data')

#
activation_sim = None
geo_file_name = next((file for file in os.listdir(paths['root']) if file.endswith('.geo.setup')), None)
#%%
######## Helper functions ########
def clean_activation_file(file_path):
    with open(file_path, "r") as file:
        lines = [line for line in file if not line.startswith("CC")]

    with open(file_path, "w") as file:
        file.writelines(lines)

    return lines

def find_highest_id(lines):
    for line in reversed(lines):
        if line.startswith("ID"):
            return int(line.split(" ")[1])
    return 0  # Return 0 if no ID found

def combine_files(activation_path, sim_path):
    with open(sim_path + ".sim", "r") as sim_file:
        sim_lines = sim_file.readlines()

    highest_sim_id = find_highest_id(sim_lines)
    activation_lines = clean_activation_file(activation_path)

    activation_event_count = find_highest_id(activation_lines)
    adjusted_activation_lines = [
        " ".join(["ID", str(int(line.split(" ")[1]) + highest_sim_id)] + line.split(" ")[2:]) if line.startswith("ID") else line
        for line in activation_lines
    ]

    en_index = next((i for i, line in enumerate(sim_lines) if line.strip() == "EN"), len(sim_lines))
    se_ai = next((i for i, line in enumerate(adjusted_activation_lines) if line.strip() == "SE"), 0)
    en_ai = next((i for i, line in enumerate(adjusted_activation_lines) if line.strip() == "EN"), len(adjusted_activation_lines))

    combined_lines = sim_lines[:en_index-1] + adjusted_activation_lines[se_ai:en_ai] + sim_lines[en_index:]
    combined_file_path = os.path.join(paths['root'], "combined_file.sim")

    with open(combined_file_path, "w") as file:
        file.writelines(combined_lines)

    return combined_file_path, activation_event_count


########### Main script ###########
# Check for the existence of the activation file
if activation_sim is not None:
    activation_file_path = os.path.join(paths['root'], activation_sim)
else:
    activation_file_path = ""

sim_file_path = os.path.join(paths['root'], sim_file_name)
print(sim_file_path)

ps_command = f"Select-String -Path \"{sim_file_path+'.sim'}\" -Pattern '^SE' | Measure-Object | %{{ $_.Count }}"
result = subprocess.run(['powershell', '-Command', ps_command], capture_output=True, text=True, check=True)
num_events = int(result.stdout.strip())
print(num_events)

if os.path.exists(activation_file_path):
    combined_file_path, activation_events_number = combine_files(activation_file_path, sim_file_path)
else:
    combined_file_path = sim_file_path
    activation_events_number = 0
    print("No activation file found; using only simulation file.")

# Rename .geo.pickle files
for filename in os.listdir(paths['root']):
    if filename.endswith(".geo.pickle"):
        new_filename = filename.replace(".geo.pickle", ".geo.setup.pickle")
        os.rename(os.path.join(paths['root'], filename), os.path.join(paths['root'], new_filename))

# Create Events object

print(combined_file_path)
events = events_tools.Events(combined_file_path.strip('.sim')[1:],
                            os.path.join(paths['root'], geo_file_name),
                            num_events + activation_events_number)

# Load geometry parameters
with open(os.path.join(paths['root'], geo_file_name) + '.pickle', 'rb') as f:
    sims_params = pickle.load(f)

# Set up and apply detector response
params = params_tools.ResponseParams(sims_params=sims_params)


STUDY = "Optimistic"
events.read_params.inputs['coarse_grids']['signal_fraction'] = 0.9

if STUDY == "Optimistic":
    events.read_params.inputs["spatial_resolution"]['sigma_xy'] = 2e-5
    events.read_params.inputs["spatial_resolution"]['spatial_resolution_multiplier'] = 0.75
    events.read_params.inputs['light']['collection'] = 0.3
    events.read_params.inputs['material']['sigma_p'] = 0.04
    events.read_params.inputs['coarse_grids']['noise'] = 10

elif STUDY == "Neutral":
    events.read_params.inputs['spatial_resolution']['sigma_xy'] = 3e-5
    events.read_params.inputs['spatial_resolution']['spatial_resolution_multiplier'] = 1
    events.read_params.inputs['light']['collection'] = 0.1
    events.read_params.inputs['material']['sigma_p'] = 0.05
    events.read_params.inputs['coarse_grids']['noise'] = 20

elif STUDY == "Pessimistic":
    events.read_params.inputs['spatial_resolution']['sigma_xy'] = 4e-5
    events.read_params.inputs['spatial_resolution']['spatial_resolution_multiplier'] = 1.5
    events.read_params.inputs['light']['collection'] = 0.05
    events.read_params.inputs['material']['sigma_p'] = 0.06
    events.read_params.inputs['coarse_grids']['noise'] = 40

events.read_params.calculate()

truth = events.truth
hits  = events.truth_hits

# 1 m2 detector
R_max = np.sqrt(2/np.pi)
mask = []
for i in range(len(hits.r)):
    if len(hits.r[i][0]) > 0:
        R = np.linalg.norm(hits.r[i][:2],axis=0)
        mask.append(np.all(R < R_max))
    else:
        mask.append(False)


new_truth = truth[mask]
new_hits = hits[mask]
events.truth = new_truth
events.truth_hits = new_hits

events.apply_detector_response()

events.reconstruct_events()
events.train_classifier_on_self()
events.classify_events()


