#%%
import os
import sys
import pickle
import awkward as ak
import numpy as np
import subprocess
import argparse

# get sim_file_name from argparse
parser = argparse.ArgumentParser(description='Input the name of the simulation file .sim')
parser.add_argument('sim_file_name', type=str, help='The name of the simulation file .sim')
args = parser.parse_args()

tools_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(tools_dir)
sys.path.append('tools')

# if sim_file_name is not provided, use the default one
if args.sim_file_name:
    sim_file_name = args.sim_file_name
else:
    sim_file_name = "FarFieldPointSource_1.000MeV_Cos1.0.inc1.id1.sim"

# extract the energy and the angle from the file name
try:
    in_energy = float(sim_file_name.split('_')[1].strip('MeV')) * 1000
    in_angle  = float(sim_file_name.split('_')[2].strip('Cos').replace('.inc1.id1.sim', ''))
except:
    in_energy = 1000
    in_angle = 1.0
    in_time = float(sim_file_name.split('background_')[1].replace('.sim', ''))
    print("Time", in_time)

print('Energy, Angle:', in_energy, in_angle)

sim_file_name = sim_file_name.strip('.sim')

import events_tools
import params_tools

# Define paths
paths = {}
paths['root'] = '.'
paths['data'] = os.path.join(paths['root'], 'data')

geo_file_name = next((file for file in os.listdir(paths['root']) if file.endswith('.geo.setup')), None)

sim_file_path = os.path.join(paths['root'], sim_file_name)
print(sim_file_path)


if os.name == 'nt':  # Windows
    command = ['powershell', '-Command', f"Select-String -Path \"{sim_file_path+'.sim'}\" -Pattern '^SE' | Measure-Object | %{{ $_.Count }}"]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
else:
    with open(f"{sim_file_path}.sim", "r") as file:
        num_events = sum(1 for line in file if line.startswith('SE'))

if os.name == 'nt':
    num_events = int(result.stdout.strip())

print(num_events)

# Rename .geo.pickle files
for filename in os.listdir(paths['root']):
    if filename.endswith(".geo.pickle"):
        new_filename = filename.replace(".geo.pickle", ".geo.setup.pickle")
        os.rename(os.path.join(paths['root'], filename), os.path.join(paths['root'], new_filename))

# Create Events object

print(sim_file_path)
events = events_tools.Events(sim_file_path.strip('.sim')[1:],
                            os.path.join(paths['root'], geo_file_name),
                            num_events)

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

# 2 m2 detector
R_max = np.sqrt(2/np.pi)
len_mask = ak.num(hits.r[:, 0]) > 0
R = np.linalg.norm(hits[len_mask].r[:, :2, 0],axis=1)
mask = np.zeros(len(truth), dtype=bool)
mask[len_mask] = R < R_max

print("Masking events", len(truth), "to", sum(mask),
      f"for R < {R_max:.3f} m. That's {sum(mask)/len(truth)*100:.2f}% of the events.")

summary_dict = {
    "energy": in_energy,
    "angle": in_angle,
    "number_of_events": len(truth),
    "number_of_events_after_cuts": sum(mask),
    "study": STUDY,
    "r_max": R_max}

with open(os.path.join(f'{sim_file_name.split(".inc")[0]}_summary.pickle'), 'wb') as f:
    pickle.dump(summary_dict, f)

events.truth = truth[mask]
events.truth_hits = hits[mask]

events.apply_detector_response()

in_vector = np.array([-np.sqrt(1-in_angle**2), 0, -in_angle])
events.reconstruct_events(IN_VECTOR=in_vector,
                          save_name=sim_file_name,
                          LEN_OF_CKD_HITS = [3,4,5,6,7,8,9,10,11])




# events.train_classifier_on_self()

# events.classify_reconstructed_events(save_name=sim_file_name)


# need to report data
# time in the simulation
# number of passed cuts sum(mask)
# number of events in the simulation len(truth)
# rate is len(truth)/time/original_area

