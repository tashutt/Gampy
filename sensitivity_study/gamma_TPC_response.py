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
parser.add_argument('study',         type=str, help='The study to be performed: Optimistic, Nominal, Pessimistic')
parser.add_argument('N_hits_to_reconstruct', type=int, default=11 ,help='The number of hits to be reconstructed')
args = parser.parse_args()

tools_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(tools_dir)
sys.path.append('tools')

import events_tools
import params_tools


sim_file_name         = args.sim_file_name
STUDY                 = args.study
N_hits_to_reconstruct = args.N_hits_to_reconstruct

# extract the energy and the angle (and time) from the file name
try:
    in_energy = float(sim_file_name.split('_')[1].strip('MeV')) * 1000
    in_angle  = float(sim_file_name.split('_')[2].strip('Cos').replace('.inc1.id1.sim', ''))
    print('Energy, Angle:', in_energy, in_angle)

except:
    in_energy = 1000
    in_angle = 1.0
    parts = sim_file_name.split('_')
    in_time = float(parts[1])
    min_energy = float(parts[3].strip("minEnergy").replace('.sim', ''))
    print("Time:", in_time)
    print("Min Energy:", min_energy)


sim_file_name = sim_file_name.strip('.sim')

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
    geo_params = pickle.load(f)

# Set up and apply detector response
params = params_tools.ResponseParams(geo_params=geo_params)


events.params.inputs['coarse_grids']['signal_fraction'] = 0.9

if STUDY == "Optimistic":
    events.params.inputs["spatial_resolution"]['sigma_xy'] = 2e-5
    events.params.inputs["spatial_resolution"]['spatial_resolution_multiplier'] = 0.75
    events.params.inputs['light']['collection'] = 0.3
    events.params.inputs['material']['sigma_p'] = 0.04
    events.params.inputs['coarse_grids']['noise'] = 10

elif STUDY == "Nominal":
    events.params.inputs['spatial_resolution']['sigma_xy'] = 3e-5
    events.params.inputs['spatial_resolution']['spatial_resolution_multiplier'] = 1
    events.params.inputs['light']['collection'] = 0.1
    events.params.inputs['material']['sigma_p'] = 0.05
    events.params.inputs['coarse_grids']['noise'] = 20

elif STUDY == "Pessimistic":
    events.params.inputs['spatial_resolution']['sigma_xy'] = 4e-5
    events.params.inputs['spatial_resolution']['spatial_resolution_multiplier'] = 1.5
    events.params.inputs['light']['collection'] = 0.05
    events.params.inputs['material']['sigma_p'] = 0.06
    events.params.inputs['coarse_grids']['noise'] = 40

events.params.calculate()

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
    "r_max": R_max,
    "N_hits_to_reconstruct": N_hits_to_reconstruct}

with open(os.path.join(f"{sim_file_name.split(".inc")[0]}_summary.pickle"), 'wb') as f:
    pickle.dump(summary_dict, f)

events.truth = truth[mask]    
events.truth_hits = hits[mask]

# DETECTOR RESPONSE
events.apply_detector_response()


# RECONSTRUCTION
in_vector = np.array([-np.sqrt(1-in_angle**2), 0, -in_angle])

# a list starting with 3 and ending with N_hits_to_reconstruct
HIT_LIST = [i for i in range(3, N_hits_to_reconstruct)]
events.reconstruct_events(IN_VECTOR=in_vector,
                          save_name=sim_file_name,
                          LEN_OF_CKD_HITS = HIT_LIST)



# CLASSIFICATION
# events.train_classifier_on_self()
# events.classify_reconstructed_events(save_name=sim_file_name)


