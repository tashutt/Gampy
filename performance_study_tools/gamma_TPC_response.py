#%%
import os
import sys
import pickle
import awkward as ak
import numpy as np
import subprocess
import argparse
import yaml

# get sim_file_name from argparse
parser = argparse.ArgumentParser(description='Input the name of the simulation file .sim')
parser.add_argument('sim_file_name', type=str, help='The name of the simulation file .sim')
parser.add_argument('--path_to_readout_inputs', type=str, default='Gampy/default_inputs/default_readout_inputs.yaml', help='The path to the readout inputs yaml file')
parser.add_argument('--path_to_computation_inputs', type=str, default='Gampy/default_inputs/default_computation_inputs.yaml', help='The path to the computation inputs yaml file')
args = parser.parse_args()

tools_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(tools_dir)
sys.path.append('tools')
sys.path.append('Gampy')

# if sim_file_name is not provided, use the default one
if args.sim_file_name:
    sim_file_name = args.sim_file_name
else:
    # this is for manual testing
    sim_file_name = "FarFieldPointSource_1.000MeV_Cos1.0.inc1.id1.sim"

# extract the energy and the angle from the file name
if "MeV" in sim_file_name:
    in_energy = float(sim_file_name.split('_')[1].strip('MeV')) * 1000
    in_angle  = float(sim_file_name.split('_')[2].strip('Cos').replace('.inc1.id1.sim', ''))
elif "SimulationStep1":
    parts = sim_file_name.split('_')
    in_energy = float(parts[1].split('to')[0])
    in_angle = 1.0
else:
    in_energy = 1000
    in_angle = 1.0

print('Energy, Angle:', in_energy, in_angle)
sim_file_name = sim_file_name.strip('.sim')


from tools import events_tools

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


events = events_tools.Events(str(sim_file_path.strip(".sim")[1:])+".sim",
                            os.path.join(paths['root'], geo_file_name),
                            num_events,
                            readout_inputs_file_name=args.path_to_readout_inputs)

truth = events.truth
hits  = events.truth_hits

total_time = truth.time[-1]


# load computation inputs
with open(args.path_to_computation_inputs, 'r') as file:
    computation_inputs = yaml.safe_load(file)
    comp_recon = computation_inputs['reconstruction_parameters']

# 2 m2 detector
R_max    = comp_recon['R_analysis']
len_mask = ak.num(hits.r[:, 0]) > 0
R        = np.linalg.norm(hits[len_mask].r[:, :2, 0],axis=1)
mask     = np.zeros(len(truth), dtype=bool)
mask[len_mask] = R < R_max

print("Masking events", len(truth), "to", sum(mask),
      f"for R < {R_max:.3f} m. That includes {sum(mask)/len(truth)*100:.2f}% of the events.")

summary_dict = {
    "energy": in_energy,
    "angle": in_angle,
    "number_of_events": len(truth),
    "number_of_events_after_cuts": sum(mask),
    "r_max": R_max,
    "time": total_time,}

with open(os.path.join(f'{sim_file_name.split(".inc")[0]}_summary.pickle'), 'wb') as f:
    pickle.dump(summary_dict, f)

events.truth      = truth[mask]
events.truth_hits = hits[mask]
events.apply_detector_response()

in_vector = np.array([-np.sqrt(1-in_angle**2), 0, -in_angle])
events.reconstruct_events(IN_VECTOR=in_vector,
                          save_name=sim_file_name,
                          LEN_OF_CKD_HITS = comp_recon["hits_reconstructed"],
                          ckd_depth = 3)



# events.train_classifier_on_self()

# events.classify_reconstructed_events(save_name=sim_file_name)


# need to report data
# time in the simulation
# number of passed cuts sum(mask)
# number of events in the simulation len(truth)
# rate is len(truth)/time/original_area

