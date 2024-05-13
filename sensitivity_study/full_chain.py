#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

7/8/23
@author: btrbalic
expanded and added args for activation, etc. by sjett

TO-DO:
fix variable names / function names / descriptions to be more clear
combine everything (?)
"""
import sys, os
sys.path.append('tools')

import params_tools
import file_tools
import events_tools 
import subprocess
import numpy as np
import awkward as ak
import time
import pickle


import argparse
parser = argparse.ArgumentParser(description='choose sim options')
parser.add_argument('--events', type=int, default=100_000, help='specify number of events. ')
parser.add_argument('-activation', action='store_true', default=True, help='Activate activation. Specify duration instead of event count.')
parser.add_argument('-analysis', action='store_true', default=False, help='Do the analysis')
parser.add_argument('-only_photons', action='store_true', default=True, help='Only have photons as sources')
parser.add_argument('-recompute_activation', action='store_true', default=False, help='Recompute the activation. Important for adding more materials')
parser.add_argument("--eng", type=str, default='2-8', help="Specify an energy range in keV, 'start-end', default 2-8")
parser.add_argument('--alt', type=int, default=550, help="Altitude in km. Default 550.")
parser.add_argument('--inc', type=int, default=0, help="Inclination in degrees. Default 0.")

parser.add_argument('--calorimeter_thickness', type=float, default=0.1, help='calorimeter thickness (in m) 0.05')
parser.add_argument('--calorimeter_material', type=str, default='CsI', help='calorimeter material, either Ar or CsI. Default CsI')
parser.add_argument('--shield_thickness', type=float, default=0.0, help='shield thickness (in m) 0.10')
parser.add_argument('--shield_material', type=str, default='Lead', help='shield material, choose Lead or Tungsten. Default Lead.')

parser.add_argument('--vessel_r_outer', type=float, default=1.5, help='vessel r_outer  0.8')
parser.add_argument('--vessel_wall_thickness', type=float, default=0.004, help='vessel wall thickness (in m) default 0.004')
parser.add_argument('--cell_h', type=float, default=0.175, help='cell height (m) default 0.175')
parser.add_argument('--cell_wall_thickness', type=float, default=0.3e-3, help='cell wall thickness (m), default 0.0003')
parser.add_argument('--acd_thickness', type=float, default=0.005, help='acd thickness (m) default 0.007')

activation_file_name = "ActivationForCsI_10cm_1.5_v4.dat"
del_sim_files = True

args = parser.parse_args()
args_dict = vars(args)
# Save the arguments as a NumPy binary file
for arg, value in args_dict.items():
    print(f"{arg}: {value}")

########################################################################


from cosmic_flux import cosmic_flux_gen

STUDY = "Optimistic"
base_directory = f'backgrounds_{STUDY}' 
i = 1
while True:
    directory_name = f'{base_directory}{i}'
    full_path = os.path.join(os.getcwd(), directory_name)
    if not os.path.exists(full_path):
        break  
    i += 1

paths = {}
paths['root'] = directory_name
paths['data'] = os.path.join(paths['root'], 'data')

#------ Params -------#
num_events = args.events
inc = args.inc
alt = args.alt
energy_low = int(args.eng.split('-')[0])
energy_high = int(args.eng.split('-')[-1])
only_photons = bool(args.only_photons)

if os.path.exists(paths['root']):    
    subprocess.run(['rm', '-rf', paths['root']])
          
os.makedirs(paths['root'], exist_ok=True)
os.makedirs(paths['data'], exist_ok=True)

#   Load default geo params
geo_params = params_tools.GeoParams(detector_geometry='geomega',
                                    cell_geometry='hexagonal')



if STUDY == "Optimistic":
    geo_params.inputs['anode_planes']['thickness'] = 0.75e-3
    geo_params.inputs['cathode_plane']['thickness'] = 0.75e-3
    geo_params.inputs['vessel']['wall_thickness'] = 2e-3
    geo_params.inputs['cells']['wall_thickness'] = 0.75e-4

elif STUDY == "Neutral":
    geo_params.inputs['anode_planes']['thickness'] = 1.5e-3
    geo_params.inputs['cathode_plane']['thickness'] = 1.5e-3
    geo_params.inputs['vessel']['wall_thickness'] = 3e-3
    geo_params.inputs['cells']['wall_thickness'] = 2e-4

elif STUDY == "Pessimistic":
    geo_params.inputs['anode_planes']['thickness'] = 3e-3
    geo_params.inputs['cathode_plane']['thickness'] = 3e-3
    geo_params.inputs['vessel']['wall_thickness'] = 4e-3
    geo_params.inputs['cells']['wall_thickness'] = 5e-4


# modify any parameter
geo_params.inputs['vessel']['r_outer'] = args.vessel_r_outer #1.85
#geo_params.inputs['vessel']['wall_thickness'] = args.vessel_wall_thickness
#geo_params.inputs['vessel']['lar_min_radial_gap'] = 0.05 #what is this param?
geo_params.inputs['cells']['height'] = args.cell_h
#geo_params.inputs['cells']['flat_to_flat'] = 0.175 # what is this parameter?
#geo_params.inputs['cells']['wall_thickness'] = args.cell_wall_thickness
geo_params.inputs['acd'] = {'thickness': args.acd_thickness}
geo_params.inputs['calorimeter']['thickness'] = args.calorimeter_thickness
geo_params.inputs['shield']['thickness'] = args.shield_thickness

geo_params.calculate()

geo_file_name = file_tools.write_geo_files(
    paths['root'],
    geo_params,
    values_id=4
    )


source_file_path =  cosmic_flux_gen.generate_cosmic_simulation(
                                            geo_file_name, 
                                            Inclination=inc, 
                                            Altitude=alt, 
                                            Elow=energy_low, 
                                            Ehigh=energy_high, 
                                            num_triggers=num_events, 
                                            output_dir=paths['root'],
                                            only_photons=only_photons)
                                
# get the name from the path 
source_file_name = os.path.basename(source_file_path).replace(".sim","")

# run cosima with the generated source file 
print("\n\nThe source file name is:",source_file_name)
time.sleep(1)
# go into the root directory
os.chdir(paths['root'])
subprocess.run(['pwd']) 

cosima_output = subprocess.run(['cosima', source_file_name], capture_output=True)
cosima_output = cosima_output.stdout.decode('utf-8')   
useful_output = cosima_output.split("Summary for run SpaceSim")[-1]

#-------
sim_particle_numbers = {}
for line in useful_output.split('\n'):
    if ":" in line:
        k,v = line.split(": ")
        if "." in v:
            sim_particle_numbers[k.strip()]=float(v.strip("sec").strip())
        else:
            sim_particle_numbers[k.strip()]=int(v.strip())
                
                 
print(useful_output,'\n\n\n\n-----------')
inc_id_tag = ".inc1.id1"
sim_file_name = source_file_name.split('.')[0] + inc_id_tag
           
# use numpy to save the events.truth dictionary to a file
np.save('data/sim_settings.npy', args_dict)
np.save('data/sim_particle_numbers.npy', sim_particle_numbers)
######################################
####### The first part is done #######
####### Now activation if enabled ####
######################################

print(sim_particle_numbers)
if args.activation:
    # go back to the working directory
    os.chdir(os.path.abspath(os.path.join(os.curdir, "..")))
    # Use cosmic_flux_gen.py for generating cosmic simulation
    sim_time = sim_particle_numbers['Observation time']
    # Call the generate_cosmic_simulation function with the activation parameter set to True
    # skip the fisrt and second step if recompute_activation is False

    if args.recompute_activation:
        print("WARNING: recompute_activation is True. This will take a long time.")
        step_1 = cosmic_flux_gen.generate_cosmic_simulation(
                                        geo_full_file_name=geo_file_name, 
                                        activation=True,  # Set activation to True
                                        Inclination=inc, 
                                        Altitude=alt, 
                                        Elow=energy_low, 
                                        Ehigh=energy_high, 
                                        duration=0.1, 
                                        output_dir=paths['root'])
        step_2 = cosmic_flux_gen.calculate_activation(
                                        geo_file_name, 
                                        Inclination=inc, 
                                        Altitude=alt, 
                                        Elow=energy_low, 
                                        Ehigh=energy_high, 
                                        output_dir=paths['root'])
    step_3 = cosmic_flux_gen.activation_events(
                                    geo_file_name, 
                                    Inclination=inc, 
                                    Altitude=alt, 
                                    Elow=energy_low, 
                                    Ehigh=energy_high, 
                                    duration=sim_time, 
                                    output_dir=paths['root'],
                                    dat_name = activation_file_name)
    
    os.chdir(paths['root'])
    subprocess.run(['pwd']) 
    if args.recompute_activation:
        #----step 1----#
        step1_file_name = os.path.basename(step_1).replace(".sim","")
        print("\n\nThe step 1 file name is:",step1_file_name)
        cosima_output1 = subprocess.run(['cosima', step1_file_name], capture_output=True)
        cosima_output1 = cosima_output1.stdout.decode('utf-8')   

        #----step 2----#
        step2_file_name = os.path.basename(step_2).replace(".sim","")
        print("\n\nThe step 2 file name is:",step2_file_name)
        cosima_output2 = subprocess.run(['cosima', step2_file_name], capture_output=True)
        cosima_output2 = cosima_output2.stdout.decode('utf-8')   
    else:
        if activation_file_name is None:
            print(" *** WARNING: activation_file_name is None. Using default. ***")
            activation_file_name = f'ActivationFor{alt}km_{energy_low}to{energy_high}keV.dat'
        #chack if the file exists
        elif not os.path.exists(f'{activation_file_name}'):
            if not os.path.exists(f'../cosmic_flux/Data/{activation_file_name}'):
                print(f'ERROR >_<: {activation_file_name} does not exist in ../cosmic_flux/Data/')
                sys.exit()
            subprocess.run(['cp', f'../cosmic_flux/Data/{activation_file_name}', '.'])

    #----step 3----#
    step3_file_name = os.path.basename(step_3).replace(".sim","")
    print("\n\nThe step 3 file name is:",step3_file_name)
    cosima_output3 = subprocess.run(['cosima', step3_file_name], capture_output=True)
    cosima_output3 = cosima_output3.stdout.decode('utf-8')   
    useful_output = cosima_output3.split("Summary for run ")[-1]  
    print(useful_output,'\n\n\n\n-----------')
    inc_id_tag = ".inc1.id1"
    activation_file_name = step3_file_name.split('.')[0] + inc_id_tag


file_a_path = sim_file_name + '.sim'  
file_b_path = activation_file_name + '.sim'
output_path = f'background_{sim_time}.sim' 

print(f"Inserting content from '{file_b_path}' into '{file_a_path}' and saving as '{output_path}'")

def find_se_en_content(file_path):
    capture = False
    content = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith("CC"):  
                continue
            if "SE" in line:
                capture = True
            if capture:
                content.append(line)
            if "EN" in line and capture:
                break
    return content

content_to_insert = find_se_en_content(file_b_path)

with open(file_a_path, 'r') as file_a, open(output_path, 'w') as output_file:
    for line in file_a:
        if "EN" in line:
            output_file.writelines(content_to_insert)
            break  
        output_file.write(line)
    for line in file_a:
        output_file.write(line)

if os.name == 'posix': 
    subprocess.run(['rm', file_a_path, file_b_path])
elif os.name == 'nt': 
    subprocess.run(['del', file_a_path, file_b_path], shell=True)

sim_file_name = output_path
print(f"Analyzing {sim_file_name}")


################## ANALYSIS ####################
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

sim_file_path = sim_file_name
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

events = events_tools.Events(sim_file_path.strip('.sim'),
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

elif STUDY == "Neutral":
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

events.truth = truth[mask]    
events.truth_hits = hits[mask]

events.apply_detector_response()

sim_file_name = os.path.basename(sim_file_name)
in_vector = np.array([-np.sqrt(1-in_angle**2), 0, -in_angle])
events.reconstruct_events(IN_VECTOR=in_vector,
                          save_name=sim_file_name,
                          LEN_OF_CKD_HITS = [3,4,5,6,7,8,9,10,11])


