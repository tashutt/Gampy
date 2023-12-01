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


import argparse
parser = argparse.ArgumentParser(description='choose sim options')
parser.add_argument('--events', type=int, default=10000, help='specify number of events. Default 5000')
parser.add_argument('-activation', action='store_true', default=False, help='Activate activation. Specify duration instead of event count.')
parser.add_argument('-analysis', action='store_true', default=False, help='Do the analysis')
parser.add_argument('-recompute_activation', action='store_true', default=False, help='Recompute the activation. Important for adding more materials')
parser.add_argument("--eng", type=str, default='2-8', help="Specify an energy range in keV, 'start-end', default 2-8")
parser.add_argument('--alt', type=int, default=550, help="Altitude in km. Default 550.")
parser.add_argument('--inc', type=int, default=0, help="Inclination in degrees. Default 0.")

parser.add_argument('--calorimeter_thickness', type=float, default=0.0, help='calorimeter thickness (in m) 0.05')
parser.add_argument('--calorimeter_material', type=str, default='CsI', help='calorimeter material, either Ar or CsI. Default CsI')
parser.add_argument('--shield_thickness', type=float, default=0.02, help='shield thickness (in m) 0.10')
parser.add_argument('--shield_material', type=str, default='Lead', help='shield material, choose Lead or Tungsten. Default Lead.')

parser.add_argument('--vessel_r_outer', type=float, default=0.67, help='vessel r_outer (in cm?) 0.8')
parser.add_argument('--vessel_wall_thickness', type=float, default=0.004, help='vessel wall thickness (in m) default 0.004')
parser.add_argument('--cell_h', type=float, default=0.175, help='cell height (m) default 0.175')
parser.add_argument('--cell_wall_thickness', type=float, default=0.3e-3, help='cell wall thickness (m), default 0.0003')
parser.add_argument('--acd_thickness', type=float, default=0.005, help='acd thickness (m) default 0.007')

del_sim_files = False
args = parser.parse_args()
args_dict = vars(args)
# Save the arguments as a NumPy binary file
for arg, value in args_dict.items():
    print(f"{arg}: {value}")

########################################################################


from cosmic_flux import cosmic_flux_gen

#   Paths
base_directory = 'cosima_run' 
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

if os.path.exists(paths['root']):    
    subprocess.run(['rm', '-rf', paths['root']])
          
os.makedirs(paths['root'], exist_ok=True)
os.makedirs(paths['data'], exist_ok=True)

#   Load default geo params
geo_params = params_tools.GeoParams(detector_geometry='geomega',
                                    cell_geometry='hexagonal')

# modify any parameter
geo_params.inputs['vessel']['r_outer'] = args.vessel_r_outer #1.85
geo_params.inputs['vessel']['wall_thickness'] = args.vessel_wall_thickness
#geo_params.inputs['vessel']['lar_min_radial_gap'] = 0.05 #what is this param?
geo_params.inputs['cells']['height'] = args.cell_h
#geo_params.inputs['cells']['flat_to_flat'] = 0.175 # what is this parameter?
geo_params.inputs['cells']['wall_thickness'] = args.cell_wall_thickness
geo_params.inputs['acd'] = {'thickness': args.acd_thickness}
geo_params.inputs['calorimeter']['thickness'] = args.calorimeter_thickness
geo_params.inputs['shield']['thickness'] = args.shield_thickness

geo_file_name = file_tools.write_geo_files(
    paths['root'],
    geo_params,
    values_id=3
    )

source_file_path =  cosmic_flux_gen.generate_cosmic_simulation(
                                            geo_file_name, 
                                            Inclination=inc, 
                                            Altitude=alt, 
                                            Elow=energy_low, 
                                            Ehigh=energy_high, 
                                            num_triggers=num_events, 
                                            output_dir=paths['root'])
                                
# get the name from the path 
source_file_name = os.path.basename(source_file_path).replace(".sim","")

# run cosima with the generated source file 
print("\n\nThe source file name is:",source_file_name)

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
        step_1 = cosmic_flux_gen.generate_cosmic_simulation(
                                        geo_full_file_name=geo_file_name, 
                                        activation=True,  # Set activation to True
                                        Inclination=inc, 
                                        Altitude=alt, 
                                        Elow=energy_low, 
                                        Ehigh=energy_high, 
                                        duration=sim_time, 
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
                                    output_dir=paths['root'])
    
    if args.recompute_activation:
        #----step 1----#
        step1_file_name = os.path.basename(step_1).replace(".sim","")
        print("\n\nThe step 1 file name is:",step1_file_name)
        os.chdir(paths['root'])
        subprocess.run(['pwd']) 
        cosima_output1 = subprocess.run(['cosima', step1_file_name], capture_output=True)
        cosima_output1 = cosima_output1.stdout.decode('utf-8')   

        #----step 2----#
        step2_file_name = os.path.basename(step_2).replace(".sim","")
        print("\n\nThe step 2 file name is:",step2_file_name)
        cosima_output2 = subprocess.run(['cosima', step2_file_name], capture_output=True)
        cosima_output2 = cosima_output2.stdout.decode('utf-8')   

    #----step 3----#
    step3_file_name = os.path.basename(step_3).replace(".sim","")
    print("\n\nThe step 3 file name is:",step3_file_name)
    cosima_output3 = subprocess.run(['cosima', step3_file_name], capture_output=True)
    cosima_output3 = cosima_output3.stdout.decode('utf-8')   
    useful_output = cosima_output3.split("Summary for run ")[-1]  
    print(useful_output,'\n\n\n\n-----------')
    inc_id_tag = ".inc1.id1"
    activation_file_name = step3_file_name.split('.')[0] + inc_id_tag


