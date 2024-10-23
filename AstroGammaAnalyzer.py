# Import necessary libraries
import os
import sys
import yaml
import argparse
from Gampy.tools import sims_tools, file_tools
from Gampy.performance_study_tools import monoenergetic_line_study as mls_study

##########################################
#### OUTLINE OF THE SIMULATION SCRIPT ####
##########################################
# 1. Load the geometry parameters
# 2. Write the geometry files
# 3. Run Monoenergetic Line Sim to get:
#    - angular resolution
#    - energy resolution 
#    - effective area/efficiency
# 4. Run the sim for backgrounds to get:
#    - background rate


# Paths setup
paths = {
    'root': '',
    'gampy': os.path.abspath(os.path.dirname(__file__)),
    'comp_params': os.path.join(os.path.dirname(__file__), 'Gampy', 'default_inputs', 'default_computation_inputs.yaml')
}
path_to_activation_setup = os.path.join(paths['gampy'], 'Gampy', 'performance_study_tools', 'activation_setup.py')
sys.path.append(paths['gampy'])

sims_params = sims_tools.Params(cell_geometry='hexagonal')

with open(paths['comp_params'], 'r') as file:
    comp_params = yaml.safe_load(file)

scenarios = {
    'optimistic': {
        'spatial_res_mult': 0.75,
        'light_collection': 0.3,
        'sigma_p': 0.04,
        'grid_noise': 10,
        'track_error_mult': 0.8,
        'anode_thickness': 0.7 /1e3 ,
        'cathode_thickness': 0.7 /1e3,
        'vessel_wall_thickness': 2 /1e3,
        'cell_wall_thickness': 0.5 /1e3
    },
    'nominal': {
        'spatial_res_mult': 1.0,
        'light_collection': 0.1,
        'sigma_p': 0.05,
        'grid_noise': 20,
        'track_error_mult': 1.0,
        'anode_thickness': 1.2 /1e3,
        'cathode_thickness': 1.2 /1e3,
        'vessel_wall_thickness': 6 /1e3,
        'cell_wall_thickness': 0.7 /1e3
    },
    'pessimistic': {
        'spatial_res_mult': 1.5,
        'light_collection': 0.05,
        'sigma_p': 0.06,
        'grid_noise': 40,
        'track_error_mult': 1.2,
        'anode_thickness': 2 /1e3,
        'cathode_thickness': 2 /1e3,
        'vessel_wall_thickness': 10 /1e3,
        'cell_wall_thickness': 1.5 /1e3
    },
    'pessimistic_good_light': {
        'spatial_res_mult': 1.5,
        'light_collection': 0.15,
        'sigma_p': 0.06,
        'grid_noise': 40,
        'track_error_mult': 1.2,
        'anode_thickness': 2 /1e3,
        'cathode_thickness': 2 /1e3,
        'vessel_wall_thickness': 10 /1e3,
        'cell_wall_thickness': 1.5 /1e3
    }

}

def apply_scenario(scenario_name):
    scenario = scenarios[scenario_name]
    
    sims_params.inputs['cells']['anode_plane_thickness'] = scenario['anode_thickness']
    sims_params.inputs['cells']['cathode_plane_thickness'] = scenario['cathode_thickness']
    sims_params.inputs['vessel']['wall_thickness'] = scenario['vessel_wall_thickness']
    sims_params.inputs['cells']['wall_thickness'] = scenario['cell_wall_thickness']

    new_geometry_params = 'new_geometry_inputs.yaml'
    with open(new_geometry_params, 'w') as file:
        yaml.dump(sims_params.inputs, file)
    
    # --------------------------------------
    with open('Gampy/default_inputs/default_readout_inputs.yaml', 'r') as file:
        readout_params = yaml.safe_load(file)
    
    readout_params['spatial_resolution']['sigma_xy'] *= scenario['spatial_res_mult']
    readout_params['light']['collection'] = scenario['light_collection']
    readout_params['material']['sigma_p'] = scenario['sigma_p']
    readout_params['charge_readout']['GAMPixG']['coarse_grids']['noise'] = scenario['grid_noise']
    
    new_readout_params = 'new_readout_inputs.yaml'
    with open(new_readout_params, 'w') as file:
        yaml.dump(readout_params, file)
    
    return os.path.abspath(new_readout_params)

####################################################################################
####################################################################################
####################################################################################


parser = argparse.ArgumentParser(description='Run Gampy simulations.')
parser.add_argument('--scenario', type=str, choices=['optimistic', 'nominal', 'pessimistic', 'pessimistic_good_light'], default='nominal', help='Select simulation scenario')
args = parser.parse_args()

# Apply the selected scenario parameters
abs_path_nrp = apply_scenario(args.scenario)
abs_path_comp_params = os.path.abspath(paths['comp_params'])

# Calculate simulation parameters
sims_params.calculate()

# Write the geometry files
geo_file_name = file_tools.write_geo_files(
    paths['root'],
    sims_params,
    values_id=5
)

# Check for existing activation file or run computation
activation_file = next((file for file in os.listdir() if file.startswith('Activation') and file.endswith('.dat')), None)
if not activation_file:
    print("Activation file not found. Starting the computation. This may take a while...")
    os.system(f"python {path_to_activation_setup} {geo_file_name}.setup -time 0.1")
    sys.exit()

#######################################
#### RUN MONOENERGETIC LINE SIMS ###### 
#######################################

mlsp = comp_params['monoenergetic_line_study']
sister_dir = "/sdf/scratch/kipac/MeV/gammatpc"

mls_study.generate_cosima_files_and_run_analysis(
    geo_file_name + ".setup", 
    mlsp['LOG_E'], 
    mlsp['ANGLES'], 
    mlsp['num_of_triggers_monoenergetic'], 
    sister_dir,
    abs_path_nrp
)

#######################################
######## RUN BACKGROUND SIMS ##########
#######################################

path_to_bckg_sim = os.path.join(paths['gampy'], 'Gampy', 'performance_study_tools', 'background_study.py')
run_command = f"python {path_to_bckg_sim} -geometry_file {geo_file_name}.setup -comp_params {abs_path_comp_params} -readout_params {abs_path_nrp} -activation_file {activation_file}"
os.system(run_command)
