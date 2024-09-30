import numpy as np
import subprocess
import os, sys
import time
import argparse
import yaml

sys.path.append("Gampy")
sys.path.append("cosmic_flux")
from cosmic_flux import cosmic_flux_gen

parser = argparse.ArgumentParser(description='Run a background study')
parser.add_argument("-geometry_file", type=str, help="Path to the geometry file")
parser.add_argument("-comp_params", type=str, help="Path to the computation parameters file")
parser.add_argument("-readout_params", type=str, help="Path to the readout parameters file")
parser.add_argument('-activation_file', type=str, help="Path to the activation file")
parser.add_argument('--alt', type=int, default=550, help="Altitude in km. Default 550.")
parser.add_argument('--inc', type=int, default=0, help="Inclination in degrees. Default 0.")

args = parser.parse_args()

geometry_path       = args.geometry_file
comp_params_path    = args.comp_params
readout_params_path = args.readout_params
activation_file     = args.activation_file

# load parameters
with open(comp_params_path, 'r') as file:
    comp_params = yaml.safe_load(file)

# load readout parameters
with open(readout_params_path, 'r') as file:
    readout_params = yaml.safe_load(file)

paths = {}
paths['root']       = os.getcwd()
paths["sister_dir"] = comp_params['server_params']['sister_dir']
paths["gtpc_response"] = os.path.join(paths['root'], "Gampy", "performance_study_tools", "gamma_TPC_response.py")
# name of current working directory
name_of_current_dir = os.path.basename(os.getcwd())

if paths["sister_dir"]:
    new_working_dir = os.path.join(paths["sister_dir"], name_of_current_dir, "working_background")
    subprocess.run(["rm", "-rf", new_working_dir])
    subprocess.run(["mkdir", "-p", new_working_dir])
    subprocess.run(["cp", "-R", "Gampy", new_working_dir])
    subprocess.run(["cp", geometry_path, new_working_dir])
    subprocess.run(["cp", activation_file, new_working_dir])



def write_run_command(file, min_e, run_dir, source_file_path):
    source_name      = os.path.basename(source_file_path).strip(".source")

    pre = f"""#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=1:59:00
#SBATCH --output=/dev/null  # Disable .out files

cd {run_dir}
cosima -s 120 -v 0 {source_file_path}
sed -i '/^CC/d' "{source_name}.inc1.id1.sim"
python {paths['gtpc_response']} {source_name}.inc1.id1.sim --path_to_readout_inputs {readout_params_path} --path_to_computation_inputs {comp_params_path}
"""
    file.write(pre)
    return True


energy_min_power = comp_params['background_study']['min_energy_power']
energy_max_power = comp_params['background_study']['max_energy_power']
how_many_runs    = comp_params['background_study']['how_many_runs']
events_in_run    = comp_params['background_study']['events_in_run']


for i in range(how_many_runs):
    low_energy = int(energy_min_power + i * (energy_max_power - energy_min_power) / (how_many_runs - 1))
    num_events = int(events_in_run / (i+1)**0.2)  
   
    #num_events = 200
    run_dir = os.path.join(new_working_dir, f"run_{i}")
    subprocess.run(["rm", "-rf", run_dir])
    os.makedirs(run_dir, exist_ok=True)
    subprocess.run(["cp", geometry_path, run_dir])

    
    source_file_path =  cosmic_flux_gen.generate_cosmic_simulation(
                                                geometry_path.replace(".setup", ""),
                                                Inclination=args.inc,
                                                Altitude=args.alt,
                                                Elow=low_energy,
                                                Ehigh=energy_max_power,
                                                num_triggers=num_events,
                                                output_dir=run_dir,
                                                only_photons=True)
    
    name_of_job = f"runJob_{i}.sh"
    with open(name_of_job, mode='w') as f:
        write_run_command(f, low_energy, run_dir, os.path.basename(source_file_path))
    
    print(f"****** Running {name_of_job} ********")
    subprocess.run(["sbatch", name_of_job])
    os.remove(name_of_job)
    time.sleep(0.05)  


                            
for i in range(how_many_runs):
    low_energy = 2
    run_dir = os.path.join(new_working_dir, f"activation_run_{i}")
    subprocess.run(["rm", "-rf", run_dir])
    os.makedirs(run_dir, exist_ok=True)
    subprocess.run(["cp", geometry_path, run_dir])
    subprocess.run(["cp", activation_file, run_dir])

    
    step_3 = cosmic_flux_gen.activation_events(
                            geometry_path.replace(".setup", ""),
                            activation_file,
                            duration=10,
                            output_dir=run_dir)

    name_of_job = f"activation_runJob_{i}.sh"

    with open(name_of_job, mode='w') as f:
        write_run_command(f, low_energy, run_dir, os.path.basename(step_3))
    
    print(f"****** Running {name_of_job} ********")
    subprocess.run(["sbatch", name_of_job])
    os.remove(name_of_job)
    time.sleep(0.05)  

