import numpy as np
import math
import subprocess
import os
import time

# Constants
how_many_runs = 10
min_energy = 200
max_energy = 8000
events_in_run = 100_000

STUDY = "Nominal"
R_analysis = np.sqrt(2.0 / np.pi)  # 2 m2 active area
activation_file_name = "ActivationForCsI_10cm_1.5_v4.dat"

# Directories
sister_dir = "/sdf/scratch/kipac/MeV/gammatpc"
cur_dir = os.getcwd()
name_of_current_dir = os.path.basename(cur_dir)
new_working_dir = os.path.join(sister_dir, name_of_current_dir)

# Create new working directory
subprocess.run(["rm", "-rf", new_working_dir])
subprocess.run(["mkdir", "-p", new_working_dir])
subprocess.run(["cp", "-R", "./", new_working_dir])

# Function to write the run command to a file
def write_run_command(file, i, energy, events):
    pre = f"""#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=0:59:00

cd {new_working_dir}
python full_chain.py --events {events} --study {STUDY} --activation_file_name {activation_file_name} --R_analysis {R_analysis} --min_energy {energy}

mv */*.pkl {cur_dir}
cd {cur_dir}
"""
    file.write(pre)

# Generate job scripts and submit them
for i in range(how_many_runs):
    energy = int(min_energy + i * (max_energy - min_energy) / (how_many_runs - 1))
    events = int(events_in_run / (i+1)**0.5)  
    print(f"Min Energy is {energy}, Events: {events}")
    name_of_job = f"runCosima_{i}.sh"

    with open(name_of_job, mode='w') as f:
        write_run_command(f, i, energy, events)
    
    time.sleep(2.5)
    print(f"Running {name_of_job}")
    subprocess.run(["sbatch", name_of_job])
    os.remove(name_of_job)
