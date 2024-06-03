import numpy as np
import math
import subprocess
import os, glob
import time


how_many_runs = 50
events_in_run = 155_000
STUDY         = "Nominal"
R_analysis    = np.sqrt(2.0/np.pi) # 2 m2 active area
activation_file_name = "ActivationForCsI_10cm_1.5_v4.dat"

sister_dir    = "/sdf/scratch/kipac/MeV/gammatpc"
cur_dir       = os.getcwd()
name_of_current_dir = os.path.basename(cur_dir)
new_working_dir     = os.path.join(sister_dir, name_of_current_dir)

subprocess.run(["rm", "-rf", new_working_dir])
subprocess.run(["mkdir", "-p", new_working_dir])
subprocess.run(["cp", "-R", "./", new_working_dir])


def write_run_command(file,i):
    pre = f"""#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=0:20:00

cd {new_working_dir}
python full_chain.py --events {events_in_run} --study {STUDY} --activation_file_name {activation_file_name} --R_analysis {R_analysis}

mv */*.pkl {cur_dir}
cd {cur_dir}
"""
    file.write(pre)

for i in range(how_many_runs):    
    name_of_job = f"runCosima_{i}.sh"

    with open(name_of_job, mode='w') as f:
        write_run_command(f,i)
    
    time.sleep(20.5)
    print(f"Running {name_of_job}")
    subprocess.run(["sbatch",name_of_job])
    os.remove(name_of_job)
