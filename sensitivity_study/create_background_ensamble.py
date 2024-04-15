import numpy as np
import math
import subprocess
import os, glob
import time


how_many_runs = 20
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
#SBATCH --time=1:00:00

cd {new_working_dir}
python full_chain.py

mv */*.pkl {cur_dir}
cd {cur_dir}
"""
    file.write(pre)

for i in range(how_many_runs):    
    name_of_job = f"runCosima_{i}.sh"

    with open(name_of_job, mode='w') as f:
        write_run_command(f,i)
    
    time.sleep(2)
    print(f"Running {name_of_job}")
    subprocess.run(["sbatch",name_of_job])
    os.remove(name_of_job)
