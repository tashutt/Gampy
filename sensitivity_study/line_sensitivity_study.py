import numpy as np
import math
import subprocess
import os, glob
import time

sister_dir = "/sdf/scratch/kipac/MeV/gammatpc"
cur_dir    = os.getcwd()
name_of_current_dir = os.path.basename(cur_dir)

new_working_dir = os.path.join(sister_dir, name_of_current_dir)

# Constants
GEOFILE = 'GammaTPC_GeoT01v04_optimistic.geo.setup'
ONE_BEAM = 'FarFieldPointSource'
LOG_E = [2.2, 2.5,2.7,3,3.2,3.5,3.7,4]#,4.2] #,3.7,4,4.2,4.5,4.7]
ANGLES = [0, 25.8, 36.9]  #, 45.6, 53.1, 60]
num_of_triggers = 100_000

# Utility Functions
def ang2cos(allAng):
    return [round(np.cos(math.radians(i)), 1) for i in allAng]

def logE2ene(allEne):
    return [int(10**ee) for ee in allEne]

# Main Script Logic
def generate_cosima_files_and_run_analysis():
    energies = logE2ene(LOG_E)
    cos_ang = ang2cos(ANGLES)

    # prepare the environment
    subprocess.run(["mkdir", "-p", new_working_dir])
    subprocess.run(["cp", "-R", "./", new_working_dir])

    for myene in energies:
        for cosTh, ang in zip(cos_ang, ANGLES):
            source_file = f'{ONE_BEAM}_{myene / 1000.:.3f}MeV_Cos{cosTh:.1f}.source'
            content = generate_source_file_content(GEOFILE, ONE_BEAM, myene, cosTh, ang)
            
            with open(source_file, 'w') as file:
                file.write(content)
            
            name_of_job = f"runCosima_{source_file}.sh"

            with open(name_of_job, mode='w') as f:
                write_run_command(f, ONE_BEAM, myene, cosTh)
            
            time.sleep(1)
            print(f"Running {name_of_job}")
            subprocess.run(["sbatch",name_of_job])
            os.remove(name_of_job)


def generate_source_file_content(geofile, oneBeam, myene, cosTh, ang):
    return f"""# A run for Cosima
# This was created with the python wrapper --> sensitivity_study.py <--

Version          1
Geometry         {geofile} 
CheckForOverlaps 1000 0.01
PhysicsListEM    LivermorePol

StoreCalibrate                 true
StoreSimulationInfo            true
StoreOnlyEventsWithEnergyLoss  true  
DiscretizeHits                 true
PreTriggerMode                 everyeventwithhits

Run FFPS
FFPS.FileName              {oneBeam}_{myene / 1000.:.3f}MeV_Cos{cosTh:.1f}
FFPS.NTriggers             {num_of_triggers}

FFPS.Source One
One.ParticleType        1
One.Beam                {oneBeam}  {ang:.1f} 0
One.Spectrum            Mono  {myene}
One.Flux                1000.0
"""



def write_run_command(file, oneBeam, myene, cosTh):
    runCode = f'{oneBeam}_{myene / 1000.:.3f}MeV_Cos{cosTh:.1f}'
    path_to_run_code = os.path.join(cur_dir, runCode)
    pre = f"""#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=1:00:00

cp {runCode}.source {new_working_dir}
cd {new_working_dir}
cosima -s 120 -v 0 {path_to_run_code}.source
python gamma_TPC_response.py {runCode}.inc1.id1.sim

mv recon_{runCode}.inc1.id1.pkl {cur_dir}
mv {runCode}_summary.pickle {cur_dir}
cd {cur_dir}

rm *.out
rm {runCode}.source
rm runCosima_{runCode}.source.sh
"""
    file.write(pre)


generate_cosima_files_and_run_analysis()







