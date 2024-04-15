import numpy as np
import math
import subprocess
import os, glob
import time

# Constants
GEOFILE = 'GammaTPC_GeoT01v04_optimistic.geo.setup'
ONE_BEAM = 'FarFieldPointSource'
LOG_E = [2.2, 2.5,2.7,3,3.2,3.5,3.7,4]#,4.2] #,3.7,4,4.2,4.5,4.7]
ANGLES = [0, 25.8, 36.9]  #, 45.6, 53.1, 60]
num_of_triggers = 10000

# Utility Functions
def ang2cos(allAng):
    return [round(np.cos(math.radians(i)), 1) for i in allAng]

def logE2ene(allEne):
    return [int(10**ee) for ee in allEne]

# Main Script Logic
def generate_cosima_files_and_run_analysis():
    energies = logE2ene(LOG_E)
    cos_ang = ang2cos(ANGLES)

    for myene in energies:
        for cosTh, ang in zip(cos_ang, ANGLES):
            source_file = f'{ONE_BEAM}_{myene / 1000.:.3f}MeV_Cos{cosTh:.1f}.source'
            content = generate_source_file_content(GEOFILE, ONE_BEAM, myene, cosTh, ang)
            write_to_file(source_file, content)
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

def write_to_file(filename, content):
    with open(filename, 'w') as file:
        file.write(content)


def write_run_command(file, oneBeam, myene, cosTh):
    runCode = f'{oneBeam}_{myene / 1000.:.3f}MeV_Cos{cosTh:.1f}'
    pre = f"""#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=1:00:00

cosima -s 120 -v 0 {runCode}.source
python gamma_TPC_complete.py {runCode}.inc1.id1.sim
echo "Done with {runCode}"
rm *.out
"""
    file.write(pre)


generate_cosima_files_and_run_analysis()


