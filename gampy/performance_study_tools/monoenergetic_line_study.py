import numpy as np
import math
import subprocess
import os
import glob

def ang2cos(allAng):
    return [round(np.cos(math.radians(i)), 1) for i in allAng]

def logE2ene(allEne):
    return [int(10**ee) for ee in allEne]

def generate_cosima_files_and_run_analysis(geofile, LOG_E, ANGLES, num_of_triggers, 
                                           sister_folder=None, readout_params=None, hours=1):
    energies = logE2ene(LOG_E)
    cos_ang = ang2cos(ANGLES)

    #if sister_folder is a string
    cur_dir    = os.getcwd()
    working_dir = cur_dir
    if sister_folder:
        name_of_current_dir = os.path.basename(cur_dir)
        new_working_dir     = os.path.join(sister_folder, name_of_current_dir)
        working_dir = new_working_dir

        # create a new directory in the sister_folder, delete the old one
        subprocess.run(["rm", "-rf", new_working_dir])
        subprocess.run(["mkdir", "-p", new_working_dir])
        subprocess.run(["cp", "-R", "Gampy", new_working_dir])
        subprocess.run(["cp", geofile,  new_working_dir])
    
    if readout_params:
        subprocess.run(["cp", readout_params, working_dir])

    
    results_dir = "monoenergetic_line_results"
    os.makedirs(results_dir, exist_ok=True)
    results_dir = os.path.abspath(results_dir)

    for myene in energies:
        for cosTh, ang in zip(cos_ang, ANGLES):
            source_file = f'FarFieldPointSource_{myene / 1000.:.3f}MeV_Cos{cosTh:.1f}.source'
            content = generate_source_file_content(geofile, myene, cosTh, ang, num_of_triggers)
            
            with open(source_file, 'w') as file:
                file.write(content)
            
            name_of_job = f"runCosima_{source_file}.sh"
            with open(name_of_job, mode='w') as f:
                write_run_command(f, myene, cosTh, results_dir, working_dir,readout_params, hours)
            
            print(f"Running {name_of_job}")
            subprocess.run(["sbatch", name_of_job])
            os.remove(name_of_job)

def generate_source_file_content(geofile, myene, cosTh, ang, num_of_triggers):
    return f"""# A run for Cosima
# This was created with the python wrapper

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
FFPS.FileName              FarFieldPointSource_{myene / 1000.:.3f}MeV_Cos{cosTh:.1f}
FFPS.NTriggers             {num_of_triggers}

FFPS.Source One
One.ParticleType        1
One.Beam                FarFieldPointSource  {ang:.1f} 0
One.Spectrum            Mono  {myene}
One.Flux                1000.0
"""

def write_run_command(file, myene, cosTh, results_dir, working_dir,readout_params,hours):
    runCode = f'FarFieldPointSource_{myene / 1000.:.3f}MeV_Cos{cosTh:.1f}'
    path_to_run_code = os.path.join(working_dir, runCode)
    pre = f"""#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time={hours}:00:00
#SBATCH --output=/dev/null  # Disable .out files

cp {runCode}.source {working_dir}
rm {runCode}.source

cd {working_dir}
cosima -s 120 -v 0 {runCode}.source
rm {runCode}.source
python Gampy/performance_study_tools/gamma_TPC_response.py {runCode}.inc1.id1.sim --path_to_readout_inputs {readout_params}

mv recon_{runCode}.inc1.id1.pkl {results_dir}
mv {runCode}_summary.pickle {results_dir}
cd {results_dir}
"""
    file.write(pre)
