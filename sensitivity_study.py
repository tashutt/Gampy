import numpy as np
import math
import subprocess
import os, glob

# Constants
GEOFILE = 'GammaTPC_GeoT01v04_optimistic.geo.setup'
ONE_BEAM = 'FarFieldPointSource'
LOG_E = [2.2, 2.5,2.7,3,3.2,3.5,3.7,4]#,4.2] #,3.7,4,4.2,4.5,4.7]#,5,5.2,5.5,5.7,6,6.2,6.5,6.7]
ANGLES = [0]#, 25.8, 36.9]  #, 45.6, 53.1, 60]

# Utility Functions
def ang2cos(allAng):
    return [round(np.cos(math.radians(i)), 1) for i in allAng]

def logE2ene(allEne):
    return [int(10**ee) for ee in allEne]

# Main Script Logic
def generate_cosima_files():
    energies = logE2ene(LOG_E)
    cos_ang = ang2cos(ANGLES)

    with open("./runCosima.sh", mode='w') as f:
        for myene in energies:
            for cosTh, ang in zip(cos_ang, ANGLES):
                source_file = f'{ONE_BEAM}_{myene / 1000.:.3f}MeV_Cos{cosTh:.1f}.source'
                content = generate_source_file_content(GEOFILE, ONE_BEAM, myene, cosTh, ang)
                write_to_file(source_file, content)
                write_run_command(f, ONE_BEAM, myene, cosTh)

def generate_source_file_content(geofile, oneBeam, myene, cosTh, ang):
    return f"""# An example run for Cosima
# This was created with the python wrapper --> create_source_file.py <--

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
FFPS.NTriggers             100000

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
    file.write(f'cosima -v 0 -s 120 {runCode}.source\n')
#bsub -R "centos7" -o a_out.txt -W 6000 
# Execute Shell Script
def execute_shell_script():
    # Append shebang line
    prepend_to_file("./runCosima.sh", "#!/bin/bash\n")
    # Append closing line
    append_to_file("./runCosima.sh", "\necho \"All simulations have been submitted.\"\n")
    # Make script executable and run
    subprocess.run(['chmod', '+x', './runCosima.sh'])
    print("Running runCosima.sh")
    subprocess.run(['./runCosima.sh'])

def prepend_to_file(filename, line):
    with open(filename, 'r+') as file:
        content = file.read()
        file.seek(0, 0)
        file.write(line + content)

def append_to_file(filename, line):
    with open(filename, 'a') as file:
        file.write(line)

def check_jobs_periodically(dt=5):
    import time
    time.sleep(2)

    while True:
        result = subprocess.run(['bjobs'], capture_output=True, text=True)
        print(result.stdout)
        if len(result.stdout) < 25:
            time.sleep(2)
            if len(subprocess.run(['bjobs'], capture_output=True, text=True).stdout) < 25:
                print("All jobs are done.")
                break
        else:
            print(f"Jobs are still running... {int(dt/5)**2 *5} s")
            time.sleep(dt)
            dt += 5  


def delete_specific_files(delete_sim=False):
    """
    Deletes files ending with .log and .source. Optionally deletes .sim files.
    """
    patterns = ['*.log', '*.source', '*.sh', '*.cfg']
    if delete_sim:
        patterns.append('*.sim')

    for pattern in patterns:
        for filename in glob.glob(pattern):
            os.remove(filename)

def run_sim_file_analysis():
    bash_script_name = 'submit_detector_response_jobs.sh'
    bash_script_content = '#!/bin/bash\n\n# This script submits jobs for all .sim files\n\n'

    # Generate the command for each .sim file
    for sim_file in glob.glob('*.sim'):
        #bash_script_content += f'bsub -R centos7 -W 6000 python gamma_TPC_complete.py {sim_file}\n'
        bash_script_content += f'python gamma_TPC_complete.py {sim_file}&\n'

    bash_script_content += "wait\n"
    bash_script_content += 'echo "All jobs submitted."\n'
    with open(bash_script_name, 'w') as bash_script:
        bash_script.write(bash_script_content)

    os.chmod(bash_script_name, 0o755)
    subprocess.run(['./' + bash_script_name])



#generate_cosima_files()
#execute_shell_script()
#check_jobs_periodically()

run_sim_file_analysis()
#check_jobs_periodically()
# delete_specific_files(delete_sim=False)
