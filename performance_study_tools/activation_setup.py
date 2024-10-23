# Given a geometry, background parameters, and a set of activation parameters, 
# this script will calculate the activation of the geometry.

import argparse
import os, sys
import subprocess

parser = argparse.ArgumentParser(description='Calculate the activation of a geometry.')
parser.add_argument('geometry', type=str, help='The geometry file.')
parser.add_argument('-time', type=float, default=1, help='The duration of the simulation in s.')
parser.add_argument('--alt', type=int, default=550, help="Altitude in km. Default 550.")
parser.add_argument('--inc', type=int, default=0, help="Inclination in degrees. Default 0.")
parser.add_argument('--energy_low', type=int, default=2, help="Low energy in keV. 100.")
parser.add_argument('--energy_high', type=int, default=8, help="High energy in keV. 10^8.")

args = parser.parse_args()


sys.path.append("Gampy")
sys.path.append("cosmic_flux")
from cosmic_flux import cosmic_flux_gen

current_dir = os.getcwd()
output_dir = os.path.join(current_dir, "files_for_activation")

if os.path.exists(output_dir):
    os.system(f"rm -r {output_dir}")
os.makedirs(output_dir)


# copy the geometry file to the output directory
geometry_file_name = os.path.basename(args.geometry)
output_geometry_file = os.path.join(output_dir, geometry_file_name)
os.system(f"cp {args.geometry} {output_geometry_file}")


step_1 = cosmic_flux_gen.generate_cosmic_simulation(
                                geo_full_file_name=geometry_file_name.replace(".setup",""),
                                activation=True,  
                                Inclination=args.inc,
                                Altitude=args.alt,
                                Elow=args.energy_low,
                                Ehigh=args.energy_high,
                                duration=args.time,
                                output_dir=output_dir,
                                data_location="Gampy/cosmic_flux/Data")
step_2 = cosmic_flux_gen.calculate_activation(
                                geo_full_file_name=geometry_file_name.replace(".setup",""),
                                Inclination=args.inc,
                                Altitude=args.alt,
                                Elow=args.energy_low,
                                Ehigh=args.energy_high,
                                output_dir=output_dir)


# at this point, we have the directory needed to execute the next steps
# move the output directory to the scratch directory
sister_dir = "/sdf/scratch/kipac/MeV/gammatpc/activation_studies"
if os.path.exists(sister_dir):
    os.system(f"rm -r {sister_dir}")
os.makedirs(sister_dir)
    
os.system(f"cp -r {output_dir} {sister_dir}")
os.system(f"rm -r {output_dir}")

new_working_dir = sister_dir+"/files_for_activation"

step1_file_name = os.path.basename(step_1).replace(".sim", "")
step2_file_name = os.path.basename(step_2).replace(".sim", "")

bash_script_content = f"""#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=3:59:00
#SBATCH --output=/dev/null  # Disable .out files

# Change to the working directory
cd {new_working_dir}

# Step 1: Run cosima for step 1
echo "Running cosima for {step1_file_name}..."
cosima {step1_file_name}

# Step 2: Run cosima for step 2
echo "Running cosima for {step2_file_name}..."
cosima {step2_file_name}

# List the files in the directory
echo "Listing files in the working directory:"
ls

# Copy .dat files back to the original directory
echo "Copying .dat files to the original directory: {current_dir}"
cp *.dat {current_dir}

echo "The activation study is complete."
"""

# Write the Bash script to a file
bash_script_path = os.path.join(new_working_dir, "activation_cosima_run.sh")
with open(bash_script_path, 'w') as bash_file:
    bash_file.write(bash_script_content)

# Make the Bash script executable
subprocess.run(['chmod', '+x', bash_script_path])
subprocess.run(["sbatch", bash_script_path])
os.system(f"rm {bash_script_path}")

print(f"Bash script executed from {bash_script_path}")