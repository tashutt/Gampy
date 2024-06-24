import numpy as np
from scipy.integrate import quad
import sys
import os

from cosmic_flux.LEOBackgroundGenerator import LEOBackgroundGenerator as LEO

def generate_cosmic_simulation(geo_full_file_name, 
                               activation=False, 
                               Inclination=0, 
                               Altitude=550, 
                               Elow=1, Ehigh=8, 
                               duration=100.0, 
                               num_triggers=10000, 
                               output_dir=None,
                               only_photons=False):
    """
    Generate cosmic flux and create simulation configuration lines, for both activation and non-activation modes.

    Args:
        geo_full_file_name (str): The full file name for the geometry setup.
        activation (bool, optional): If set to True, generates activation files. If False, does not.
        Inclination (float, optional): Inclination of the orbit in degrees. Default is 0.
        Altitude (float, optional): Altitude of the orbit in km. Default is 550.
        Elow (float, optional): Log10 of the lowest energy limit in keV. Default is 1.
        Ehigh (float, optional): Log10 of the highest energy limit in keV. Default is 8.
        duration (float, optional): Duration of the simulation in seconds. Relevant only if activation is True. Default is 2.0.
        num_triggers (int, optional): Number of triggers. Relevant only if activation is False. Default is 10000.
        output_dir (str, optional): Output directory path. If not provided, defaults to the working directory.
    
    Returns:
        source_file_name (str): The path to the created source file.
    """

    Geomlat = -0.1


    # Set the source file name based on whether it's for activation or not
    source_file_name = f"{'Activation' if activation else 'Simulation'}Step1For{Altitude}km_{int(10**Elow)}to{int(10**Ehigh)}keV.source"
    
    lines = ['']
    lines.append('Version          1 \n')
    lines.append('Geometry         ' + geo_full_file_name + '.setup\n')

    if activation:
        lines.append('DetectorTimeConstant                 0.000005\n')
        lines.append('PhysicsListHD    qgsp-bic-hp\n')
        lines.append('PhysicsListEM    LivermorePol\n')
        lines.append('DecayMode    ActivationBuildup\n')
        lines.append('\n')
        lines.append('StoreSimulationInfo            all\n')
        lines.append('DefaultRangeCut            0.1\n')
        lines.append('\n')
        lines.append('Run SpaceSim' + '\n')
        lines.append('SpaceSim.FileName           ' + source_file_name.replace(".source", "") + '\n')
        lines.append('SpaceSim.Time          ' + f'{duration} ' + '\n')
        lines.append('SpaceSim.IsotopeProductionFile          ' + 'Isotopes' + '\n')
        lines.append('\n')
    else:
        lines.append('CheckForOverlaps 1000 0.01 \n')
        lines.append('PhysicsListEM    LivermorePol\n')
        lines.append('\n')
        #lines.append('StoreCalibrate                 true\n')
        #lines.append('StoreSimulationInfo            true\n')
        #lines.append('StoreOnlyEventsWithEnergyLoss  true  ' + '// Only relevant if no trigger criteria is given!\n')
        lines.append('DiscretizeHits                 true\n')
        lines.append('PreTriggerMode                 everyeventwithhits\n')
        lines.append('\n')
        lines.append('Run SpaceSim' + '\n')
        lines.append('SpaceSim.FileName           ' + source_file_name.replace(".source", "") + '\n')
        lines.append('SpaceSim.NTriggers          ' + f'{num_triggers:5.0f} ' + '\n')
        lines.append('\n')



    particles_ID = {
        "AtmosphericNeutrons": 6,
        "PrimaryProtons": 4,
        "SecondaryProtonsUpward": 4,
        "SecondaryProtonsDownward": 4,
        "PrimaryAlphas": 21,
        "CosmicPhotons": 1,
        "AlbedoPhotons": 1,
        "PrimaryPositrons": 2,
        "SecondaryPositrons": 2,
        "PrimaryElectrons":3,
        "SecondaryElectrons":3,}
    
    # TODO Rcut 
        
    LEOClass = LEO(1.0*Altitude, 1.0*Inclination,Geomlat)
    ViewAtmo = 2*np.pi * (np.cos(np.deg2rad(LEOClass.HorizonAngle)) + 1)
    ViewSky  = 2*np.pi * (1-np.cos(np.deg2rad(LEOClass.HorizonAngle)))

    Particle = ["AtmosphericNeutrons", "CosmicPhotons", "PrimaryProtons",
        "SecondaryProtonsUpward","SecondaryProtonsDownward", "PrimaryAlphas", "PrimaryElectrons",
        "PrimaryPositrons", "SecondaryElectrons", "SecondaryPositrons",
        "AlbedoPhotons"
        ]
    
    relevant_for_activation = ["AtmosphericNeutrons", "PrimaryProtons","SecondaryProtonsUpward","SecondaryProtonsDownward", "PrimaryAlphas"]
    photons = ["CosmicPhotons", "AlbedoPhotons"]

    fac = [ViewAtmo, ViewSky,ViewSky, 2*np.pi, 2*np.pi, ViewSky, ViewSky, ViewSky,4*np.pi,4*np.pi,ViewAtmo]    

    if output_dir is None:
        output_dir = os.getcwd()

    dat_name = f"For{Altitude}km_{int(10**Elow)}to{int(10**Ehigh)}keV"
    output_folder = os.path.join(output_dir, dat_name)

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    source_file_name = os.path.join(output_dir,
                                    source_file_name)
    
    particle_beam_mapping = {
        "PrimaryElectrons": "CosmicElectronsMizuno.beam.dat",
        "SecondaryElectrons": "AlbedoElectronsAlcarazMizuno.beam.dat",
        "PrimaryPositrons": "CosmicPositronsMizuno.beam.dat",
        "SecondaryPositrons": "AlbedoPositronsAlcarazMizuno.beam.dat",
        "CosmicPhotons": "CosmicPhotonsGruber.beam.dat",
        "PrimaryProtons": "CosmicProtonsSpenvis.beam.dat",
        "PrimaryAlphas": "CosmicAlphasSpenvis.beam.dat",
        "SecondaryProtonsUpward": "AlbedoProtonMizuno.beam.dat",
        "SecondaryProtonsDownward": "AlbedoProtonMizuno.beam.dat",
        "AtmosphericNeutrons":"AlbedoNeutronsKole.beam.dat",
        "AlbedoPhotons": "AlbedoPhotonsTuerlerMizunoAbdo.beam.dat",
    }

    #copy the beam files to Data
    data_dir = "cosmic_flux/Data"

    for particle, beam_file in particle_beam_mapping.items():
        source_file = os.path.join(data_dir, beam_file)
        destination_file = os.path.join(output_folder, beam_file)

        # Use sys commands to copy
        sys_command = f"cp {source_file} {destination_file}"
        os.system(sys_command)
    
    with open(source_file_name, 'w') as runfile:
        runfile.writelines(lines)
        runfile.write('\n')

        for i in range(0, len(Particle)):
            # Only write the activation sources if we're doing activation
            if activation and Particle[i] not in relevant_for_activation:
                print(f"Skipping {Particle[i]} for activation")
                continue
              
            if only_photons and Particle[i] not in photons:
                print(f"Skipping {Particle[i]} for only photons regime")
                continue
              
            Energies = np.logspace(Elow, Ehigh, num=100, endpoint=True, base=10.0)
            filename = f"{dat_name}/{Particle[i]}.dat"
            Output = os.path.join(output_dir if output_dir else '', filename)

            IntSpectrum = np.trapz(getattr(LEOClass, Particle[i])(Energies), Energies)
            print(Particle[i], IntSpectrum * fac[i], " #/cm^2/s")

            runfile.write(f"SpaceSim.Source {Particle[i]}\n")
            runfile.write(f"{Particle[i]}.ParticleType {particles_ID[Particle[i]]}\n")
            
            beam_file = particle_beam_mapping.get(Particle[i], "FileNotFound.beam.dat") 
        
            runfile.write(f"{Particle[i]}.Beam FarFieldFileZenithDependent {dat_name}/{beam_file}\n")
            runfile.write(f"{Particle[i]}.Spectrum File {filename}\n")
            runfile.write(f"{Particle[i]}.Flux {IntSpectrum * fac[i]}\n")
            runfile.write("\n\n")

            # Create the .dat file
            with open(Output, 'w') as f:
                print(f'# {Particle[i]} spectrum ', file=f)
                print('# Format: DP <energy in keV> <shape of differential spectrum [XX/keV]>', file=f)
                print('# Although cosima doesn\'t use it the spectrum here is given as a flux in #/cm^2/s/keV', file=f)
                print(f'# Integrated over {fac[i]} sr', file=f)
                print(f'# Integral Flux: {IntSpectrum * fac[i]} #/cm^2/s', file=f)
                print('', file=f)
                print('IP LOGLOG', file=f)
                print('', file=f)
                for j in range(0, len(Energies)):
                    print('DP', Energies[j], getattr(LEOClass, Particle[i])(Energies[j]), file=f)
                print('EN', file=f)


    return source_file_name


def calculate_activation(geo_full_file_name, Inclination=0, Altitude=550, Elow=1, Ehigh=8, duration=31556736, output_dir=None):
    
    source_file_name2 = f"ActivationStep2For{Altitude}km_km_{int(10**Elow)}to{int(10**Ehigh)}keV.source"

    lines = ['']
    lines.append('Version          1 \n')
    lines.append('Geometry         ' + geo_full_file_name + '.setup\n')
    lines.append('DetectorTimeConstant                 0.000005\n')
    lines.append('PhysicsListHD    qgsp-bic-hp\n')
    lines.append('PhysicsListEM    LivermorePol\n')
    lines.append('\n')
    lines.append('Activator    A\n')
    lines.append('A.IsotopeProductionFile          '
                 + 'Isotopes.inc1.dat' + '\n')
    lines.append('A.ActivationMode          '
                 + 'ConstantIrradiation  ' + f'{duration} ' + '\n')
    lines.append(f'A.ActivationFile          ActivationFor{Altitude}km_{int(10**Elow)}to{int(10**Ehigh)}keV.dat')
    lines.append('\n')
    lines.append('\n')
    
    if output_dir is None:
        output_dir = os.getcwd()

    dat_name = f"For{Altitude}km_{int(10**Elow)}to{int(10**Ehigh)}keV"
    output_folder = os.path.join(output_dir, dat_name)

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    source_file_name2 = os.path.join(output_dir,
                                    source_file_name2)
    runfile = open(source_file_name2, 'w')

    # Write the lines to the runfile
    runfile.writelines(lines)
    runfile.write('\n')
    runfile.close()
    
    return source_file_name2

def activation_events(geo_full_file_name, Inclination=0, Altitude=550, Elow=1, Ehigh=8, duration=31556736, output_dir=None, dat_name=None):
    
    source_file_name3 = f"ActivationStep3For{Altitude}km_{int(10**Elow)}to{int(10**Ehigh)}keV.source"
    
    lines = ['']
    lines.append('Version          1 \n')
    lines.append('Geometry         ' + geo_full_file_name + '.setup\n')
    lines.append('DetectorTimeConstant                 0.000005\n')
    lines.append('PhysicsListHD    qgsp-bic-hp\n')
    lines.append('PhysicsListEM    LivermorePol\n')
    lines.append('DecayMode    ActivationDelayedDecay\n')
    lines.append('\n')
    lines.append('StoreSimulationInfo    all\n')

    lines.append('DefaultRangeCut     0.1\n')

    lines.append('Run ActivationStep3\n')

    lines.append(f'ActivationStep3.FileName                         ActivationStep3For{Altitude}km_{int(10**Elow)}to{int(10**Ehigh)}keV\n')
    lines.append('ActivationStep3.Time          ' + f'{duration} ' + '\n')
    if dat_name is None:
        lines.append(f'ActivationStep3.ActivationSources          ActivationFor{Altitude}km_{int(10**Elow)}to{int(10**Ehigh)}keV.dat')
    else:
        lines.append(f'ActivationStep3.ActivationSources          {dat_name}' )
        
    lines.append('\n')
    lines.append('\n')
    
    if output_dir is None:
        output_dir = os.getcwd()

    dat_name = f"For{Altitude}km_{int(10**Elow)}to{int(10**Ehigh)}keV"
    output_folder = os.path.join(output_dir, dat_name)

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    source_file_name3 = os.path.join(output_dir,
                                    source_file_name3)
    runfile = open(source_file_name3, 'w')

    # Write the lines to the runfile
    runfile.writelines(lines)
    runfile.write('\n')
    runfile.close()
    
    return source_file_name3
