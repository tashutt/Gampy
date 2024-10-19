#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 16:13:42 2023

Collection of tools for using PENELOPE to simulate electron tracks

March 23, split this off from tracks_tools

@author: tshutt
"""
def simple_penelope_track_maker(p,
                        steering,
                        delete_penelope_data=True,
                        initial_direction=[0, 0, -1],
                        reset_origin=True,
                        wipe_folders=False,
                        fresh_seed=True,
                        compression_bin_size=None
                        ):
    """
    Runs Penelope to create electron tracks in a single material, over
    a range of energies.  Then parses these to python track object.
    Currently removes any old data.

    The number of energies and numbers of tracks per energy are in steering

    The tracks are parsed in python and output written in a pair of
    a numpy (.npz) file nd pickle files, one pair per track.  Tracks from
    each energy goes into a separate folder.

    initial_direction is a vector (array or list), or 'random'.
        Note - this assume tracks simulated in direction (0,0,1)
    reset_origin - if true, track mean is at origin (0,0,0).
    parse_penelope - see comments there

    Steering has energies in keV and number of tracks at each energy

    Output is mks

    NOTE: What deposited energy is used to generate electron should be
    checked.

    Improvement: more rational handling of material and read_params - material
        should be in read_params

    11/7/21 TS port from matlab
    """
    import numpy as np
    import os
    import glob
    from datetime import datetime

    import tracks_tools

    #   Simulation input

    #   Maximum number of tracks to simulate in penelope at one time - disk
    #   space issue, as large raw penelope tracks aren't crunched to smaller
    #   parsed python files and removed until this batch is completed.
    max_num_tracks_batch = 50

    #   Material and read_params to give recombination.  This is really not
    #   handled well - material shoul be in read_params
    material = 'LAr'
    if 'material' in steering:
        material = steering['material']
    import readout_tools
    read_params = readout_tools.Params()

    #   Particles. 1=electron, 2=photon, 3=positron
    #   Set to electrons if missing
    if not 'particles' in steering:
        particles = 1
    else:
        particles = steering['particles']

    if particles==1:
        ptag = 'electrons'
    if particles==2:
        ptag = 'photons'
    if particles==3:
        ptag = 'positrons'
    particle_ids = {'electrons': 1, 'photons': 2, 'positrons': 3}

    #   Prep and check folders
    if not os.path.isdir(p['output']):
        os.mkdir(p['output'])

    if not os.path.isdir(p['executable']):
        print('*** Error: No executable folder ***')
        exit()

    #%%  Simulate - loop through energies

    for ne, energy in enumerate(steering['energies']):

        #   Output folders.  Order is  material / particles /energy.
        #   Create as needed, and can wipe.

        etag = f'E{energy:07.0f}'

        p0 = os.path.join(p['output'], material)
        if not os.path.isdir(p0):
            os.mkdir(p0)
        p1 = os.path.join(p0, ptag)
        if not os.path.isdir(p1):
            os.mkdir(p1)
        p2 = os.path.join(p1, etag)
        if os.path.isdir(p2):
            os.chdir(p2)
            if wipe_folders:
                for f in glob.glob("*"):
                    os.remove(f)
        else:
            os.mkdir(p2)

        #   Blab
        print(f'E = {energy} keV ' + ptag + ', '
              + str(steering['num_tracks'][ne]) + ' tracks '
              + 'in ' + material)

        #   These control looping over calls to penelope
        num_full_bunches = np.fix(
            steering['num_tracks'][ne] / max_num_tracks_batch
            ).astype(int)
        num_bunches = num_full_bunches
        if (steering['num_tracks'][ne] % max_num_tracks_batch) > 0:
            last_bunch_size \
                = steering['num_tracks'][ne] % max_num_tracks_batch
            num_bunches += 1
        else:
            last_bunch_size = max_num_tracks_batch

        #   Copy executable and materials file to data directory
        os.system(
            'cp '
            + os.path.join(p['executable'], 'pentracks')
            + ' '
            + os.path.join(p2, 'pentracks')
            )
        os.system(
            'cp '
            + os.path.join(p['executable'], material + '.mat')
            + ' '
            + os.path.join(p2, material + '.mat')
            )

        #   Loop over bunches
        for nb in range(num_bunches):

            if nb < num_bunches-1:
                num_penelope_tracks = max_num_tracks_batch
            else:
                num_penelope_tracks = last_bunch_size

            #   blab
            print('\r   bunch ' + str(nb+1) + '/' + str(num_bunches) \
                + ' of ' + str(num_penelope_tracks) + ' tracks', end='')

            #   Create pentracks.in in the data folder, specifying
            #   energies and number of tracks
            penelope_in_image = pentracks_in_file_image(
                energy,
                particles,
                material,
                num_penelope_tracks,
                fresh_seed
                )
            with open(
                    os.path.join(p2, 'pentracks.in'),
                    'w'
                    ) as file:
                for line in penelope_in_image:
                    file.write(line + '\n')

            #   Run penelope
            os.chdir(p2)
            os.system('./pentracks < pentracks.in')

            print('   Parsing PENELOPE output to python tracks')
            if not compression_bin_size==None and compression_bin_size>0:
                print('Compressing to ' +
                      f'{compression_bin_size*1e6:5.0f} microns')

            #   These are data files to process
            full_file_name_list \
                = glob.glob(os.path.join(p2, 'TrackE*.dat'))

            for full_file_name, nf in zip(
                    full_file_name_list,
                    np.arange(0, len(full_file_name_list))
                    ):

                #   Parse file
                track = parse_penelope_file(
                    p2,
                    full_file_name,
                    read_params,
                    initial_direction=initial_direction,
                    reset_origin=reset_origin,
                    compression_bin_size=compression_bin_size
                    )

                #   Save meta data
                track['meta']['material'] = material
                track['meta']['initial_particle'] = particles
                track['meta']['energy'] = energy
                track['meta']['particle_ids'] = particle_ids

                #  Save penelope_tracks
                file_name = full_file_name.split(os.path.sep)[-1]
                out_file_name = (
                    file_name.split('_')[0]
                    + '_'
                    + datetime.now().strftime('D%Y%m%d_T%H%M%f')
                    )
                tracks_tools.save_penelope_track(
                    os.path.join(p2, out_file_name),
                    track
                    )

                #   Delete raw penelope track, if requested
                if delete_penelope_data:
                    os.remove(full_file_name)

        #   Delete executable
        os.system('rm pentracks')

def parse_penelope_file(
        p_data,
        full_file_name,
        read_params,
        initial_direction=[0, 0, -1],
        reset_origin=True,
        compression_bin_size=200e-6
        ):
    """
    Reads Penelope output track files in folder p_data, parses them to create
        track and saves them, one fle per track
        Works on set of files of same energy in single folder.  Each file,
        input and output, is a single track.

    initial_direction is a vector (array or list), or 'random'.
        Note - this assume tracks simulated in direction (0,0,1)

    reset_origin - if true, translates track so that the mean is at
    origin (0,0,0).  This translation is applied after the track is
    rotated (if it is rotated).


    The number of charges generated by Penelope is somewhat off, and a
    crude correction factor which needs more careful study is hard
    coded here.

    N.B. Penelope output file units are eV and cm.  In this routine we
        switch to keV and m.

    To do:
        + Improve / check number of charges: w value and recombination
        + Check if code is right for delta rays (manual says 7, I used 6)

    11/4/21  TS - matlab port
    """

    import numpy as np
    from math import pi
    import os
    import sys

    import tracks_tools
    import math_tools

    #   Get penelope settings from penelope_in file
    penelope_input = parse_penelope_in(p_data)

    #   define w
    #   Fudge was some early rough kludge; should be revisited. Also,
    #   badly implemented - fudge should be in read_params also.
    #   TODO[ts]: Clean up
    w_fudge = 0.8
    w = read_params.material['w'] * w_fudge

    # print('\r  file  ' + str(nf) + ' ', end='')

    file_name = full_file_name.split(os.path.sep)[-1]
    with open(os.path.join(p_data, file_name)) as file:
        lines = file.readlines()

    #   First line blank, next is simulation time
    simulation_date = lines[1]

    data_block = np.zeros((len(lines)-2, 15))
    n = 0
    for line in lines[2:]:
        data_block[n, :] = np.array(line.split())
        n += 1

    #   Parse raw Penelope output.  Units change:
    #      Penelope eV, cm.
    #      Output: keV, m.
    #   Fortran format:
    #  IF(DE.GT.0.0) THEN
    #     WRITE(28,'(i12, i7,4i3,i9,2i3,6e20.12)')
    # 1     NSTEP, NPARTICLE,KPAR,ILB(1),ILB(2),ILB(3),ILB(4),ICOL,
    # 1     IABS,E,DE,DS,X,Y,Z
    #  ENDIF
    #
    #   ICOL is interaction mechanism:
    #   For electrons (KPAR=1):
    #   icol = 1:  soft_event
    #   icol = 2:  hard_elastic
    #   icol = 3:  hard_inelastic
    #   icol = 4:  hard_brem
    #   icol = 5:  inner_shell_ionization
    #   icol = 6:  not used
    #   icol = 7:  "delta" - not actual interactions, no energy deposited
    #   icol = 8:  auxiallary interaction
    #
    #   For photons (KPAR=2):
    #   icol = 1:  coherent (Rayleigh) scattering
    #   icol = 2:  incoherent (Compton) scattering
    #   icol = 3:  photoelectric absorption
    #   icol = 4:  electron-positron pair production
    #   icol = 6:  not used
    #   icol = 6:  not used
    #   icol = 7:  delta interaction
    #   icol = 8:  auxiallary interaction
    #
    #   For positrons (KPAR=3):
    #   icol = 1:  soft_event
    #   icol = 2:  hard_elastic
    #   icol = 3:  hard_inelastic
    #   icol = 4:  hard_brem
    #   icol = 5:  inner_shell_ionization
    #   icol = 6:  annhilation
    #   icol = 7:  "delta" - not actual interactions, no energy deposited
    #   icol = 8:  auxiallary interaction

    penelope_data = {}

    penelope_data['step'] = data_block[:, 0]
    penelope_data['id'] = data_block[:, 1]
    penelope_data['particle'] = data_block[:, 2]
    penelope_data['generation'] = data_block[:, 3]
    penelope_data['parent'] = data_block[:, 4]
    penelope_data['parent_interaction'] = data_block[:, 5]
    penelope_data['atomic_relaxation'] = data_block[:, 6]
    penelope_data['interaction'] = data_block[:, 7]
    penelope_data['absorbed'] = data_block[:, 8]
    penelope_data['energy'] = data_block[:, 9] / 1000
    penelope_data['delta_energy'] = data_block[:, 10] / 1000
    penelope_data['delta_step'] = data_block[:, 11] * 100
    penelope_data['r'] = data_block[:, 12:].transpose() / 100

    #   For initial photons, note first interaction
    first_interaction = 'unknown'
    if (penelope_data['step'][0]==1) & (penelope_data['particle'][0]==2):
        if penelope_data['interaction'][0]==2:
            first_interaction = 'compton'
        elif penelope_data['interaction'][0]==4:
            first_interaction = 'pair'

    #   This is probably not a good way to do this - use step
    penelope_data['birth_step'] \
        = np.diff(np.insert(penelope_data['id'], 0, 0)) > 0

    #   Note that hard_inelastic, hard_brem, and inner_shell_ionization
    #   do not contribute to ionization - the energy  of these I think
    #   goes to secondary particles.
    #   Charge generated only in soft interactions (this is debatable)
    active_interactions = (
        (penelope_data['interaction']==1)
        & (penelope_data['delta_energy'] > w)
        & ~((penelope_data['particle']==3) & (penelope_data['absorbed']==1))
        )
    num_e = np.round(np.fix(
        penelope_data['delta_energy']
        / w * (1 - read_params.material['recombination'])
        ) * active_interactions).astype(int)
    deposits_mask = num_e>0
    num_e = num_e[deposits_mask]

    #   Number of charges at each site is num_e from above
    r = penelope_data['r'][:, deposits_mask]

    #   Origin and initial diretion are from Penelope input
    origin = penelope_input['r_o'] / 100
    penelope_direction = penelope_input['s_o']

    #   Track initial directions
    if isinstance(initial_direction, str):
        if initial_direction!='random':
            sys.exit('Error in parse_penelope_file: bad initial_direction')

        #   generate unit vector with random direction in 4pi
        rng = np.random.default_rng()
        theta = np.arccos(1 - 2 * rng.random(1))
        phi = 2 * pi * rng.random(1)
        s = math_tools.sph2cart(theta, phi)

    #   Rotate to specified directions
    else:
        s = np.array(initial_direction)

    #   Rotate track, initial vector
    r = math_tools.rotate_ray(r, s)
    initial_direction = math_tools.rotate_ray(penelope_direction, s)

    #   Center track on mean
    if reset_origin:
        center = r.mean(axis=1)
        r = (r.transpose() - center).transpose()
        origin += -center

    #   Compress
    if not compression_bin_size==None and compression_bin_size>0:
        r, num_e = tracks_tools.compress_track(
            r,
            num_e,
            compression_bin_size
            )

    #   Assign everything to track
    track = {}

    track['r'] = r
    track['num_e'] = num_e
    track['particle'] = penelope_data['particle'][deposits_mask]
    track['generation'] = penelope_data['generation'][deposits_mask]
    track['parent'] = penelope_data['parent'][deposits_mask]
    track['interaction'] = penelope_data['interaction'][deposits_mask]
    track['birth_step'] = penelope_data['birth_step'][deposits_mask]

    track['truth'] = {}

    track['truth']['origin'] = origin
    track['truth']['initial_direction'] = initial_direction
    track['truth']['first_interaction'] = first_interaction

    #   Energy and charge
    track['truth']['num_electrons'] = np.sum(num_e)
    track['truth']['track_energy'] \
        = penelope_input['electron_energy']

    #   Save Peneleope job meta data
    track['meta'] = {}
    track['meta']['penelope_input'] = penelope_input
    track['meta']['simulation_date'] = simulation_date

    return track

def parse_penelope_in(p):
    """ Reads pentracks.in file in folder p; returns penelope_input """

    import numpy as np
    import os

    #   Read and parse 'pentracks.in', the PENELOPE steering file
    with open(os.path.join(p, 'pentracks.in')) as file:
        lines = file.readlines()
    penelope_input = {}
    penelope_input['raw_lines'] = [line.rstrip() for line in lines]

    #   Save a few important parameters from penelope.in
    for line in penelope_input['raw_lines']:

        #   Settings start with names of up to six characters in length
        if line[0:6].count(' ') < 6:

            name = line[0:6].replace(' ', '')
            values_string = line[7:].split(r'[')[0].split()

            #   These are saved

            if name == 'MSIMPA':
                values = np.array(
                    [float(value_string)for value_string in values_string])
                penelope_input['eabs'] = values[0:3]
                penelope_input['c1'] = values[3]
                penelope_input['c2'] = values[4]
                penelope_input['wcc'] = values[5]
                penelope_input['wcr'] = values[6]

            if name == 'SENERG':
                penelope_input['electron_energy'] \
                    = float(values_string[0]) / 1000

            if name == 'SPOSIT':
                values = np.array(
                    [float(value_string)for value_string in values_string])
                penelope_input['r_o'] = values

            if name == 'SCONE':
                values = np.array(
                    [float(value_string)for value_string in values_string])
                theta = values[0]
                phi = values[1]
                alpha = values[2]

                s_o = np.zeros(3)
                if alpha==0:
                    s_o[0] = np.cos(phi) * np.sin(theta)
                    s_o[1] = np.cos(phi) * np.sin(theta)
                    s_o[2] = np.cos(theta)
                else:
                    print('Warning: initial direction not known')

                penelope_input['s_o'] = s_o

    return penelope_input

def pentracks_in_file_image(energy,
                            particles=1,
                            material='LAr',
                            num_tracks=1000,
                            fresh_seed=True
                            ):
    """ return pentrack.in file as image, for energy and num_tracks """

    import numpy as np

    #   Energy dependent penelope settings
    energy_settings = {}
    energy_settings['max_energy'] = []  # keV - upper energy for this set
    energy_settings['eabs'] = []
    energy_settings['c1'] = []
    energy_settings['c2'] = []
    energy_settings['wcc'] = []
    energy_settings['wcr'] = []
    energy_settings['timelimit'] = []

    energy_settings['max_energy'].append(10001)
    energy_settings['eabs'].append(50)
    energy_settings['c1'].append(0.01)
    energy_settings['c2'].append(0.01)
    energy_settings['wcc'].append(50)
    energy_settings['wcr'].append(50)
    energy_settings['timelimit'].append(12000)

    energy_settings['max_energy'].append(int(1.001e5))
    energy_settings['eabs'].append(1000)
    energy_settings['c1'].append(0.02)
    energy_settings['c2'].append(0.02)
    energy_settings['wcc'].append(1000)
    energy_settings['wcr'].append(1000)
    energy_settings['timelimit'].append(12000)

    energy_settings['max_energy'].append(int(1.0001e6))
    energy_settings['eabs'].append(5000)
    energy_settings['c1'].append(0.02)
    energy_settings['c2'].append(0.02)
    energy_settings['wcc'].append(5000)
    energy_settings['wcr'].append(5000)
    energy_settings['timelimit'].append(12000)

    #   Create new penelope.in file with these settings

    #   Initial default pentrack_in file image
    penelope_in = default_pentracks_in_file()

    #   This is index of the relevant energy settings - the lowest energy
    #   set
    nei = np.asarray(
        energy <= np.array(energy_settings['max_energy'])
        ).nonzero()[0][0]

    nl = 0
    for line in penelope_in:

        #   Primary particle
        if line[0:6].strip()=='SKPAR':
            penelope_in[nl] = (
                f'SKPAR  {particles:d}'
                ).ljust(line.find('[')) + line[line.find('['):]

        #   Track energy
        if line[0:6].strip()=='SENERG':
            penelope_in[nl] = (
                f'SENERG {energy:0.0f}e3'
                ).ljust(line.find('[')) + line[line.find('['):]

        #   Target material
        if line[0:6].strip()=='MFNAME':
            penelope_in[nl] = (
                'MFNAME ' + material + '.mat'
                ).ljust(line.find('[')) + line[line.find('['):]

        #   Number of tracks
        if line[0:6].strip()=='NSIMSH':
            penelope_in[nl] = (
                f'NSIMSH {num_tracks:0.0f}'
                ).ljust(line.find('[')) + line[line.find('['):]

        #   EABS, etc.   Handle a bit different, as may get too long
        if line[0:6].strip()=='MSIMPA':
            new_line = (
                'MSIMPA '
                + f'{energy_settings["eabs"][nei]:0.0f} '
                + f'{energy_settings["eabs"][nei]:0.0f} '
                + f'{energy_settings["eabs"][nei]:0.0f} '
                + f'{energy_settings["c1"][nei]:0.0f} '
                + f'{energy_settings["c2"][nei]:0.0f} '
                + f'{energy_settings["wcc"][nei]:0.0f} '
                + f'{energy_settings["wcr"][nei]:0.0f}'
                ).ljust(line.find('[')) + line[line.find('['):]
            penelope_in[nl] = new_line[0:len(line)]

        #   Random number generator seed
        if fresh_seed & (line[0:6].strip()=='RSEED'):
            iseed1 = np.random.randint(1, 2**31)
            iseed2 = np.random.randint(1, 2**31)
            penelope_in[nl] = (
                f'RSEED  {iseed1:0.0f} '
                + f'{iseed2:0.0f} '
                ).ljust(line.find('[')) + line[line.find('['):]

        nl += 1

    return penelope_in

def default_pentracks_in_file():
    """
    Returns base pentracks.in file as list of lines.
    Note that Penelope units are eV and cm.
    """

    base_file = []

    base_file.append(
        'TITLE  Electrons in homogeneous cylinder.'
        )
    base_file.append(
        '       .'
        )
    base_file.append(
        'GSTART >>>>>>>> Beginning of the geometry definition list.'
        )
    base_file.append(
        'LAYER  -1e3 1e3                                 '
        + '[Z_lower and Z_higher]'
        )
    base_file.append(
        'CYLIND 1 0 1e3                      '
        + '[Material, R_inner and R_outer]'
        )
    base_file.append(
        'GEND   <<<<<<<< End of the geometry definition list.'
        )
    base_file.append(
        '       The labels KL,KC denote the KC-th cylinder in the KL-th layer.'
        )
    base_file.append(
        '       .'
        )
    base_file.append(
        '       >>>>>>>> Source definition.'
        )
    base_file.append(
        'SKPAR  1        [Primary particles: 1=electron, 2=photon, 3=positron]'
        )
    base_file.append(
        'SENERG 20e3              [Initial energy (monoenergetic sources only)]'
        )
    base_file.append(
        'SPOSIT 0 0 0                 [Coordinates of the source center]'
        )
    base_file.append(
        'SCONE  0 0 0                             [Conical beam; angles in deg]'
        )
    base_file.append(
        '       .'
        )
    base_file.append(
        '       >>>>>>>> Material data and simulation parameters.'
        )
    base_file.append(
        '                Up to MAXMAT materials; 2 base_file for each material.'
        )
    base_file.append(
       'MFNAME LAr.mat                         [Material file, up to 20 chars]'
        )
    base_file.append(
        'MSIMPA 50 50 50 0.01 0.01 100 100                 '
        + '[EABS(1:3),C1,C2,WCC,WCR]'
        )
    base_file.append(
        '       .'
        )
    base_file.append(
        '       >>>>>>>> Local maximum step lengths and absorption energies.'
        )
    base_file.append(
        'DSMAX  1 1 1.0D35                  [Mmaximum step length in body KL,KC]'
        )
    base_file.append(
        '       .'
        )
    base_file.append(
        '       >>>>>>>> Job properties.'
        )
    base_file.append(
        'RESUME dump.dat                [Resume from this dump file, 20 chars]'
        )
    base_file.append(
        'DUMPTO dump.dat                   [Generate this dump file, 20 chars]'
        )
    base_file.append(
        'DUMPP  60                                    [Dumping period, in sec]'
        )
    base_file.append(
        '       .'
        )
    base_file.append(
        'RSEED  -1 -1                   [Seeds of the random-number generator]'
        )
    base_file.append(
        'NSIMSH 50                       [Desired number of simulated showers]'
        )
    base_file.append(
        'TIME   12000                       [Allotted simulation time, in sec]'
        )
    base_file.append(
        'END                                  [Ends the reading of input data]'
        )

    return base_file
