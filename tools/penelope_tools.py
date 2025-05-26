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
                                full_output=False,
                                ):
    """
    Runs Penelope to create electron tracks in a single material, over
    a range of energies.  Then parses these to python track object.

    steering - dictionary with keys:
        particles - 'photons', 'electrons', or 'positrons'
        material - 'LAr' or 'Lxe'
        energies -  scalar or list, in keV
        num_tracks - scalar or list for each energy in energies
        eabs - (optional) scalar or list, assigned to all EABS(KPAR,M)
        c - (optional) scalar or list, assigned to C1 and C2
        wc = (optional) ascalar or list, assigned to WCC and WCR
        'folder_tag' - (optional) for all folders from this simluation run
    initial_direction:
        vector (array or list) - this becomes initial direction
            This assume tracks simulated in direction (0,0,1)
        'random' - initial direction chosen randomly over 4 pi
        None - keeps direction from sims (should be (0,0,1))
    reset_origin - if true, track mean is at origin (0,0,0).
    fresh_seed - if False, then sim runs start with same RNG seed

    The tracks are parsed in python and output written in a pair of
    a numpy (.npz) file nd pickle files, one pair per track.  Tracks from
    each energy goes into a separate folder.

    Output is mks

    NOTE: What deposited energy is used to generate electron should be
    checked.

    Improvement: more rational handling of material and read_params - material
        should be in read_params

    11/7/21 TS port from matlab
    """
    import numpy as np
    import os
    import sys
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

    #   Particles are electrons by default.  Penelope particle ids are
    #   1=electron, 2=photon, 3=positron
    if not 'particles' in steering:
        particles = 'electrons'
    else:
        particles = steering['particles']
    if (particles!='photons') and (particles!='electrons') \
        and (particles!='positrons'):
            sys.exit('Unrecognized particles definition')
    particle_ids = {'electrons': 1, 'photons': 2, 'positrons': 3}

    #   Unpack eneriges and number of traks
    if type(steering['energies']) is list:
        energies = steering['energies']
    else:
        energies = [steering['energies']]
    if type(steering['num_tracks']) is list:
        num_trackss = steering['num_tracks']
    else:
        num_trackss = [steering['num_tracks']] * len(energies)

    #   Deal with optional simulation resolution parameters
    if 'eabs' in steering:
        if type(steering['eabs']) is list:
            eabss = steering['eabs']
        else:
            eabss = [steering['eabs']]
    else:
        eabss = [False] * len(energies)
    if 'c' in steering:
        if type(steering['c']) is list:
            cs = steering['c']
        else:
            cs = [steering['c']]
    else:
        cs = [False] * len(energies)
    if 'wc' in steering:
        if type(steering['wc']) is list:
            wcs = steering['wc']
        else:
            wcs = [steering['wc']]
    else:
        wcs = [False] * len(energies)


    #   Prep and check folders
    if not os.path.isdir(p['output']):
        os.mkdir(p['output'])

    if not os.path.isdir(p['executable']):
        sys.exit('No executable folder')

    #   folder tag if supplied
    if (not 'folder_tag' in steering) or (steering['folder_tag']==''):
        folder_tag = ''
    else:
        folder_tag = '_' + steering['folder_tag']

    #%%  Simulate - loop through energies

    for ne, energy in enumerate(energies):

        #   Output folders.  Order is  material / particles /energy.
        #   Create as needed, and can wipe.

        etag = f'E{energy:07.0f}'

        p0 = os.path.join(p['output'], material)
        if not os.path.isdir(p0):
            os.mkdir(p0)
        p1 = os.path.join(p0, particles)
        if not os.path.isdir(p1):
            os.mkdir(p1)
        p2 = os.path.join(p1, etag + folder_tag)
        if os.path.isdir(p2):
            os.chdir(p2)
            if wipe_folders:
                for f in glob.glob("*"):
                    os.remove(f)
        else:
            os.mkdir(p2)

        #   Blab
        print(f'E = {energy} keV ' + particles + f', {num_trackss[ne]:d}'
              + ' tracks in ' + material)

        #   These control looping over calls to penelope
        num_full_bunches = np.fix(
            num_trackss[ne] / max_num_tracks_batch
            ).astype(int)
        num_bunches = num_full_bunches
        if (num_trackss[ne] % max_num_tracks_batch) > 0:
            last_bunch_size \
                = num_trackss[ne] % max_num_tracks_batch
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
            #   energies, number of tracks, settings
            penelope_in_image = pentracks_in_file_image(
                energy,
                particles,
                material,
                num_penelope_tracks,
                fresh_seed,
                cs[ne],
                wcs[ne],
                eabss[ne],
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

            #   These are data files to process
            full_file_name_list \
                = glob.glob(os.path.join(p2, 'TrackE*.dat'))

            for full_file_name, nf in zip(
                    full_file_name_list,
                    np.arange(0, len(full_file_name_list))
                    ):

                #   Parse file
                track = parse_penelope_file(
                    full_file_name,
                    read_params,
                    initial_direction=initial_direction,
                    reset_origin=reset_origin,
                    full_output=full_output,
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
        full_file_name,
        read_params,
        initial_direction=[0, 0, -1],
        reset_origin=True,
        full_output=False,
        ):
    """
    Reads Penelope output track files in folder p_data, parses them to create
        track and saves them, one fle per track
        Works on set of files of same energy in single folder.  Each file,
        input and output, is a single track.

    initial_direction:
        vector (array or list) - this becomes initial direction
            This assume tracks simulated in direction (0,0,1)
        'random' - initial direction chosen randomly over 4 pi
        None - keeps direction from sims (should be (0,0,1))
    reset_origin - if true, translates track so that the mean is at
        origin (0,0,0).  Applied after any rotation.

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

    import math_tools

    #   Get penelope settings from penelope_in file
    penelope_input = parse_penelope_in(os.path.split(full_file_name)[0])

    #   define w
    #   Fudge was some early rough kludge; should be revisited. Also,
    #   badly implemented - fudge should be in read_params also.
    #   TODO[ts]: Clean up
    w_fudge = 0.8
    w = read_params.material['w'] * w_fudge

    # print('\r  file  ' + str(nf) + ' ', end='')

    with open(full_file_name) as file:
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
    #   note: NPARTICLE is a sequential counter for each particle, staring
    #       at 1 for the initial particle
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
    #   icol = 5:  not used
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

    #   Assign values
    step = data_block[:, 0].astype(int)
    particle_id = data_block[:, 2].astype(int)
    interaction = data_block[:, 7].astype(int)
    absorbed = data_block[:, 8].astype(int)
    delta_energy = data_block[:, 10] / 1000
    delta_step = data_block[:, 11] * 100
    r = data_block[:, 12:].transpose() / 100

    if full_output:
        particle_num = data_block[:, 1].astype(int)
        energy = data_block[:, 9] / 1000
        atomic_relaxation = data_block[:, 6].astype(int)
        generation = data_block[:, 3].astype(int)
        parent = data_block[:, 4].astype(int)
        parent_interaction = data_block[:, 5].astype(int)

    #   These are important interactions
    brem = ((particle_id==1) | (particle_id==3)) & (interaction==4)
    photo = (particle_id==2) & (interaction==3)
    compton = (particle_id==2) & (interaction==2)
    pair = (particle_id==2)& (interaction==4)

    #   First interactions of photons.
    first_interaction = 'n/a'
    if penelope_input['initial_particle']=='photon':
        if interaction[0]==2:
            first_interaction = 'compton'
        elif interaction[0]==4:
            first_interaction = 'pair'

    #   Note that hard_inelastic, hard_brem, and inner_shell_ionization
    #   do not contribute to ionization - the energy  of these I think
    #   goes to secondary particles.
    #   Charge generated only in soft interactions (this is debatable)
    active_interactions = (
        (interaction==1)
        & (delta_energy > w)
        & ~((particle_id==3) & (absorbed==1))
        )
    num_e = np.round(np.fix(
        delta_energy
        / w * (1 - read_params.material['recombination'])
        ) * active_interactions).astype(int)
    deposits_mask = num_e>0

    #   Initial diretion from Penelope input
    penelope_direction = penelope_input['s_o']

    #   This didn't accomoplish what was intended. Keep for reference.
    # birth_step = np.diff(np.insert(particle_num, 0, 0)) > 0

    #   Track initial direction

    #   check for bad input
    if (isinstance(initial_direction, str)) & (initial_direction!='random'):
        sys.exit('Error in parse_penelope_file: bad initial_direction')

    if not initial_direction:
        initial_direction = penelope_direction

    else:

        #   Random
        if initial_direction=='random':
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

    #   Origin is from Penelope input unless a photon. Then is first step
    #   location
    if penelope_input['initial_particle']=='photon':
        origin = r[:, 0].T
    else:
        origin = penelope_input['r_o'] / 100

    #   Center track on mean
    if reset_origin:
        center = r.mean(axis=1)
        r = (r.transpose() - center).transpose()
        origin += -center


    #   Assign everything to track
    track = {}

    if full_output:
        track['particle_id'] = particle_id[deposits_mask]
        track['interaction'] = interaction[deposits_mask]
        track['deposits'] = deposits_mask
        track['parent_interaction'] = parent_interaction
        track['atomic_relaxation'] = atomic_relaxation
        track['absorbed'] = absorbed
        track['energy'] = energy
        track['delta_energy'] = delta_energy
        track['delta_step'] = delta_step
        track['particle_num'] = particle_num
        track['parent'] = parent
        track['generation'] = generation

    #   If full output deposits_mask is null
    if full_output:
        deposits_mask = np.ones_like(deposits_mask, dtype=bool)

    #   r, num_e
    track['r'] = r[:, deposits_mask]
    track['num_e'] = num_e[deposits_mask]

    #   Truth
    track['truth'] = {}

    track['truth']['num_electrons'] = np.sum(num_e)
    track['truth']['track_energy'] = penelope_input['initial_energy']

    track['truth']['origin'] = origin
    track['truth']['initial_direction'] = initial_direction
    track['truth']['first_interaction'] = first_interaction

    track['truth']['r_brem'] = r[:, brem]
    track['truth']['r_photo'] = r[:, photo]
    track['truth']['r_compton'] = r[:, compton]
    track['truth']['r_pair'] = r[:, pair]

    track['truth']['energy_brem'] = delta_energy[brem]
    track['truth']['energy_photo'] = delta_energy[photo]
    track['truth']['energy_compton'] = delta_energy[compton]
    track['truth']['energy_pair'] = delta_energy[pair]

    #   Peneleope job meta data
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

            if name == 'SKPAR':
                penelope_input['initial_particle_id'] = int(values_string[0])
                if int(values_string[0])==1:
                    penelope_input['initial_particle'] = 'electron'
                elif int(values_string[0])==2:
                    penelope_input['initial_particle'] = 'photon'
                elif int(values_string[0])==3:
                    penelope_input['initial_particle'] = 'positron'

            if name == 'MSIMPA':
                values = np.array(
                    [float(value_string)for value_string in values_string])
                penelope_input['eabs'] = values[0:3]
                penelope_input['c1'] = values[3]
                penelope_input['c2'] = values[4]
                penelope_input['wcc'] = values[5]
                penelope_input['wcr'] = values[6]

            if name == 'SENERG':
                penelope_input['initial_energy'] \
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
                            particles='electrons',
                            material='LAr',
                            num_tracks=1000,
                            fresh_seed=True,
                            c=False,
                            wc=False,
                            eabs=False,
                            ):
    """ return pentrack.in file as image, for energy and num_tracks """

    import numpy as np

    #   These are energy dependent settings
    settings = {}
    settings['max_energy'] = []  # keV - upper energy for this set
    settings['eabs'] = []
    settings['c1'] = []
    settings['c2'] = []
    settings['wcc'] = []
    settings['wcr'] = []

    settings['max_energy'].append(10000.1)
    settings['eabs'].append(50)
    settings['c1'].append(0.01)
    settings['c2'].append(0.01)
    settings['wcc'].append(50)
    settings['wcr'].append(50)

    settings['max_energy'].append(int(1.001e5))
    settings['eabs'].append(1000)
    settings['c1'].append(0.02)
    settings['c2'].append(0.02)
    settings['wcc'].append(1000)
    settings['wcr'].append(1000)

    settings['max_energy'].append(int(1.0001e6))
    settings['eabs'].append(5000)
    settings['c1'].append(0.02)
    settings['c2'].append(0.02)
    settings['wcc'].append(5000)
    settings['wcr'].append(5000)

    #   Find settings to use, overulling defaults if supplied
    nei = np.asarray(
        energy <= np.array(settings['max_energy'])
        ).nonzero()[0][0]
    if not eabs:
        eabs = settings['eabs'][nei]
    if not c:
        c1 = settings['c1'][nei]
        c2 = settings['c2'][nei]
    else:
        c1 = c
        c2 = c
    if not wc:
        wcc = settings['wcc'][nei]
        wcr = settings['wcr'][nei]
    else:
        wcc = wc
        wcr = wc

    #   Penelope uses particle IDs
    particle_ids = {'electrons': 1, 'photons': 2, 'positrons': 3}
    particles_id = particle_ids[particles]

    #   Create new penelope.in file with these settings

    #   Initial default pentrack_in file image
    penelope_in = default_pentracks_in_file()

    for nl, line in enumerate(penelope_in):

        #   Primary particle
        if line[0:6].strip()=='SKPAR':
            penelope_in[nl] = (
                f'SKPAR  {particles_id:d}'
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
                + f'{eabs:0.0f} '
                + f'{eabs:0.0f} '
                + f'{eabs:0.0f} '
                + f'{c1:0.3f} '
                + f'{c2:0.3f} '
                + f'{wcc:0.0f} '
                + f'{wcr:0.0f}'
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
        'MSIMPA 50 50 50 0.010 0.010 100 100               '
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
