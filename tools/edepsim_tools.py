consts = {'E_FIELD': 0,
          'LAR_DENSITY': 0,
          'BOX_ALPHA': 0,
          'BOX_BETA': 0,
          'BIRKS_Ab': 0,
          'BIRKS_kb': 0,
          'W_ION': 0,
          }
          
def quench(tracks, mode = "BIRKS"):
    """
    Do a thing
    """
    # for itrk in tqdm(range(tracks.shape[0])):
    for itrk in range(tracks.shape[0]):

        if 'dEdx_ionizing' in tracks.dtype.names:
            dEdx = tracks[itrk]["dEdx_ionizing"]
            dE = tracks[itrk]["dE_ionizing"]
        else:
            dEdx = tracks[itrk]["dEdx"]
            dE = tracks[itrk]["dE"]

        recomb = 0
        if mode == "BOX":
            # Baller, 2013 JINST 8 P08005
            csi = consts['BOX_BETA'] * dEdx / (consts['E_FIELD'] * consts['LAR_DENSITY'])
            recomb = max(0, log(consts['BOX_ALPHA'] + csi)/csi)
        elif mode == "BIRKS":
            # Amoruso, et al NIM A 523 (2004) 275
            recomb = consts['BIRKS_Ab'] / (1 + consts['BIRKS_kb'] * dEdx / (consts['E_FIELD'] * consts['LAR_DENSITY']))
        else:
            raise ValueError("Invalid recombination mode: must be 'consts['BOX']' or 'consts['BIRKS']'")

        if isnan(recomb):
            raise RuntimeError("Invalid recombination value")

        tracks[itrk]["n_electrons"] = recomb * dE / consts['W_ION']

# insert segment data for a single event_id. trajectories can include all the trajectories or just those within the event
# division_size, origin_shift in mm
# origin_shift is relative change of origin of position basis (added to original x,y,z)
# use after running dE_to_dQ
# currently make_pickle just adds a placeholder metadata file.
# track energy in keV for the pickle file 
# simulation_extent in mm gives the volume to be passed in (size of detector; points outside will be ignored)
def h5_convert(segments, trajectories,
               filename = None,
               make_pickle = True, 
               division_size = 0.1,
               track_energy = 1E6,
               **kwargs):
    

    #check that all segments are from the same event
    assert((segments['event_id'] == segments[0]['event_id']).all())
    
    #arrays to fill in
    r_array          = np.empty((3,0))
    trackID_array    = np.array([])
    pdgID_array      = np.array([])
    n_elec_array     = np.array([])

    for track in segments:
        
        ### find number of subdivisions
        dL = track['dx'] #cm
        if dL <= division_size:
            N_sub = 1
        else:
            N_sub = math.floor(dL/division_size)
        
        ### calculate n_elec 
        n_electrons = track['n_electrons']
        if np.round(n_electrons) == 0:
            continue #skip if no electrons
        elif np.round(n_electrons/N_sub) == 0:
            while np.round(n_electrons/N_sub) == 0:
                N_sub -= 1 #don't subdivide so finely that there's no electrons per subsegment  
        n_elec_subs = np.full(N_sub, np.round(n_electrons/N_sub), dtype = np.int64)

        ### calculate r
        x_start = track['x_start'] + origin_shift[0] #mm
        y_start = track['y_start'] + origin_shift[1]
        z_start = track['z_start'] + origin_shift[2] #shift z to make negative 
    
        x_end = track['x_end'] + origin_shift[0]
        y_end = track['y_end'] + origin_shift[1]
        z_end = track['z_end'] + origin_shift[2]
        
        #ignore points outside of simulation_extent
        if simulation_extent:
            if x_start < simulation_extent[0][0] or x_start > simulation_extent[0][1] or \
                y_start < simulation_extent[1][0] or y_start > simulation_extent[1][1] or \
                z_start < simulation_extent[2][0] or z_start > simulation_extent[2][1] or \
                x_end < simulation_extent[0][0] or x_end > simulation_extent[0][1] or \
                y_end < simulation_extent[1][0] or y_end > simulation_extent[1][1] or \
                z_end < simulation_extent[2][0] or z_end > simulation_extent[2][1]:
                continue
            
        vec = np.array([x_end - x_start,
                        y_end - y_start,
                        z_end - z_start])     
        dvec = vec/N_sub

        r = np.transpose(  #initial position + half a division (since we want midpoints) + increments
                   np.array([x_start, y_start, z_start])
                 + dvec/2
                 + np.array([n*dvec for n in np.arange(0, N_sub)])
                         )/1000 #convert to m from mm
        
        ### find trackID
        trackID = track['traj_id']
        trackID_subs = np.full(N_sub, trackID)        
        
        ### find pdgID
        pdgID = track['pdg_id']
        pdgID_subs = np.full(N_sub, pdgID)
                
        ### add values
        r_array = np.concatenate((r_array, r), axis = 1)
        trackID_array = np.cast[np.int64](np.append(trackID_array, trackID_subs))
        pdgID_array = np.cast[np.int64](np.append(pdgID_array, pdgID_subs))
        n_elec_array = np.cast[np.int64](np.append(n_elec_array, n_elec_subs))
        
    #check r_array is not empty
    if r_array.shape[1] == 0:
        r_array = np.zeros((3,1))
        n_elec_array = np.ones(1, dtype = np.int64)
        trackID_array = np.zeros(1)
        pdgID_array = np.array([11])
    if filename:
        np.savez(filename + '.npz', 
                 r = r_array,
                 num_e = n_elec_array,
                 trackID = trackID_array,
                 pdgID = pdgID_array)
    
    #make pickle file with metadata
    if make_pickle:
        
        #find initial direction. TODO: this doesn't work for GENIE inputs that don't have the primary neutrino
        eventMask = trajectories['event_id'] == segments[0]['event_id']
        trackMask = trajectories['traj_id'] == 0 #first track
        initial_trajectory = trajectories[eventMask & trackMask]
        initial_direction = np.array(initial_trajectory['pxyz_start']/np.linalg.norm(
                                initial_trajectory['pxyz_start'])).reshape(3,1) #must be this shape
                                
        
        #track_extent: approximate size of simulated volume in cm
        track_extent = np.max([
            abs(max(r_array[0]) - min(r_array[0])),
            abs(max(r_array[1]) - min(r_array[1])),
            abs(max(r_array[2]) - min(r_array[2])),
                                 ])
        
        #num_electrons
        num_electrons = np.sum(n_elec_array)
        
        #origin
        origin = (np.array(initial_trajectory['xyz_start']) + np.array(origin_shift)).reshape(3,)/10 #origin currently in cm!
        
        attributes = {
            'truth': 
                {   'origin'            : origin,
                    'initial_direction' : initial_direction,
                    'num_electrons'     : num_electrons,
                    'track_energy'      : track_energy, #keV (1GeV)
                    'track_extent'      : track_extent}, 
            'meta': 
                {'penelope_input': 
                    {'raw_lines': 
                        [
                        'TITLE  Electrons in homogeneous cylinder. [PLACEHOLDER METADATA -- DUNE SIM]',
                        '       .',
                        'GSTART >>>>>>>> Beginning of the geometry definition list.',
                        'LAYER  -1e3 1e3                                 [Z_lower and Z_higher]',
                        'CYLIND 1 0 1e3                      [Material, R_inner and R_outer]',
                        'GEND   <<<<<<<< End of the geometry definition list.',
                        '       The labels KL,KC denote the KC-th cylinder in the KL-th layer.',
                        '       .',
                        '       >>>>>>>> Source definition.',
                        'SKPAR  1        [Primary particles: 1=electron, 2=photon, 3=positron]',
                        'SENERG 1500e3            [Initial energy (monoenergetic sources only)]',
                        'SPOSIT 0 0 0                 [Coordinates of the source center]',
                        'SCONE  0 0 0                             [Conical beam; angles in deg]',
                        '       .',
                        '       >>>>>>>> Material data and simulation parameters.',
                        '                Up to MAXMAT materials; 2 base_file for each material.',
                        'MFNAME LAr.mat                         [Material file, up to 20 chars]',
                        'MSIMPA 500 500 500 0 0 200 200                    [EABS(1:3),C1,C2,WCC,WCR]',
                        '       .',
                        '       >>>>>>>> Local maximum step lengths and absorption energies.',
                        'DSMAX  1 1 1.0D35                  [Mmaximum step length in body KL,KC]',
                        '       .',
                        '       >>>>>>>> Job properties.',
                        'RESUME dump.dat                [Resume from this dump file, 20 chars]',
                        'DUMPTO dump.dat                   [Generate this dump file, 20 chars]',
                        'DUMPP  60                                    [Dumping period, in sec]',
                        '       .',
                        'RSEED  296588484 69159203      [Seeds of the random-number generator]',
                        'NSIMSH 50                       [Desired number of simulated showers]',
                        'TIME   12000                       [Allotted simulation time, in sec]',
                        'END                                  [Ends the reading of input data]'
                        ],
                    'electron_energy': track_energy, #keV (1GeV)
                    'r_o': np.array([0., 0., 0.]),
                    's_o': np.array([ 0.53138345, -0.70392913, -0.47129122]),
                    'eabs': np.array([500., 500., 500.]),
                    'c1': 0.0,
                    'c2': 0.0,
                    'wcc': 200.0,
                    'wcr': 200.0
                    },
                'simulation_date': '   Date and time: 21th Apr 2023. 16:41:19\n',
                'file_name': filename
                }
            }

        if filename:
            #save pickle file
            pickle.dump(attributes, open(filename + '.pickle', 'wb'))  

    return r_array, n_elec_array, pdgID_array, attributes
