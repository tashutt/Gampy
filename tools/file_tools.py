#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Colletion of tools to read .sim file and crate events

Parsing has long list of problems / needed improvements - see comments

TODO: Move any file file handling out of events_tools and to here
TODO: Consider splitting sim file tools from other file tools

@author: tshutt
awkward implemented
calorimeter implemented
"""

class Sim_File:
    """"
    Class for sim file handling

    At initialization, opens file and reads up to first event.

    Subsequent calls to read_next_event reads the next event and returns it
    in a raw form as self.raw_event.  parse_raw_event creates hits
    from the events in the .ht and .ia lines, returning self.parsed_event

    make_event_tree and show_event_tree are for diagnostics

    Units from Megablib: cm and keV
    """

    def __init__(self, full_file_name):
        """ Opens a sim file and reads to the first event.
            Returns file "handle" f

            full_file_name includes path, but not extension

            TODO: parse and return header information
            8/20 TS
        """
        #   Open
        f = open(full_file_name)

        #   Skip to first event - text_line = 'SE'
        text_line = f.readline().strip('\n')

        while text_line=="" or text_line[0:2]!='SE':
            text_line = f.readline().strip('\n')
            print(text_line)

        self.f = f

    def read_next_event(self):
        """
        Reads the next event in the file, and returns all of
        the raw information in the dictionary self.raw_event

        Assumes that the file has previously be read up to and including
        the 'SE' line demarking the start of this event, and reads the
        next event 'SE' before returning.  If no next 'SE' is encountered,
        thensim['end_of_file'] is set True.

        There is minimal processing of the raw information.  The
        Megalib units of cm and keV are preserved.
        """

        import numpy as np
        from collections import deque

        #   Initial presumption of innocence
        self.good_event = True

        #   Clear any previous parsed event
        if hasattr(self, 'parsed_event'):
            delattr(self, 'parsed_event')

        #   Particle and Interction types are meta data

        #   All interaction types, first as name to number
        interaction_types_str2num = {}
        interaction_types_str2num['INIT'] = 1
        interaction_types_str2num['PAIR'] = 2
        interaction_types_str2num['COMP'] = 3
        interaction_types_str2num['PHOT'] = 4
        interaction_types_str2num['BREM'] = 5
        interaction_types_str2num['ANNI'] = 6
        interaction_types_str2num['RAYL'] = 7
        interaction_types_str2num['IONI'] = 8
        interaction_types_str2num['INEL'] = 9
        interaction_types_str2num['CAPT'] = 10
        interaction_types_str2num['DECA'] = 11
        interaction_types_str2num['ESCP'] = 12
        interaction_types_str2num['ENTR'] = 13
        interaction_types_str2num['EXIT'] = 14
        interaction_types_str2num['BLAK'] = 15
        #   Number to name, dealing with zero start index
        interaction_types_num2str = ['null']
        interaction_types_num2str[1:] \
            = [key for key in interaction_types_str2num.keys()]

        #   Select particles types
        particle_num2str = [''] * 27
        particle_num2str[1] = 'gamma'
        particle_num2str[2] = 'e+'
        particle_num2str[3] = 'e-'
        particle_num2str[4] = 'p'
        particle_num2str[5] = 'pbar'
        particle_num2str[6] = 'n'
        particle_num2str[8] = 'mu+'
        particle_num2str[9] = 'mu-'
        particle_num2str[21] = 'alpha'
        particle_num2str[22] = 'ion'
        particle_num2str[23] = 'pi+'
        particle_num2str[24] = 'pio'
        particle_num2str[25] = 'pi-'

        meta = {}
        meta['interaction_types_str2num'] = interaction_types_str2num
        meta['interaction_types_num2str'] = interaction_types_num2str
        meta['particle_num2str'] = particle_num2str

        #   event_info is essentially meta data for the event
        event_info = {}

        #   Event ID
        text_line = self.f.readline().strip('\n')
        numstring = text_line[2:].split()
        event_info['triggered_event_id'] = int(numstring[0])
        event_info['simulated_event_id']= int(numstring[1])

        if event_info['triggered_event_id'] == 6861:
            print(event_info['triggered_event_id'])

        #   Event time
        text_line = self.f.readline().strip('\n')
        event_info['time'] = float(text_line[2:])

        #   Energies: active, escaped and passive energy
        text_line = self.f.readline().strip('\n')
        event_info['deposited_energy'] = float(text_line[2:])
        text_line = self.f.readline().strip('\n')
        event_info['escaped_energy'] = float(text_line[2:])
        text_line = self.f.readline().strip('\n')
        event_info['passive_energy'] = float(text_line[2:])

        #   Specific passive material - PM
        #   The type of material is recored, but I'm not trackign that
        #   yet
        text_line = self.f.readline().strip('\n')
        event_info['energy_special_passive'] = []
        while text_line[0:2]=='PM':
            splitline=text_line[2:].split()
            event_info['energy_special_passive'].append(float(splitline[1]))
            text_line = self.f.readline().strip('\n')

        #   The event is returned as the dictionary raw_event
        raw_event = {}
        raw_event['meta'] = meta
        raw_event['event_info'] = event_info

        #%%   IA - interaction lines.  First initialize things
        ia = {}

        #   Initialize values - appended in loop below
        ia['interaction_type'] = deque()
        ia['interaction_id'] = deque()
        ia['parent_interaction_id'] = deque()
        ia['detector'] = deque()
        ia['time'] = deque()
        ia['rx'] = deque()
        ia['ry'] = deque()
        ia['rz'] = deque()
        ia['particle_primary'] = deque()
        ia['s_primary_x'] = deque()
        ia['s_primary_y'] = deque()
        ia['s_primary_z'] = deque()
        ia['polarization_in_x'] = deque()
        ia['polarization_in_y'] = deque()
        ia['polarization_in_z'] = deque()
        ia['energies_primary'] = deque()
        ia['particle_secondary'] = deque()
        ia['s_secondary_x'] = deque()
        ia['s_secondary_y'] = deque()
        ia['s_secondary_z'] = deque()
        ia['polarization_out_x'] = deque()
        ia['polarization_out_y'] = deque()
        ia['polarization_out_z'] = deque()
        ia['energies_secondary'] = deque()

        #%%   Read IA lines and put into ia structure
        while text_line[0:2]=='IA':

            ia['interaction_type'].append(
                interaction_types_str2num[text_line[3:7]]
                )

            splitline = text_line[8:].split(';')

            ia['interaction_id'].append(int(splitline[0]))
            ia['parent_interaction_id'].append(int(splitline[1]))
            ia['detector'].append(int(splitline[2]))
            ia['time'].append(float(splitline[3]))
            ia['rx'].append(float(splitline[4]))
            ia['ry'].append(float(splitline[5]))
            ia['rz'].append(float(splitline[6]))
            ia['particle_primary'].append(int(splitline[7]))
            ia['s_primary_x'].append(float(splitline[8]))
            ia['s_primary_y'].append(float(splitline[9]))
            ia['s_primary_z'].append(float(splitline[10]))
            ia['polarization_in_x'].append(float(splitline[11]))
            ia['polarization_in_y'].append(float(splitline[12]))
            ia['polarization_in_z'].append(float(splitline[13]))
            ia['energies_primary'].append(float(splitline[14]))
            ia['particle_secondary'].append(int(splitline[15]))
            ia['s_secondary_x'].append(float(splitline[16]))
            ia['s_secondary_y'].append(float(splitline[17]))
            ia['s_secondary_z'].append(float(splitline[18]))
            ia['polarization_out_x'].append(float(splitline[19]))
            ia['polarization_out_y'].append(float(splitline[20]))
            ia['polarization_out_z'].append(float(splitline[21]))
            ia['energies_secondary'].append(float(splitline[22]))

            #   Read next text_line
            text_line = self.f.readline().strip('\n')

            #   If eof or other file error, break
            if text_line=='':
                break

        # Convert deques to numpy arrays
        for key in ia:
            ia[key] = np.array(ia[key])

        raw_event['ia'] = ia

        #   Check for no IA line after the first - have seen this in .sim files
        if len(ia['interaction_id'])==1:
            self.good_event = False
            print(f'Event {event_info["triggered_event_id"]:d} bad: ' +
                  'has only one IA line')

        #%%   Read HT (hit) lines and put into ht structure

        #   If eof or other error, return
        if text_line != '':

            ht = {}

            #   Initialize values - appended in loop below
            ht['detector'] = deque()
            ht['cell'] = deque()
            ht['rx'] = deque()
            ht['ry'] = deque()
            ht['rz'] = deque()
            ht['energy'] = deque()
            ht['time'] = deque()
            ht['interaction_id'] = deque()
            ht['interactions_id'] = deque()


            #   Loop over HTsim lines
            n = 0
            while text_line[0:2] == 'HT':

                n += 1

                #   Split line into text sections
                splitline = text_line[6:].split(';')

                #   Detector ID
                detector_id = int(splitline[0])
                ht['detector'].append(detector_id)

                #   Some ugliness here for Oliver's hack (documented at:
                #   https://confluence.slac.stanford.edu
                #       /display/GTPC/MEGALib+notes)
                #   which added a volume tag to the .sim file, but only
                #   for drift chambers - detector_id = 5.
                #   For other detector_ids, set cell to 0.
                if detector_id==5:
                    ht['cell'].append(int(splitline[1].split('_')[1]))
                else:
                    ht['cell'].append(0)

                # if the detector id is 2 = calorimeter
                # AND the 7th index (interaction id) is a number less than 5
                # then that event should be flagged as a
                # bad calorimeter event... implemented in parse raw event


                ht['rx'].append(float(splitline[2]))
                ht['ry'].append(float(splitline[3]))
                ht['rz'].append(float(splitline[4]))
                ht['energy'].append(float(splitline[5]))
                ht['time'].append(float(splitline[6]))
                ht['interaction_id'].append(int(splitline[7]))
                ht['interactions_id'].append(
                    np.array(splitline[7:], dtype=int))

                text_line = self.f.readline().strip('\n')
                if text_line == '':
                    break

            #   Convert lists to arrays, apart from interactions_idss
            for key in ht:
                if key!='interactions_id':
                    ht[key] = np.array(ht[key])

            #   Assign HT to sim structure, if it exists
            if len(ht['detector'])>0:
                raw_event['ht'] = ht

            #   Otherwise, a bad event
            else:
                self.good_event = False
                print(f'Event {event_info["triggered_event_id"]:d} bad: ' +
                      'no HT lines')

        #   Check for end of file.  If text_line is 'SE', then no
        if text_line[0:2] == 'SE':
            end_of_file = False
        else:
            end_of_file = True
        raw_event['end_of_file'] = end_of_file

        #   Add raw_event to file object
        self.raw_event = raw_event

    def parse_raw_event(self, sims_params=None):

        """
        Parses the raw event information in self.raw_event, returning
        self.parsed_event.

        The information in the .sim is split between IA lines and HT lines.
        The IA lines describe the interactions, while the HT lines keep
        track of energy deposited by dE/dx "ionization".  Here we generate
        truth_hits which have the locations of interactions (from the IA
        lines) that result in charged particles (for gamma ray, this means
        compton, photo, pair), and the total energy of the resulting tracks,
        which is found by summing the energy HT lines corresponding to each
        IA line.
        The hits are listed in the order of the interactions - though
        note the comlication described below when tracks spans multiple cells.

        There are several complications to interperting IA and HT lines
        which are handled in this module:

        When X-rays are absorbed, this creates a separate (photo)
        interaction by Cosima, but given their tiny range, we here
        add their energy to the correct parent interaction or track that
        gave rise to the x-ray.
        This takes several forms, as is detailed in comments below.
        Similarly, low energy bremsstrahlung (defined with a
        threshold hard coded in this routine) are also added to their
        parent track.

        Another complication is tracks that span multiple cells.  This is
        is a promient effect for incident charged paritcles, but also
        happens with the electron recoil tracks from gamma ray interactions.
        In this case the first hit is the interaction that gives rise
        to the charged particle, and energy in additional cells is put into
        new "hits" at the end of the hits arrays.

        Note also that dE/dx ionizing energy deposition does not constitute
        an "interaction", but is associated with the interaction that
        generated the charged particle, e.g., a Compton scatter creating
        an electron.  For incident charged particles, this corresponding
        "interaction" id is 1, which is the event initialization. The type
        of incident particle is in events.truth['incident_particle'],
        for which events.meta['particle_num2str'] translates number
        to particle. Gammas are id = 1.
        events.truth['incident_charged_particle'] is a convenient boolean.

        The results of this routine are packaged in in parsed_event

        These variables keep track of mulit-cell tracks:
            'multi_cell_interaction', a boolean which
                is true for any interaction that splits into multiple cells
            'interaction' - the interaction ID in the .sim file.
                This value is the same for all hits generated by
                multi-cell tracks, and refers to the original interaction.
                multi_cell_interaction will be true only for that
                original interaction.
                In general 'interaction' is not a sequential list
                (1, 2, 3 ...), but has missing interactions since many
                IA line interactions have been droped because they had
                no energy (e.g., pair production) or were x-rays or bream
                whose energy was added to the parent interaction.

        These variables provide information about cells
            cell - cell # for each hit, size(max_num_hits, num_events)
            cells_list - list of struck cells
            num_cells - number of sturck cells,
            cell_index - hit index for cells_list
            cell_hit_count - number of hits in each cell

        The parsed event units are MKS and keV (Megalib's cm converted to m)

        Note the spatial information in HT lines and IA lines are not
            consistent - HT lines are in cell coordinates.

        TODO: Many issues to clean up:
            Capture polarizations
            Implement full (compressed) track output.  For incident
                charged particles, and/or tracks above some energy
            Two input parameters need to be handled: thresholds for
                x-ray and brem merging.
            Test on high energy particles with "interesting" interactions
                - i.e., muon, or pion creating
            Consider merging x-rays and brem with parent track based on
                spatial clustering instead of energy.  Could go further
                and use only clustering in this module.
        """

        import sys
        import numpy as np
        import copy

        import sims_tools

        if not 'ht' in self.raw_event:
            sys.exit('*** Error in parse_raw_event - no ht lines ***')

        #   These used for diagnostics
        blab = False
        fuss = False
        event_num = self.raw_event["event_info"]["triggered_event_id"]

        #   Settings for processing - should possibly be supplied params
        x_ray_energy_threshold = 4.0
        merged_brem_threshold = 10.0

        #   Incident particle ID, energy, location and direction
        #   extracted from first IA line.  The location - I think -
        #   is where the particle is on the disk tangent to the
        #   surrounding sphere:
        incident_particle = self.raw_event['ia']['particle_secondary'][0]
        incident_energy = self.raw_event['ia']['energies_secondary'][0]
        incident_r = np.zeros((3,))
        incident_r[0] = self.raw_event['ia']['rx'][0]
        incident_r[1] = self.raw_event['ia']['ry'][0]
        incident_r[2] = self.raw_event['ia']['rz'][0]
        incident_s = np.zeros((3,))
        incident_s[0] = self.raw_event['ia']['s_secondary_x'][0]
        incident_s[1] = self.raw_event['ia']['s_secondary_y'][0]
        incident_s[2] = self.raw_event['ia']['s_secondary_z'][0]

        decay_list = self.raw_event['ia']['interaction_type'] == 11 # decay
        inert_material_list = self.raw_event['ia']['detector'] == 0
        decay_in_inert_material = np.any(decay_list & inert_material_list)
        calorimeter_involved = self.raw_event['ia']['detector'] == 2
        happend_in_first_3   = self.raw_event['ia']['interaction_id'] < 5
        calorimeter_in_first_3 \
            = np.any(calorimeter_involved & happend_in_first_3)


        #   The deposited energy, as reported by Cosima
        deposited_energy = self.raw_event['event_info']['deposited_energy']

        #   These keep track of interactions.
        #   Usage notes for interactions and parents:
        #     + These are interaction numbers, which are one greater
        #       than their array indices, must subtract 1 when indexing.
        #     + There is not a one-to-one correspondence between
        #       parents and children, which prohibits certain types
        #       of vectorized indexing
        interactions = copy.copy(self.raw_event['ia']['interaction_id'])
        parents = self.raw_event['ia']['parent_interaction_id']
        types = self.raw_event['ia']['interaction_type']
        detectors = self.raw_event['ia']['detector']

        #   Types and detectors of parents need finesse, since the
        #   first entry, which is the init line, doesn't have a well
        #   defined parent.
        #   So we make new varibles, with first entry given value 0.
        parent_types = types[parents-1]
        parent_types[0] = 0

        parent_detectors = detectors[parents-1]
        parent_detectors[0] = 0

        #   Boolean for incident charged particle.  For now, everything
        #   except gammas and neutrons are charged particles
        if ((incident_particle == 1) | (incident_particle == 6)):
            incident_charged_particle = False
        else:
            incident_charged_particle = True

        s_primary = np.zeros((3, interactions.size))
        s_primary[2,:] = self.raw_event['ia']['s_primary_z'][interactions-1]

        #   Here we find various measures of event quality

        #   Keep track of escaped energy depending on direction
        #   up or down (through or back-scatter, assuming planar geometry)
        escaped = self.raw_event['ia']['interaction_type']==12
        back = (
            self.raw_event['ia']['s_secondary_z'][0]
            * self.raw_event['ia']['s_primary_z'][escaped] < 0
            )
        escaped_back_energy = np.sum(
            self.raw_event['ia']['energies_primary'][escaped][back]
            )
        escaped_through_energy = np.sum(
            self.raw_event['ia']['energies_primary'][escaped][~back]
            )

        #   "Entrance scatter" for gammas: initial scatter in inert material
        if (incident_particle==5) & (detectors[0]==0):
            clean_entrance = False
            entrance_scatter_energy \
                = self.raw_event['ia']['energies_secondary'][1]
        else:
            clean_entrance = True
            entrance_scatter_energy = 0

        #   Missing energy - escaped or passive, above 1 meV
        if (self.raw_event['event_info']['escaped_energy']
            + self.raw_event['event_info']['passive_energy']) > 0.001:
            missing_energy = True
        else:
            missing_energy = False

        #   Missing energy after 1st scatter, independent of
        #   whether first scatter was in inert material (i.e., whether
        #   there was an  entrance scatter)
        if (np.any(detectors[1:]==0)
            | (self.raw_event['event_info']['escaped_energy'] > 0.001)):
            missing_energy_after_entrance = True
        else:
            missing_energy_after_entrance = False

        #   Number of Rayleigh scatters, and number of bremstrahlung
        num_rayleigh = np.sum(types==7)
        num_brem = np.sum(types==5)

        #   Find energy in cells.   This is done by looping over all
        #   interactions that created particles which deposit energy via
        #   dE/dx, and summing the energy in HT lines associated with those
        #   inteactions.  This gets messy to deal with tracks depositing
        #   energy over multiple cells.

        #   This restricts interactions to those that make HT energy.
        #   HT energy only comes from pair, compton, photo, and incident
        #   charged particles.
        #   This mask doesn't insist on active material, as the track
        #   can enter an active region, and does not distinguish
        #   detector types (e.g., cells vs calorimeter)

        # interaction_types_str2num['INIT'] = 1
        # interaction_types_str2num['PAIR'] = 2
        # interaction_types_str2num['COMP'] = 3
        # interaction_types_str2num['PHOT'] = 4
        # interaction_types_str2num['BREM'] = 5
        # interaction_types_str2num['ANNI'] = 6
        # interaction_types_str2num['RAYL'] = 7
        # interaction_types_str2num['IONI'] = 8
        # interaction_types_str2num['INEL'] = 9
        # interaction_types_str2num['CAPT'] = 10
        # interaction_types_str2num['DECA'] = 11
        # interaction_types_str2num['ESCP'] = 12
        # interaction_types_str2num['ENTR'] = 13
        # interaction_types_str2num['EXIT'] = 14
        # interaction_types_str2num['BLAK'] = 15

        interaction_mask = (
            ((types == 1) & incident_charged_particle)
            | (types == 2)
            | (types == 3)
            | (types == 4)
            #| (types == 11) # Bahrudin added
            )

        #   Arrays for energies and cell(s) of each interaction
        energies = np.zeros(len(interactions), dtype=float)
        cells = np.zeros(len(interactions), dtype=int)

        #   These needed for mulit-cell interactions
        splits_count = np.zeros(len(interactions), dtype=int)
        split_cells = [[] for n in range(types.size)]
        split_energies = [[] for n in range(types.size)]
        split_rs = [[] for n in range(types.size)]

        #   Loop through relevant interactions.  Note the
        #   subtraction to get to indices.
        for ni in (interactions[interaction_mask]-1):

            #   HT hits: this interation, and in cells. Note: HT cell
            #   number set to zero if detector is not a cell.
            ht_mask = (
                (self.raw_event['ht']['interaction_id']==interactions[ni])
                & (self.raw_event['ht']['cell'] != 0)
                )

            #   Only proceed if there is HT energy
            if np.any(ht_mask):

                #   List of cells for this interaction, and count
                #   of that list
                all_cells \
                    = np.unique(self.raw_event['ht']['cell'][ht_mask])
                splits_count[ni] = all_cells.size

                #   Single cell interactions
                if all_cells.size==1:
                    cells[ni] = all_cells
                    energies[ni] \
                        = sum(self.raw_event['ht']['energy'][ht_mask])

                #   Multi cell interactions take a lot of work.
                #   Store output in separate arrays, to be merged later.
                #   The energy location taken from HT lines, which are in
                #   cell coordinates
                else:
                    int_cells = []
                    int_energies = []
                    int_r = []
                    for cell, nc in zip(all_cells, range(all_cells.size)):
                        ht_cell_mask = (
                            ht_mask
                            & (self.raw_event['ht']['cell'] == cell)
                            )
                        int_cells.append(cell)
                        ht_energies \
                            = self.raw_event['ht']['energy'][ht_cell_mask]
                        int_energies.append(ht_energies.sum())
                        r_raw = np.array([
                            self.raw_event['ht']['rx'][ht_cell_mask],
                            self.raw_event['ht']['ry'][ht_cell_mask],
                            self.raw_event['ht']['rz'][ht_cell_mask]
                            ]) / 100
                        r_weighted = np.average(
                            r_raw,
                            axis=1,
                            weights=ht_energies)
                        r_global = sims_tools.global_to_cell_coordinates(
                            r_weighted,
                            cell,
                            sims_params,
                            reverse = True
                            )
                        int_r.append(r_global)
                    split_cells[ni] = int_cells
                    split_energies[ni] = int_energies
                    split_rs[ni] = int_r

                    #   We want the cell with the interaction to be
                    #   first in the split cells list, as this is the
                    #   cell that gets assigned to the original location
                    #   in the interaction arrays.  We do this by finding
                    #   which cell has its center closest to the interaction.
                    #   This doesn't apply to initial charged particles, which
                    #   don't have an interaction.  Possibly the natural
                    #   order of the HT lines makes this distance check
                    #   unnecessary.
                    #   TODO: check if distance test is needed, and remove
                    #   if not
                    if ni != 0:
                        r_int = np.zeros(3, dtype=float)
                        r_int[0] = self.raw_event['ia']['rx'][ni]
                        r_int[1] = self.raw_event['ia']['ry'][ni]
                        r_int[2] = self.raw_event['ia']['rz'][ni]
                        r_int = r_int / 100
                        interaction_cell = (all_cells[np.sqrt((((
                            sims_params.cells['centers']
                            [:, all_cells-1].T - r_int).T
                            )**2).sum(axis=0)).argmin()])
                        split_cells[ni].remove(interaction_cell)
                        split_cells[ni].insert(0, interaction_cell)
                        if interaction_cell == 0:
                            print('Correct interaction split cell, '
                                  + f'event {event_num:d}')
                        else:
                            print('Incorrect interaction split cell, '
                                  + f'event {event_num:d}')
                        if fuss:
                            print('Multi cell interaction, '
                                     + 'not from first track '
                                     + f'event {event_num:d} '
                                     + 'Need to check handling ')

        #   These are original interactions which are multi-cell
        multi_cell_interactions_mask = splits_count > 1

        #   ACDs.
        #   Calorimeter names not working in Cosima, so a workaround,
        #   using the z location to figure out which ACD detector
        front_acd_energy = 0.0
        back_acd_energy = 0.0
        ht_acd_mask = self.raw_event['ht']['detector']==4
        for nh in np.nonzero(ht_acd_mask)[0]:
            if float(self.raw_event['ht']['rz'][nh])>0:
                front_acd_energy += self.raw_event['ht']['energy'][nh]
            else:
                back_acd_energy += self.raw_event['ht']['energy'][nh]

        calorimeter_energy = 0.0
        ht_cal_mask = self.raw_event['ht']['detector']==2
        for nh in np.nonzero(ht_cal_mask)[0]:
            calorimeter_energy += self.raw_event['ht']['energy'][nh]

        #TODO: get rid of this when finished checking
        def blab_first():
            split_energy_sum = sum([energy for split in split_energies
                        for energy in split])
            acd_energy = front_acd_energy + back_acd_energy
            all_energy = energies.sum() + split_energy_sum + acd_energy
            print('Initial')
            print(f'   single cells: {energies.sum():5.3f} keV')
            print(f'   in scatters: {energies[scatters_in_mask].size:d}, '
                  + f'{energies[scatters_in_mask].sum():5.3f} keV')
            print(f'   split cells: {split_energy_sum:5.3f} keV')
            print(f'   acd front & back: {acd_energy:5.3f} keV')
            print(f'   Total: {all_energy:5.3f} keV')

        def blab_on(headline):
            split_energy_sum = sum([energy for split in split_energies
                        for energy in split])
            acd_energy = front_acd_energy + back_acd_energy
            all_energy = energies.sum() + split_energy_sum + acd_energy
            print(headline)
            print(f'   single cells: {energies.sum():5.3f} keV')
            print(f'      child_x_rays: {energies[child_x_rays-1].size:d}, '
                  + f'{energies[child_x_rays-1].sum():5.3f} keV')
            if 'following_x_rays_mask' in locals():
                print('      following_x_rays: '
                      + f'{following_x_rays_mask.sum():d}, '
                      + f'{energies[following_x_rays_mask].sum():5.3f} keV')
            if 'merged_brem_photo' in locals():
                print('      merged_brem_photo: '
                      + f'{energies[merged_brem_photo-1].size:d}, '
                      + f'{energies[merged_brem_photo-1].sum():5.3f} keV')
            print(f'   split cells: {split_energy_sum:5.3f} keV')
            print(f'   acd front & back: {acd_energy:5.3f} keV')
            print(f'   Total: {all_energy:5.3f} keV')

        def blab_last():
            split_energy_sum = sum([energy for split in split_energies
                        for energy in split])
            acd_energy = front_acd_energy + back_acd_energy
            all_energy = energies.sum() + split_energy_sum + acd_energy
            print('')
            print('Cell energies:')
            print(f'   init: {energies[types==1].size:d}, '
                  + f'{energies[types==1].sum():5.3f} keV')
            print(f'   pair: {energies[types==2].size:d}, '
                  + f'{energies[types==2].sum():5.3f} keV')
            print(f'   comp: {energies[types==3].size:d}, '
                  + f'{energies[types==3].sum():5.3f} keV')
            print(f'   phot: {energies[types==4].size:d}, '
                  + f'{energies[types==4].sum():5.3f} keV')
            print(f'   brem: {energies[types==5].size:d}, '
                  + f'{energies[types==5].sum():5.3f} keV')
            print(f'   anni: {energies[types==6].size:d}, '
                  + f'{energies[types==6].sum():5.3f} keV')
            print(f'   total: {energies.size:d}, {energies.sum():5.3f} keV')
            print(f'   split cell energies: {split_energy_sum:5.3f} keV')
            print( '   sum energies + split cell: '
                  + f'{energies.sum()+split_energy_sum:5.3f} keV')
            print('ACDs')
            print(f'   front: {front_acd_energy:5.3f} keV')
            print(f'   back: {back_acd_energy:5.3f} keV')
            print(f'   total: {acd_energy:5.3f} keV')
            print(f'All energy: {all_energy:5.3f} keV')

        if blab: blab_first()

        #   We now have to sort out x-rays and low energy brem that should
        #   be merged with the parent tracks or interactions.  This is
        #   sadly complicated.
        #
        #   For context, these interactions produce gammas:
        #       bremsstrahlung
        #       annihilation
        #       an incident gammma
        #   These interactions produce charged particles:
        #       pair production
        #       compton
        #       photo-absorption
        #       an incident charged particle
        #
        #   X-rays are produced both at a photo-absorption interactions,
        #       by the photo-absorbing atom,
        #       and also along charged particle tracks. Note that for
        #       incident charged particles the "interaction" is 1, the
        #       inititalization interaction.
        #
        #   Note that for Cosima the secondary gamma in compton scatters
        #       is the same as the initial.  Hence the only "daughter"
        #       of compton scatterings is e-, and not a gamma.
        #
        #   With that said, the are three types of low energy gamma rays
        #   to combine with parent interactions or tracks:
        #
        #   1. x-rays which are children of charged-particle producing
        #       interaction.  Note, this doesn't distinguish between
        #       x-rays produced along the resulting track, or produced
        #       at the interaction site.  The energy is assigned to parent
        #       interaction, as long as the parent interaction was in
        #       an active material (otherwise the X-ray remains a separate
        #       interaction)
        #   2. x-rays which sequentially follow a photo-absorption, but
        #       are not tagged as their children - perhaps erroneously.
        #       These have the same parent as photo-absorption, and
        #       that parent is a photon (to distinguish this
        #       from x-rays from charged particle tracks).
        #       Energy is added to the photo-absorption interaction.
        #   3. photo-abosorption after brem at sufficiently low energy
        #       that it should be merged with track that procuced brem.
        #       Assign to the interaction which generated the track, i.e.,
        #       the grandparent interaction of the photo-absorption.
        #
        #   Interaction codes:
        #       INIT = 1
        #       PAIR = 2
        #       COMP = 3
        #       PHOT = 4
        #       BREM = 5
        #       ANNI = 6

        #   Child x-rays are defined as:
        #      + energy less than pre-defined threshold, but > 0
        #      + photoabsorption
        #      + parent detectors is active - otherwise is orphaned x-ray
        child_x_rays = interactions[
            (0.0 < energies) & (energies < x_ray_energy_threshold)
            & (types == 4)
            & (   (parent_types == 2)
                | (parent_types == 3)
                | (parent_types == 4)
                | ((parent_types == 1) & (incident_particle != 1)))
            & (parent_detectors != 0)
            ]

        #   Add energy to track by looping over parents.  Deal with
        #   split cell parents
        for npt in np.unique(parents[child_x_rays-1]):
            if ~multi_cell_interactions_mask[npt-1]:
                energies[npt-1] += energies[child_x_rays[
                    parents[child_x_rays-1] == npt
                    ]-1].sum()
            else:
                for cell, nc in zip(split_cells[npt-1],
                                    range(len(split_cells[npt-1]))):
                    split_energies[npt-1][nc] += energies[child_x_rays[
                        (parents[child_x_rays-1] == npt)
                        & (cells[child_x_rays-1] == cell)
                        ]-1].sum()

        if blab: blab_on('child x rays added')

        #   Zero the merged energy
        energies[child_x_rays-1] = 0

        if blab: blab_on('child x rays zeroed')

        #   Following x-rays:
        #       + energy less than pre-defined threshold, but > 0
        #       + photoabsorption
        #       + follows photoabsorption in interaction sequence
        #       + this and previous interactions share same parent
        #       + that parent is gamma from brem, annihilation or
        #           incident gamma.
        following_x_rays_mask = (
            (0.0 < energies) & (energies < x_ray_energy_threshold)
            & (types == 4)
            & (np.insert(types[:-1], 0, 0) == 4)
            & np.insert(parents[0:-1] == parents[1:], 0, False)
            & (   (parent_types == 5)
                | (parent_types == 6)
                | ((parent_types == 1) & (incident_particle == 1)))
            )
        #   Add these energies to parent.  Those with single cell
        #   previous interaction are straightforward
        single_followers = interactions[1:][
            ~multi_cell_interactions_mask[0:-1]
            & following_x_rays_mask[1:]
            ]
        energies[single_followers-2] += energies[single_followers-1]
        #   Multi-cell previous interaction is messier - needs to be checked,
        #   hence sys.exit if encountered.
        multi_followers = interactions[1:][
            multi_cell_interactions_mask[0:-1]
            & following_x_rays_mask[1:]
            ]
        for nfp in multi_followers:
            for cell, nc in zip(split_cells[nfp-1],
                                range(len(split_cells[nfp-1]))):
                if cells[nfp-1] == cell:
                    split_energies[nfp-1][nc] += energies[nfp-1]
            sys.exit('Following x-ray parent has split cell - check this')

        if blab: blab_on('following x rays added')

        #   Zero the merged energy
        energies[following_x_rays_mask] = 0

        if blab: blab_on('following x rays added')

        #   Photo absorbed brem: low energy photobsorption from
        #   parent brem, gets added to originating track.
        #   These are identified by energy, by type, and parent type.
        merged_brem_photo_mask = (
            (0.0 < energies) & (energies < merged_brem_threshold)
            & (types == 4)
            & (parent_types == 5)
            )
        merged_brem_photo = interactions[merged_brem_photo_mask]
        #   Originating track is grandparent of the photo-absorption.
        #   Add energy to track by looping over grandparents.
        #   Also need to deal with very rare case of grand parent not
        #   having any HT lines - if so, then there is no track to
        #   merge with, so remove from this treatment
        bad = np.zeros(merged_brem_photo.size, dtype=bool)
        for ngp in np.unique(parents[parents[merged_brem_photo-1]-1]):
            gp_has_hits = (self.raw_event['ht']['interaction_id']
                           == interactions[ngp-1]).sum()
            if gp_has_hits > 0:
                if ~multi_cell_interactions_mask[ngp-1]:
                    energies[ngp-1] += energies[merged_brem_photo[
                        parents[parents[merged_brem_photo-1]-1] == ngp
                        ]-1].sum()
                else:
                    for cell, nc in zip(split_cells[ngp-1],
                                        range(len(split_cells[ngp-1]))):
                        split_energies[ngp-1][nc] \
                            += energies[merged_brem_photo[
                                (parents[parents[merged_brem_photo-1]-1]==ngp)
                                & (cells[merged_brem_photo-1] == cell)
                                ]-1].sum()
            else:
                bad[parents[parents[merged_brem_photo-1]-1]==ngp] = True

        #   Remove the rare case
        merged_brem_photo = merged_brem_photo[~bad]

        if blab: blab_on('following x rays added')

        #   Zero the merged energy
        energies[merged_brem_photo-1] = 0

        if blab: blab_on('following x rays added')

        #TODO: remove these checks after testing
        if energies[types==5].sum()>0:
            sys.exit('ERROR: Brem with energy ')
        if energies[types==6].sum()>0:
            sys.exit('ERROR: Annihilation with energy ')
        if energies[types==7].sum()>0:
            sys.exit('ERROR: Rayleigh with energy ')

        #   Remove "null" interactions.  These have no energy at this
        #   this point, either becuase of the type of interaction
        #   (brem, annihilation, rayleigh, etc.), or because of merging
        #   (x-rays), or in passive material.  Also, don't remove location
        #   of split cell interactions, though these are currently empty
        remove = (energies==0) & ~(splits_count>1)

        #   Remove these interactions
        interactions = interactions[~remove]
        types = types[~remove]
        energies = energies[~remove]
        cells = cells[~remove]
        multi_cell_interactions_mask = multi_cell_interactions_mask[~remove]
        detectors = detectors[~remove]

        if blab: blab_last()

        #   Now, with reduced set of interactions to keep, find
        #   locations and vectors for these interactions
        r = np.zeros((3, interactions.size), dtype=float)
        r[0, :] = self.raw_event['ia']['rx'][interactions-1]
        r[1, :] = self.raw_event['ia']['ry'][interactions-1]
        r[2, :] = self.raw_event['ia']['rz'][interactions-1]
        r = r / 100
        s_primary = np.zeros_like(r)
        s_primary[0, :] \
            = self.raw_event['ia']['s_primary_x'][interactions-1]
        s_primary[1, :] \
            = self.raw_event['ia']['s_primary_y'][interactions-1]
        s_primary[2, :] \
            = self.raw_event['ia']['s_primary_z'][interactions-1]
        s_secondary = np.zeros_like(r)
        s_secondary[0, :] \
            = self.raw_event['ia']['s_secondary_x'][interactions-1]
        s_secondary[1, :] \
            = self.raw_event['ia']['s_secondary_y'][interactions-1]
        s_secondary[2, :] \
            = self.raw_event['ia']['s_secondary_z'][interactions-1]

        #   Scatters in are interactions where the primary interaction
        #   is in passive material, but energy is deposited in active region
        #   These are defined as:
        #       + non zero energy from HT lines
        #       + interaction in passive material
        #       + not initial interaction, which can be charged particle
        #   Their energy and any adjusts due to x-rays should have
        #   been correctly handled so far.  But the interacdtion location
        #   cannot be taken from the passive material.  Here we assign
        #   it to the location of the first HT line for this interacion.
        scatters_in_mask = (
            (energies>0) & (detectors==0) & (interactions>1)
            )
        for nsi in np.nonzero(scatters_in_mask)[0]:
            #   HT hits: this interation, and in cells
            if fuss:
                print(f'Scatter in, should be checked, event {event_num:d}')
            ht_mask = (
                (self.raw_event['ht']['interaction_id']==interactions[nsi])
                & (self.raw_event['ht']['cell'] != 0)
                )
            print(f'interaction: {interactions[nsi]:d}')
            print(f'ht_mask.sum: {ht_mask.sum():d}')
            r_cell = np.array([
                self.raw_event['ht']['rx'][np.nonzero(ht_mask)[0][0]],
                self.raw_event['ht']['ry'][np.nonzero(ht_mask)[0][0]],
                self.raw_event['ht']['rz'][np.nonzero(ht_mask)[0][0]]
                ]) / 100
            r_global = sims_tools.global_to_cell_coordinates(
                r_cell,
                cells[nsi],
                sims_params,
                reverse = True
                )
            r[:, nsi] = r_global

        #   Add split off part of multi cell interaction to end of
        #   arrays.

        #   Need to remember this below
        pre_append_length = interactions.size

        #   Numbers needed for enlargement
        num_split_interactions = (splits_count>1).sum()
        num_split_cells = len([cell for splits in split_cells
                           for cell in splits])
        add_num = num_split_cells - num_split_interactions
        int_zeros = np.zeros(add_num, dtype=int)
        bool_zeros = np.zeros(add_num, dtype=bool)
        float_zeros = np.zeros(add_num, dtype=float)

        #   Enlarge arrays
        interactions = np.append(interactions, int_zeros)
        types = np.append(types, int_zeros)
        energies = np.append(energies, float_zeros)
        r = np.append(
            r,
            np.zeros((3, add_num), dtype=float),
            axis=1
            )
        s_primary = np.append(
            s_primary,
            np.zeros((3, add_num), dtype=float),
            axis=1
            )
        s_secondary = np.append(
            s_secondary,
            np.zeros((3, add_num), dtype=float),
            axis=1
            )
        cells = np.append(cells, int_zeros)
        multi_cell_interactions_mask \
            = np.append(multi_cell_interactions_mask, bool_zeros)
        scatters_in_mask \
            = np.append(scatters_in_mask, bool_zeros)

        #   Loop over multi-cell interactions, and add split cells
        ni = pre_append_length - 1
        for nmi in interactions[multi_cell_interactions_mask]:

            #   Index of interaction nmi in shorted arrays
            nsi = np.nonzero(interactions==nmi)[0][0]

            #   First cell goes in original location
            cells[nsi] = split_cells[nmi-1][0]
            energies[nsi] = split_energies[nmi-1][0]
            #   Use split r for location only if initial charged particle
            if (nmi==1) & incident_charged_particle:
                r[:, nsi] = split_rs[nmi-1][0]

            #   Now append remaining cells to end of arrays
            for nc in range(1, splits_count[nmi-1]):

                ni += 1

                cells[ni] = split_cells[nmi-1][nc]
                energies[ni] = split_energies[nmi-1][nc]
                r[:, ni] = split_rs[nmi-1][nc]

                types[ni] = types[nsi]
                s_primary[:, ni] = s_primary[:, nsi]
                s_secondary[:, ni] = s_secondary[:, nsi]

                interactions[ni] = interactions[nsi]

        #   Create track energy, though this is contained in eneriges.
        #   Note that there should be no energy
        #   in the first interaction for incident_partile == 1 (gammas).
        track_energy = energies[interactions==1].sum()
        if (incident_particle == 1) and (track_energy > 0):
            sys.exit(f'Error, event {event_num:d}: erroneous track energy')

        #   List of cells with hits (length is <= number of hits),
        #   index into this list for each hit (length = number of hits),
        #   and number of counts for each cell struck
        cells_list, cell_index, cell_hit_count \
            = np.unique(cells, return_inverse=True, return_counts=True)

        #   Number of cells hit per event
        num_cells = cells_list.size

        #   Total energy directly from HT lines
        total_energy = self.raw_event['ht']['energy'].sum()

        #   Final checks on energy.
        energy_difference =  abs(total_energy - (
            energies.sum()
            + front_acd_energy
            + back_acd_energy
            + calorimeter_energy
            ))
        if abs(energy_difference) > 1:
            print(f'Warning, event {event_num:d}: ht sum energy'
                + 'differs from hits + acd + caoloritmeter by '
                + f'{energy_difference:5.2f} keV, '
                + f'with initial energy {total_energy:5.2f}' )
        if abs(deposited_energy - total_energy) / deposited_energy > 1e-4:
            print(f'Warning, event {event_num:d}: cosima "deposited" energy'
                + 'differs from hits + acd + caoloritmeter by 1e-4, '
                + f'with initial energy {total_energy:5.2f}' )

        #   Package output
        parsed_event = {}

        #   Per event information
        parsed_event['incident_energy'] = incident_energy
        parsed_event['incident_particle'] = incident_particle
        parsed_event['incident_charged_particle'] \
            = incident_charged_particle
        parsed_event['incident_r'] = incident_r
        parsed_event['incident_s'] = incident_s
        parsed_event['total_energy'] = total_energy
        parsed_event['front_acd_energy'] = front_acd_energy
        parsed_event['back_acd_energy'] = back_acd_energy
        parsed_event['calorimeter_energy'] = calorimeter_energy
        parsed_event['track_energy'] = track_energy
        parsed_event['clean_entrance'] = clean_entrance
        parsed_event['entrance_scatter_energy'] = entrance_scatter_energy
        parsed_event['missing_energy'] = missing_energy
        parsed_event['missing_energy_after_entrance'] \
            = missing_energy_after_entrance
        parsed_event['num_rayleigh'] = num_rayleigh
        parsed_event['num_brem'] = num_brem
        parsed_event['escaped_back_energy'] = escaped_back_energy
        parsed_event['escaped_through_energy'] = escaped_through_energy
        parsed_event['num_cells'] = num_cells
        parsed_event["decay_in_inert_material"] = decay_in_inert_material
        parsed_event["calorimeter_in_first_3"] = calorimeter_in_first_3


        #   Arrays of per hit information
        parsed_event['energy'] = energies
        parsed_event['r'] = r
        parsed_event['s_primary'] = s_primary
        parsed_event['s_secondary'] = s_secondary
        parsed_event['cell'] = cells
        parsed_event['cell_index'] = cell_index
        parsed_event['interaction_type'] = types
        parsed_event['interaction'] = interactions
        parsed_event['multi_cell_interaction'] = multi_cell_interactions_mask
        parsed_event['scatters_in_mask'] = scatters_in_mask

        #   These are per struck cell information
        parsed_event['cells_list'] = cells_list
        parsed_event['cell_hit_count'] = cell_hit_count

        #   Assign to self
        self.parsed_event = parsed_event

    def summarize_event(self, only_problems=False):
        """
        New, hopefully faster replacement for tree builder and
        display below.

        Unfinished: need to display graph, and also probably use graph
        for stuff above that, which also probably needs overhaul.
        """

        import numpy as np
        import copy

        #   convenient variable
        event_num = self.raw_event['event_info']['triggered_event_id']

        #   Event num and event-level summed energies
        if not only_problems:
            print('\nEvent %d, '
                  ' %s, '
                  'E(keV) = inc.: %4.1f, '
                  'active: %4.1f, '
                  'escaped:  %4.1f, '
                  'passive: %4.1f'
                   % (
                   event_num,
                   self.raw_event['meta']['particle_num2str'][
                       self.raw_event['ia']['particle_secondary'][0]],
                   self.raw_event['ia']['energies_secondary'][0],
                   self.raw_event['event_info']['deposited_energy'],
                   self.raw_event['event_info']['escaped_energy'],
                   self.raw_event['event_info']['passive_energy']
                   ))

        # HT energy deposits, sorted by interactions
        ia_interaction_types = self.raw_event['ia']['interaction_type']

        passive_interactions = self.raw_event['ia']['detector']==0

        for nic in np.unique(ia_interaction_types):
            these_ia_interactions = ia_interaction_types==nic
            passive_count = 0
            etot = 0
            for nia in np.nonzero(these_ia_interactions)[0]:
                ht_mask = self.raw_event['ht']['interaction_id']==(nia+1)
                this_energy = sum(self.raw_event['ht']['energy'][ht_mask])
                if (this_energy>0) and passive_interactions[nia]:
                    passive_count +=1
                etot += this_energy

                if nic==5 and (this_energy>0.0):
                    print(
                        f'Event {event_num:d}, '
                        + f'interaction {nia[0]+1:d} - Brem, has'
                        + f'{etot:9.1f} keV'
                        )

            if not only_problems:
                print(
                    '   '
                    + self.raw_event['meta']['interaction_types_num2str'][nic]
                    + f': #={these_ia_interactions.sum():3d}, '
                    + f'{passive_count:d} passive w/ active energy, '
                    + f'E = {etot:9.1f} keV'
                    )

        for (iid, interactions_ids) in zip(
                range(self.raw_event['ht']['interaction_id'].size),
                self.raw_event['ht']['interactions_ids']
                ):
            if interactions_ids.size>1:
                print(
                    '   '
                    + f'HT line {iid+1:3d}, interactions = '
                    + str(interactions_ids)
                    )

        #   Adapted from:
        #   https://stackoverflow.com/questions/45460653/
        #   given-a-flat-list-of-parent-child-create-a-
        #   hierarchical-dictionary-tree

        kids = copy.copy(self.raw_event['ia']['interaction_id'])
        parents = self.raw_event['ia']['parent_interaction_id']

        lst = [(parent, child) for (parent, child) in zip(parents, kids)]

        # Build a directed graph and a list of all names that have no parent
        graph = {inter: set() for tup in lst for inter in tup}
        has_parent = {inter: False for tup in lst for inter in tup}
        for parent, child in lst:
            graph[parent].add(child)
            has_parent[child] = True

        # All interactions with no parent:
        roots = [inter for inter, parent in has_parent.items() if not parent]

        # traversal of the graph (doesn't care about duplicates and cycles)
        def traverse(hierarchy, graph, inters):
            for inter in inters:
                hierarchy[inter] = traverse({}, graph, graph[inter])
            return hierarchy

        tree = traverse({}, graph, roots)

        return tree

    def make_event_tree(self):
        """
        Constructs a tree diagram of event

        8/24/2020   TS
        """

        import numpy as np

        tree = {}

        if not 'ht' in self.raw_event:
            print('*** Warning in DiagramEvent:'
                  + 'no HT lines, event skipped ***')
            self.tree = None
            return

        #   Add hits to interactions.
        tree['num_hits'] = np.empty(
            len(self.raw_event['ia']['interaction_id']),dtype=int)
        tree['hit_energy'] \
            = np.empty(len(self.raw_event['ia']['interaction_id']))
        for ni in range(len(self.raw_event['ia']['interaction_id'])):
            hits = self.raw_event['ht']['interaction_id']==(ni+1)
            tree['num_hits'][ni] = sum(hits)
            tree['hit_energy'][ni]= sum(self.raw_event['ht']['energy'][hits])

        #   These next section creates all_hit_energies and all_secondries.
        #   For each interaction these are lists of all secondary energies
        #   and interactions, including the interaction itself

        #   Start with list of interactions and associated parents
        interaction_ids = self.raw_event['ia']['interaction_id'].tolist()
        parents = self.raw_event['ia']['parent_interaction_id'].tolist()

        #   Prepopulate the "all" fields with interactions/energies for this
        #   interaction
        tree['all_hit_energies'] = []
        tree['all_secondries'] = []
        for ni in range(len(interaction_ids)):
            tree['all_secondries'].append([ni+1])
            tree['all_hit_energies'].append([tree['hit_energy'][ni]])

        #   This iterative procedure starts at the "bottom" of the event  -
        #   that is, interactions with no secondary, and works its way up,
        #   at each iteraction removing the bottom set of interactions,
        #   and so working back to the primary
        while len(interaction_ids)>1:

            parent_list=set(parents)

            #   "bottom" of the chain is all interactions that are not
            #   a parent
            bottoms=[]
            for ni in range(len(interaction_ids)):
                if set({interaction_ids[ni]}).isdisjoint(parent_list):
                    bottoms.append(interaction_ids[ni])

            #   Assign these bottom interactions to their parents
            these_parents = \
                self.raw_event['ia']['parent_interaction_id'] \
                    [np.array(bottoms)-1]
            bottom_cut = np.array(np.zeros(len(interaction_ids)),dtype=bool)
            for nb in range(len(bottoms)):
                for nnb in range(len(tree['all_secondries'][bottoms[nb]-1])):
                    tree['all_secondries'][these_parents[nb]-1].append(
                        tree['all_secondries'][bottoms[nb]-1][nnb])
                bottom_cut[np.array(interaction_ids)==(bottoms[nb])] = True

            #   Remove current bottom interactions from both
            #   interactions and parents
            interaction_ids = \
                np.ndarray.tolist(np.array(interaction_ids)[~bottom_cut])
            parents = \
                np.ndarray.tolist(np.array(parents)[~bottom_cut])

        #   Sort list of all interactions for convenience, then assign
        #   hit energies associated with each secondary
        for ni in range(len(tree['all_secondries'])):
            tree['all_secondries'][ni].sort()
            tree['all_hit_energies'][ni] = tree['hit_energy'][
                np.array(tree['all_secondries'][ni])-1]

        #   Create direct_secondaries, and direct_hit_energies.
        #   For each interaction,
        #   this is a list of only those interactions and eneriges that are
        #   direct secondaries of that interaction.
        tree['direct_secondaries'] = \
            [[] for i in range(len(self.raw_event['ia']['interaction_id']))]
        tree['direct_hit_energies'] = \
            [[] for i in range(len(self.raw_event['ia']['interaction_id']))]
        for ni in range(len(self.raw_event['ia']['interaction_id'])):
            if self.raw_event['ia']['parent_interaction_id'][ni]>0:
                tree['direct_secondaries'][
                    self.raw_event['ia']['parent_interaction_id'][ni]-1
                    ].append(ni+1)
                tree['direct_hit_energies'][
                    self.raw_event['ia']['parent_interaction_id'][ni]-1
                    ].append(tree['hit_energy'][ni])

        self.tree = tree

    def show_event_tree(self, startindex=0):

        """ Displays tree to command line """

        #   Recursive display of direct secondary information
        def dig_down(event, startindex, indentcounter):
            """ Called by show_event_tree, displays tree information.
            Calls itself recursively """

            c = ' ' * 3 * indentcounter

            for nn in range(len(
                    event.tree['direct_secondaries'][startindex])):

                ns=event.tree['direct_secondaries'][startindex][nn]
                nsi = ns - 1

                #   Need this tag
                if (
                        (event.raw_event['ia']['detector'][nsi]!=0)
                        | (event.raw_event['ia']['detector'][nsi]!=0)
                        ):
                    activetag='active '
                else:
                    activetag='passive'

                # if interaction['type{nsi}=='COMP'
                #     or interaction['type{nsi}=='BREM'
                #     or interaction['type{nsi=='ANNI':
                #                iaetag='IA E_secondary = ' ...
                #         sprintf('#4.1f',interaction['energies.out(nsi)) ', ']
                # else
                #     iaetag=[]

                print(
                    '%s'
                    'Int. %d '
                    '%s, '
                    '(%s, '
                    '%s), '
                    '%s, '
                    'E_hits: Tot = %4.2f, '
                    'This int = %4.2f'
                    % (
                        c,
                        ns,
                        event.raw_event['meta']['interaction_types_num2str'][
                            event.raw_event['ia']['interaction_type'][nsi]],
                        event.raw_event['meta']['particle_num2str'][
                            event.raw_event['ia']['particle_primary'][nsi]],
                        event.raw_event['meta']['particle_num2str'][
                            event.raw_event['ia']['particle_secondary'][nsi]],
                        activetag,
                        # iaetag ...
                        sum(event.tree['all_hit_energies'][nsi]),
                        event.tree['hit_energy'][nsi]
                        )
                    )

                dig_down(event, nsi, indentcounter+1)

        #   Make tree if it doesn't exist
        if not 'tree' in self.raw_event['event_info']:
            self.make_event_tree()

        #   Bail if still no tree.  Should really have
        #   set error in make_event_tree
        if not self.tree:
            print('Bad event - no tree created')
            return

        #   Event num and event-level summed energies
        print('Event %d, '
              '%s, '
              'E(keV): inc. =  %4.1f, '
              'tot =  %4.1f, '
              'escaped =  %4.1f, '
              'passive = %4.1f'
               % (
               self.raw_event['event_info']['triggered_event_id'],
               self.raw_event['meta']['particle_num2str'][
                   self.raw_event['ia']['particle_secondary'][0]],
               self.raw_event['ia']['energies_secondary'][0],
               self.raw_event['event_info']['deposited_energy'],
               self.raw_event['event_info']['escaped_energy'],
               self.raw_event['event_info']['passive_energy']
               ))

        #   Other summary infomration
        # print(
        #     '  IA %d '
        #     '%s, '
        #     'Hit energies: this interaction = %4.2f, '
        #     'this + secondaries = %4.2f'
        #     % (
        #         self.raw_event['interaction['interaction[startindex],
        #         self.raw_event['interaction['interaction_type[startindex],
        #         self.tree['hit_energy[startindex],
        #         sum(self.tree['all_hit_energies[startindex])
        #         ))

        #   Recursively blab about secondaries
        dig_down(self, startindex, 1)

######## Helper functions ##########
def particle_name(iso, ia=2):

    particle_ids = {
        0: "?", 1: "photon", 2: "positron", 3: "electron", 4: "proton",
        5: "anti_proton",
        6: "neutron", 7: "anti_neutron", 8: "anti_muon", 9: "muon",
        10: "anti_tau",
        11: "tau", 12: "electron_neutrino", 13: "anti_electron_neutrino",
        14: "muon_neutrino", 15: "anti_muon_neutrino",
        16: "tau_neutrino", 17: "anti_tau_neutrino", 18: "deuteron",
        19: "triton", 20: "helium_3", 21: "alpha"
    }

    elements_dict = {
        1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O",
        9: "F", 10: "Ne",
        11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P", 16: "S", 17: "Cl",
        18: "Ar", 19: "K", 20: "Ca",
        21: "Sc", 22: "Ti", 23: "V", 24: "Cr", 25: "Mn", 26: "Fe", 27: "Co",
        28: "Ni", 29: "Cu", 30: "Zn",
        31: "Ga", 32: "Ge", 33: "As", 34: "Se", 35: "Br", 36: "Kr", 37: "Rb",
        38: "Sr", 39: "Y", 40: "Zr",
        41: "Nb", 42: "Mo", 43: "Tc", 44: "Ru", 45: "Rh", 46: "Pd", 47: "Ag",
        48: "Cd", 49: "In", 50: "Sn",
        51: "Sb", 52: "Te", 53: "I", 54: "Xe", 55: "Cs", 56: "Ba", 57: "La",
        58: "Ce", 59: "Pr", 60: "Nd",
        61: "Pm", 62: "Sm", 63: "Eu", 64: "Gd", 65: "Tb", 66: "Dy", 67: "Ho",
        68: "Er", 69: "Tm", 70: "Yb",
        71: "Lu", 72: "Hf", 73: "Ta", 74: "W", 75: "Re", 76: "Os", 77: "Ir",
        78: "Pt", 79: "Au", 80: "Hg",
        81: "Tl", 82: "Pb", 83: "Bi", 84: "Po", 85: "At", 86: "Rn", 87: "Fr",
        88: "Ra", 89: "Ac", 90: "Th",
        91: "Pa", 92: "U", 93: "Np", 94: "Pu", 95: "Am", 96: "Cm", 97: "Bk",
        98: "Cf", 99: "Es", 100: "Fm",
        101: "Md", 102: "No", 103: "Lr", 104: "Rf", 105: "Db", 106: "Sg",
        107: "Bh", 108: "Hs", 109: "Mt",
        110: "Ds", 111: "Rg", 112: "Cn", 113: "Nh", 114: "Fl", 115: "Mc",
        116: "Lv", 117: "Ts", 118: "Og"
    }

    if int(iso) == 1:
        if ia == 2:
            return particle_ids[int(iso)]
        if ia > 0.21:
            return particle_ids[int(iso)] + "_atm"
        else:
            return particle_ids[int(iso)] + "_cos"

    if int(iso) in particle_ids.keys():
        return particle_ids[int(iso)]
    ele = int(iso / 1000)
    av = iso - ele * 1000
    H = elements_dict[ele]

    return f"{H}{av}" if av > 0 else H

######## Helper functions ##########

def read_events_from_sim_file(full_file_name,
                              sims_params=None,
                              num_events_to_read=1e10):
    """
    Opens .sim file and reads and parses all events to create hits, which
    are (ideally) separate electron recoil tracks.

    These are then put into a "flattened" set of arrays for vectorized
    calculations.  This is returned as the diciontaries 'truth', which is
    per event information, and 'truth_hits', which is mostly
    quanities of dimension [num_hits, num_events] but also include per
    struck cell information

    In most cases, this is intended to be called by events_tools
    instead of directly by the user.

    TODO clean up the poorly thought out use of copy, and replace the
        matrix representation with awkward structures
    """

    import awkward as ak
    import numpy as np
    import sims_tools

    #   Open file
    sim_file = Sim_File(full_file_name)

    # Initialize lists to store per-event and per-hit data
    truth_list = []
    truth_hits_list = []

    #   Loop over events
    for n in range(int(num_events_to_read)):

        #   Read next event
        sim_file.read_next_event()

        #   This skips several types of bad events
        if sim_file.good_event:

            #   Parse raw sim file information
            try:
                sim_file.parse_raw_event(sims_params)
            except:
                print(f"ERROR, skipping {n}")
                continue

            #   These have values per event, as opposed to per hit
            truth_event = {
                'time':
                sim_file.raw_event['event_info']['time'],
                'triggered_id':
                sim_file.raw_event['event_info']['triggered_event_id'],
                'incident_energy':
                sim_file.parsed_event['incident_energy'],
                'incident_particle':
                sim_file.parsed_event['incident_particle'],
                'incident_charged_particle':
                sim_file.parsed_event['incident_charged_particle'],
                'incident_s':
                sim_file.parsed_event['incident_s'],
                'total_energy':
                sim_file.parsed_event['total_energy'],
                'deposited_energy':
                sim_file.raw_event['event_info']['deposited_energy'],
                'escaped_energy':
                sim_file.raw_event['event_info']['escaped_energy'],
                'passive_energy':
                sim_file.raw_event['event_info']['passive_energy'],
                'front_acd_energy':
                np.sum(sim_file.parsed_event['front_acd_energy']),
                'back_acd_energy':
                np.sum(sim_file.parsed_event['back_acd_energy']),
                'calorimeter_energy':
                np.sum(sim_file.parsed_event['calorimeter_energy']),
                'track_energy':
                np.sum(sim_file.parsed_event['track_energy']),
                'clean_entrance':
                sim_file.parsed_event['clean_entrance'],
                'entrance_scatter_energy':
                sim_file.parsed_event['entrance_scatter_energy'],
                'missing_energy':
                sim_file.parsed_event['missing_energy'],
                'missing_energy_after_entrance':
                sim_file.parsed_event['missing_energy_after_entrance'],
                'num_rayleigh':
                sim_file.parsed_event['num_rayleigh'],
                'num_brem':
                sim_file.parsed_event['num_brem'],
                'escaped_back_energy':
                sim_file.parsed_event['escaped_back_energy'],
                'escaped_through_energy':
                sim_file.parsed_event['escaped_through_energy'],
                'num_hits':
                len(sim_file.parsed_event['energy']),
                'num_cells':
                sim_file.parsed_event['num_cells'],
                'decay_in_inert_material':
                sim_file.parsed_event["decay_in_inert_material"],
                'calorimeter_in_first_3':
                sim_file.parsed_event["calorimeter_in_first_3"],
            }

            #   This is values per hit
            truth_hits_event = {
                'energy':
                sim_file.parsed_event['energy'],
                'r':
                sim_file.parsed_event['r'],
                's_primary':
                sim_file.parsed_event['s_primary'],
                's_secondary':
                sim_file.parsed_event['s_secondary'],
                'cell':
                sim_file.parsed_event['cell'],
                'cell_index':
                sim_file.parsed_event['cell_index'],
                'interaction_type':
                sim_file.parsed_event['interaction_type'],
                'interaction':
                sim_file.parsed_event['interaction'],
                'multi_cell_interaction':
                sim_file.parsed_event['multi_cell_interaction'],
                'cells_list':
                sim_file.parsed_event['cells_list'],
                'cell_hit_count':
                sim_file.parsed_event['cell_hit_count']
            }

            #   Several calculated things

            #   "true" theta - from geometry of incident and first
            #   scattered vectors
            if truth_hits_event['s_primary'].size > 0:
                truth_event['theta'] = (
                    np.arccos(np.dot(truth_hits_event['s_primary'].T[0],
                                     truth_event['incident_s']))
                    )
            else:
                truth_event['theta'] = np.nan

            #   Drift distance is distance from z anode
            truth_hits_event['z_drift'] = (
                sims_params.cells['z_anode'][truth_hits_event['cell'] - 1]
                - truth_hits_event['r'][2, :]
                )

            #   Particle name - combines partice ID and direction
            truth_event['particle_name'] = particle_name(
                truth_event['incident_particle'],
                truth_event['incident_s'][2]
                )

            #   Add these results to lists
            truth_list.append(truth_event)
            truth_hits_list.append(truth_hits_event)

        #   Reached end of file
        if sim_file.raw_event['end_of_file']:
            break

    #   Close .sim file
    sim_file.f.close()

    # Convert lists of dictionaries to awkward arrays
    truth = ak.Array(truth_list)
    truth_hits = ak.Array(truth_hits_list)

    #   Generate locations in cell coordinates
    r_cell = sims_tools.global_to_cell_coordinates(truth_hits['r'],
                                                       truth_hits['cell'],
                                                       sims_params,
                                                       reverse=False)
    truth_hits['r_cell'] = r_cell

    #   Meta data from the last read sim events
    meta = sim_file.raw_event['meta']

    return truth, truth_hits, meta

def write_events_file(events, full_file_name):
    """
    Saves events to disk in .hdf5 and .pickle files
    full_file_stub includes path but not extensions
    """

    import pickle
    import h5py
    import awkward as ak

    with open(full_file_name.replace('.sim', '.meta.pickle'), 'wb') as f:
        pickle.dump(events.meta, f)

    with h5py.File(full_file_name.replace('.sim', '.hdf5'), 'w') as f:
        for key in ['truth', 'truth_hits']:
            group = f.create_group(key)
            ak_array = getattr(events, key)
            form, length, container \
                = ak.to_buffers(ak.to_packed(ak_array), container=group)
            group.attrs['form'] = form.to_json()
            group.attrs['length'] = length

def read_events_file(full_file_name):
    """
    Loads events from .hdf5 and .pickle files
    full_file_stub includes path but not extensions
    """

    import h5py
    import pickle
    import awkward as ak
    import numpy as np

    with h5py.File(full_file_name.replace('.sim', '.hdf5'), 'r') as f:
        truth = ak.from_buffers(
            ak.forms.from_json(f['truth'].attrs['form']),
            f['truth'].attrs['length'],
            {k: np.asarray(v) for k, v in f['truth'].items()}
        )

        truth_hits = ak.from_buffers(
            ak.forms.from_json(f['truth_hits'].attrs['form']),
            f['truth_hits'].attrs['length'],
            {k: np.asarray(v) for k, v in f['truth_hits'].items()}
        )

    with open(full_file_name.replace('.sim', '.meta.pickle'), 'rb') as f:
        meta = pickle.load(f)

    # return meta, truth, truth_hits
    return truth, truth_hits, meta

def is_max_80_percent_larger_than_total(list_values):
    if len(list_values) > 3:
        return False
    if len(list_values) == 1:
        return False
    # Find the largest value
    max_value = max(list_values)
    total_sum = sum(list_values)

    # Check if the largest is at least 80% of the sum
    if max_value >= 0.8 * total_sum:
        if min(list_values) < 20:
            return True

def write_evta_file(events, paths, bad_events, version='200'):
    """
    Writes events in events['measured_hits'] to an .evta file
    # bad implementation: fix the position/energy uncertainty
    """

    import numpy as np
    import awkward as ak
    import os

    #   Get evta file name
    events.meta['file_names'] = events.meta['sim_file_name']

    #   Open file, write header
    f = open(os.path.join(paths['root'],
                          events.meta['file_names']) + '.evta', 'w')

    f.write('Version ' + version + '\n')
    f.write('Type EVTA\n')
    f.write('\n')

    if version=='200':

        #   THIS IGNORES RECOMBINATION FLUCTUATIONS FOR MULTIPLE CELL HITS
        #   AND IN ANY CASE THIS CALCULAITON SHOULD HAPPEN
        #   IN response_tools

        #   Start with convenient varibles
        sigmaxy = 0.1/1000
        sigmaz  = 0.1/1000
        calorimeter_xy = 0
        calorimeter_z = -5

        # replace all events.measured_hits['energy'] with mh
        mh = events.measured_hits
        truth_mask = mh['_good_mask']
        acd_pass = ~mh['ACD_activated']
        cal_pass = ~mh['calorimeter_activated']
        decay_in_inert_material = events.truth['decay_in_inert_material'][truth_mask]
        calorimeter_in_first_3  = events.truth['calorimeter_in_first_3'][truth_mask]



        sigma_energy = np.sqrt((0.015*mh['energy'])**2 + 2**2)
        print("\n\n\n BAD EVENTS: ", bad_events, "\n\n\n")

        for ne in range(len(mh['energy'])):
            ID = events.truth["triggered_id"][truth_mask][ne]
            if int(ID) in bad_events:
                print("Bad event, skipping")
                #continue

            if (acd_pass[ne] and cal_pass[ne]
                and not decay_in_inert_material[ne]):
                if is_max_80_percent_larger_than_total(mh['energy'][ne]):
                    print('*** Warning in write_evta_file: '
                          + 'One hit dominates ***', mh["energy"][ne])
                    pass

                f.write('SE\n')
                f.write(f'ID {ID :1.0f}\n')
                f.write(f'TI {events.measured_hits["time"][ne]:2.9f}\n')

                num_hits = ak.num(mh['energy'])
                for nh in range(num_hits[ne]):
                    z = (
                        mh['r'][ne,2,nh]
                        #+ events.meta['params'].cells['geomaga_reference_zo']
                        )
                    f.write(
                        'HT 5;'
                        + f'{mh["r"][ne,0,nh]*100:10.7f};'
                        + f'{mh["r"][ne,1,nh]*100:10.7f};'
                        + f'{mh["r"][ne,2,nh]*100:10.7f};'
                        + f'{mh["energy"][ne,nh]:10.7f}'
                        + f'{sigmaxy*100:10.7f}; '
                        + f'{sigmaxy*100:10.7f}; '
                        + f'{sigmaz*100:10.7f}; '
                        + f'{sigma_energy[ne,nh]:10.7f}\n'
                        )

                cal_ene = mh["calorimeter_energy"][ne]
                if cal_ene > 30:
                    f.write(f'HT 2;'
                            + f'{calorimeter_xy:10.7f};'
                            + f'{calorimeter_xy:10.7f};'
                            + f'{calorimeter_z:10.7f};'
                            + f'{cal_ene:10.7f}'
                            + f'{98:10.7f}; '
                            + f'{98:10.7f}; '
                            + f'{15:10.7f}; '
                            + f'{4*cal_ene/100 + 3:10.7f}\n'
                            )

def fix_sim_file_ht_lines(full_sim_file_name_in,
                          full_geo_file_name,
                          full_sim_file_name_out
                          ):

    """
    Fixes problem: ht lines in .sim file have positions based
    on a sub-detecor coordinate system, but should be global coordinates

    This routine not well checked.

    BIG KNOWN PROBLEM: LAST FEW LINES OF FILE ARE NOT CORRECTLY HANDLED

    @author: tshutt
    """
    import numpy as np
    import pickle

    import response_tools
    import readout_tools

    #   Load sims_params that were generated for Cosima, then
    #   with these generate default response params
    with open(full_geo_file_name + '.pickle', 'rb') as f:
        sims_params = pickle.load(f)
    read_params = readout_tools.Params(cells=sims_params.cells)

    #   Open file_names
    f_in = open(full_sim_file_name_in + '.sim')
    f_out = open(full_sim_file_name_out + '.sim', 'w')

    #   Skip to first event - text_line = 'SE'
    text_line = f_in.readline().strip('\n')
    f_out.write(text_line + '\n')
    while text_line=="" or text_line[0:2]!='SE':
        text_line = f_in.readline().strip('\n')
        if text_line[0:2]!='SE':
            print(text_line)
        f_out.write(text_line + '\n')

    r = np.zeros(3)

    #   Read to end of data, denoted by a blank line
    nl = 0
    while text_line!="":

        nl += 1

        if (nl % 1000)==0:
            print('Event ' + str(nl))

        #   Read to HT line
        while text_line[0:2]!='HT':
            text_line = f_in.readline().strip('\n')
            if text_line[0:2]!='HT':
                f_out.write(text_line + '\n')

        #   Read all HT lines, use response_tools cell <-> global
        #   coordinate translation.  For this need cell of each hit. Also,
        #   that tool is in mks, so change r units
        while text_line[0:2]=='HT':

            splitline = text_line[6:].split(';')

            cell = np.array([int(splitline[1].split('_')[1])])

            r[0] = float(splitline[2]) / 100
            r[1] = float(splitline[3]) / 100
            r[2] = float(splitline[4]) / 100

            r_global =response_tools.global_to_cell_coordinates(
                r,
                cell,
                read_params,
                reverse = True
                ) * 100

            #   reconstruct text line and write out
            text_line_out = text_line[0:6] + ''.join([
                ';'.join(splitline[0:2]),
                ';', f'{r_global[0]:10.5f}',
                ';', f'{r_global[1]:10.5f}',
                ';', f'{r_global[2]:10.5f}',
                ';',
                ';'.join(splitline[5:])
                ])
            f_out.write(text_line_out + '\n')

            #   Read next text_line
            text_line = f_in.readline().strip('\n')

        f_out.write(text_line + '\n')

        #   Blank line is end of events
        if text_line=='':
            f_out.write(text_line + '\n')
            break

    #   End lines - last line starts with "TS"
    while 1:
        text_line = f_in.readline().strip('\n')
        f_out.write(text_line + '\n')
        if text_line[0:2]=='TS':
            break

    #   Apparently file open/close here is bad practice
    f_in.close()
    f_out.close()

def get_geo_tag(topology_id, values_id):
    """ Standard geometry tag for file names """
    geo_tag = f'_GeoT{topology_id:02d}v{values_id:02d}'
    return geo_tag

def get_theta_tag(cos_theta):
    """ Standard cos_theta tag for .sim and related file names """
    theta_tag = f'_Cos{cos_theta:2.1f}'
    return theta_tag

def get_sim_file_name(
        beam,
        gamma_energy,
        topology_id,
        values_id=0,
        cos_theta=None,
        ):
    """ Base sim file name, without inc or id #s or extension """

    import sys

    #   Theta tag, if point source.
    if beam=='FarFieldPointSource':
        theta_tag = get_theta_tag(cos_theta)
    elif beam=='FarFieldIsotropic':
        theta_tag = ''
    else:
        sys.exit('Error: unsupported beam type in get_sim_file_name')

    #   Geo tag
    geo_tag = get_geo_tag(topology_id, values_id)

    #   Base and geo names
    file_name = (
        beam
        + f'_{gamma_energy/1000:5.3f}MeV'
        + theta_tag
        + geo_tag
        )

    return file_name

def get_geo_file_name(topology_id, values_id):
    """ Base geometry file name, without extension """

    #   Geo tag
    geo_tag = get_geo_tag(topology_id, values_id)

    #   Base and geo names
    file_name = 'GammaTPC' + geo_tag + '.geo'

    return file_name

def write_geo_files(path, sims_params, values_id=0):
    """
    Writes geometry files - .setup, and copy of sims_params into
        folder set by path

    values_id - integer tags for geometry values constants

    returns file_names
    """

    import os
    import pickle

    #   File names
    file_name = get_geo_file_name(sims_params.topology_id, values_id)

    #   Calculate sims_params to update file image
    sims_params.calculate()

    #   Write .setup from file image in sims_params
    with open(os.path.join(path, file_name + '.setup'), 'w') as f:
        for line in sims_params.setup_file_lines:
            f.write(line)

    #   Write sims_params
    with open(os.path.join(path, file_name + '.pickle'), 'wb') as f:
            pickle.dump(sims_params, f)

    return file_name

def write_source_file(data_path,
                      geo_full_file_name,
                      num_triggers,
                      beam,
                      energy,
                      cos_theta=None,
                      ):

    import os
    from math import acos

    topology_id, values_id = geo_full_file_name.split('_')[1].split('.')[0] \
        .split('T')[1].split('v')
    topology_id = int(topology_id)
    values_id = int(values_id)

    sim_file_name \
        = get_sim_file_name(
            beam,
            energy,
            topology_id,
            values_id,
            cos_theta
            )

    if beam=='FarFieldPointSource':
        theta = acos(cos_theta)
        beam_tag = 'FFPS'
        beam_values = f' {theta:7.5f}   0'
    elif beam=='FarFieldIsotropic':
        beam_tag = 'FFI'
        beam_values = ' '

    lines = ['']
    lines.append('Version          1 \n')
    lines.append('Geometry         ' + geo_full_file_name + '.setup\n')
    lines.append('CheckForOverlaps 1000 0.01 \n')
    lines.append('PhysicsListEM    Livermore \n')
    lines.append('\n')
    lines.append('StoreCalibrate                 true\n')
    lines.append('StoreSimulationInfo            true\n')
    lines.append('StoreOnlyEventsWithEnergyLoss  true  '
                 +'// Only relevant if no trigger criteria is given!\n')
    lines.append('DiscretizeHits                 true\n')
    lines.append('PreTriggerMode                 everyeventwithhits\n')
    lines.append('\n')
    lines.append('Run ' + beam_tag + '\n')
    lines.append(beam_tag + '.FileName           ' + sim_file_name + '\n')
    lines.append(beam_tag + '.NTriggers          '
                 + f'{num_triggers:5.0f} ' + '\n')
    lines.append('\n')
    lines.append('\n')
    lines.append(beam_tag + '.Source One \n')
    lines.append('One.ParticleType        1 \n')
    lines.append('One.Beam                ' + beam + beam_values + '\n')
    lines.append('One.Spectrum            Mono  ' + f'{energy:5.1f}' + '\n')
    lines.append('One.Flux                1000.0')

    #   Write .source file
    with open(os.path.join(data_path, sim_file_name + '.source'), 'w') as f:
        for line in lines:
            f.write(line)

def add_evta_file_names(file_names, events):
    """ First attempt at unified handling for file names.  Work
        in progress. Also returns meta information
    """

    import os

    # #   Study tag, if study present
    study_tag = ''
    # if 'study' in events.meta['params'].meta:
    #     # study_tag = events.meta['params']. \
    #     #     meta['study'].labels['study_tag']
    #     case = events.meta['params'].meta['case']
    #     case_tag = events.meta['params']. \
    #         meta['study'].labels['case_tag'][case]
    #     study_tag = '.s' + case_tag

    start, end = file_names['base'].split('.inc')

    file_names['evta'] = start + study_tag + '.inc' + end

    file_names['path_evta'] = os.path.join(
        file_names['paths']['data'],
        file_names['evta']
        )

    return file_names

# %%
