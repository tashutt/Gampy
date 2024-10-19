#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 21:08:02 2022

@author: tshutt
"""

class Events:
    """"
    Flattend representation of events in file.  Arrays padded in the
    scatters dimension. Dimensions are [scatters, event] or
    [space, scatters, event]

    When first called, returns truth, truth_hits and meta.  Also sets
    default readout parameters and attaches those to object, with
    GAMPixG charge readout.

    TODO: When relevant, add selection of charge readout.  Follow
        approach in tracks_tools, including defined
        method for resetting params with different

    TODO: Review, extend various performance routines: pointing,
        pointing error, CKD ordering, etc.  These neither well
        checked nor reviewed

    Units are mks and keV, requiring a conversion from Megalib cm and keV
    """
    def __init__(self,
                 sim_file_name,
                 sims_inputs_file_name,
                 num_events_to_read=1e10,
                 readout_inputs_file_name='default',
                 read_sim_file=False,
                 write_events_files=True,
                 ):
        """
        Reads events from .hdf5 file if found, otherwise from
        .sim file and writes hdf5 files unless directed not to.

        See sims_tools and readout_tools for options for
        sims_inputs_file_name and readout_inputs_file_name
        """

        import os
        import sys

        import file_tools
        import sims_tools
        import readout_tools
        import awkward as ak

        #   Get geometry_params
        self.sims_params \
            = sims_tools.Params(sims_inputs_file_name)

        #   This flag prevents updating sims_params
        self.sims_params.live = False

        #   If .hdf5 files present, read those
        if os.path.isfile(sim_file_name + '.hdf5') \
            and not read_sim_file:

            print('Reading .hdf5 files')

            #   Read .hdf5 truth and truth hits, and pickle meta files
            self.truth, self.truth_hits, self.meta \
                = file_tools.read_events_file(sim_file_name)

            #   Trim arrays to only contain requested # o events
            if num_events_to_read < self.meta['num_events']:
                self = trim_events(self, num_events_to_read)

            elif num_events_to_read > self.meta['num_events']:
                print('*** Warning: saved files have '
                      + str(self.meta['num_events'])
                      + ' of ' + str(num_events_to_read) + ' requested'
                      )

        #   If not, read and parse .sim file.  Then write .h5 files
        #   unless told not to.
        #   Load geometry description, preferably from .yaml input file,
        #   but for backwards compatabilty if no .yaml file, read minimal
        #   cell description from .source file.
        elif os.path.isfile(sim_file_name):

            print('Reading .sim file')

            #   Now read and parse .sim file
            self.truth, self.truth_hits, self.meta \
                = file_tools.read_events_from_sim_file(
                    sim_file_name,
                    self.sims_params,
                    num_events_to_read,
                    )

            #   Add file names, number of events, geo_parms to meta data
            self.meta['sim_file_name'] = sim_file_name
            self.meta['sims_inputs_file_name'] \
                = sims_inputs_file_name
            self.meta['num_events'] = ak.num(self.truth['num_hits'], axis=0)

            #   Write .hdf5 files
            if write_events_files:
                print('Writing .hdf5 files')
                file_tools.write_events_file(self, sim_file_name)

        #   Otherwise file not found - error
        else:
            sys.exit('Error in events tool: File ' \
                     + sim_file_name + '.sim not found')

        #   Generate default response params, and assign to events
        self.read_params = readout_tools.Params(
            inputs_file_name=readout_inputs_file_name,
            charge_readout_name='GAMPixG',
            cells=self.sims_params.cells,
            )

    def reset_params(self):
        """ Reset readout parameters to default values.
            Charge readout is GAMPixG
            Removes measured events, cones.
            NOTE: this here only as placeholder until charge readout
            is implemented for events
        """

        import readout_tools

        #   Create fresh set of readout params
        self.read_params = readout_tools.Params(
            charge_readout_name='GAMPixG',
            cells=self.sims_params.cells,
            )

        #   Remove attributes that depend on params
        kill_keys = ['events_measured', 'cones']
        for kill_key in kill_keys:
            if hasattr(self, kill_key):
                delattr(self, kill_key)

    def apply_detector_response(self, ):
        """ Applies response defined by params to truth_hits,
        creating measure_hits """

        import response_tools

        #   Calculate cell params
        self.read_params.calculate()

        #   Apply response
        self = response_tools.apply_detector_response(self)

    ###### PILEUP START ######
    def pileup_analysis(self, drift_len_via_diffusion_enabled=0):
        """
        Add a boolean to events.measured that describes if the measured energy
        will be confused with a hit from another event.

        TO consider: should this be implemented on the truth hits too?
        """
        import awkward as ak

        self.drift_est_acc = 0.02

        if 'measured_hits' not in self.__dict__:
            self.apply_detector_response()

        num_of_events = len(self.measured_hits['total_energy'])
        num_of_hits_in_event = ak.num(self.measured_hits['energy'])
        velocity = self.read_params.charge_drift['velocity']

        time_allocation = self._compute_time_allocation(
            drift_len_via_diffusion_enabled, num_of_events,
            num_of_hits_in_event, velocity)

        self.time_allocation = time_allocation
        self.measured_hits['pileup_detected'] \
            = self._compute_pileup_flag(time_allocation)

    def _compute_time_allocation(self,
                                 drift_enabled,
                                 num_events,
                                 num_hits,
                                 velocity
                                 ):

        allocation = {}

        for event_number in range(num_events):
            if num_hits[event_number] > 0:
                hit_time = self.truth["time"][event_number]
                affected_cells = self.truth_hits["cell"][event_number]
                z_drifts = self.truth_hits["z_drift", event_number]
                drift_times = z_drifts / velocity

                for i, hit_cell_index in enumerate(affected_cells):
                    timeframe = self._compute_timeframe(drift_enabled,
                                                        hit_time,
                                                        drift_times[i]
                                                        )
                    allocation.setdefault(event_number, {}).setdefault(
                        hit_cell_index,
                        []
                        ).append(timeframe)

        return allocation

    def _compute_timeframe(self, drift_enabled, hit_time, drift_time):
        if drift_enabled:
            delta_t = min(
                self.drift_est_acc / self.read_params.charge_drift['velocity'],
                drift_time
                )
            return (hit_time + drift_time, hit_time + drift_time + delta_t)
        else:
            return (hit_time, hit_time + drift_time)

    def _compute_pileup_flag(self, time_allocation):
        import awkward as ak

        pileup_detected = []

        for event_id, cell_hits in time_allocation.items():
            event_has_overlap = any(
                self._check_overlap(time_allocation,
                                    event_id,
                                    cell_id, hit_start,
                                    hit_end
                                    )
                for cell_id, hit_timeframes in cell_hits.items()
                for hit_start, hit_end in hit_timeframes
            )

            if event_has_overlap:
                pileup_detected.append(event_id)

        return ak.Array(pileup_detected)

    def _check_overlap(self, event_hits, event_id, cell_id, beg0, end0):
        import numpy as np

        events = {k: v for k, v in event_hits.items()
                  if k != event_id and cell_id in v}
        if not events:
            return False

        timings = np.concatenate([events[e][cell_id] for e in events], axis=0)
        overlaps = np.logical_and((beg0 < timings[:, 1]),
                                  (end0 > timings[:, 0])
                                  )

        return np.any(overlaps)

    def write_evta_file(self, paths, bad_events=[], evta_version='200'):
        """ Writes events structure into evta file """

        import file_tools
        file_tools.write_evta_file(self, paths, bad_events, evta_version)
        return True

    ###### Native reconstruction tools ######
    # There are 2 distinct motivations to do reconstruction:
    # 1. To calculate the ARM and efficiency of the reconstruction
    # --- for this, there has to be a point source with known angle
    # 2. To calculate the energy and cone angle of the reconstructed events

    # ARM is calculated within the reconstruction_tools.py if IN_VECTOR is given

    # default is to use the measured hits, but if use_truth_hits is True, then
    # the truth hits are used


    def reconstruct_events(self,
                        IN_VECTOR = None,
                        LEN_OF_CKD_HITS = [3,4,5,6,7,8],
                        use_truth_hits=False,
                        save_name=""):
        """
        Reconstructs the events using the measured hits.
        If use_truth_hits is True, the truth hits are used instead.
        """
        import reconstruction_tools

        db = reconstruction_tools.reconstruct(self,
                                            LEN_OF_CKD_HITS,
                                            IN_VECTOR,
                                            use_truth_hits,
                                            outside_mask=None,
                                            MIN_ENERGY=0.1,
                                            filename=save_name)

        self.reconstructed_data = db

    def train_classifier_on_self(self, database=None):
        """
        Trains a classifier on the reconstructed data
        Trainings should be done on more than one set of events
        Training is using truth data to learn (in the output),
        so it's cheating if it's used on itself [kind of]
        """
        import reconstruction_tools

        if database is None:
            clf = reconstruction_tools.train_classifier(self.reconstructed_data,
                                                        filename='classifier.pkl',
                                                        plot_confusion_matrix=True)
        else:
            clf = reconstruction_tools.train_classifier(database,
                                                        filename='a_major_classifier.pkl',
                                                        plot_confusion_matrix=True)
        self.classifier = clf


    def classify_reconstructed_events(self,save_name=None,load_classifier=None):
        import joblib
        from scipy.optimize import curve_fit
        import matplotlib.pyplot as plt
        import numpy as np

        if load_classifier is None:
            print('Trying to use self-clasiffier. Bad practice')
            classifier = self.classifier
        else:
            classifier = joblib.load(load_classifier)

        features = ['e_out_CKD', 'min_hit_distance', 'kn_probability',
                    'calculated_energy', 'num_of_hits', 'first_scatter_angle']

        df = self.reconstructed_data
        X = df[features]
        df['use'] = classifier.predict(X)

        ### fitting the ARM histogram ##########
        def lorentzian(x, x0, gamma, A):
            return A * (gamma**2 / ((x - x0)**2 + gamma**2))

        hist, bins = np.histogram(df.query("use==1").ARM, bins=50, range=(-10,10))
        x = (bins[1:] + bins[:-1]) / 2
        y = hist

        popt, _ = curve_fit(lorentzian, x, y, p0=[0, 1, max(y)])

        plt.clf()
        plt.hist(df.query("use==1").ARM, bins=50, range=(-10,10), alpha=0.6, label='ARM histogram')
        plt.xlabel('ARM [degrees]')
        plt.ylabel('Counts')
        plt.title('ARM histogram')
        plt.xlim(-10, 10)
        plt.plot(x, lorentzian(x, *popt), label='Lorentzian fit', color='red')

        fwhm = 2 * popt[1]
        plt.annotate(f'FWHM = {fwhm:.2f} degrees', xy=(0.1, 0.89),
                    xycoords='axes fraction', fontsize=14)

        plt.legend()
        if save_name is not None:
            plt.savefig(f'{save_name}_ARM_histogram.png')
        plt.show()
        print(f"FWHM from fit: {fwhm:.2f}")
        ########################################

        self.reconstructed_data = df











def calculate_stats(
            self,
            mask=None,
            indices=None,
            num_bins=None,
            expanded_stats=False
            ):
        """
        Calculate statistics

        Inputs:
            mask - along events axis
            indices - for distributions, are indices from digitization
                of some events-oriented variable
            num_bins - length of disributions, must span range of indices
            expanded_stats -  If true, expands set of statistics

        Outputs:
            stat_sums dictionary
        """

        import numpy as np

        def stats_core(self, mask=None):
            """
            Define statistics while generating lowest level sums.
            """

            #   If no mask provided, is true for all events
            if mask is None:
                mask = np.zeros(len(self.truth['num_hits']), dtype=bool)

            stats_sums = {}

            #   Averages
            stats_sums['full_energy'] \
                = np.sum(
                    ~self.truth['missing_energy'][mask]
                    )
            stats_sums['clean_entrance_missing_energy'] \
                =  np.sum(
                    self.truth['clean_entrance'][mask]
                    & ((self.truth['passive_energy'][mask]>0.001)
                    | (self.truth['escaped_energy'][mask]>0.001))
                    )
            stats_sums['bad_entrance_missing_energy'] \
                =  np.sum(
                        ~self.truth['clean_entrance'][mask]
                        &((self.truth['passive_energy'][mask]>0.001)
                          |(self.truth['escaped_energy'][mask]>0.001))
                        )

            #   Ordering stats
            if 'order' in self.measured_hits:
                stats_sums['full_energy_ordered'] \
                    = np.sum(
                        ~self.truth['missing_energy'][mask]
                        &self.measured_hits['order']['tried'][mask]
                        &self.measured_hits['order']['ordered'][mask]
                        )
                stats_sums['full_energy_disordered'] \
                    = np.sum(
                        ~self.truth['missing_energy'][mask]
                        &self.measured_hits['order']['tried'][mask]
                        &~self.measured_hits['order']['ordered'][mask]
                        )
                stats_sums['clean_entrance_missing_energy_ordered']\
                    =  np.sum(
                        self.truth['clean_entrance'][mask]
                        &((self.truth['passive_energy'][mask]>0.001)
                          |(self.truth['escaped_energy'][mask]>0.001))
                        &self.measured_hits['order']['tried'][mask]
                        &self.measured_hits['order']['ordered'][mask]
                        )
                stats_sums['clean_entrance_missing_energy_disordered'] \
                    =  np.sum(
                        self.truth['clean_entrance'][mask]
                        &((self.truth['passive_energy'][mask]>0.001)
                          |(self.truth['escaped_energy'][mask]>0.001))
                        &self.measured_hits['order']['tried'][mask]
                        &~self.measured_hits['order']['ordered'][mask]
                        )
                stats_sums['bad_entrance_missing_energy_ordered']\
                    =  np.sum(
                        ~self.truth['clean_entrance'][mask]
                        &((self.truth['passive_energy'][mask]>0.001)
                          |(self.truth['escaped_energy'][mask]>0.001))
                        &self.measured_hits['order']['tried'][mask]
                        &self.measured_hits['order']['ordered'][mask]
                        )
                stats_sums['bad_entrance_missing_energy_disordered'] \
                    =  np.sum(
                        ~self.truth['clean_entrance'][mask]
                        &((self.truth['passive_energy'][mask]>0.001)
                          |(self.truth['escaped_energy'][mask]>0.001))
                        &self.measured_hits['order']['tried'][mask]
                        &~self.measured_hits['order']['ordered'][mask]
                        )

            #   Expanded set of stats, off by default
            if expanded_stats:
                stats_sums['escaped'] \
                    = np.sum(
                        self.truth['escaped_energy'][mask]>0.001
                        )
                stats_sums['escaped_back'] = np.sum(
                        self.truth['escaped_back_energy'][mask]>0.001
                        )
                stats_sums['escaped_through'] = np.sum(
                        self.truth['escaped_through_energy'][mask]>0.001
                        )
                stats_sums['passive'] = np.sum(
                        self.truth['passive_energy'][mask]>0.001
                        )
                stats_sums['clean_entrance'] = np.sum(
                        self.truth['clean_entrance'][mask]
                        )
                stats_sums['clean_entrance_escaped'] = np.sum(
                        self.truth['clean_entrance'][mask]
                        & (self.truth['escaped_energy'][mask]>0.001)
                        )
                stats_sums['clean_entrance_passive'] = np.sum(
                        self.truth['clean_entrance'][mask]
                        & (self.truth['passive_energy'][mask]>0.001)
                        )
                stats_sums['clean_entrance_passive_and_escaped'] \
                    = np.sum(
                        self.truth['clean_entrance'][mask]
                        & (self.truth['passive_energy'][mask]>0.001)
                        & (self.truth['escaped_energy'][mask]>0.001)
                        )
                stats_sums['clean_entrance_passive_or_escaped'] \
                    = np.sum(
                        self.truth['clean_entrance'][mask]
                        & ((self.truth['passive_energy'][mask]>0.001)
                        | (self.truth['escaped_energy'][mask]>0.001))
                        )
                stats_sums['clean_entrance_no_escaped_back'] =  \
                    np.sum(
                        self.truth['clean_entrance'][mask]
                        & (self.truth['escaped_back_energy'][mask]<0.001)
                        )

            return stats_sums

        #   stats is dictionary
        self.stats = {}

        #   Find scalar sums over all events
        sums = stats_core(self, mask)
        self.stats['scalar_sums'] = {}
        for key in sums.keys():
            self.stats['scalar_sums'][key] = sums[key]
        self.stats['scalar_sums_sum'] = np.sum(mask)

        #   Disributions: calculate stats in num_bins bins with
        #   bin indices of all events provided
        if not indices is None and not num_bins is None:

            #  Find distribution sums, unfortunately looping over bins
            for bin_index in range(num_bins):

                #   Index from np.digitize starts with 1.
                index = bin_index + 1

                #   Get sums for events in this bin
                sums = stats_core(
                    self,
                    (indices==index) & mask
                     )

                #   For first bin, initialize distribution sums
                if index==1:
                    distribution_sums = {}
                    for key in sums.keys():
                        distribution_sums[key] = []
                    distribution_all_sums = []

                #   Append current bin to distribution_sums
                distribution_all_sums.append(
                    self.truth['time'][(indices==index) & mask].size
                    )
                for key in distribution_sums.keys():
                    if distribution_all_sums[bin_index] > 0:
                        distribution_sums[key].append(
                            sums[key]
                            )
                    else:
                        distribution_sums[key].append(0)

            #   Final results are dictionay of numpy arrays
            self.stats['distribution_sums'] = {}
            for key in distribution_sums.keys():
                self.stats['distribution_sums'][key] = np.array(
                    distribution_sums[key]
                    )
            self.stats['distribution_all_sums'] = distribution_all_sums

def trim_events(events, num_trimmed_events):
    """ Used when creating events instance, trims events structure to
    contain only first num_trimmed_events

    TODO: make work with awkward arrays.
    """

    import numpy as np

    mask = np.zeros(len(events.truth['num_hits']), dtype=bool)
    if num_trimmed_events > len(mask):
        print('ERROR: num trimmed events exceeds number of events')
        return

    mask[0:num_trimmed_events] = True

    max_hits = np.max(events.truth['num_hits'][mask])

    for field in events.truth.fields:
        if events.truth[field].ndim==1:
            events.truth[field] \
                = events.truth[field][mask]
        elif events.truth[field].ndim==2:
            events.truth[field] \
                = events.truth[field][0:max_hits, mask]
        elif events.truth[field].ndim==3:
            events.truth[field] \
                = events.truth[field][:, 0:max_hits, mask]

    for field in events.truth_hits.keys():
        if events.truth_hits[field].ndim==1:
            events.truth_hits[field] \
                = events.truth_hits[field][mask]
        elif events.truth_hits[field].ndim==2:
            events.truth_hits[field] \
                = events.truth_hits[field][0:max_hits, mask]
        elif events.truth_hits[field].ndim==3:
            events.truth_hits[field] \
                = events.truth_hits[field][:, 0:max_hits,  mask]

    events.meta['num_events'] = num_trimmed_events

    return events


