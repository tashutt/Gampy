"""
TODO: make these all dictionaries instead of classes

"""

class PixelPitchDriftDistance:
    """
      Separately scan:
          - drift distance
          - pixels pitch

      null is optional, default is False.  If True, then uses uses 5 cm drft
      and default pixel pitch - that is, nullifies the study

      As in all 2 parameter scans, the results can be thought of in two
      basis sets: (a) in the linear space of cases, or (b) in the 2d space
      of [depth][pitch]
      Tools to access both basis sets are supplied in study.kit

      10/20 phython port   TS
    """
    def __init__(self, null=False):

        import numpy as np

        from gampy.tools import readout_tools

        self.kit = {}

        #   The two sets of values - used unless null
        depths = np.array([0.002, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, \
                            0.075, 0.1, 0.2])
        pitches = np.array([0.4e-3, 0.5e-3, 0.6e-3])
        # pitches = np.array([0.2e-3, 0.3e-3, 0.4e-3, 0.5e-3, 0.75e-3])

        #   Assign values
        if not null:
            self.kit['pitches'] = pitches
            self.kit['depths'] = depths
            self.kit['select_depth_indices'] = [2, 6, 9]
            self.kit['select_pitch_indices'] = [0, 1, 2]
            self.kit['default_depth_index'] = 6
            self.kit['default_pitch_index'] = 3
        else:
            read_params = readout_tools.Params()
            self.kit['depths'] = [0.05]
            self.kit['pitches'] = [read_params.pixels['pitch']]
            self.kit['select_depth_indices'] = [0]
            self.kit['select_pitch_indices'] = [0]
            self.kit['default_depth_index'] = 0
            self.kit['default_pitch_index'] = 0

        #  Indices and such
        self.kit['drift_case_index'] = []
        self.kit['pitch_case_index'] = []
        self.kit['num_pitches'] = len(pitches)
        self.kit['num_depths'] = len(depths)
        self.kit['num_cases'] = len(pitches) * len(depths)
        self.kit['case'] = \
            [[0 for i in range(len(pitches))] \
             for j in range(len(depths))]

        #   Labels
        self.labels = {}

        self.labels['study_name'] = 'Pixel Pitch Drift'
        self.labels['study_tag'] = 'PixPitDri'

        self.labels['case'] = []
        self.labels['case_tag'] = []
        self.labels['depth'] = []
        self.labels['pitch'] = []

        #   Assign values, looping over both variables
        self.fields = []
        self.sub_fields = []
        self.values = []
        nc = 0
        for nd in range(len(depths)):
            for npp in range(len(pitches)):

                #   Indices
                self.kit['case'][nd][npp] = nc
                self.kit['pitch_case_index'].append(npp)
                self.kit['drift_case_index'].append(nd)
                nc += 1

                #   labels and tags
                self.labels['case'].append(
                    f'{depths[nd]*100:3.1f} cm drift'
                    + ', {pitches[npp]*1e6:3.0f}'
                    + r' $\mu$' + ' pitch'
                    )
                self.labels['case_tag'].append(
                    'D' + str(nd) + 'P' + str(npp))
                self.labels['depth'].append(
                    f'{depths[nd]*100:3.1f} cm drift'
                    )
                self.labels['pitch'].append(
                    f'{pitches[npp]*1e6:4.0f}'
                    + r' $\mu$m' + ' pitch'
                    )

                #   Save params fields and sub-fields, and values
                self.fields.append(['pixels'])
                self.sub_fields.append(['pitch'])
                self.values.append([pitches[npp]])
                # self.values.append([depths[nd], pitches[npp]])

class GAMPixDNoiseDepth():
    """
      Separately scan:
          - event depth
          - pixels noise

      null is optional, default is False.  If True, then uses uses 3 m drft
      and default pixel noise - that is, nullifies the study

      As in all 2 parameter scans, the results can be thought of in two
      basis sets: (a) in the linear space of cases, or (b) in the 2d space
      of [depth][noise]
      Tools to access both basis sets are supplied in study.kit

      3/32   TS
    """
    def __init__(self, null=False):

        import numpy as np
        from gampy.tools import readout_tools

        self.kit = {}

        #   The two sets of values - used unless null
        depths = np.arange(0.5, 7, 0.5)
        noises = np.array([50, 1000])

        #   Assign values
        if not null:
            self.kit['depths'] = depths
            self.kit['noises'] = noises
            self.kit['select_depth_indices'] = [3, 11]
        else:
            read_params = readout_tools.Params()
            self.kit['depths'] = [3]
            self.kit['noises'] = [read_params.pixels['noise']]
            self.kit['select_depth_index'] = 0

        #  Indices and such
        self.kit['depth_case_index'] = []
        self.kit['noise_case_index'] = []
        self.kit['num_depths'] = len(depths)
        self.kit['num_noises'] = len(noises)
        self.kit['num_cases'] = len(depths) * len(noises)
        self.kit['case'] = \
            [[0 for i in range(len(noises))] \
             for j in range(len(depths))]

        #   Labels
        self.labels = {}

        self.labels['study_name'] = 'Pixel Noise Depth'
        self.labels['study_tag'] = 'NoiseDepth'

        self.labels['case'] = []
        self.labels['case_tag'] = []
        self.labels['depth'] = []
        self.labels['noise'] = []

        #   Assign values, looping over both variables
        self.fields = []
        self.sub_fields = []
        self.values = []
        nc = 0
        for nd in range(len(depths)):
            for nn in range(len(noises)):

                #   Indices
                self.kit['case'][nd][nn] = nc
                self.kit['depth_case_index'].append(nd)
                self.kit['noise_case_index'].append(nn)
                nc += 1

                #   labels and tags
                self.labels['case'].append(
                    f'{depths[nd]:3.1f} m depth'
                    + ', '
                    + f'{noises[nn]:3.0f} e$^-$ ENC'
                    )
                self.labels['case_tag'].append(
                    'D' + str(nd) + 'N' + str(nn))
                self.labels['depth'].append(
                    f'{depths[nd]:3.1f} m depth'
                    )
                self.labels['noise'].append(
                    f'{noises[nn]:3.0f} e$^-$ ENC'
                    )

                #   Save params fields and sub-fields, and values
                self.fields.append(['pixels'])
                self.sub_fields.append(['noise'])
                self.values.append([depths[nd], noises[nn]])

class DUNEReadoutLifetimeDepth():
    """
    For DUNE, scan:
        - Charge readou - GAMPIxD, AnodeGridD
        - electron lifetime
        - event depth

    For readout. lifetime and depth indice nr, nl, nd, case index nc
        is study.kit['case'][nr][nl][nd]

    4/23  TS
    """
    def __init__(self):

        import numpy as np

        self.kit = {}

        #   Assign values to scan over. Here we have three variables.
        readouts = ['GAMPixD', 'AnodeGridD', 'LArPix']
        lifetimes = np.array([3., 5., 10.]) / 1000
        depths = np.arange(0.5, 7, 0.5)

        #   Assign values
        self.kit['readouts'] = readouts
        self.kit['lifetimes'] = lifetimes
        self.kit['depths'] = depths
        self.kit['select_lifetime_indices'] = [1]
        self.kit['select_depth_indices'] = [3, 11]

        #  Indices and such
        self.kit['readout_case_index'] = []
        self.kit['lifetime_case_index'] = []
        self.kit['depth_case_index'] = []
        self.kit['num_readouts'] = len(readouts)
        self.kit['num_lifetimes'] = len(lifetimes)
        self.kit['num_depths'] = len(depths)
        self.kit['num_cases'] = len(readouts) * len(lifetimes) * len(depths)
        self.kit['case'] = [[[0     # note here we reverse index order
            for j in range(len(depths))]
            for i in range(len(lifetimes))]
            for k in range(len(readouts))]
        self.kit['readout_indices'] = range(len(readouts))
        self.kit['lifetime_indices'] = range(len(lifetimes))
        self.kit['depth_indices'] = range(len(depths))

        #   Labels
        self.labels = {}

        self.labels['study_name'] = 'DUNE Readout, Lifetime and Depth'
        self.labels['study_tag'] = 'DUJNEReadoutLifetimeDepth'

        self.labels['case'] = []
        self.labels['case_tag'] = []
        self.labels['readout'] = []
        self.labels['lifetime'] = []
        self.labels['depth'] = []

        #   Variable labels
        for nr in range(len(readouts)):
            self.labels['readout'].append(
                readouts[nr]
                )
        for nl in range(len(lifetimes)):
            self.labels['lifetime'].append(
                f'{lifetimes[nl]*1e3:3.1f} ms e- lifetime'
                )
        for nd in range(len(depths)):
            self.labels['depth'].append(
                f'{depths[nd]:3.1f} m depth'
                )

        #   Assign values, looping over both variables
        self.fields = []
        self.sub_fields = []
        self.values = []
        nc = 0
        for nr in range(len(readouts)):
            for nl in range(len(lifetimes)):
                for nd in range(len(depths)):

                    #   Indices
                    self.kit['case'][nr][nl][nd] = nc
                    self.kit['readout_case_index'].append(nr)
                    self.kit['lifetime_case_index'].append(nl)
                    self.kit['depth_case_index'].append(nd)
                    nc += 1

                    #   labels and tags
                    self.labels['case'].append(
                        self.labels['readout'][nr]
                        + ', '
                        + self.labels['lifetime'][nl]
                        + ', '
                        + self.labels['depth'][nd]
                        )
                    self.labels['case_tag'].append(
                        'R' + str(nr)
                        + 'L' + str(nl)
                        + 'D' + str(nd)
                        )

                    #   For params inputs,put fields, sub-fields,
                    #   and values into list of arrays
                    self.fields.append(['charge_drift'])
                    self.sub_fields.append(['electron_lifetime'])
                    self.values.append([lifetimes[nl]])

class SpatialResolution:
    """
      Scan over spatial resolution

      null is optional, default is False.  If True, then uses default

      11/20 python port   TS
    """
    def __init__(self, null=False):

        #   Spatial resolutions to change, and default
        if not null:
            resolution_xy = [0.2e-3, 0.3e-3, 0.4e-3, 0.5e-3, 0.75e-3, 1e-3]
            resolution_z = resolution_xy
        else:
            from gampy.tools import response_definition
            response = response_definition.Response()
            resolution_xy = response.spatial_resolution.sigma_xy
            resolution_z = response.spatial_resolution.sigma_z

        #   Build study structure
        self.fields = \
            [['spatial_resolution', 'spatial_resolution']] \
            * len(resolution_xy)
        self.sub_fields =\
            [['sigma_xy', 'sigma_z']] \
            * len(resolution_xy)
        self.values = []
        for nc in range(len(resolution_xy)):
            self.values.append(
                [resolution_xy[nc], resolution_z[nc]]
                )

        #  Kit contains useful meta information
        self.kit = {}
        self.kit['num_cases'] = len(self.values)

        #   Study labels
        self.labels = {}

        self.labels['study_name'] = 'Spatial Resolution'
        self.labels['study_tag'] = 'SpaRes'

        self.labels['case'] = []
        self.labels['case_tag'] = []
        for nc in range(len(resolution_xy)):
            self.labels['case'].append(
                r'$\sigma_{xzy}$:'
                + f' {resolution_xy[nc]*1e6:3.0f}'
                + r' $\um$'
                )
            self.labels['case_tag'].append(
                'sxyz'
                + f'{resolution_xy[nc]*1e6:04.0f}'
                )

class FullResponse:
    """
      Vary several parameters to create optimistic, nominal and pessimistic
      values for both energy and spatial response

      12/19 TS
    """
    def __init__(self, null=False):

        self.fields = []
        self.sub_fields = []
        self.values = []

        self.labels = {}

        self.labels['study_name'] = 'Full Repsponse'
        self.labels['study_tag'] = 'FullRes'

        self.labels['case'] = []
        self.labels['case_tag'] = []

        #   Optimistic
        self.labels['case'].append('Optimistic')
        self.labels['case_tag'].append('Opt')
        self.fields.append([
            'spatial_resolution',
            'spatial_resolution',
            'coarse_grids',
            'light',
            'material'
            ])
        self.sub_fields.append([
            'sigma_xy',
            'sigma_z',
            'noise',
            'collection',
            'sigma_p'
            ])
        self.values.append([
            200e-6,
            200e-6,
            10,
            0.3,
            0.04
            ])

        #   Nominal
        self.labels['case'].append('Nominal')
        self.labels['case_tag'].append('Nom')
        self.fields.append([
            'spatial_resolution',
            'spatial_resolution',
            'coarse_grids',
            'light',
            'material'
            ])
        self.sub_fields.append([
            'sigma_xy',
            'sigma_z',
            'noise',
            'collection',
            'sigma_p'
            ])
        self.values.append([
            400e-6,
            400e-6,
            20,
            0.1,
            0.06
            ])

        #   Pessimistic
        self.labels['case'].append('Pessimistic')
        self.labels['case_tag'].append('Pes')
        self.fields.append([
            'spatial_resolution',
            'spatial_resolution',
            'coarse_grids',
            'light',
            'material'
            ])
        self.sub_fields.append([
            'sigma_xy',
            'sigma_z',
            'noise',
            'collection',
            'sigma_p'
            ])
        self.values.append([
            750e-6,
            750e-6,
            30,
            0.05,
            0.06
            ])

        #  Kit contains useful meta information
        self.kit = {}
        self.kit['num_cases'] = len(self.fields)



