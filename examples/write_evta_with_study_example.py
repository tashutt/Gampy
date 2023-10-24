#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test / example script. Read and process sim file, read study and
add detector response for each study case and write .evta file

@author: tshutt
11/17/20
"""

import file_tools
import events_tools
import params_tools
import APRA2021_readout_studies

import os

#%%   Paths
paths = {}
paths['root'] = '/Users/tshutt/Documents/Work/Simulations/MEGALib'
paths['data'] = os.path.join(paths['root'], 'Data')
paths['plots'] =  os.path.join(paths['root'], 'Plots')

#%%   What to do

study = APRA2021_readout_studies.FullResponse()

evta_version = '200'

num_events = 100
gamma_energy = 1000

source = 'FarFieldPointSource'
cos_theta = '1.0'
# cos_theta = '0.6'

geo_version = '2.0'

inc = 1

file_names = file_tools.get_file_names(
    paths,
    source,
    gamma_energy,
    cos_theta,
    geo_version,
    inc
    )

#%% Get events
events = events_tools.Events(
    file_names,
    num_events,
    read_sim_file=False,
    write_events_files=False
    )

#%%   Loop on study cases
for nc in range(study.kit['num_cases']):

    #    Apply detector response for this case
    params = params_tools.Params(file_names['path_geometry'])
    params.apply_study_case(study, nc)
    events.apply_detector_response(params)

    #   Write evta file
    events.write_evta_file(evta_version)







