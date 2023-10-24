#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summer 2021

Example of applying response study to events from .sim file.  

@author: tshutt
"""

import file_tools
import events_tools
import params_tools
import readout_studies

import os
import numpy as np
import matplotlib.pyplot as plt

#%%   Paths
paths = {}
paths['root'] = '/Users/tshutt/Documents/Work/Simulations/MEGALib'
paths['data'] = os.path.join(paths['root'], 'Data')
paths['plots'] =  os.path.join(paths['root'], 'Plots')

#%%   What to do

study = readout_studies.FullResponse()

num_events = 1e6
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

#%% Get events, apply response

events = events_tools.Events(
    file_names,
    num_events,
    read_sim_file=False,
    write_events_files=True
    )

params = params_tools.Params(file_names['path_geometry'])

#%%  Loop over study cases, applying detector response and plotting
#   energy resolution

fig, ax = plt.subplots(1, 1)

edges = np.arange(0, 1.1, 0.001) * gamma_energy
centers = edges[0:-1] + np.diff(edges)/2

for nc in range(study.kit['num_cases']):

    #   Define parameters and apply values for this study
    params = params_tools.Params(file_names['path_geometry'])
    params.apply_study_case(study, nc)

    #   Add detector response to MC truth
    events.apply_detector_response(params)

    counts, _ = np.histogram(
        events.measured_hits['total_energy'],
        edges
        )

    ax.step(centers, counts, label=study.labels['case'][nc])

ax.legend()
ax.grid(linestyle=':',linewidth='1')
ax.set_xlim(gamma_energy*0.95, gamma_energy*1.025)
ax.set_xlabel('Measured Energy (keV)')
ax.set_ylabel('Counts')
ax.set_title(file_names['base'])

