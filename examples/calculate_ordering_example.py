#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick routine to test ordering calculation.  Currently doesn't do anything
with calculated ordering

Created on Wed Aug  4 00:13:17 2021

@author: tshutt
"""

import file_tools
import events_tools
import params_tools

import os

#%%   Paths
paths = {}
paths['root'] = '/Users/tshutt/Documents/Work/Simulations/MEGALib'
paths['data'] = os.path.join(paths['root'], 'Data')
paths['plots'] =  os.path.join(paths['root'], 'Plots')

#%%   What to do

num_hits_list = [4, 5, 6]

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
events.apply_detector_response(params)

#%%     Cacluate ordering
events.calculate_ordering(num_hits_list)





