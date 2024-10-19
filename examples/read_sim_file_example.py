#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example of reading a .sim file to make events object

Created on Mon Jun 26 01:04:09 2023

@author: tshutt
"""
import os

import file_tools
import events_tools

#%%   Paths
paths = {}
paths['root'] = '/Users/tshutt/Documents/Work/Simulations/MEGALib'
paths['data'] = os.path.join(paths['root'], 'Data')
paths['plots'] =  os.path.join(paths['root'], 'Plots')

#%%   What to do

topology_id=1
values_id=3

num_events = int(1e4)
gamma_energy = 1000
beam = 'FarFieldIsotropic'
# beam = 'FarFieldPointSource'

cos_theta = 1.0

inc_id_tag = '.inc1.id1'

#%% Get events

# sim_file_name = 'AllSky550km_GeoT01v03.inc2.id1'
sim_file_name = 'FarFieldIsotropic_1.000MeV_GeoT01v03.inc1.id1'
# sim_file_name = file_tools.get_sim_file_name(
#     beam,
#     gamma_energy,
#     topology_id,
#     values_id,
#     )

geo_file_name = file_tools.get_geo_file_name(topology_id, values_id)

events = events_tools.Events(
    os.path.join(paths['data'], sim_file_name), # + inc_id_tag),
    os.path.join(paths['root'], geo_file_name),
    num_events,
    read_sim_file=False,
    )

#%% Apply response
# events.apply_detector_response()
