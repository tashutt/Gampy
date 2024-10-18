#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 22:26:32 2021

@author: tshutt
"""

import file_tools

import os
import time

#%%   Paths
paths = {}
paths['root'] = '/Users/tshutt/Documents/Work/Simulations/MEGALib'
paths['data'] = os.path.join(paths['root'], 'Data')
paths['plots'] =  os.path.join(paths['root'], 'Plots')

#%%   What to do

display_speed = 0.5

num_events = 20

topology_id=1
values_id=3

num_events = 1e4
gamma_energy = 1000
beam = 'FarFieldIsotropic'
# beam = 'FarFieldPointSource'
cos_theta = 1.0

inc_id_tag = '.inc1.id1'

sim_file_name = file_tools.get_sim_file_names(
    beam,
    gamma_energy,
    topology_id,
    values_id,
    )

#%%   Open

#   Open file
event = file_tools.Sim_Event(
    os.path.join(paths['data'], sim_file_name + inc_id_tag)
    )

# Read all events, put hit variables for each into long list
while 1:

    #   Load next event
    event.next_event()

    # #   End of File
    #  if text_line== '':
    #      break

    if 'ht' in event.sim:
         event.make_tree()
         event.show_tree()
         time.sleep(display_speed)
