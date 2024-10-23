#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example of creating a Geomega geometry .source file

6/16/23

@author: tshutt
"""

import os

from gampy.tools import sims_tools
from gampy.tools import file_tools

#   Paths
paths = {}
paths['root'] = '/Users/tshutt/Documents/Work/Simulations/MEGALib'
paths['data'] = os.path.join(paths['root'], 'data')

#%% Generate geometry file with default inputs, write to root path

#   Load default geo params
sims_params = sims_tools.Params(detector_geometry='geomega',
                                    cell_geometry='hexagonal')

#   Write file
geo_file_name = file_tools.write_geo_files(paths['root'], sims_params)

#%%  Generate file with modified parameter value

#   Change the overall radius
sims_params.inputs['vessel']['r_outer'] = 0.6

#   Write again, but add values id to distinguis from default
#   (default is values_id=0)
mod_geo_file_name = file_tools.write_geo_files(
    paths['root'],
    sims_params,
    values_id=3
    )

#%% Write source file for Cosima run.

#   Run settings
num_events = 1e4
gamma_energy = 1000
beam = 'FarFieldIsotropic'
# beam = 'FarFieldPointSource'
cos_theta = 1.0

file_tools.write_source_file(
    paths['data'],
    os.path.join(paths['root'], mod_geo_file_name),
    num_events,
    beam,
    gamma_energy,
    cos_theta=cos_theta,
    )
