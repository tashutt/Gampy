"""
This example runs PENELOPE to generate tracks.

steering specifies energies and numbers of tracks per energy.

The PENELOPE output files are large text files, which are parsed
in python and the output written in a pair of
a numpy (.npz) nd pickle files, one pair per track.  The raw text files
are then deleted.
Tracks from each energy go into a separate folder.

energies in keV

11/9/21     TS
"""

import penelope_tools

import numpy as np

#  What to do

#   Pick sets of energies, particles and numbers of tracks - put
#   in steering file
#   Particle: 1=e-, 2=g, 3=e+

steering = {}

steering['particles'] =  2
steering['material'] =  'LAr'
# steering['material'] =  'LXe'

# steering['energies'] =  np.array([300, 500, 750, 1000, 1250, 1500, 2000,
#                                   2500, 3000, 3500, 4000, 5000])
# steering['num_tracks'] = 300 * np.ones_like(steering['energies'], dtype=int)

energy = 10000
num_events = 500

steering ['energies'] =  np.array([energy], dtype=int)
steering['num_tracks'] = num_events \
    * np.ones(len(steering ['energies']), dtype=int)

#   Use this for big tracks, otherwise omit.
compression_bin_size = None

#   Paths
p={}
p['executable'] \
    ='/Users/tshutt/Documents/Work/AnalysisCode/Penelope/Code/execute'
p['output'] = '/Users/tshutt/Documents/Work/Simulations/Penelope/Tracks'

#%%  Launch simulation

penelope_tools.simple_penelope_track_maker(
    p,
    steering,
    random_initial_direction=True,
    reset_origin=False,
    fresh_seed=True,
    delete_penelope_data=False,
    compression_bin_size=compression_bin_size
    )


