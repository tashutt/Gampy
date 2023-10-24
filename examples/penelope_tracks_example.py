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

#   Pick sets of energies, and numbrer of tracks at each energy.
#   Put in steering file
steering = {}
# steering['energies'] =  np.array([300, 500, 750, 1000, 1250, 1500, 2000,
#                                   2500, 3000, 3500, 4000, 5000])
# steering['num_tracks'] = 300 * np.ones_like(steering['energies'], dtype=int)

steering['energies'] =  np.array([7500, 10000])
steering['num_tracks'] = 100 * np.ones_like(steering['energies'], dtype=int)

# steering ['energies'] =  np.array([1000000])
# steering['num_tracks'] = 10 * np.ones(len(steering ['energies']), dtype=int)

#   Use this for big tracks, otherwise omit.
compression_bin_size = None

#   Paths
p={}
p['executable'] \
    ='/Users/tshutt/Documents/Work/AnalysisCode/Penelope/Code/execute'
p['output'] = '/Users/tshutt/Documents/Work/Simulations/Penelope/LAr/Tracks'

#%%  Launch simulation

penelope_tools.simple_penelope_track_maker(
    p,
    steering,
    random_initial_direction=True,
    reset_origin=False,
    wipe_folders=False,
    fresh_seed=True,
    delete_penelope_data=True,
    compression_bin_size=compression_bin_size
    )


