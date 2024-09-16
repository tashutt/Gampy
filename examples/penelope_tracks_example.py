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

# steering ['energies'] \
#     =  np.array([100, 200, 500, 1000, 2000, 5000, 10000], dtype=float)
# steering ['num_tracks'] \
#     =  np.array([2000, 2000, 2000, 1000, 1000, 500, 500], dtype=int)

# steering ['energies'] = np.array([20000, 50000, 100000,
#                                   200000, 500000, 1000000], dtype=float)
# steering ['num_tracks'] \
#     =  np.array([25, 15, 10, 5, 2, 2], dtype=int)

steering ['energies'] = np.array([200000, 500000, 1000000], dtype=float)
steering ['num_tracks'] =  np.array([5, 2, 2], dtype=int)



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
    initial_direction=[0, 0, -1],
    wipe_folders = False,
    reset_origin=False,
    fresh_seed=True,
    delete_penelope_data=True,
    compression_bin_size=compression_bin_size
    )


