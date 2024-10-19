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

# steering ['energies'] \
#     =  np.array([100, 200, 500, 1000, 2000, 5000, 10000], dtype=float)
# steering ['num_tracks'] \
#     =  np.array([2000, 2000, 2000, 1000, 1000, 500, 500], dtype=int)

steering ['energies'] =  np.array([10000], dtype=float)
steering ['num_tracks'] \
    = np.ones_like(steering ['energies'], dtype=int) * int(4000)

# steering ['energies'] = np.array([10000, 20000, 50000, 100000], dtype=float)
# steering ['num_tracks'] \
#       = np.ones_like(steering ['energies'], dtype=int) * int(1e4)

#   Use this for big tracks, otherwise omit.
compress = True
compression_scale = 30e-6
delete_raw = True

#   Paths
p={}
p['executable'] \
    ='/Users/tshutt/Documents/Work/AnalysisCode/Penelope/Code/execute'
p['output'] = '/Users/tshutt/Documents/Work/Simulations/Penelope/Tracks'

#%%  Launch simulation

penelope_tools.simple_penelope_track_maker(
    p,
    steering,
    initial_direction='random',
    wipe_folders=False,
    reset_origin=False,
    fresh_seed=True,
    delete_penelope_data=True,
    )

