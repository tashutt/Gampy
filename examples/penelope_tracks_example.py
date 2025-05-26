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

steering['particles'] = 'photons'
steering['material'] =  'LAr'

steering['folder_tag'] = ''

# steering ['energies'] \
#     =  np.array([100, 200, 500, 1000, 2000, 5000, 10000], dtype=float)
# steering ['num_tracks'] \
#     =  np.array([2000, 2000, 2000, 1000, 1000, 500, 500], dtype=int)

# steering ['energies'] =   np.array([100,   200,  500, 1000, 2000, 5000], dtype=float)
# steering ['num_tracks'] = int(1e4) * np.ones_like(steering ['energies'], dtype=int)
# steering ['num_tracks'] = np.array([6000, 6000, 6000, 8000, 8000, 9000], dtype=int)

steering['energies'] = 30000
steering['num_tracks'] = 10
# steering['eabs'] = np.tile(75, steering['energies'].size)
# steering['cs'] = np.tile(.02, steering['energies'].size)
# steering['wcs'] = np.tile(75, steering['energies'].size)

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
    wipe_folders=True,
    reset_origin=False,
    fresh_seed=True,
    delete_penelope_data=False,
    )

