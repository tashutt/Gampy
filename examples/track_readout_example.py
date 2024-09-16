#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example of reading a Penelope track from disk and then reading
it out with different charge readout architectures

Created on Tue Apr 25 08:31:33 2023

@author: tshutt
"""

import os
import glob

import tracks_tools

#   Track energy and particle
energy = 750
particle = 'electrons'

#   Depth is a drift distance added to all the z values of
#   the raw track.  That is, each element of the track drifts through
#   a distance = depth - track.raw_track[2, :].   If depth = 0, then
#   track.raw_track[2, :] needs to be negative.
depth = 0.05

#   Find files - get list of files in folder with single energy tracks.
etag = f'E{energy:07.0f}'
p = {'root'  : \
     '/Users/tshutt/Documents/Work/Simulations/Penelope/Tracks/LAr'}
p['tracks'] = os.path.join(p['root'], particle, etag)
if not p['tracks']:
    print('*** Error: folder ' + etag + ' not found')
    raise SystemExit()
files= glob.glob(os.path.join(p['tracks'], 'TrackE*.npz'))

#   Read one track, with default charge readout
file_num = 1
track = tracks_tools.Tracks(files[file_num].strip('.npz'))

print(f'{energy/1000:3.02f} keV track {file_num:1.0f}'
      + f', with {track.truth["num_electrons"]:4.0f} e-'
      + f', at {depth:2.1f} m depth')

#%%   Set the charge readout to GAMPixD, then read out and blab.
track.reset_params(charge_readout_name='GAMPixD')
track.readout_charge(depth)

print(track.read_params.charge_readout_name)
print( '  charge surviving drift: '
      + f'{track.drifted_track["num_e"].sum():4.0f} e-')
print( '  pixels charge: '
      + f'{track.pixel_samples["samples_triggered"].sum():4.0f} e-')
print( '  coarse tiles charge: '
      + f'{track.coarse_tiles_samples["samples_triggered"].sum():4.0f} e-')

#   Display track alone, then display pixel readout alone
track.display(pixels=False)
track.display(raw_track=False)

#%%   Read out with LArPix
track.reset_params(charge_readout_name='LArPix')
track.readout_charge(depth)

print(track.read_params.charge_readout_name)
print( '  pixels charge: '
      + f'{track.pixel_samples["samples_triggered"].sum():4.0f} e-')

#%%   Read out with AnodeGrid
track.reset_params(charge_readout_name='AnodeGridD')
track.readout_charge(depth)

print(track.read_params.charge_readout_name)
print( '  anode grid charge: '
      + f'{track.anode_grid_samples["samples_triggered"].sum():4.0f} e-')

#%%   Back to GAMPixD, also changing some settings
noise = 1000
lifetime = 5e-3
track.reset_params(charge_readout_name='GAMPixD')
track.read_params.inputs['charge_drift']['electron_lifetime'] = lifetime
track.read_params.inputs['pixels']['noise'] = noise
track.readout_charge(depth)
print(track.read_params.charge_readout_name)

print(f'  Mod: pixels noise = {noise:3.0f} e-'
      + f', lifetime = {lifetime*1e3:3.1f} ms')
print( '  charge surviving drift: '
      + f'{track.drifted_track["num_e"].sum():4.0f} e-')
print( '  pixels charge: '
      + f'{track.pixel_samples["samples_triggered"].sum():4.0f} e-')
print( '  coarse tiles charge: '
      + f'{track.coarse_tiles_samples["samples_triggered"].sum():4.0f} e-')

