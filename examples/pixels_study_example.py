#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 12 17:14:22 2024

@author: tshutt
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 14:35:43 2020

@author: tshutt
"""

"""
   Generate statistics of tracks of a given energy, using pixel readout study

#  4/20    TS
"""

import numpy as np
import os
import sys
import glob
import matplotlib.pyplot as plt

import electron_track_tools
import readout_studies

#%% What to do

#   Conditions
energy = 1000

#   Pixel readout trade study
study  =  readout_studies.PixelPitchDriftDistance()

#   What
particles = 'electrons'

#   Folders
etag = f'E{energy:07.0f}'

p = {'base' : '/Users/tshutt/Documents/Work/Simulations/Penelope/LAr'}
p['tracks'] = os.path.join(p['base'],'Tracks', particles, etag)
if not os.path.isdir(p['tracks']):
    sys.exit('Folder ' + p['tracks'] + ' not found')
p['plots'] = os.path.join(p['base'], 'Plots/Tracks')

#%%   Initialize things

#   number of samples, for each drift distance and pitches
num_triggered_pixels = np.zeros((
    study.kit['num_depths'],
    study.kit['num_pitches']
    ))

#   number of samples, for each drift distance and pitches
num_triggered_electrons = np.zeros((
    study.kit['num_depths'],
    study.kit['num_pitches']
    ))

num_raw_electrons = np.zeros((
    study.kit['num_depths'],
    study.kit['num_pitches']
    ))

files = glob.glob(os.path.join(p['tracks'], 'TrackE*.npz'))

#   Load first track in folder
track = electron_track_tools.Track(files[0].strip('.npz'))

#%%   Loop over study cases and calculate

for nd in range(0, study.kit['num_depths']):
    for npp in range(study.kit['num_pitches']):

        #   Case number
        nc = study.kit['case'][nd][npp]

        #     Apply study case
        track.read_params.apply_study_case(study, nc)

        #   Readout pixels, at depth from study
        track.readout_charge(depth=study.kit['depths'][nd])

        #   Number of triggered pixels
        num_triggered_pixels[nd, npp] \
            = track.pixel_samples['samples_triggered'].size

        #   Sum of triggered electrons
        num_triggered_electrons[nd, npp] = np.sum(
                track.pixel_samples['samples_triggered']
                )

        #   Number of raw electrons
        num_raw_electrons[nd, npp] = np.sum(
                track.pixel_samples['samples_raw']
                )

#%%  Plot vs pitch, for selected drifts

fig, ax = plt.subplots()

drift_indices_to_plot = [0, 3, study.kit['num_depths']-1]

for nd in range(len(drift_indices_to_plot)):
    nc = study.kit['case'][drift_indices_to_plot[nd]][0]
    ax.plot(
        study.kit['pitches'] * 1e6,
        num_triggered_electrons[drift_indices_to_plot[nd], :],
        label = study.labels['depth'][nc]
        )

ax.legend()

ax.grid(linestyle=':',linewidth='1')
ax.set_xlim(study.kit['pitches'][0]*1e6, study.kit['pitches'][-1]*1e6)
ax.set_ylim(0, ax.get_ylim()[1])

ax.set_xlabel('Pixel Pitch (µm)')
ax.set_ylabel('Measured Electrons')
ax.set_title(f'{energy:4.0f} keV electrons, Triggered Pixels')

#%%  Plot vs pitch, for selected drifts

fig, ax = plt.subplots()

pitch_indices_to_plot = [0, 3, study.kit['num_pitches']-1]

for npp in range(len(pitch_indices_to_plot)):
    nc = study.kit['case'][0][pitch_indices_to_plot[npp]]
    ax.plot(
        study.kit['depths'] * 100,
        num_triggered_electrons[:, pitch_indices_to_plot[npp]],
        label = study.labels['pitch'][nc]
        )

ax.legend()

ax.grid(linestyle=':',linewidth='1')
# ax.set_xlim(study.kit['pitches'][0]*1e6, study.kit['pitches'][-1]*1e6)
ax.set_ylim(0, ax.get_ylim()[1])

ax.set_xlabel('Depth (cm)')
ax.set_ylabel('Measured Electrons')
ax.set_title(f'{energy:4.0f} keV electrons, Triggered Pixels')

