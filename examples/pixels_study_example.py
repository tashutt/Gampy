#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 14:35:43 2020

@author: tshutt
"""

"""
   Statistics of pixel readout of drifted tracks, with study of 
    pixel params

#  4/20    TS
"""

import numpy as np
import os
import matplotlib.pyplot as plt

import electron_track_tools
import params_tools
import readout_studies

#%% What to do

#   Conditions
energy = 1000
drift_distance = 0.05

#   Pixel readout trade study
study  =  readout_studies.PixelPitchDriftDistance()

#   Folders
paths = {'root': 
         '/Users/tshutt/Documents/Work/Simulations/Penelope/LAr'}
paths['plots'] =  os.path.join(paths['root'], 'Plots')
paths['data'] = os.path.join(paths['root'], 'Tracks', f'E{energy:06.0f}')

#%%   Initialize things

#   number of samples, for each drift distance and pitches
num_triggered_pixels = np.zeros((
    study.kit['num_drift_distances'],
    study.kit['num_pitches']
    ))

#   number of samples, for each drift distance and pitches
num_electrons = np.zeros((
    study.kit['num_drift_distances'],
    study.kit['num_pitches']
    ))

num_electrons_raw = np.zeros((
    study.kit['num_drift_distances'],
    study.kit['num_pitches']
    ))

#   Load first track in folder
track = electron_track_tools.Track(
    os.path.join(paths['data'],'Track' + f'E{energy:06.0f}_00001')
    )

#%%   Loop over study cases and calculate

for nd in range(study.kit['num_drift_distances']):
    for npp in range(study.kit['num_pitches']):
        
        #   Case number
        nc = study.kit['case'][nd][npp]

        #   Define params (to be safe, do this every time), then
        #   add this study case to it
        params = params_tools.Params()
        params.apply_study_case(study,nc)
    
        #   Drift track
        track.drift_track(drift_distance, params)
        
        #   Readout pixels
        track.readout_pixels(params)
                   
        #   Number of triggered pixels
        num_triggered_pixels[nd, npp] = np.sum(
                track.pixels['samples']['triggered_pixels']
                )

        #   Number of triggered pixels
        num_electrons[nd, npp] = np.sum(
                track.pixels['samples']['flat']['with_noise']
                )

        num_electrons_raw[nd, npp] = np.sum(
                track.pixels['samples']['raw']
                )

#%%  Plot vs pitch, for selected drifts

fig, ax = plt.subplots()

drift_indices_to_plot = [0, 3, study.kit['num_drift_distances']-1]

for nd in range(len(drift_indices_to_plot)):
    nc = study.kit['case'][drift_indices_to_plot[nd]][0]
    ax.plot(
        study.kit['pitches'] * 1e6,
        num_triggered_pixels[drift_indices_to_plot[nd], :],    
        label = study.labels['drift_distance'][nc]
        )

ax.legend()
        
ax.grid(linestyle=':',linewidth='1')
ax.set_xlim(study.kit['pitches'][0]*1e6, study.kit['pitches'][-1]*1e6)
ax.set_ylim(0, ax.get_ylim()[1])

ax.set_xlabel('Pixel Pitch (µm)')
ax.set_ylabel('Number of Triggered Pixels')
ax.set_title(f'{energy:4.0f} keV electrons, Triggered Pixels')

#%%  Plot vs pitch, for selected drifts

fig, ax = plt.subplots()

pitch_indices_to_plot = [0, 3, study.kit['num_pitches']-1]

for npp in range(len(pitch_indices_to_plot)):
    nc = study.kit['case'][0][pitch_indices_to_plot[npp]]
    ax.plot(
        study.kit['drift_distances'] * 100,
        num_electrons[:, pitch_indices_to_plot[npp]],
        label = study.labels['pitch'][nc]
        )

ax.legend()
        
ax.grid(linestyle=':',linewidth='1')
# ax.set_xlim(study.kit['pitches'][0]*1e6, study.kit['pitches'][-1]*1e6)
# ax.set_ylim(0, ax.get_ylim()[1])

ax.set_xlabel('Drift distance (cm)')
ax.set_ylabel('Number of Triggered Pixels')
ax.set_title(f'{energy:4.0f} keV electrons, Triggered Pixels')



