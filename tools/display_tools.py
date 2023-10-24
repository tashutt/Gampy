#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 18:50:42 2023

Collection of event display routines

TODO[ts] display_track: huge amount of clean up
  - projections not finished.
  - can't select raw_track and drifted_track at same time

@author: tshutt
"""

def display(
        self,
        raw_track=True,
        drifted_track=False,
        pixels=True,
        projections=False,
        max_sample=None,
        plot_lims=None,
        initial_vector=True,
        track_center_at_origin=False,
        units='mm',
        view_angles=None
        ):
    """
    Display track in new fig, ax.
    old notes: set axes to same span, espeically for tracks only.
        something about plot lims for pixels
    """

    import numpy as np
    import matplotlib.pyplot as plt

    import electron_track_tools

    #   Prep variable to plot initial vector
    def prep_s(s, origin, scale):
        s_p = np.zeros((3, 2))
        s_p[0, :] = np.array([0, s[0]]) + origin[0]
        s_p[1, :] = np.array([0, s[1]]) + origin[1]
        s_p[2, :] = np.array([0, s[2]]) + origin[2]
        s_p = s_p * scale
        return s_p

    def prep_track(r, num_e, plot_lims, offset, delta):

        #   If needed, change delta to getreasonable bin size
        max_num_bins = 100
        if ((plot_lims[0, 1] - plot_lims[0, 0]) / delta) > max_num_bins:
            delta = (plot_lims[0, 1] - plot_lims[0, 0]) / max_num_bins

        #   Digitize
        bins = [
            np.arange(plot_lims[0, 0], plot_lims[0, -1], delta),
            np.arange(plot_lims[1, 0], plot_lims[1, -1], delta),
            np.arange(plot_lims[2, 0], plot_lims[2, -1], delta),
            ]
        samples, _ = np.histogramdd(
            [r[0, :], r[1, :], r[2, :]],
            bins,
            weights=num_e
            )

        #   Mesgrid of locations needed below for flat list of locations
        locations_x, locations_y, locations_z = np.meshgrid(
            bins[0][0:-1] + np.diff([bins[0][0:2]]) / 2,
            bins[1][0:-1] + np.diff([bins[1][0:2]]) / 2,
            bins[2][0:-1] + np.diff([bins[2][0:2]]) / 2,
            indexing='ij'
            )

        #   Raw is all charge, without noise
        mask = samples > 0
        n_e_p = samples[mask].flatten()
        r_p = np.zeros((3, np.sum(mask)))
        r_p[0, :] = locations_x[mask]
        r_p[1, :] = locations_y[mask]
        r_p[2, :] = locations_z[mask]

        return r_p, n_e_p

    #   Check input and deal with defaults

    if drifted_track and not hasattr(self, 'drifted_track'):
        print('*** Warning - no drifted track to display')
        drifted_track = False
    if pixels and not hasattr(self, 'pixel_samples'):
        pixels = False
    if track_center_at_origin:
        offset = self.raw_track['r'].mean(axis=1)
    else:
        offset = np.zeros((3,))

    #
    if not (raw_track or drifted_track or pixels):
        print('*** Nothing to display')
        return None, None, None

    if pixels and max_sample is None:
        max_sample = np.max(
            self.pixel_samples['samples_triggered']
            )

    if units=='m':
        scale = 1
    elif units=='mm':
        scale = 1000
    elif units=='cm':
        scale = 100
    elif units=='µm':
        scale = 1e6
    else:
        print('*** Error in display_track - bad units ***')

    #   Raw or drfited track - assign, and subtract offset
    if raw_track:
        r = (self.raw_track['r'] - offset.reshape(3,1)) * scale
        num_e = self.raw_track['num_e']
    elif drifted_track or pixels:
        r = (self.drifted_track['r'] - offset.reshape(3,1)) * scale
        num_e = self.drifted_track['num_e']

    #   Plot limits
    if plot_lims is None:
        if pixels:
            if 'r_raw' in self.pixel_samples:
                plot_lims = electron_track_tools.find_bounding_cube(
                    self.pixel_samples['r_raw']* scale, buffer=1.1)
            else:
                plot_lims = electron_track_tools.find_bounding_cube(
                    self.pixel_samples['r_triggered'] * scale, buffer=1.1)
        else:
            plot_lims = electron_track_tools.find_bounding_cube(
                r, buffer=1.1)

    #   Track gets dititized and scaled
    if raw_track or drifted_track:
        r_p, n_e_p = prep_track(r, num_e, plot_lims, offset, 1e-5 * scale)


    #   Origin
    origin = self.truth['origin'] - offset

    #   Initial direction, scaled.
    extent = np.diff(electron_track_tools.find_bounding_box(
        r,
        buffer=0.001
        )).max()
    s = self.truth['initial_direction'] * extent / 5

    #   Convient plotting variables
    s_p = prep_s(s, origin, scale)

    #   Tags
    track_tag \
        = f'{self.truth["track_energy"]:4.0f} keV, ' \
        + f'{self.truth["num_electrons"]/1000:4.1f}K e- \n'

    if pixels:
        pitch = self.params.pixels['pitch']
        noise = self.params.pixels['noise']
        # sampling_time = self.pixel_samples['sensors']['sampling_time']
        # threshold = self.pixel_samples['sensors']['noise'] \
        #     * self.pixel_samples['sensors']['threshold_sigma']
        pixel_tag \
            = f'{pitch*1e6:4.0f}' \
            + ' $\mu$m pitch, ' \
            + '$\sigma_e$ = ' \
            + f'{noise:4.1f} e-\n'
            # + ' ns sampling, ' \
            # + f' / {threshold:4.1f}  e- \n'
            # + f'{sampling_time*1e9:3.0f}' \
    else:
        pixel_tag = ''
    if (pixels or drifted_track) and ('depth' in self.drifted_track):
        drift_tag \
            = f'{self.drifted_track["depth"]*100:5.3f}' \
                + ' cm depth\n'
    else:
        drift_tag = ''

    #   Start plotting
    fig = plt.figure()

    #   Projections
    if projections:

        fig.set_size_inches(10, 10)

        fig.suptitle(track_tag)

        ax = fig.add_subplot(2, 2, 1)
        ax.plot(r_p[0,], r_p[2,], '.', ms=1)
        ax.scatter(origin[0], origin[1], marker='o', c='r', s = 36)
        ax.plot(s_p[0,], s_p[2,])

        ax.set_xlim(plot_lims[0, 0], plot_lims[0, 1]);
        ax.set_ylim(plot_lims[2, 0], plot_lims[2, 1]);
        ax.set_xlabel('x (' + units + ')')
        ax.set_ylabel('z (' + units + ')')
        ax.set_title('X-Z')

        ax = fig.add_subplot(2, 2, 2)
        ax.plot(r_p[1,], r_p[2,], '.', ms=1)
        ax.scatter(origin[0], origin[1], marker='o', c='r', s = 36)
        ax.plot(s_p[1,], s_p[2,])

        ax.set_xlim(plot_lims[1, 0], plot_lims[1, 1]);
        ax.set_ylim(plot_lims[2, 0], plot_lims[2, 1]);
        ax.set_xlabel('y (' + units + ')')
        ax.set_ylabel('z (' + units + ')')
        ax.set_title('Y-Z')

        ax = fig.add_subplot(2, 2, 4)
        ax.plot(r_p[0,], r_p[1,], '.', ms=1)
        ax.scatter(origin[0], origin[1], marker='o', c='r', s = 36)
        ax.plot(s_p[0,], s_p[1,])

        ax.set_xlim(plot_lims[0, 0], plot_lims[0, 1]);
        ax.set_ylim(plot_lims[1, 0], plot_lims[1, 1]);
        ax.set_xlabel('x (' + units + ')')
        ax.set_ylabel('y (' + units + ')')
        ax.set_title('X-Y')

    #   3D
    if projections:
        ax = fig.add_subplot(2, 2, 3, projection='3d')
    else:
        ax = fig.add_subplot(projection='3d')

    if raw_track or drifted_track:
        ax.scatter(r_p[0,], r_p[1,], r_p[2,],
            s = n_e_p / np.max(n_e_p) * 5**2,
            c = n_e_p
            )
        ax.scatter(origin[0], origin[1], origin[2],
                   marker='o', c='r', s = 12)
        if initial_vector:
            ax.plot(s_p[0,], s_p[1,], s_p[2], linewidth=1, c='r')

    #   Plot pixel samples
    if pixels and self.pixel_samples['r_triggered'].size:
        ax.scatter(
            (self.pixel_samples['r_triggered'][0, :]
             - offset[0]) * scale,
            (self.pixel_samples['r_triggered'][1, :]
             - offset[1]) * scale,
            (self.pixel_samples['r_triggered'][2, :]
             - offset[2]) * scale,
            s = self.pixel_samples['samples_triggered'] \
                / max_sample * 14**2,
            c = self.pixel_samples['samples_triggered']
            )

    # if drifted_track:
    # ax.text(-1.5*plot_lims[0, 1], 0, 1.1*plot_lims[0, 1],
    #     f'{self.drifted_track["drift_distance"]*100:4.2f} cm drift',
    #     fontsize=16
    #     )

    if view_angles:
        ax.view_init(elev=view_angles[0], azim=view_angles[1])

    ax.set_xlim(plot_lims[0, 0], plot_lims[0, 1]);
    ax.set_ylim(plot_lims[1, 0], plot_lims[1, 1]);
    ax.set_zlim(plot_lims[2, 0], plot_lims[2, 1]);
    ax.set_xlabel('x (' + units + ')')
    ax.set_ylabel('y (' + units + ')')
    ax.set_zlabel('z (' + units + ')')
    ax.set_title(track_tag + pixel_tag + drift_tag)

    return fig, ax, plot_lims
