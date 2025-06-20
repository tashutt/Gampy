#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 18:50:42 2023

Collection of event display routines

TODO[ts] display_track: huge amount of clean up
  - projections not finished.
  - can't select raw and drifted at tracks same time

@author: tshutt
"""

def display_track(track,
                  raw=True,
                  drifted=False,
                  pixels=True,
                  resolution=1e-5,
                  max_num_e=None,
                  max_pixel_samples=None,
                  plot_lims=None,
                  units='cm',
                  show_initial_vector = True,
                  center = False,
                  show_origin = True,
                  view_angles = None,
                  no_axes = False,
                  show_title = True,
                  ):

    """
    Display track in new fig, ax.
    old notes: set axes to same span, espeically for tracks only.
        something about plot lims for pixels
    """

    import numpy as np
    import matplotlib.pyplot as plt

    from . import tracks_tools

    #   Prep variable to plot initial vector
    def prep_s(s, origin, scale):
        s_p = np.zeros((3, 2))
        s_p[:, 0] = origin * scale
        s_p[:, 1] = (s.squeeze() + origin) * scale
        return s_p

    def prep_track(r, num_e, resolution, scale, track_extent):

        #   If needed, change resolution to getreasonable bin size
        max_num_bins = 500
        if (track_extent / resolution) > max_num_bins:
            resolution = track_extent  / max_num_bins

        plot_lims = tracks_tools.find_bounding_box(
            r,
            buffer = 0.01 * track_extent
            )

        #   Digitize
        bins = [
            np.arange(plot_lims[0, 0], plot_lims[0, -1], resolution),
            np.arange(plot_lims[1, 0], plot_lims[1, -1], resolution),
            np.arange(plot_lims[2, 0], plot_lims[2, -1], resolution),
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

        #   Scale r_p for plotting
        r_p = r_p * scale

        # r_p = r * scale
        # n_e_p = num_e

        return r_p, n_e_p

    #   Check inpus and assign defaults

    if drifted and not hasattr(track, 'drifted'):
        drifted = False
    if pixels and not hasattr(track, 'pixel_samples'):
        pixels = False
    if not (raw or drifted or pixels):
        print('*** Nothing to display')
        return None, None, None

    if units=='m':
        scale = 1
    elif units=='mm':
        scale = 1000
    elif units=='cm':
        scale = 100
    elif units=='µm':
        scale = 1e6
    elif units=='nm':
        scale = 1e9
    else:
        print('*** Error in display_track - bad units ***')

    if center:
        offset = track.raw['r'].mean(axis=1)
    else:
        offset = np.zeros((3,))

    #   Raw, or drifted track - assign, and subtract offset
    if (raw  and  hasattr(track, 'raw')):
        r = (track.raw['r'] - offset.reshape(3,1))
        num_e = track.raw['num_e']
    elif drifted or pixels:
        r = (track.drifted['r'] - offset.reshape(3,1))
        num_e = track.drifted['num_e']
    else:
        print('Warningin display_track: inconsistent inputs')
        return

    #   Extent of track and pixels
    track_extent = np.diff(tracks_tools.find_bounding_cube(
        r,
        buffer=0
        )).max()

    #   Track gets digitized at physical scale resolution, and converted
    #   to scale for plotting
    if raw or drifted:
        r_p, n_e_p = prep_track(r, num_e, resolution, scale, track_extent)

    #   Plot limits
    if plot_lims is None:
        if pixels: # and not raw:
            pitch = track.read_params.pixels['pitch']
            if 'r_raw' in track.pixel_samples:
                plot_lims = tracks_tools.find_bounding_cube(
                    track.pixel_samples['r_raw'] * scale, buffer = pitch / 2)
            else:
                plot_lims = tracks_tools.find_bounding_cube(
                    track.pixel_samples['r_triggered'] * scale,
                    buffer = pitch / 2)
        else:
            plot_lims = tracks_tools.find_bounding_cube(
                r_p, buffer = track_extent * 0.05)

    #   Origin
    origin = track.truth['origin'] - offset

    #   Initial direction, scaled.
    s = track.truth['initial_direction'] * track_extent / 5

    #   Convient plotting variables
    s_p = prep_s(s, origin, scale)
    o_p = origin * scale

    #   Tags
    if track.truth["num_electrons"] < 1000:
        num_tag = f', {track.truth["num_electrons"]:d} e-'
    elif track.truth["num_electrons"] < 1e6:
        num_tag = f', {track.truth["num_electrons"]/1000:4.1f}K e-'
    else:
        num_tag = f', {track.truth["num_electrons"]/1e6:4.1f}K e-'
    if 'material' in track.meta:
        material_tag = ' in ' + track.meta['material']
    else:
        material_tag = ''
    # if track.meta['initial_particle']==1:
    #     particle_tag = 'e'
    track_tag = (
        f'{track.truth["initial_particle_energy"]:4.0f} keV'
        + material_tag
        + num_tag
        )

    if pixels:
        track_tag = track_tag + '; '
        pixel_tag \
            = f'\n{track.read_params.pixels["pitch"]*1e6:4.0f}' \
            + ' $\mu$m pitch, ' \
            + '$\sigma_e$ = ' \
            + f'{track.read_params.pixels["noise"]:4.1f} e-'
    else:
        pixel_tag = ''
    if (pixels or drifted) and ('depth' in track.drifted):
        drift_tag \
            = f'\n{track.drifted["depth"]*100:5.2f}' \
                + ' cm depth\n'
    else:
        drift_tag = ''

    #   Set maximum charges or samples
    if  (raw or drifted) and max_num_e is None:
        max_num_e = n_e_p.max()
    if  pixels and max_pixel_samples is None:
        max_pixel_samples = np.max(
            track.pixel_samples['samples_triggered']
                )

    #   Start plotting
    fig = plt.figure()

    #   3D
    ax = fig.add_subplot(projection='3d')

    if raw or drifted:
        ax.scatter(
            r_p[0,],
            r_p[1,],
            r_p[2,],
            s = n_e_p / max_num_e * 5**2,
            c = n_e_p / max_num_e
            )
        # plt.colorbar(im, shrink = 0.5, aspect=15, pad=0.15)

        if show_origin:
            ax.scatter(o_p[0], o_p[1], o_p[2], marker='o', c='r', s = 12)

        if show_initial_vector:
            ax.plot(s_p[0,], s_p[1,], s_p[2,], linewidth=1, c='brown')

    #   Plot pixel samples
    if pixels and track.pixel_samples['r_triggered'].size:
        ax.scatter(
            (track.pixel_samples['r_triggered'][0, :]
             - offset[0]) * scale,
            (track.pixel_samples['r_triggered'][1, :]
             - offset[1]) * scale,
            (track.pixel_samples['r_triggered'][2, :]
             - offset[2]) * scale,
            s = track.pixel_samples['samples_triggered'] \
                / max_pixel_samples * 10**2,
            c = track.pixel_samples['samples_triggered'] \
                / max_pixel_samples
            )

    # if drift_tag!='':
    #     ax.text(-1.5*plot_lims[0, 1], -1.5*plot_lims[0, 1], 1*plot_lims[0, 1],
    #         drift_tag,
    #         fontsize=12,
    #         color='blue',
    #         )

    if view_angles:
        ax.view_init(elev=view_angles[0], azim=view_angles[1])

    #   Turn of axes
    if no_axes:
        ax.set_axis_off()

    ax.set_xlim(plot_lims[0, 0], plot_lims[0, 1]);
    ax.set_ylim(plot_lims[1, 0], plot_lims[1, 1]);
    ax.set_zlim(plot_lims[2, 0], plot_lims[2, 1]);
    ax.set_xlabel('x (' + units + ')')
    ax.set_ylabel('y (' + units + ')')
    ax.set_zlabel('z (' + units + ')')
    if show_title:
        ax.set_title(track_tag + pixel_tag + drift_tag, fontweight='bold')

    plt.tight_layout()

    return fig, ax, plot_lims
