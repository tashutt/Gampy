#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 18:50:42 2023

Collection of event display routines

TODO[ts] display_track: huge amount of clean up
  - projections not finished.
  - can't select raw_track and drifted_track at same time

@author: tshutt

Added reconstructed events visualisation tools:
run visualize_events to get:
 - 3D sphere with circles
 - 2D projection of that
 - Image of intensities

TODO:
    - Circles with electron track reconstructions (arches) 
    - point source detection and stats
@author: bahrudint
"""

def display(
        self,
        raw_track=True,
        drifted_track=False,
        pixels=True,
        resolution=1e-5,
        max_num_e=None,
        max_pixel_samples=None,
        plot_lims=None,
        show_initial_vector=True,
        show_origin=True,
        track_center_at_origin=False,
        units='mm',
        view_angles=None,
        no_axes=False,
        show_title=True,
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
        s_p[:, 0] = origin * scale
        s_p[:, 1] = (s.squeeze() + origin) * scale
        return s_p

    def prep_track(r, num_e, resolution, scale, track_extent):

        #   If needed, change resolution to getreasonable bin size
        max_num_bins = 100
        if (track_extent / resolution) > max_num_bins:
            resolution = track_extent  / max_num_bins

        plot_lims = electron_track_tools.find_bounding_box(
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

        return r_p, n_e_p

    #   Check input and deal with defaults

    if drifted_track and not hasattr(self, 'drifted_track'):
        print('*** Warning - no drifted track to display')
        drifted_track = False
    if pixels and not hasattr(self, 'pixel_samples'):
        print('*** Error in display: no pixel redaout to display')
        pixels = False
    if track_center_at_origin:
        offset = self.raw_track['r'].mean(axis=1)
    else:
        offset = np.zeros((3,))

    #
    if not (raw_track or drifted_track or pixels):
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
    else:
        print('*** Error in display_track - bad units ***')

    #   Raw or drfited track - assign, and subtract offset
    if raw_track:
        r = (self.raw_track['r'] - offset.reshape(3,1))
        num_e = self.raw_track['num_e']
    elif drifted_track or pixels:
        r = (self.drifted_track['r'] - offset.reshape(3,1))
        num_e = self.drifted_track['num_e']

    #   Extent of track and pixels
    track_extent = np.diff(electron_track_tools.find_bounding_cube(
        r,
        buffer=0
        )).max()

    #   Track gets digitized at physical scale resolution, and converted
    #   to scale for plotting
    if raw_track or drifted_track:
        r_p, n_e_p = prep_track(r, num_e, resolution, scale, track_extent)

    #   Plot limits
    if plot_lims is None:
        if pixels:
            pitch = self.params.pixels['pitch']
            if 'r_raw' in self.pixel_samples:
                plot_lims = electron_track_tools.find_bounding_cube(
                    self.pixel_samples['r_raw'] * scale, buffer = pitch / 2)
            else:
                plot_lims = electron_track_tools.find_bounding_cube(
                    self.pixel_samples['r_triggered'] * scale,
                    buffer = pitch / 2)
        else:
            plot_lims = electron_track_tools.find_bounding_cube(
                r_p, buffer = track_extent * 0.05)

    #   Origin
    origin = self.truth['origin'] - offset

    #   Initial direction, scaled.
    s = self.truth['initial_direction'] * track_extent / 5

    #   Convient plotting variables
    s_p = prep_s(s, origin, scale)
    o_p = origin * scale

    #   Tags
    track_tag \
        = f'{self.truth["track_energy"]:4.0f} keV, ' \
        + f'{self.truth["num_electrons"]/1000:4.1f}K e-'

    if pixels:
        track_tag = track_tag + '; '
        pixel_tag \
            = f'\n{self.params.pixels["pitch"]*1e6:4.0f}' \
            + ' $\mu$m pitch, ' \
            + '$\sigma_e$ = ' \
            + f'{self.params.pixels["noise"]:4.1f} e-'
    else:
        pixel_tag = ''
    if (pixels or drifted_track) and ('depth' in self.drifted_track):
        drift_tag \
            = f'\n{self.drifted_track["depth"]*100:5.2f}' \
                + ' cm depth\n'
    else:
        drift_tag = ''

    #   Set maximum charges or samples
    if  (raw_track or drifted_track) and max_num_e is None:
        max_num_e = n_e_p.max()
    if  pixels and max_pixel_samples is None:
        max_pixel_samples = np.max(
            self.pixel_samples['samples_triggered']
                )

    #   Start plotting
    fig = plt.figure()

    #   3D
    ax = fig.add_subplot(projection='3d')

    if raw_track or drifted_track:
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
    if pixels and self.pixel_samples['r_triggered'].size:
        ax.scatter(
            (self.pixel_samples['r_triggered'][0, :]
             - offset[0]) * scale,
            (self.pixel_samples['r_triggered'][1, :]
             - offset[1]) * scale,
            (self.pixel_samples['r_triggered'][2, :]
             - offset[2]) * scale,
            s = self.pixel_samples['samples_triggered'] \
                / max_pixel_samples * 10**2,
            c = self.pixel_samples['samples_triggered'] \
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



###  Display tools for 3D circles and Mollweide projection
###  # 1. Compute points on a circle given an axis and an angle.
###  # 2. Plot 3D circles on the provided axis based on the given events data.
###  # 3. Plot a quiver arrow indicating the direction based on the given angle.
###  # 4. Project 3D points to a Mollweide projection.
###  # 5. Rotate points to align with the z-axis based on the given vector.
###  # 6. Plot circles on a Mollweide projection based on the given events data.
###  # 7. Accumulate 2D points into a grid for visualization.
###  # 8. Plot the accumulated map using imshow.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pyproj import Proj

def compute_circle_points(axis, angle, num_points=100):
    """
    Compute points on a circle given an axis and an angle.

    Parameters:
    axis (array-like): Array of shape (2,) representing the axis angles (psi, xsi).
    angle (float): The compton angle.
    num_points (int, optional): Number of points to generate on the circle. Default is 100.

    Returns:
    np.ndarray: Array of shape (num_points, 3) containing the 3D coordinates of the circle points.
    """
    theta = np.linspace(0, 2 * np.pi, num_points)
    circle_x = np.sin(angle) * np.cos(theta)
    circle_y = np.sin(angle) * np.sin(theta)
    circle_z = np.cos(angle) * np.ones_like(theta)
    
    rotation_matrix = np.array([
        [np.cos(axis[0]) * np.cos(axis[1]), -np.sin(axis[0]), np.cos(axis[0]) * np.sin(axis[1])],
        [np.sin(axis[0]) * np.cos(axis[1]),  np.cos(axis[0]), np.sin(axis[0]) * np.sin(axis[1])],
        [-np.sin(axis[1]), 0, np.cos(axis[1])]
    ])
    
    circle_points = np.vstack((circle_x, circle_y, circle_z)).T
    rotated_circle_points = circle_points.dot(rotation_matrix.T)
    
    return rotated_circle_points

def project_to_mollweide(points):
    """
    Project 3D points to a Mollweide projection.

    Parameters:
    points (np.ndarray): Array of shape (n, 3) containing the 3D coordinates of the points.

    Returns:
    np.ndarray: Array of shape (n, 2) containing the projected 2D coordinates.
    """
    lon = np.arctan2(points[:, 1], points[:, 0]) * 180 / np.pi
    lat = np.arcsin(points[:, 2]) * 180 / np.pi
    proj = Proj(proj='moll')
    x, y = proj(lon, lat)
    return np.vstack((x, y)).T

def rotate_to_align_with_z(points, vector):
    """
    Rotate points to align with the z-axis based on the given vector.

    Parameters:
    points (np.ndarray): Array of shape (n, 3) containing the 3D coordinates of the points.
    vector (np.ndarray): Array of shape (3,) representing the direction vector.

    Returns:
    np.ndarray: Array of shape (n, 3) containing the rotated 3D coordinates.
    """
    vector = vector / np.linalg.norm(vector)
    z_axis = np.array([1, 0, 0])
    rotation_axis = np.cross(vector, z_axis)
    rotation_angle = np.arccos(np.dot(vector, z_axis))
    
    # Create the rotation matrix using the Rodrigues' rotation formula
    K = np.array([
        [0, -rotation_axis[2], rotation_axis[1]],
        [rotation_axis[2], 0, -rotation_axis[0]],
        [-rotation_axis[1], rotation_axis[0], 0]
    ])
    I = np.eye(3)
    R = I + np.sin(rotation_angle) * K + (1 - np.cos(rotation_angle)) * np.dot(K, K)
    
    rotated_points = points.dot(R.T) 
    return rotated_points

def accumulate_in_grid(image_data, grid_size):
    """
    Accumulate 2D points into a grid for visualization.

    Parameters:
    image_data (np.ndarray): Array of shape (n, 100, 2) containing the 2D coordinates of the points.
    grid_size (int): Size of the grid.

    Returns:
    np.ndarray: Accumulated grid of shape (grid_size, grid_size).
    """
    grid = np.zeros((grid_size, grid_size))

    for k in range(image_data.shape[0]):
        circle_points_2d = image_data[k]
        if np.sum(circle_points_2d[:, 0]) == 0:
            continue
        x = circle_points_2d[:, 0]
        y = circle_points_2d[:, 1]
        
        # Normalize points to fit in the grid
        x = (x + 180) / 360 * (grid_size - 1)
        y = (y + 90) / 180 * (grid_size - 1)
        
        # Accumulate values in the grid
        for i in range(len(x)):
            grid[int(y[i]), int(x[i])] += 1
    
    return grid

def plot_3d_circles(ax, events, threshold=3.5, num_events=1000):
    """
    Plot 3D circles on the provided axis based on the given events data.

    Parameters:
    ax (Axes3D): 3D axis to plot on.
    events (pd.DataFrame): DataFrame containing event data with columns 'use', 'compton_angle', 'psi_angle', and 'xsi_angle'.
    threshold (float, optional): Threshold for the compton angle to filter events. Default is 3.5.
    num_events (int, optional): Number of events to plot. Default is 1000.
    """
    for k in range(min(num_events, len(events))):
        event = events.iloc[k]
        if event['use'] == 1 and event['compton_angle'] < threshold:
            psi = event['psi_angle']
            xsi = event['xsi_angle']
            compton_angle = event['compton_angle']

            axis = np.array([psi, xsi])
            circle_points = compute_circle_points(axis, compton_angle)

            ax.plot(circle_points[:, 0], 
                    circle_points[:, 1], 
                    circle_points[:, 2], 
                    alpha=0.07,
                    linewidth=2,
                    color='b')

def plot_quiver(ax, angle):
    """
    Plot a quiver arrow indicating the direction based on the given angle.

    Parameters:
    ax (Axes3D): 3D axis to plot on.
    angle (float): Angle to determine the direction of the quiver.
    """
    angle_rad = np.arccos(angle)
    ax.quiver(0, 0, 0, 
              -np.sin(angle_rad), 0, 
              np.cos(angle_rad), color='r',
              alpha=0.6,
              linewidth=2)

def plot_mollweide_circles(ax, events, direction_vector, threshold=0.3, num_events=10000):
    """
    Plot circles on a Mollweide projection based on the given events data.

    Parameters:
    ax (Axes): 2D axis to plot on.
    events (pd.DataFrame): DataFrame containing event data with columns 'use', 'compton_angle', 'psi_angle', and 'xsi_angle'.
    direction_vector (np.ndarray): Array of shape (3,) representing the direction vector.
    threshold (float, optional): Threshold for the compton angle to filter events. Default is 0.3.
    num_events (int, optional): Number of events to plot. Default is 10000.

    Returns:
    np.ndarray: Array of shape (num_events, 100, 2) containing the 2D coordinates of the circles.
    """
    image_data = np.zeros((num_events, 100, 2))

    for k in range(min(num_events, len(events))):
        event = events.iloc[k]
        if event['use'] == 1 and event['compton_angle'] > threshold:
            psi = event['psi_angle']
            xsi = event['xsi_angle']
            compton_angle = event['compton_angle']
            
            axis = np.array([psi, xsi])
            circle_points = compute_circle_points(axis, compton_angle)
            rotated_circle_points = rotate_to_align_with_z(circle_points, direction_vector)
            circle_points_2d = project_to_mollweide(rotated_circle_points)
            image_data[k] = circle_points_2d / 1e5
            
            ax.plot(circle_points_2d[:, 0] / 1e5, 
                    circle_points_2d[:, 1] / 1e5, 
                    alpha=0.04,
                    linewidth=2,
                    color='b')

    return image_data

def plot_accumulated_map(grid):
    """
    Plot the accumulated map using imshow.

    Parameters:
    grid (np.ndarray): Accumulated grid of shape (grid_size, grid_size).
    """
    plt.figure(figsize=(10, 5))
    plt.imshow(grid, origin='lower', cmap='hot')
    plt.colorbar(label='Number of circles')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Aggregated Circles Map')
    plt.show()

def visualize_events(rec_events, energy, angle):
    """
    Visualize events data by generating 3D and 2D plots.

    Parameters:
    rec_events (pd.DataFrame): DataFrame containing event data with columns 'use', 'compton_angle', 'psi_angle', and 'xsi_angle'.
    energy (float): Energy parameter for visualization.
    angle (float): Angle parameter for visualization.
    """
    # 3D Plot
    fig = plt.figure()
    ax_3d = fig.add_subplot(111, projection='3d')

    # Plot 3D circles
    plot_3d_circles(ax_3d, rec_events)

    # Plot quiver
    plot_quiver(ax_3d, angle)

    ax_3d.set_xlabel('X')
    ax_3d.set_ylabel('Y')
    ax_3d.set_zlabel('Z')
    ax_3d.set_xlim(-1, 1)
    ax_3d.set_ylim(-1, 1)
    ax_3d.set_zlim(-1, 1)
    plt.show()

    # 2D Mollweide Projection
    fig, ax_2d = plt.subplots()
    angle_rad = np.arcsin(angle) - np.pi / 2
    direction_vector = np.array([np.sin(angle_rad), 0, np.cos(angle_rad)])

    image_data = plot_mollweide_circles(ax_2d, rec_events, direction_vector)

    ax_2d.set_xlabel('Longitude')
    ax_2d.set_ylabel('Latitude')
    plt.show()

    # Accumulated Grid
    grid_size = 180
    grid = accumulate_in_grid(image_data, grid_size)

    # Display the accumulated map using imshow
    plot_accumulated_map(grid)




# Example usage
"""
from tools import display_tools  

ENERGY = 10000
ANGLE = 0.9

for (energy, angle), group in groups:
    if energy == ENERGY and angle == ANGLE:
        group_clean = group.dropna(subset=features).copy()
        X_group = group_clean[features]
        group_clean.loc[:, 'use'] = classifier.predict(X_group)
        rec_events = group_clean.query('use == 1').copy()
        break
    
%matplotlib qt
display_tools.visualize_events(rec_events, energy=ENERGY, angle=ANGLE)

"""
