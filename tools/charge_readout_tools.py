#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 11:16:17 2022

Collection of routines for the charge readout of electron recoil tracks
described by r, the location of charges and num_e, the number of charges
at each r.  For each type of readout, a description of the sensos
must also be provided, which can be created in params_tool.


TODO: Implement coarse trigger for pixel chips
TODO: Revive hexagonal cells.  Chip array readout. Update naming: x,y -> 0,1
TODO: Revisit focusing, add de-focussing to output
TODO[ts]: Implement overall z timing
TODO[ts]: Implement signals spanning cells.
TODO[ts]: Integrate Bahrudin's code???
TODO[ts]: Add cube method to flesh out pixel data for ML

@author: tshutt
"""

def readout_dual_scale_pixels(r, num_e, coarse_sensors, pixel_sensors,
                        triggered_output=True, raw_output=True,
                        stats_output=False):
    """
    Readout of fine scale pixels in a dual scale readout system.
        Assumes each logical element of coarse_sensors corresponds
        to a single corresonding single logical "chip" of fine scale pixels.
        The readout of those pixels is done by readout_pixels.

        This currently uses an idealized zero noise trigger for the summed
        signal in each logical coarse voxel, and does not yet used
        a trigger found separately from the coarse sensors.  The latter
        is a signficantly simplification in the case of coarse wires.

    Input:
        r  - electron locations, dimension [0:2, :]
        num_e - number of electron at each entry of r
        coarse_sensors - description of coarse_tiles or chip_array
        pixel_sensors - descripotion of pixels
        triggered_output - if true, triggered samples returned
        raw_output - if true, all noisless samples with any charge returned

    Output: pixel_samples, with fields:
        if triggered_output:
            samples_triggered = pixel samples with noise, and
                above trigger threshold.  1d array
            r_triggered - locations of samples. array of size
                [3, samples_triggered.size]
        if raw_output:
            samples_raw = pixel samples of noiseless signal, no
                trigger applied
            r_raw - locations of samples. array of size
                [3, samples_triggered.size]

    Note that edges of sampling in z generated here.

    10/22 TS
    """

    import numpy as np

    import geometry_tools

    #   Find coarse voxels defined by coarse_sensors that contain charge
    voxels = find_voxels(r, coarse_sensors)

    #   Calculate pixels, looping over coarse voxels
    for n in range(voxels['voxel_indices'].shape[1]):

        #   Conveneient variable - indices of this voxel, as a column vector
        voxel_indices = voxels['voxel_indices'][:, n]

        #   This mask selects entries in r which are in this voxel index
        mask = np.all(
            np.equal(voxels['charge_indices'], voxel_indices[:, None]),
            axis=0)

        #   For these entries r, change to pixel chip coordinates
        #   (note the called routine does not alter the calling r)
        r_chip = geometry_tools.cell_to_chip_coordinates(
            r[:, mask],
            voxel_indices[:, None],
            coarse_sensors['centers']
            )

        #   Masked charge
        num_e_chip = num_e[mask]

        #   Focus electrons uniformly onto pixels
        #   TODO: review focus_factor.  Need to add defocussing
        r_chip[0, :] = r_chip[0, :] * pixel_sensors['focus_factor']
        r_chip[1, :] = r_chip[1, :] * pixel_sensors['focus_factor']

        #   Readout out pixels for these r
        chip_samples = readout_pixels(
            r_chip,
            num_e_chip,
            pixel_sensors,
            voxels['z_edges'][voxel_indices[2] : voxel_indices[2] + 2],
            triggered_output=triggered_output,
            raw_output=raw_output,
            stats_output = stats_output,
            )

        #   Transform pixel centers back to cell coordinates
        if triggered_output:
            r_triggered_this_chip \
                = geometry_tools.cell_to_chip_coordinates(
                    chip_samples['r_triggered'],
                    voxel_indices[:, None],
                    coarse_sensors['centers'],
                    reverse=True
                    )
        if raw_output:
            r_raw_this_chip \
                = geometry_tools.cell_to_chip_coordinates(
                    chip_samples['r_raw'],
                    voxel_indices[:, None],
                    coarse_sensors['centers'],
                    reverse=True
                    )

        #   Stitch voxel to output
        if triggered_output:
            if not 'r_triggered' in locals():
                r_triggered = r_triggered_this_chip
                samples_triggered = chip_samples['samples_triggered']
            else:
                r_triggered = np.append(
                    r_triggered,
                    r_triggered_this_chip,
                    axis=1
                    )
                samples_triggered = np.append(
                    samples_triggered,
                    chip_samples['samples_triggered']
                    )
        if raw_output:
            if not 'r_raw' in locals():
                r_raw = r_raw_this_chip
                samples_raw = chip_samples['samples_raw']
            else:
                r_raw = np.append(
                    r_raw,
                    r_raw_this_chip,
                    axis=1
                    )
                samples_raw = np.append(
                    samples_raw,
                    chip_samples['samples_raw']
                    )

    #   Package output
    pixel_samples = {}
    if triggered_output:
        pixel_samples['samples_triggered'] = samples_triggered
        pixel_samples['r_triggered'] = r_triggered
    if raw_output:
        pixel_samples['samples_raw'] = samples_raw
        pixel_samples['r_raw'] = r_raw

    return pixel_samples

def readout_pixels(r, num_e, pixel_sensors, z_limits=None, raw_output=True,
                   triggered_output=True, stats_output=False):
    """
    Reads out pixels, for two types of pixel_sensors which are
        functionally very similar:
        pixels on a single pixel chip, in which case the top and bottom
            z limits come from the supplied z_limits.  This applies to
            both versions of GAMPix pixels and LArPix
        coarse tile electrodes, in which case the z sampling
            is fully calculated within this routine

    Input:
        r  - electron locations, dimension [0:2, :]
        num_e - number of electron at each entry of r
        pixel_sensors - description of either pixels or coarse tiles
        z_limits - Used for dual scale readout, array with lower and upper
            edges in z of the coarse voxel being sampled.  If missing,
            then span in z chosen to contain data.
        triggered_output - triggered pixels returned
        raw_output - all noisless pixels with any charge returned

    Output: samples, with fields:
        if triggered_output:
            samples_triggered = pixel samples with noise, and above
                trigger threshold.  1d array
            r_triggered - locations of samples. array of size
                [3, samples_triggered.size]
        if raw_output:
            samples_raw = pixel samples of noiseless signal, no
                trigger applied
            r_raw - locations of samples. array of size
                [3, samples_triggered.size]
        z_edges = edges of sampling in z, generated in find_sensor_span

    Digitization currently highly idealized:
       + z is treated like x and y.
       + all dimensions are purely voxelized - no cross-talk between
          voxels is considered
    Improvements:
        + Add full treatment of coarse grid trigger and pixel
            readout time
        + Separate trigger thresholds for hardware pixel trigger, and
            software sample trigger
        + Appropriately average of space and time based on study of
            electron optics and induced signals
        + Treat buffers as they will be done in asic

    4/20    TS  cleaned up from prior versions
    10/22   TS  overhaul, and generalized
    """

    # from sklearn.decomposition import PCA
    import numpy as np

    #   Find span of struck pixels in x-y, and also save first index
    xy_edges = []
    xy_centers = []
    first_index = []
    for n in range(2):
        idx =  find_span(r[n, :], pixel_sensors['edges'][n])
        xy_edges.append(pixel_sensors['edges'][n][idx])
        xy_centers.append(
            xy_edges[n][0:-1]
            + np.diff(xy_edges[n][0:2]) / 2
            )
        first_index.append(idx[0])

    #   Create z sampling.  Pixels don't fit normal pattern.
    if z_limits is not None:
        num_z_samples = pixel_sensors['edges'][0].size
        z_edges = np.linspace(z_limits[0], z_limits[1], num_z_samples + 1)
        idx =  find_span(r[2, :], z_edges)
        z_edges = z_edges[idx]
        z_centers = z_edges[0:-1] + np.diff(z_edges[0:2]) / 2
    else:
        z_edges, z_centers \
            = create_z_sampling(r[2, :], pixel_sensors['sampling_pitch'])

    #   Digitize
    samples_raw, _ = np.histogramdd(
        [r[0, :], r[1, :], r[2, :]],
        bins = [xy_edges[0],
                xy_edges[1],
                z_edges],
        weights=num_e
        )

    #   Meshgrid of locations needed below for flat list of locations
    locations_x, locations_y, locations_z = np.meshgrid(
        xy_centers[0],
        xy_centers[1],
        z_centers,
        indexing='ij'
        )

    #   Trigger, for now, treats all samples the same.  This is not
    #   what the chip hardware architecture will actually do.
    if triggered_output:
        samples_noisy = samples_raw \
            + pixel_sensors['noise'] \
            * np.random.randn(
                samples_raw.shape[0],
                samples_raw.shape[1],
                samples_raw.shape[2]
                )
        trigger_mask = samples_noisy \
            > pixel_sensors['noise'] \
                * pixel_sensors['threshold_sigma']
        samples_triggered = samples_noisy[trigger_mask].flatten()
        r_triggered = np.zeros((3, np.sum(trigger_mask)))
        r_triggered[0, :] = locations_x[trigger_mask]
        r_triggered[1, :] = locations_y[trigger_mask]
        r_triggered[2, :] = locations_z[trigger_mask]

    #   Raw is all charge, without noise
    if raw_output:
        signal_mask = samples_raw > 0
        samples_raw = samples_raw[signal_mask].flatten()
        r_raw = np.zeros((3, np.sum(signal_mask)))
        r_raw[0, :] = locations_x[signal_mask]
        r_raw[1, :] = locations_y[signal_mask]
        r_raw[2, :] = locations_z[signal_mask]

    #   Stats output
    if stats_output:
        num_pixels_triggered = np.any(
            samples_noisy > pixel_sensors['trigger_noise'],
            axis=2
            ).sum()

    #   Save output
    samples = {}
    if triggered_output:
        samples['samples_triggered'] = samples_triggered
        samples['r_triggered'] = r_triggered
    if raw_output:
        samples['samples_raw'] = samples_raw
        samples['r_raw'] = r_raw
    if stats_output:
        samples['num_pixels_triggered'] = num_pixels_triggered
    samples['z_edges'] = z_edges

    return samples

def readout_coarse_grids(r, num_e, coarse_grid_sensors, stats_output=False):
    """
    Readout of tracks using coarse induction grids

    Input:
        r  - electron locations, dimension [0:2, :]
        num_e - number of electron at each entry of r
        coarse_grid_sensors - descripotion coarse grids

    Returned: coarse_grid_samples, with fields:
        samples - samples on wires, as list in x[0] and y[1] of arrays
            of signals with fields:
                raw - signal with no noise, in all wires with signal
                noisy - signal with noise, in all wires with signal
                triggered - noisy signal above single wire threshold
        where - locations and indices of wires with signal.  Organized as
            lists of lists in x-z, and y-z, i.e. [0][0] is x wires, [0][1]
            is z for these x wires, and [1][:] is the same for y.
            The fields are as follows:
                locations - wire locations and centers of z samples.
                indices - indices of locations
                triggered_locations - same as locations, but only for samples
                    above single wire threshold
                triggered_indices - indices of triggered locations
        voxel_trigger - trigger for pixel chips, generated from coarse
            grid signals.  Voxel defined as 3d object of pixel chips
            in x,y, and bins in z. Fields:
                trigger - array of triggered voxels.  array of shape [3, :]
                z_centers_span - locations of z bin centers spanning
                    triggered voxels
                z_edges_span - locations of z bin edges spanning
                    triggered voxels

    Readout is highly idealized - see below in code

    TODO[ts]: Check that coarse trigger still works, post overhaul

    Work in progress

    5/20    TS, based on earlier versions
    4/21    TS, overhauled
    10/22   TS, overhaul, generalization
    """

    import numpy as np

    # #   Find span of wires and times containing signal.  The various items
    # #   in span are lists, with first
    # sensor_span = find_sensor_span(r, coarse_grid_sensors)

    #   Find set wires that span the signal.  Note that details are
    #   different for these induction wires compared to all other
    #   non induction cases
    xy_centers = []
    xy_indices = []
    for n in range(2):
        idx \
            = find_span(r[n, :], coarse_grid_sensors['centers'][n])
        xy_centers.append(coarse_grid_sensors['centers'][n][idx])
        xy_indices.append(idx)

    #   Create z sampling
    z_edges, z_centers \
        = create_z_sampling(r[2, :], coarse_grid_sensors['pitch'])
    z_indices = np.arange(0, z_centers.size)

    #   Final output adds noise triggering.  Restrict output to wires
    #   with signal (or triggered).
    #   In all cases, output is list of x, y
    samples_raw = []
    samples_noisy = []
    samples_triggered = []
    samples_wire_locations = []
    samples_wire_indices = []
    samples_triggered_wire_locations = []
    samples_triggered_wire_indices = []

    #   These are local
    samples_span = []
    samples_span_noisy = []

    #   Threshold in a wire
    wire_threshold = coarse_grid_sensors['threshold_sigma'] \
        * coarse_grid_sensors['noise']

    #   Loop over x, y
    for n in range(2):

        #   Read out samples - arrays in x-z and y-z, in a list
        samples_span.append(readout_induction_grid_span(
            r[n, :],
            r[2, :],
            num_e,
            xy_centers[n],
            z_edges
            ))

        #   Add noise to span samples
        samples_span_noisy.append(
            samples_span[n]
            + coarse_grid_sensors['noise'] * np.random.randn(
                samples_span[n].shape[0],
                samples_span[n].shape[1]
                )
            )

        #   Booleans masks for presence signal, and triggered signal.
        #   Note the trigger we account for the sharing amongst 4 wires,
        #   with signal sensors['signa_fraction']
        signal_mask = samples_span[n] > 0
        triggered_mask = (
            (samples_span_noisy[n]
             * coarse_grid_sensors['signal_fraction'] / 4)
            > wire_threshold
            )

        #   These are flattened arrays of samples only in wires with
        #   signal or triggered
        samples_raw.append(samples_span[n][signal_mask])
        samples_noisy.append(samples_span_noisy[n][signal_mask])
        samples_triggered.append(samples_span_noisy[n][triggered_mask])

        #   Here we construct the locations and indices of wires with
        #   signal or above threshold, by first creating meshes of
        #   locations and indices, and then selecting the valid subsets.
        #   Note that indices in x, y refer to the full set of wire
        #   indices from span
        centers_mesh_xy, centers_mesh_z = np.meshgrid(
            xy_centers[n],
            z_centers,
            indexing='ij'
            )
        indices_mesh_xy, indices_mesh_z = np.meshgrid(
            xy_indices[n],
            z_indices,
            indexing='ij'
            )

        #   Raw signal locations and indices
        samples_wire_locations.append(
            [centers_mesh_xy[signal_mask],
            centers_mesh_z[signal_mask]]
            )
        samples_wire_indices.append(
            [indices_mesh_xy[signal_mask],
             indices_mesh_z[signal_mask]]
             )

        #   Triggered signal locations and indices
        samples_triggered_wire_locations.append(
            [centers_mesh_xy[triggered_mask],
             centers_mesh_z[triggered_mask]]
            )
        samples_triggered_wire_indices.append(
            [indices_mesh_xy[triggered_mask],
             indices_mesh_z[triggered_mask]]
            )

    #   We now find the voxel trigger, where a voxel is the volume between
    #   4 wires (2 in x, 2 in y) and a slice in z

    #   The trigger for voxel is found when sum of the signals in the
    #   surrounding wires in a z slice are above threshodl.  This
    #   of course is not perfet - if multiplite voxels in a given z
    #   slice contain signal, this results in ghost triggers.
    #   Here we loop over z slices, and choose to loop on
    #   rows = y, vectorizing over columns = x (the opposite x-y order
    #   gives the same result).
    #   Note we adjust the signal size by the separately calculated
    #   signal fraction in 4 wires, divided by 2 since we separately
    #   compare 2 wires at a time.
    #   This calculationg encomapses the range of sensors in span, resulting
    #   in a boolean array over a "box" in x, y, z.
    voxel_trigger_box = np.zeros(
        (samples_span_noisy[0][:, 0].size - 1,
         samples_span_noisy[1][:, 0].size - 1,
         z_centers.size),
        dtype=bool
        )
    for nz in range(z_centers.size):
        for ny in range(samples_span_noisy[1][:, nz].size - 1):
            voxel_trigger_box[:, ny, nz] = (
                (((samples_span_noisy[1][ny, nz]
                   + samples_span_noisy[1][ny + 1, nz])
                  * coarse_grid_sensors['signal_fraction'] / 2)
                 > wire_threshold)
                & (((samples_span_noisy[0][:-1, nz]
                     + samples_span_noisy[0][1:, nz])
                  * coarse_grid_sensors['signal_fraction'] / 2)
                   > wire_threshold)
                )

    #   Here we flatten the voxel trigger to an array of indices of
    #   triggered voxels
    cix, ciy, ciz = np.meshgrid(
        np.arange(0, voxel_trigger_box.shape[0], dtype=int)
            + xy_indices[0][0],
        np.arange(0, voxel_trigger_box.shape[1], dtype=int)
            + xy_indices[1][0],
        np.arange(0, voxel_trigger_box.shape[2], dtype=int),
        indexing='ij'
        )
    voxel_trigger \
        = np.zeros((3, np.sum(voxel_trigger_box)), dtype=int)
    voxel_trigger[0, :] = cix[voxel_trigger_box]
    voxel_trigger[1, :] = ciy[voxel_trigger_box]
    voxel_trigger[2, :] = ciz[voxel_trigger_box]

    #   Find coarse voxels defined by coarse_sensors that contain charge
    if stats_output:
        voxels = find_voxels(r, coarse_grid_sensors)

    #   Package output
    coarse_grid_samples = {}

    coarse_grid_samples['samples_raw'] = samples_raw
    coarse_grid_samples['samples_noisy'] = samples_noisy
    coarse_grid_samples['samples_triggered'] = samples_triggered

    coarse_grid_samples['where'] = {}
    coarse_grid_samples['where']['locations'] = samples_wire_locations
    coarse_grid_samples['where']['indices'] = samples_wire_indices
    coarse_grid_samples['where']['triggered_locations'] \
        = samples_triggered_wire_locations
    coarse_grid_samples['where']['triggered_indices'] \
        = samples_triggered_wire_indices

    coarse_grid_samples['voxel_trigger'] = {}
    coarse_grid_samples['voxel_trigger']['trigger'] \
        = voxel_trigger
    coarse_grid_samples['voxel_trigger']['z_centers_span'] \
        = z_centers
    coarse_grid_samples['voxel_trigger']['z_edges_span'] \
        = z_edges

    if stats_output:
        coarse_grid_samples['voxels'] = voxels

    return coarse_grid_samples

def readout_induction_grid_span(u, z, num_e, u_centers, z_edges):
    """
    For single set of inducton wires, and charge specified by lateral
        dimension u, depth z and num_e,
        returns samples read out in all the wires (or strips)
        contained within sensor_span.

    Returns:
        samples_raw - the noiseless signals in the wires.  Format is
            separate arrays for x-z and y-z, put in a list.
            Example: samples[1][ny, nz] is signal on the nth
            wire in y, and nth sample in z.

    This routine is the place to implement a detailed treatment of
        signals and wires. The current calculation is quite simplified,
        as detailed in comments below.u
    """

    import sys
    import numpy as np

    def find_samples(u, wire_locations):
        """
        For a set of charges at a 1d spatial location u, read out by
        wires (strips) at wire_locations, finds the measured charge signal
        in wire.

        Currently has very crude and inaccurate treatment of sharing
        of signals between wires: signal between two wires contributes
        to each proportionally to distance, with correct integral.
        The real signal is more complicated: bi-polar, and strongly
        position dependent, and does not preserve the integral.
        """

        #   Find bins for each charge - used in subsequent calculation
        #   Subtract 1 to get first-index-zero indexing.  This is index
        #   that, for n wires, has n-1 values.
        gap_indices = np.digitize(u, wire_locations) - 1

        #   Check that all charges are in range
        in_space_range = \
            (gap_indices >= 0) & \
            (gap_indices < wire_locations.size)
        if np.any(~in_space_range):
            sys.exit('Error in find_samples_1d - signal out of range')

        #   Distances to two nearest wires
        u_pitch = np.diff(wire_locations[:2])
        d_1 = (u - (wire_locations[0] + gap_indices * u_pitch))
        d_2 = ((wire_locations[0] + (gap_indices + 1) * u_pitch) - u)

        #   For now, signal is linearly proportional to d - this is
        #   clearly wrong and should be changed
        samples_1 = d_1 / u_pitch
        samples_2 = d_2 / u_pitch

        #   span_sample are sum over signal from all charges in both wires
        span_samples = np.zeros(wire_locations.size)
        for gap_index in np.unique(gap_indices):

            span_samples[gap_index] = span_samples[gap_index] \
                + np.sum(samples_1[gap_indices==gap_index])

            span_samples[gap_index + 1] = span_samples[gap_index + 1] \
                + np.sum(samples_2[gap_indices==gap_index])

        return span_samples

    #   Initialize output
    samples_raw = np.zeros((u_centers.size, z_edges.size - 1), dtype=int)

    #   Find the z bin for each entry in r.  Subtract 1 to get
    #   first-index-zero indexing
    z_bins = np.digitize(z, z_edges) - 1

    #   Now loop over z bins with signal, calculating samples in x and y
    for z_bin in np.unique(z_bins):

        #   Mask for charges in this z bin
        z_mask = z_bins==z_bin

        #   Find measured samples in x and y
        samples_raw[:, z_bin] = find_samples(u[z_mask], u_centers)

    return samples_raw

def readout_anode_grid(r, num_e, anode_grid_sensors):
    """
    Readout of drifted tracks with 1D anode (i.e., non inductive) grid,
        with "x" wires which lie along the y axis.

    Currently not implemented as a dual scale readout

    Input:
        r  - electron locations, dimension [0:2, :]
        num_e - number of electron at each entry of r
        anode_grid_sensors - descripotion anode grids

    Returned: coarse_grids, with fields:
        samples - samples on wires, with fields:
            raw - signal with no noise, in all wires with signal
            noisy - signal with noise, in all wirese with signal
            triggered - noisy signal above single wire threshold
        where - locations and indices of wires with signal, as follows:
            locations - wire + z location for any signal.  array of
                size [3, samples['raw'].size]
            indices - indices of locations
            triggered_locations - same as locaitons, but only for samples
                above single wire threshold
            triggered_indices - indices of triggered locations
        voxel_trigger - trigger for pixel chips, generated from coarse
            grid signals.  Voxel defined as 3d object of pixel chips
            in x,y, and bins in z. Fields:
            trigger - array of triggered voxes.  array of shape [3, :]
            z_centers_span - locations of z bin centers spanning
                triggered voxels
            z_edges_span - locations of z bin edges spanning
                triggered voxels

    4/23
    """

    import numpy as np

    #   Find centers and edges that span data
    ix_min, ix_max = find_span(r[0, :], anode_grid_sensors['edges'])
    x_edges = anode_grid_sensors['edges'][ix_min : ix_max + 1]
    x_centers = x_edges[0:-1] + np.diff(x_edges[0:2]) / 2
    x_indices = np.arange(ix_min, ix_max)

    #   Create z sampling, based on anodge grid pitch
    z_edges, z_centers \
        = create_z_sampling(r[2, :], anode_grid_sensors['pitch'])
    z_indices = np.arange(0, z_centers.size)

    #   Find the z bin for each entry in r.  Subtract 1 to get
    #   first-index-zero indexing
    z_bins = np.digitize(r[2, :], z_edges) - 1

    #   Now loop over z bins with signal, calculating samples for all
    #   sensors that span the sensors
    span_samples = np.zeros((x_centers.size, z_centers.size), dtype=int)
    for z_bin in np.unique(z_bins):

        #   Selects charges in this z bin, and assign this z index
        z_mask = z_bins==z_bin

        #   Signal is at wire and in z_bin
        span_samples[:, z_bin], _ = np.histogram(
            r[0, z_mask],
            bins=x_edges,
            weights=num_e[z_mask]
            )

    #   Threshold in a wire
    wire_threshold = anode_grid_sensors['threshold_sigma'] \
        * anode_grid_sensors['noise']

    #   Add noise
    span_samples_noisy = (
        span_samples
        + anode_grid_sensors['noise']
        * np.random.randn(span_samples.shape[0], span_samples.shape[1])
        )

    #   Wires with any signal, and with signal above single-wire threshold.
    signal_mask = span_samples > 0
    triggered_mask = span_samples_noisy > wire_threshold

    #   Output starts with flatttened arrays of sample values
    anode_grid_samples = {}
    anode_grid_samples['samples_raw'] = span_samples[signal_mask]
    anode_grid_samples['samples_noisy'] = span_samples_noisy[signal_mask]
    anode_grid_samples['samples_triggered'] \
        = span_samples_noisy[triggered_mask]

    #   Meshgrids of centers and indices needed
    centers_mesh_x, centers_mesh_z = np.meshgrid(
        x_centers,
        z_centers,
        indexing='ij'
        )
    indices_mesh_x, indices_mesh_z = np.meshgrid(
        x_indices,
        z_indices,
        indexing='ij'
        )

    #   Assign raw samples locations and indices
    anode_grid_samples['where'] = {}
    anode_grid_samples['where']['centers'] = np.array(
        [centers_mesh_x[signal_mask], centers_mesh_z[signal_mask]]
        )
    anode_grid_samples['where']['indices'] = np.array(
        [indices_mesh_x[signal_mask], indices_mesh_z[signal_mask]]
        )

    #   Assign triggered samples locations and indices
    anode_grid_samples['where']['triggered_centers'] = np.array(
        [centers_mesh_x[triggered_mask], centers_mesh_z[triggered_mask]]
        )
    anode_grid_samples['where']['triggered_indices'] = np.array(
        [indices_mesh_x[triggered_mask], indices_mesh_x[triggered_mask]]
        )

    return anode_grid_samples

def find_voxels(r, coarse_sensors):
    """
    Finds indices of the voxels in which electrons in r reside.
    coarse_sensors are pixel chips or coarse tiles.  The z sampling is
    generated here with pitch equal to the sensors.
    The z sampling bins are returned.

    Input:
        r  - electron locations, dimension [0:2, :]
        coarse_sensors - from params, either chip_array or coarse_tiles

    Returned: xy_wire_output dictionary with these fields:
        voxels, with fields:
            charge_indices - for each entry in r, the indices of
                the voxel containg r, where for x and y the the
                voxels are defined as pixel chips, and in z are
                bins.  Array of size [3, r[0, :].size].
            voxels_indices - the set of voxels with signal, returned
                as indices of the voxels. Array of size [3, :].
            z_centers - centers of voxel z bins that span the data
            z_edges - edges of voxel z bins that span the data

    11/22   TS
    """

    import numpy as np
    from scipy import stats

    #   Find span of struck pixels in x-y, and also save first index
    xy_edges = []
    first_index = []
    for n in range(2):
        idx =  find_span(r[n, :], coarse_sensors['edges'][n])
        xy_edges.append(coarse_sensors['edges'][n][idx])
        first_index.append(idx[0])

    #   Create z sampling, based on array sensors pitch
    z_edges, z_centers = create_z_sampling(r[2, :], coarse_sensors['pitch'])

    #   Digitize to find indices of voxel for each charge
    _, _, charge_voxel_indices = stats.binned_statistic_dd(
        r.transpose(),
        None,
        statistic='count',
        bins = [xy_edges[0],
                xy_edges[1],
                z_edges],
        expand_binnumbers=True
        )

    #   Subtract 1 from indices to get first-index-zero indexing
    charge_voxel_indices -= 1

    #   Adjust x-y indices to refer to the the full set of chips
    for n in range(2):
        charge_voxel_indices[n, :] += first_index[n]

    #   Indices of the voxels with signal
    voxels_with_signal_indices = np.unique(charge_voxel_indices, axis=1)

    #   Package output
    voxels = {}
    voxels['charge_indices'] = charge_voxel_indices
    voxels['voxel_indices'] = voxels_with_signal_indices
    voxels['z_centers'] = z_centers
    voxels['z_edges'] = z_edges

    return voxels

def find_span(u, sensor_boundaries_1d):
    """ For array of locations, u, finds min and max indices of
    sensor_boundaries_1d that minimally span u. """

    import numpy as np

    i_min = (np.abs(sensor_boundaries_1d - u.min())).argmin()
    if sensor_boundaries_1d[i_min] > u.min(): i_min += -1

    i_max = (np.abs(sensor_boundaries_1d - u.max())).argmin()
    if sensor_boundaries_1d[i_max] < u.max(): i_max += 1

    return np.arange(i_min, i_max + 1)

def create_z_sampling(z, pitch):
    """
    Generates z samples edges and centers, based on pitch

    returns arrays edges, centers
    """

    import numpy as np
    import math

    #   Create buffered span of z edges
    buffer = 1.2
    span = (z.max() - z.min())
    if span==0: span = pitch
    i_span = int(math.ceil(buffer * span / pitch))
    edges =  z.mean() + pitch * np.arange(-i_span, i_span + 1)

    #   Now narrow to minimum span, and create centers
    idx =  find_span(z, edges)
    edges = edges[idx]
    centers = edges[:-1] + pitch / 2

    return edges, centers


