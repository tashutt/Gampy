#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Routines to set up geometries, both for use with MEGALIB and electron
tracks from PENELOPE or other code.

TODO: Deal with versioning.  Add file name generator?

@author: tshutt

calorimeter and shield added by sjett 8/10/23
"""

def calculate_geomega_geometry_v1(params):
    """
    Generates geomaga geometry, with following "topology":
        + Flat cylindrical geometry, with carbon fiber vessel, and variable
            diameter
        + Two indientical layers of cells, either rectangular or hexagonal
        + Two identical ACD layers with full footprint of cylinder
        + Internal dead material (anode, cathode, cell walls) of LAr
        + No separate micrometeor shied

    From geomega_inputs, calculate all params needed to describe the
    detector, along with geomega file image, and add these to params.

    Note that this detector description does not include the readout.

    Cosima and Geomega units are: cm, keV, g, deg

    TODO: Add z coordinate translation for TPC cells
    TODO: Add versioning.  This routine currently decribes
    a detector with:
        + Layers of ACD on front and back, implemented as tiles with
            the same footprint as the cells
        + Two layers of cells
    """

    import sys
    import math
    import numpy as np

    #   ID for this topology - no checking of this
    params.topology_id = 1

    #   Geomega z coordinate for various layers.  This is the center
    #   of each volume in the coordinates of its mother volume, which is
    #   currently the vessel (or Ar)
    params.z_centers = {}

    params.z_centers['front_acd'] = (
        params.acd['thickness'] / 2
        + params.anode_planes['thickness']
        + params.cells['height']
        + params.cathode_plane['thickness'] / 2
        )
    params.z_centers['front_anode_plane'] = (
        + params.anode_planes['thickness'] / 2
        + params.cells['height']
        + params.cathode_plane['thickness'] / 2
        )
    params.z_centers['front_cells'] = (
        + params.anode_planes['thickness'] / 2
        + params.cells['height'] / 2
        )
    params.z_centers['cathode_plane'] = 0
    params.z_centers['back_cells'] \
        = - params.z_centers['front_cells']
    params.z_centers['back_anode_plane'] \
        = - params.z_centers['front_anode_plane']
    params.z_centers['back_acd'] \
        = - params.z_centers['front_acd']

    #   Vessel height
    params.vessel['height'] = 2 * (
        params.vessel['wall_thickness']
        + params.acd['thickness']
        + params.anode_planes['thickness']
        + params.cells['height']
        + params.cathode_plane['thickness'] / 2
        + params.shield['thickness']/2
        + params.calorimeter['thickness']/2
        )

    #   Generate cell array, either for hexagonal or rectangular cells.
    if params.cells['geometry']=='hexagonal':
        centers, corners = make_honeycomb(
            params.cells['flat_to_flat'],
            params.vessel['r_outer'] - params.vessel['lar_min_radial_gap']
            )
    elif params.cells['geometry']=='rectangular':
        centers, corners = make_square_array(
            params.cells['flat_to_flat'],
            params.vessel['r_outer'] - params.vessel['lar_min_radial_gap']
            )
    else:
        sys.exit('Error in calculate_geomega_geometry: bad cell geometry')

    #   Tile two layers to cells - "front" and "back", with front having
    #   z > 0,
    front_layer_mask = np.concatenate(
        (np.ones(centers.shape[1], dtype=bool),
        np.zeros(centers.shape[1], dtype=bool))
        )

    params.cells['centers'] = np.zeros((3, 2 * centers.shape[1]))
    params.cells['centers'][0:2, :] = np.tile(centers, 2)
    params.cells['centers'][2, front_layer_mask] \
        = params.z_centers['front_cells']
    params.cells['centers'][2, ~front_layer_mask] \
        = params.z_centers['back_cells']

    params.cells['corners'] = np.zeros(
        (3, corners.shape[1], 2 * corners.shape[2])
        )
    params.cells['corners'][0:2, :, :] = np.tile(corners, 2)
    params.cells['corners'][2, :, front_layer_mask] \
        = params.z_centers['front_cells']
    params.cells['corners'][2, :, ~front_layer_mask] \
        = params.z_centers['back_cells']

    #   These booleans select front and back cells
    params.cells['front_layer'] = front_layer_mask
    params.cells['back_layer'] = np.logical_not(front_layer_mask)

    #   Cell dimensions in x, y
    if params.cells['geometry']=='hexagonal':
        corner_to_corner = params.cells['flat_to_flat'] * math.sqrt(3) / 2
        params.cells['width_xo'] = params.cells['flat_to_flat']
        params.cells['width_yo'] = corner_to_corner
        params.cells['width_xi'] \
            = params.cells['width_xo'] \
                - params.cells['wall_thickness'] / 2
        params.cells['width_yi'] \
            = params.cells['width_yo'] \
                - params.cells['wall_thickness'] * math.sqrt(3) / 2 / 2
        params.cells['rotation'] = 30.
        params.cells['segment_length'] \
            = params.cells['flat_to_flat'] / math.sqrt(3)
        params.cells['area'] \
            = 2 * math.sqrt(3) * (params.cells['flat_to_flat'] /2 )**2
    elif params.cells['geometry']=='rectangular':
        params.cells['width_xo'] = params.cells['flat_to_flat']
        params.cells['width_xi'] \
            = params.cells['width_xo'] \
                - params.cells['wall_thickness'] / 2
        params.cells['width_yo'] = params.cells['width_xo']
        params.cells['width_yi'] = params.cells['width_xi']
        params.cells['rotation'] = 45.
        params.cells['area'] = params.cells['width_xo']**2

    #%% Now make geomega file image


    def get_tpc_cell_line(cell_number, cell_center, cell_geometry):
        '''
        For a cell number and location of center, returns the
        line calling out cell position
        '''

        if cell_geometry=='rectangular':
            rotation_tag = '0.'
        else:
            rotation_tag = '30.'

        cell_tag = f'Cell_{cell_number:03.0f}'

        line = ['BaseTPCCell.Copy ' + cell_tag + '\n']
        line.append(
            cell_tag + '.Position    '
            + f'{cell_center[0]*100:10.7f} '
            + f'{cell_center[1]*100:10.7f} '
            + f'{cell_center[2]*100:10.7f} '
            + '\n'
            )
        line.append(
            cell_tag + '.Rotation    0. 0. ' + rotation_tag + '\n')
        line.append(
            cell_tag + '.Mother      AllArgon\n')

        return line

    r_surround = 1.5 * params.vessel['r_outer']

    vessel_r_inner = params.vessel['r_outer'] \
        - params.vessel['wall_thickness']

    all_ar_height \
        = params.vessel['height'] \
            - 2 * params.vessel['wall_thickness']

    # cell_corner_to_corner = params.cells['flat_to_flat'] * math.sqrt(3)/ 2

    cell_flat_to_flat_inner = params.cells['flat_to_flat'] \
        - params.cells['wall_thickness']

    #%%  Generate geomega file lines

    lines = ['Include $(MEGAlib)/megalibgeo/materials/Materials.geo\n']

    lines.append('\n')

    lines.append('//serves as the sphere from which '+
                 'farfield sources are instantiated\n')
    lines.append(
        'SurroundingSphere '
        + f'{r_surround*100:3.0f} '
        + '0. 0. 0. '
        + f'{r_surround*100:3.0f}'
        + '\n'
        )
    lines.append('ShowSurroundingSphere False\n')

    lines.append('\n')

    lines.append('//////////////////////////////////////////////////////\n')
    lines.append('/////// VOLUMES //////////////////////////////////////\n')
    lines.append('//////////////////////////////////////////////////////\n')

    lines.append('\n')

    #   World volume
    lines.append('//World volume\n')
    lines.append('Volume WorldVolume\n')
    lines.append('WorldVolume.Material Vacuum\n')
    lines.append('WorldVolume.Visibility 0\n')
    lines.append('WorldVolume.Shape BRIK 5000. 5000. 5000.\n')
    lines.append('WorldVolume.Mother 0\n')

    lines.append('\n')

    #   Template volumes

    #   Base TPC Cell
    lines.append('//Base unit hex cell (copy all "real" cells from this)\n')
    lines.append('Volume BaseTPCCell\n')
    lines.append('BaseTPCCell.Material Argon\n')
    lines.append(
        'BaseTPCCell.Shape PGON'
        + ' 0. 360. '
        + f'{params.cells["corners"].shape[1]}'
        + ' 2 '
        + f'{params.cells["height"]/2*100:10.7f} '
        + '0. '
        + f'{cell_flat_to_flat_inner/2*100:10.7f} '
        + f'{-params.cells["height"]*100/2:10.7f} '
        + '0. '
        + f'{cell_flat_to_flat_inner/2*100:10.7f} '
        + '\n'
        )
    lines.append('BaseTPCCell.Color 9\n')

    lines.append('\n')

    #   Base ACD layer
    lines.append('//Base unit ACD layer (copy all "real" ACDs from this)\n')
    lines.append('Volume BaseACDLayer\n')
    lines.append('BaseACDLayer.Material Argon\n')
    lines.append(
        'BaseACDLayer.Shape TUBE '
        + '0. '
        + f'{vessel_r_inner*100:10.7f} '
        + f'{(params.acd["thickness"]/2)*100:10.7f} '
        + '0. '
        + '360.'
        + '\n'
        )
    lines.append('BaseACDLayer.Color 14\n')

    lines.append('\n')

    # Outer ACD cylinder
    # outer_act_thickness = params.acd["thickness"]
    outer_act_thickness = 0.02
    calAndShield =( 0* params.calorimeter["thickness"] 
                +  params.shield["thickness"] 
                )
    lines.append('// Outer ACD cylinder\n')
    lines.append('Volume OuterACD\n')
    lines.append('OuterACD.Material Argon\n')
    lines.append(
        'OuterACD.Shape TUBE '
        + f'{vessel_r_inner*100:10.7f} ' # inner radius
        + ' '
        + f'{(vessel_r_inner + outer_act_thickness)*100:10.7f}'  # Outer radius (inner + thickness)
        + ' '
        + f'{(params.vessel["height"]/2 + calAndShield/2)*100:10.7f} '
        + '0. '
        + '360.'
        + '\n'
    )
    lines.append(f'OuterACD.Position 0.0 0.0 {(params.vessel["height"]/2 - calAndShield)*100:10.7f}\n')
    lines.append('OuterACD.Color 15\n')
    lines.append('OuterACD.Mother WorldVolume\n\n')


    #   Now the actual volumes

    #   Vessel volume defined by outer boundary
    lines.append('//Vessel volume\n')
    lines.append('Volume Vessel\n')
    lines.append('Vessel.Material CarbonFiber\n')
    lines.append(
        'Vessel.Shape TUBE '
        + '0. '
        + f'{params.vessel["r_outer"]*100:10.7f} '
        + f'{params.vessel["height"]/2*100:10.7f} '
        + '0. '
        + '360.'
        + '\n'
        )
    lines.append(f'Vessel.Position 0.0 0.0 {(params.vessel["height"] / 2)*100:10.7f}\n')
    lines.append('Vessel.Color 14\n')
    lines.append('Vessel.Mother WorldVolume\n')

    lines.append('\n')

    #   Argon volume is inside vessel walls
    lines.append('//All Ar volume\n')
    lines.append('Volume AllArgon\n')
    lines.append('AllArgon.Material Argon\n')
    lines.append(
        'AllArgon.Shape TUBE '
        + '0. '
        + f'{vessel_r_inner*100:10.7f} '
        + f'{all_ar_height/2*100:10.7f} '
        + '0. '
        + '360.'
        + '\n'
        )
    lines.append('AllArgon.Position 0.0 0.0 0.0\n')
    lines.append('AllArgon.Color 38\n')
    lines.append('AllArgon.Mother Vessel\n')

    lines.append('\n')

    lines.append('//////////////////////////////////////////////////////\n')
    lines.append('//////////////////Detector Characteristics////////////\n')
    lines.append('//////////////////////////////////////////////////////\n')

    lines.append('\n')

    lines.append('DriftChamber            Chamber\n')
    lines.append('Chamber.DetectorVolume  BaseTPCCell\n')
    lines.append('Chamber.SensitiveVolume BaseTPCCell\n')
    lines.append('Chamber.LightSpeed              1.34E+10\n')
    lines.append('Chamber.DriftConstant			0.0\n')
    lines.append('Chamber.LightDetectorPosition	3\n')
    lines.append('Chamber.EnergyPerElectron		0.1\n')
    lines.append('Chamber.EnergyResolution		Ideal\n')
    lines.append('Chamber.TimeResolutionAt      1000    0.00000000001 //s\n')
    lines.append('Chamber.LightEnergyResolutionAt	100000	10.0\n')
    lines.append('Chamber.LightEnergyResolutionAt	10000	5.0\n')
    lines.append('Chamber.LightEnergyResolutionAt	1000	1.0\n')
    lines.append('Chamber.LightEnergyResolutionAt	100	0.1   \n')
    lines.append('Chamber.FailureRate             0.0\n')
    lines.append('Chamber.Offset                  {0.} {0.}\n')
    lines.append('Chamber.StripNumber             1000 1000\n')

    lines.append('\n')

    lines.append('Scintillator   ACD\n')
    lines.append('ACD.DetectorVolume  BaseACDLayer\n')
    lines.append('ACD.SensitiveVolume BaseACDLayer\n')
    lines.append('ACD.TriggerThreshold 50\n')
    lines.append('ACD.EnergyResolution Gauss 50 50 5\n')
    lines.append('ACD.EnergyResolution Gauss 500 500 50\n')

    lines.append('\n')

    lines.append('Scintillator   ACD_outer\n')
    lines.append('ACD_outer.DetectorVolume  OuterACD\n')
    lines.append('ACD_outer.SensitiveVolume OuterACD\n')
    lines.append('ACD_outer.TriggerThreshold 50\n')
    lines.append('ACD_outer.EnergyResolution Gauss 50 50 5\n')
    lines.append('ACD_outer.EnergyResolution Gauss 500 500 50\n')

    lines.append('\n')

    lines.append('/////////////////////////////////////////\n')
    lines.append('//////////// Cells //////////////////////\n')
    lines.append('/////////////////////////////////////////\n')

    lines.append('\n')

    #   Loop over cells
    for nc in range(params.cells['centers'].shape[1]):
        cell_lines = get_tpc_cell_line(
            nc + 1,
            params.cells['centers'][:, nc],
            params.cell_geometry
            )
        for cell_line in cell_lines:
            lines.append(cell_line)
        lines.append('\n')

    lines.append('/////////////////////////////////////////\n')
    lines.append('////////////  ACD  //////////////////////\n')
    lines.append('/////////////////////////////////////////\n')

    lines.append('\n')

    lines.append('BaseACDLayer.Copy Top_ACD\n')
    lines.append(
        'Top_ACD.Position  0.0 0.0 '
        + f'{params.z_centers["front_acd"]*100:10.7f}'
        + '\n'
        )
    lines.append('Top_ACD.Rotation   0. 0. 0.\n')
    lines.append('Bottom_ACD.Material Argon\n')
    lines.append('Top_ACD.Mother     WorldVolume\n')

    lines.append('\n')

    shield_z = - ( params.calorimeter["thickness"] 
                   +  params.shield["thickness"] 
                   )
    lines.append('BaseACDLayer.Copy Bottom_ACD\n')

    lines.append(
        'Bottom_ACD.Position  0.0 0.0 '
        + f'{(params.z_centers["back_acd"] + shield_z) *100:10.7f}'
        + '\n'
        )
    lines.append('Bottom_ACD.Rotation  0. 0. 0.\n')
    lines.append('Bottom_ACD.Material Argon\n')
    lines.append('Bottom_ACD.Mother   WorldVolume\n')

    if params.calorimeter['thickness'] != 0.0:
        lines.append('/////////////////////////////////////////\n')
        lines.append('////////////  Calorimeter  //////////////\n')
        lines.append('/////////////////////////////////////////\n')

        lines.append('Material CsI\n')
        lines.append('CsI.Density 4.53\n')
        lines.append('CsI.ComponentByAtoms Cs 1 // Cs\n')
        lines.append('CsI.ComponentByAtoms I 1 // I\n')
        lines.append('\n')
        lines.append('Volume BaseCalorimeter\n')
        lines.append('BaseCalorimeter.Material CsI\n')
        

        lines.append(
            'BaseCalorimeter.Shape BOX '
            + f'{((vessel_r_inner*100) / (2 ** 0.27)):10.7f} '
            + f'{((vessel_r_inner*100) / (2 ** 0.27)):10.7f} '
            + f'{params.calorimeter["thickness"]/2*100:10.7f} '
            #+ '0. '
            #+ '360.'
            + '\n'
            )
        lines.append('BaseCalorimeter.Color 11\n')

        lines.append('\n')
        lines.append('Calorimeter       CsICal\n')
        lines.append('CsICal.DetectorVolume  BaseCalorimeter\n')
        lines.append('CsICal.SensitiveVolume BaseCalorimeter\n')
        lines.append('CsICal.TriggerThreshold 50\n')
        lines.append('CsICal.EnergyResolution Gauss 662 662 12.65\n')
        lines.append('CsICal.EnergyResolution Gauss 661 661 12.43\n')
        lines.append('CsICal.DepthResolution 662 0.21\n')
        lines.append(
            f'BaseCalorimeter.Position  0.0 0.0 {(-params.calorimeter["thickness"] / 2)*100:10.7f}\n'
            )
        lines.append('BaseCalorimeter.Rotation   0. 0. 0.\n')
        lines.append('BaseCalorimeter.Mother     WorldVolume\n')

        lines.append('\n')

    if params.shield['thickness'] != 0.0:
        lines.append('/////////////////////////////////////////\n')
        lines.append('////////////  Earth Sheild  /////////////\n')
        lines.append('/////////////////////////////////////////\n')   

        lines.append('Material Lead\n')
        lines.append('Lead.Density 11.3\n')
        lines.append('Lead.ComponentByAtoms Pb 1 // Pb\n')
        lines.append('\n')
        lines.append('Material Tungsten\n')
        lines.append('Tungsten.Density 19.3\n')
        lines.append('Tungsten.ComponentByAtoms W 1 // W\n')
        lines.append('\n')

        lines.append('Volume Shield\n')
        lines.append(f'Shield.Material {params.shield["material"]}\n')
        lines.append(
            'Shield.Shape TUBE '
            + '0. '
            + f'{(vessel_r_inner + params.acd["thickness"])*100:10.7f} '
            + f'{params.shield["thickness"]*100:10.7f} '
            + '0. '
            + '360.'
            + '\n'
            )
        shield_z = - ( params.calorimeter["thickness"] 
                   +  params.shield["thickness"] 
                   )
                   
        lines.append(f'Shield.Position 0.0 0.0 {(shield_z)*100:10.7f}\n')
        lines.append('Shield.Color 1\n')
        lines.append('Shield.Mother WorldVolume\n')

        lines.append('\n')

    params.setup_file_lines = lines

    return params

def make_honeycomb(cell_width, max_radius=1.75):
    """
    Returns array of honeycomb of cells that fit within max_radius,
        with flat_to_flat oriented along x, corner_to_corner along y.

        cell_width - flat-to-flat width
        max_radius - honeycomb covers points to this radius
        num_layers -

        centers[2, num_cells] - locations of cell centers in x any.
        corners[2, 6, num_cells] - locations of corners of all hex cells

    Dec 21, 2021   Port from Matlab - TS
    """

    import numpy as np
    from math import sqrt
    import math
    import copy

    #   Step lengths in x and y
    dx = cell_width * 1 / 2
    dy = cell_width * sqrt(3) / 2

    #   Number of layers needed to cover max_radius, not including center
    #   hex
    num_layers = math.ceil(max_radius / dy - 1 / 3)

    #   These are step directions to create layer
    steps = []
    steps.append((0, 2))
    steps.append((-2, 0))
    steps.append((-1, -1))
    steps.append((1, -1))
    steps.append((2, 0))
    steps.append((1, 1))
    steps.append((-1, 1))

    #   Now create lists centers['x'], centers['y'] of locations of segments

    #   Fictitious start at 1,-1
    centers = {}
    centers['x'] = [dx]
    centers['y'] = [-dy]

    #   nc is the cell number
    nc = 0

    for nl in range(num_layers):

        nls = -1

        #   move to starting point in upper right
        nc += 1
        nls += 1
        centers['x'].append(centers['x'][nc-1] + dx * steps[0][0])
        centers['y'].append(centers['y'][nc-1] + dy * steps[0][1])


        #   Now move around the layer up to last step
        for ns in range(1, len(steps)-1):
            for nss in range(nl+1):
                nc = nc+1
                centers['x'].append(centers['x'][nc-1] + dx * steps[ns][0])
                centers['y'].append(centers['y'][nc-1] + dy * steps[ns][1])

        #   Last step is one time fewer
        for nss in range(nl):
            nc = nc+1
            centers['x'].append(centers['x'][nc-1] + dx * steps[-1][0])
            centers['y'].append(centers['y'][nc-1] + dy * steps[-1][1])

    #   Fix up actual start at 0,0
    nc = 0
    centers['x'][nc] = 0
    centers['y'][nc] = 0

    #   centers become arrays
    old_centers = copy.copy(centers)
    centers = np.zeros((2, len(old_centers['x'])), dtype=float)
    centers[0, :] = np.array(old_centers['x'])
    centers[1, :] = np.array(old_centers['y'])

    #   Find corners of all cells

    #   Vertcies of corners of single hexagon
    xv = cell_width / 2 * np.array((1, 0, -1, -1, 0, 1))
    yv = cell_width / 2 * 1 / sqrt(3) * np.array((1, 2, 1, -1, -2, -1))

    #   Add these vertices to centers
    corners = np.zeros((2, len(xv), len(centers[0, :])))
    for nc in range(len(centers[0, :])):
        corners[0, :, nc] = centers[0, nc] + xv
        corners[1, :, nc] = centers[1, nc] + yv

    #   Now restrict to inside max radius
    in_mask \
        = np.sqrt(corners[0, :, :]**2 + corners[1, :, :]**2).max(axis=0) \
            < max_radius
    centers =  centers[:, in_mask]
    corners =  corners[:, :, in_mask]

    return centers, corners

def make_square_array(cell_width, max_radius=1.75):
    """
    Returns array of rectangular cells that fit within max_radius

        cell_width - flat-to-flat width
        max_radius - honeycomb covers points to this radius
        num_layers -

        centers[2, num_cells] - locations of cell centers in x any.
        corners[2, 6, num_cells] - locations of corners of all hex cells

    Dec 21, 2021   Port from Matlab - TS
    """

    import numpy as np
    import math

    #   Find 1d set of centers
    num_edges = math.floor(2 * max_radius / cell_width)
    edges_1d = np.arange(-num_edges / 2, num_edges / 2 + 1) * cell_width
    centers_1d = edges_1d[0:-1] + cell_width / 2

    #   Generate array
    xc, yc = np.meshgrid(centers_1d, centers_1d)
    centers = np.zeros((2, xc.size), dtype=float)
    centers[0, :] = xc.reshape(xc.size)
    centers[1, :] = yc.reshape(yc.size)

    #   Find corners of all cells

    #   Vertcies of corners of single hexagon
    xv = cell_width / 2 * np.array((1, -1, -1, 1))
    yv = cell_width / 2 * np.array((1, 1, -1, -1))

    #   Add these vertices to centers
    corners = np.zeros((2, len(xv), len(centers[0, :])))
    for nc in range(len(centers[0, :])):
        corners[0, :, nc] = centers[0, nc] + xv
        corners[1, :, nc] = centers[1, nc] + yv

    #   Now restrict to inside max radius
    in_mask \
        = np.sqrt(corners[0, :, :]**2 + corners[1, :, :]**2).max(axis=0) \
            < max_radius
    centers =  centers[:, in_mask]
    corners =  corners[:, :, in_mask]

    return centers, corners

def init_simple_geometry(bounding_box):
    """
    Simple "detector" geometry with single rectangual cell
    intended for use with tracks found in uniform fluid,
    such as PENELOPE tracks.

    Note that this detector description does not include the readout.

    Cell is centered about x=y=0, and fully spans bounding_box, along
    with a buffer distance added to all side to e.g., cope with diffusion
    """

    import numpy as np

    #   Cells - here only one

    cells = {}

    #   Location is center of bounding box in x-y, upper edge in z
    cells['positions'] = np.zeros((3,1), dtype=float)
    cells['thetas'] = np.zeros(1, dtype=float)

    #   Dimension of cell
    cells['geometry'] = 'rectangular'
    cells['num_cells'] = 1
    cells['width_xi'] = 2 * np.max(np.abs(bounding_box[0, :]))
    cells['width_xo'] = cells['width_xi']
    cells['width_yi'] = 2 * np.max(np.abs(bounding_box[1, :]))
    cells['width_yo'] = cells['width_yi']
    cells['height'] = np.diff(bounding_box[2, :])

    #   These are used for display - should be moved elsewhere
    cells['plotting'] = {}

    cells['plotting']['base_cell_perimeter'] = {}
    cells['plotting']['base_cell_perimeter']['x'] \
        = cells['width_xi'] / 2 * np.array([1, -1, -1, 1, 1])
    cells['plotting']['base_cell_perimeter']['y'] \
        = cells['width_yi'] / 2 * np.array([1, 1, -1, -1, 1])

    px = []
    py = []
    cx = []
    cy = []
    px.append(cells['plotting']['base_cell_perimeter']['x'])
    py.append(cells['plotting']['base_cell_perimeter']['y'])
    cx.append(0.)
    cy.append(0.)
    cells['plotting']['cell_centers'] = {}
    cells['plotting']['cell_centers']['x'] = cx
    cells['plotting']['cell_centers']['y'] = cy
    cells['plotting']['cell_perimeters'] = {}
    cells['plotting']['cell_perimeters']['x'] = px
    cells['plotting']['cell_perimeters']['y'] = py

    return cells


def global_to_cell_coordinates(r_in, cell, params, reverse=False):
    """
    Transforms coordinates based on cell properties and rotation.
    
    Parameters:
    - r_in (awkward Array): The input coordinates, can be of shape (N, 3), (N, M, 3), or a scalar.
    - cell (awkward Array or scalar): Indicates which cell the coordinates belong to.
    - params (object): An object that contains cell properties like 'centers' and 'rotation'.
    - reverse (bool): Whether to reverse the transformation.
    
    Returns:
    - r_transformed (awkward Array): Transformed coordinates.
    """

    import numpy as np
    import awkward as ak

    # Extract cell parameters
    x_o = params.cells['centers'][0, :]
    y_o = params.cells['centers'][1, :]
    z_o = params.cells['centers'][2, :]
    theta = params.cells['rotation']

    # Determine the sign for reverse transformation
    sign = -1 if reverse else 1
    theta_rad = sign * np.radians(theta)

    # Handle different shapes of r_in
    if r_in.ndim == 3:
        pass
    elif r_in.ndim == 2:
        r_in = ak.Array([r_in])
    else:
        r_in = ak.Array([[r_in]])
        cell = ak.Array([cell])

    modified_x_list = []
    modified_y_list = []
    modified_z_list = []

    # Create mask based on cell presence
    cell_mask = ak.num(cell) > 0

    # Perform reverse transformation. 
    # Reverse: shift + rorate.
    if reverse:
        for i, val in enumerate(cell_mask):
            if val:
                modified_x = r_in[i,0] + sign * ak.Array([x_o[cell[i] - 1]][0])
                modified_y = r_in[i,1] + sign * ak.Array([y_o[cell[i] - 1]][0])
                modified_z = r_in[i,2:] + sign * ak.Array([z_o[cell[i] - 1]][0])

                # Perform rotation
                temp_x = modified_x * np.cos(theta_rad) - modified_y * np.sin(
                    theta_rad)
                temp_y = modified_x * np.sin(theta_rad) + modified_y * np.cos(
                    theta_rad)
            else:
                temp_x = r_in[i, 0]
                temp_y = r_in[i, 1]
                modified_z = r_in[i, 2:]

            modified_x_list.append(temp_x)
            modified_y_list.append(temp_y)
            modified_z_list.append(modified_z)

    # regular: rotate + shift
    else:
        temp_x = r_in[:, 0] * np.cos(theta_rad) - r_in[:,
                                                       1] * np.sin(theta_rad)
        temp_y = r_in[:, 0] * np.sin(theta_rad) + r_in[:,
                                                       1] * np.cos(theta_rad)

        for i, val in enumerate(cell_mask):
            if val:
                modified_x = temp_x[i] + sign * ak.Array([x_o[cell[i] - 1]][0])
                modified_y = temp_y[i] + sign * ak.Array([y_o[cell[i] - 1]][0])
                modified_z = r_in[i,
                                  2:] + sign * ak.Array([z_o[cell[i] - 1]][0])
            else:
                modified_x = temp_x[i]
                modified_y = temp_y[i]
                modified_z = r_in[i, 2:]

            modified_x_list.append(modified_x)
            modified_y_list.append(modified_y)
            modified_z_list.append(modified_z)

    # Convert lists to awkward arrays
    modified_x_array = ak.Array(modified_x_list)
    modified_y_array = ak.Array(modified_y_list)
    modified_z_array = ak.Array(modified_z_list)

    # Concatenate arrays to form the final result
    r_transformed = ak.concatenate([
        modified_x_array[:, None, :], modified_y_array[:, None, :],
        modified_z_array
    ], axis=1)

    return r_transformed


def cell_to_tile_coordinates(
        r_in,
        tile_indices,
        tile_locations,
        reverse=False
        ):
    """
    Translates r_in to r_out, between cell coordinates and "tile"
        coordinates, where tiles are either pixel chips or
        coarse tiles.

    Default is from cell to pixel_chip; if reverse then is pixel_chip to cell

    r_in - rray of locations, dimension [3, :] or [2, :].
        The z coordinate, if present, is not affected.

    tile_indices - array of indices of tiles for each entry in r_in

    pixel_chips is from params

    r_in is currenty on 2 dimensions [space, element]

    TODO[ts]: implemnt for hits - model on global to cell, including comments.
    """

    import numpy as np
    import copy

    r_out = copy.copy(r_in)

    #   If one tile, transpose it
    if tile_indices.ndim==1: tile_indices = tile_indices[:, None]

    x_mesh, y_mesh = np.meshgrid(tile_locations[0], tile_locations[1])

    tile_centers = np.zeros_like(tile_indices[0:2, :], dtype=float)
    tile_centers[0, :] = x_mesh[tile_indices[1, :], tile_indices[0, :]]
    tile_centers[1, :] = y_mesh[tile_indices[1, :], tile_indices[0, :]]

    #   Translate chip to cell: add the center(s)
    if reverse:
        r_out[0:2, :] += tile_centers

    #   Translate cell to chip: subtract the center(s)
    else:
        r_out[0:2, :] -= tile_centers

    return r_out

def find_struck_chips_hits(hits, params):
    """
    Using r_cell in hits, finds chips that are struck for each hit,
    returns hits['chip'], the index of chips struck.

    TODO[ts]: reconcile this will find_span in charge_readout_tools
    TODO[bt]: Convert to awkward when needed
    """

    import numpy as np

    hits['chip'] = np.zeros(hits['r'][0, :, :].shape, dtype=int)

    for nh in range(hits['r'][0,:,0].size):

        x = hits['r_cell'][0, nh, hits['alive'][nh, :]]
        y = hits['r_cell'][1, nh, hits['alive'][nh, :]]

        #   Subtract 1 to get first-index-zero indexing
        rows = np.digitize(
            y,
            params.pixel_chips['edges']['y']
            ) - 1

        in_row_bounds = \
            (rows >= 0) & \
            (rows < params.pixels['num_tile_rows'])

        chip = np.zeros(x.shape, dtype=int) - 1

        for row in np.unique(rows[in_row_bounds]):

            in_row = rows==row

            #   Subtract 1 to get first-index-zero indexing
            columns = np.digitize(
                x[in_row],
                params.pixel_chips['edges']['x'][row]
                ) - 1

            in_column_bounds = \
                (columns>=0) & \
                (columns<params.pixels['num_tile_columns'][row])

            these_chips = np.zeros(in_column_bounds.size, dtype=int) - 1

            these_chips[in_column_bounds] = np.array(
                params.pixels['linear_index'][row]
                )[columns[in_column_bounds]]

            chip[in_row] = these_chips

        hits['chip'][nh,  hits['alive'][nh, :]] = chip

    return hits




