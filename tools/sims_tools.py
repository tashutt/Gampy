#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Routines that define and prepare the geometry for the simulation, including
the parameters class Params, and related geometry tools.

Note that any parameters not part of the simulation, such as wire grids
dimensions, are not defined here.

TODO: Deal with versioning?

@author: tshutt

calorimeter and shield added by sjett 8/10/23
"""

class Params():
    """
    Create parameters object for geomega geometry according to settings, and
        cell_geometry if supplied.

    Inputs:
        settings_file_name - file with input settings.  default is
            standard values in Gampy/settings
        cell_geometry - 'rectangular' or 'hexagonal'. If supplied overides
            values in settings file
    """

    def __init__(self, settings_file_name='default', cell_geometry=None):
        """
        Loads inputs to generate Geomega geometry, then calculates
        parameters from these and cell_geometry.

        Has option for finding limited set of parameters from Cosima .setup
        files.

        Inputs:
            settings_file_name - full file name for inputs .yaml file, or
                Cosima .setup file.  If .setup, then only generates a
                limited set of cells parameters.  'default' is standard
                file in Gampy/settings
            cell_geomtry - "rectangular" or "hexagonal".  If not supplied,
                then taken from settings file.
        """

        import sys

        #   This flag used to turn off calculations when used after
        #   Cosima has run.   Perhaps better done with a decorator?
        self.live = True

        #   Load inputs, calculate parameters.
        if ((settings_file_name == 'default')
            or (settings_file_name.split('.')[-1] == 'yaml')):
            self.inputs = get_params_inputs(settings_file_name, cell_geometry)
            self.calculate()

        #   Loads minimal parameters from .setup file, for backward
        #   compatabile data file opeining.  Kill one day.
        elif settings_file_name.split('.')[-1] == 'setup':
            self.cells = get_cell_params_from_setup(settings_file_name)
            self.live = False

        else:
            sys.exit('Error: unrecognized .setup file: ' + settings_file_name)


    def calculate(self):
        """
        Calculates parameters based on input constants.
        """

        import sys

        #   This probbly not very robust method prevents changing
        #   geometry when working with simulated data
        if not self.live:
            sys.exit('Error in sims_tools.GeometryParams: changing'
                     + ' geomtry not allowed after simulation performed.')

        #   Assign all inputs to attributes of params
        for key in self.inputs.keys():
            attribute_dictionary = {}
            for sub_key in self.inputs[key]:
                attribute_dictionary[sub_key] \
                    = self.inputs[key][sub_key]
            setattr(self, key, attribute_dictionary)

        #   Calcaulate geomaga geometry.
        #   Different versions will be handled here.
        self = calculate_geomega_geometry_v1(self)

    def apply_study_case(self, study, case):
        """ Apply case of study to inputs, and calculate params"""

        if not hasattr(self, 'meta'):
            self.meta = {}

        #   Save study and case to params
        self.meta['study'] = study
        self.meta['case'] = case

        #   Mod study parameters
        for ni in range(len(study.fields[case])):
            self.inputs[study.fields[case][ni]] \
                [study.sub_fields[case][ni]] \
                    = study.values[case][ni]

        #   Recalculate derived values
        self.calculate()

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

    TODO: Add versioning.  This routine currently describes
    a detector with:
        + Layers of ACD on front and back, implemented as tiles with
            the same footprint as the cells
        + Two layers of cells
    """

    import sys
    import math
    import numpy as np

    #   ID for this topology - planar 2 layers + optional scintillator
    #   and shield
    params.topology_id = 1

    #   Calculate a set of temporary variables for convenience.
    surround_r = 1.5 * params.vessel['r_outer']

    vessel_height_outer = 2 * (
        2.0 * params.vessel['wall_thickness']
        + 2.0 * params.planar_acd['thickness']
        + 2.0 * params.cells['height']
        + params.shield['thickness']
        + params.calorimeter['thickness']
        )
    vessel_r = params.vessel['r_outer'] \
        - params.vessel['wall_thickness']

    all_ar_height = vessel_height_outer \
        - 2 * params.vessel['wall_thickness']

    planar_acd_z_center = (
            params.cells['cathode_plane_thickness']
            + params.cells['height']
            + params.cells['anode_plane_thickness']
            + params.planar_acd['thickness'] / 2
            )

    flat_to_flat_outer = params.cells['flat_to_flat'] \
        + 2.0 * params.cells['wall_thickness']
    cell_z_center = (
        params.cells['cathode_plane_thickness']
        + params.cells['height'] / 2.0
        )
    cell_z_anode = (
        params.cells['cathode_plane_thickness']
        + params.cells['height']
        )

    # #   Geomega z coordinate for various layers.  This is the center
    # #   of each volume in the coordinates of its mother volume, which is
    # #   currently the vessel (or Ar)
    # params.z_centers = {}

    # params.z_centers['cathode_plane'] = (
    #     params.cells['cathode_plane_thickness'] / 2.0
    #     )
    # params.z_centers['cells'] = (
    #     params.cells['cathode_plane_thickness']
    #     + params.cells['height'] / 2.0
    #     )
    # params.z_centers['anode_plane'] = (
    #     params.cells['cathode_plane_thickness']
    #     + params.cells['height']
    #     + params.cells['anode_plane_thickness'] / 2.0
    #     )
    # params.z_centers['planar_acd'] = (
    #     params.cells['cathode_plane_thickness']
    #     + params.cells['height']
    #     + params.cells['anode_plane_thickness']
    #     + params.planar_acd['thickness'] / 2
    #     )
    # params.z_centers['vessel'] = (
    #     params.cells['cathode_plane_thickness']
    #     + params.cells['height']
    #     + params.cells['anode_plane_thickness']
    #     + params.planar_acd['thickness']
    #     + params.vessel['thickness']
    #     )

    #   Cell parameters.
    if params.cells['geometry']=='hexagonal':
        params.cells['corner_to_corner'] \
            = params.cells['flat_to_flat'] * 2 / math.sqrt(3)
        params.cells['rotation'] = 30.
        params.cells['wall_length'] \
            = params.cells['flat_to_flat'] / math.sqrt(3)
        params.cells['area'] \
            = 2 * math.sqrt(3) * (params.cells['flat_to_flat'] /2.0)**2
    elif params.cells['geometry']=='rectangular':
        params.cells['rotation'] = 45.
        params.cells['area'] \
            = params.cells['width_x'] * params.cells['width_y']

     #%%  Cell arrays

    #   Generate cell array, either for hexagonal or rectangular cells.
    if params.cells['geometry']=='hexagonal':
        centers, corners = make_honeycomb(
            flat_to_flat_outer,
            params.vessel['r_outer'] - params.vessel['lar_min_radial_gap']
            )
    elif params.cells['geometry']=='rectangular':
        centers, corners = make_square_array(
            params.cells['width'],
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
    params.cells['front_layer_mask'] = front_layer_mask

    params.cells['num_cells'] = front_layer_mask.size

    #   Center and corner locations need handling of z for layers
    params.cells['centers'] = np.zeros((3, 2 * centers.shape[1]), dtype=float)
    params.cells['centers'][0:2, :] = np.tile(centers, 2)
    params.cells['centers'][2, front_layer_mask] = cell_z_center
    params.cells['centers'][2, ~front_layer_mask] = -cell_z_center
    params.cells['corners'] = np.zeros(
        (3, corners.shape[1], 2 * corners.shape[2])
        )
    params.cells['corners'][0:2, :, :] = np.tile(corners, 2)
    params.cells['corners'][2, :, front_layer_mask] \
        = cell_z_center
    params.cells['corners'][2, :, ~front_layer_mask] \
        = -cell_z_center

    #   z location of surface of anode for each cell
    params.cells['z_anode'] = np.zeros_like(params.cells['centers'][0, :])
    params.cells['z_anode'] = (
        cell_z_anode * front_layer_mask
        - cell_z_anode * ~front_layer_mask
        )

    #%% Now make geomega file image

    #   Helper routine to create text lines for cell copies
    def get_tpc_cell_line(cell_number, cell_center, cell_geometry,
                          cell_rotation):
        '''
        For a cell number and location of center, returns the
        line calling out cell position
        '''

        cell_tag = f'Cell_{cell_number:03.0f}'

        line = ['BaseTPCCell.Copy ' + cell_tag + '\n']
        line.append(
            cell_tag + '.Position    '
            + f'{cell_center[0]*100.0:10.7f} '
            + f'{cell_center[1]*100.0:10.7f} '
            + f'{cell_center[2]*100.0:10.7f} '
            + '\n'
            )
        line.append(
            cell_tag + '.Rotation    0. 0. ' + f'{cell_rotation:4.2f}' + '\n')
        line.append(
            cell_tag + '.Mother      AllArgon\n')

        return line

    #%%  Generate geomega file lines

    lines = ['Include $(MEGAlib)/megalibgeo/materials/Materials.geo\n']

    lines.append('\n')

    lines.append('//serves as the sphere from which '+
                 'farfield sources are instantiated\n')
    lines.append(
        'SurroundingSphere '
        + f'{surround_r*100.0:3.0f} '
        + '0. 0. 0. '
        + f'{surround_r*100.0:3.0f}'
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
        + f'{params.cells["height"]/2*100.0:10.7f} '
        + '0. '
        + f'{params.cells["flat_to_flat"]/2*100.0:10.7f} '
        + f'{-params.cells["height"]*100/2:10.7f} '
        + '0. '
        + f'{params.cells["flat_to_flat"]/2*100.0:10.7f} '
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
        + f'{vessel_r*100.0:10.7f} '
        + f'{(params.planar_acd["thickness"]/2)*100.0:10.7f} '
        + '0. '
        + '360.'
        + '\n'
        )
    lines.append('BaseACDLayer.Color 14\n')

    lines.append('\n')

    # Outer ACD cylinder
    # outer_acd_thickness = params.planar_acd["thickness"]
    #   TODO:  remove this kludge - put in params

    calAndShield =( 0* params.calorimeter["thickness"]
                +  params.shield["thickness"]
                )
    lines.append('// Outer ACD cylinder\n')
    lines.append('Volume OuterACD\n')
    lines.append('OuterACD.Material Argon\n')
    lines.append(
        'OuterACD.Shape TUBE '
        + f'{vessel_r*100.0:10.7f} ' # inner radius
        + f'{(vessel_r + params.outer_acd["thickness"])*100.0:10.7f} '
        + f'{(vessel_height_outer/2 + calAndShield/2)*100.0:10.7f} '
        + '0. '
        + '360.'
        + '\n'
    )
    lines.append(
        'OuterACD.Position 0.0 0.0 '
        + f'{(vessel_height_outer/2 - calAndShield)*100.0:10.7f}\n'
        )
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
        + f'{params.vessel["r_outer"]*100.0:10.7f} '
        + f'{vessel_height_outer/2*100.0:10.7f} '
        + '0. '
        + '360.'
        + '\n'
        )
    lines.append('Vessel.Position 0.0 0.0 '
                 + f'{(vessel_height_outer / 2)*100.0:10.7f}\n')
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
        + f'{vessel_r*100.0:10.7f} '
        + f'{all_ar_height/2*100.0:10.7f} '
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
            params.cells['geometry'],
            params.cells['rotation'],
            )
        for cell_line in cell_lines:
            lines.append(cell_line)
        lines.append('\n')

    lines.append('/////////////////////////////////////////\n')
    lines.append('//////////////  ACD  ////////////////////\n')
    lines.append('/////////////////////////////////////////\n')

    lines.append('\n')

    lines.append('BaseACDLayer.Copy Top_ACD\n')
    lines.append(
        'Top_ACD.Position  0.0 0.0 '
        + f'{planar_acd_z_center*100.0:10.7f}'
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
        + f'{-planar_acd_z_center*100.0:10.7f}'
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
            + f'{((vessel_r*100) / (2 ** 0.27)):10.7f} '
            + f'{((vessel_r*100) / (2 ** 0.27)):10.7f} '
            + f'{params.calorimeter["thickness"]/2*100.0:10.7f} '
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
            'BaseCalorimeter.Position  0.0 0.0 '
            + f'{(-params.calorimeter["thickness"] / 2)*100.0:10.7f}\n'
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
            + f'{(vessel_r + params.planar_acd["thickness"])*100.0:10.7f} '
            + f'{params.shield["thickness"]*100.0:10.7f} '
            + '0. '
            + '360.'
            + '\n'
            )
        shield_z = - ( params.calorimeter["thickness"]
                   +  params.shield["thickness"]
                   )

        lines.append(f'Shield.Position 0.0 0.0 {(shield_z)*100.0:10.7f}\n')
        lines.append('Shield.Color 1\n')
        lines.append('Shield.Mother WorldVolume\n')

        lines.append('\n')

    params.setup_file_lines = lines

    return params

def make_honeycomb(cell_width, max_radius=1.75):
    """
    Returns array of 2D honeycomb of cells that fit within max_radius,
        with flat-to-flat oriented along x, corner_to_corner along y.

        cell_width - flat-to-flat width
        max_radius - honeycomb covers points to this radius

        centers[2, num_cells] - x, y locations of cell centers
        corners[2, 6, num_cells] - x, y locations of corners of all hex cells

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
    Returns array of square cells that fit within max_radius

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

def global_to_cell_coordinates(r_in, cell, params, reverse=False):
    """
    Transforms coordinates based on cell properties and rotation.

    Parameters:
    - r_in (awkward Array): The input coordinates, can be of shape
        (N, 3), (N, M, 3), or a scalar.
    - cell (awkward Array or scalar): Indicates which cell the coordinates
        belong to.
    - params (object): An object that contains cell properties like
        'centers' and 'rotation'.
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
                modified_x = r_in[i,0] \
                    + sign * ak.Array([x_o[cell[i] - 1]][0])
                modified_y = r_in[i,1] \
                    + sign * ak.Array([y_o[cell[i] - 1]][0])
                modified_z = r_in[i,2:] \
                    + sign * ak.Array([z_o[cell[i] - 1]][0])

                # Perform rotation
                temp_x = modified_x * np.cos(theta_rad) \
                    - modified_y * np.sin(theta_rad)
                temp_y = modified_x * np.sin(theta_rad) \
                    + modified_y * np.cos(theta_rad)
            else:
                temp_x = r_in[i, 0]
                temp_y = r_in[i, 1]
                modified_z = r_in[i, 2:]

            modified_x_list.append(temp_x)
            modified_y_list.append(temp_y)
            modified_z_list.append(modified_z)

    # regular: rotate + shift
    else:
        temp_x = r_in[:, 0] * np.cos(theta_rad) \
            - r_in[:, 1] * np.sin(theta_rad)
        temp_y = r_in[:, 0] * np.sin(theta_rad) \
            + r_in[:,1] * np.cos(theta_rad)

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

def get_params_inputs(settings_file_name='default', cell_geometry=None):
    """Returns inputs for params.  Cell geometry comes from seetings if
    not supplied."""

    import os
    import yaml

    #   Set file to default if not supplied
    if settings_file_name == 'default':
        settings_file_name = os.path.join(
            os.path.dirname(os.path.split(__file__)[0]),
            'settings',
            'default_sims_settings.yaml'
            )

    #   Load inputs
    with open(settings_file_name, 'r') as file:
        inputs = yaml.safe_load(file)

    #   Take cell_geometry from inputs if not supplied, and remove
    #   from inputs in any case
    if not cell_geometry:
        cell_geometry = inputs['cell_geometry']
    inputs.pop('cell_geometry')

    #   Assign appropriate cell geometry inputs
    if cell_geometry == 'hexagonal':
        for key in inputs['cell_geometries']['hexagonal']:
            inputs['cells'][key] \
                = inputs['cell_geometries']['hexagonal'][key]
    if cell_geometry == 'rectangular':
        for key in inputs['cell_geometries']['rectangular']:
            inputs['cells'][key] \
                = inputs['cell_geometries']['rectangular'][key]
    inputs['cells']['geometry'] = cell_geometry

    #   Remove cell_geometries altogether
    inputs.pop('cell_geometries')

    return inputs

def get_cell_params_from_setup(setup_file_name):
    """ Hack routine to recover minimal cell geometry information
    from .setup file.  Used to recover data before Aug/Sept 2024 overhaul
    of parameters routines.
    TODO: Should be deleted when no longer needed
    """

    import sys
    import numpy as np
    import math

    try:
        with open(setup_file_name, 'r') as f:
            lines = [line for line in f]
    except FileNotFoundError:
        sys.exit('Bad .setup file: ' + setup_file_name)

    #   Geometry lines
    tpc_line = [line for line in lines if
            len(line)>2 and line.split()[0]=='BaseTPCCell.Shape'][0]
    # vessel_line = [line for line in lines if
    #         len(line)>2 and line.split()[0]=='Vessel.Shape'][0]
    # base_acd_line = [line for line in lines if
    #         len(line)>2 and line.split()[0]=='BaseACDLayer.Shape'][0]
    # bottom_acd_line = [line for line in lines if
    #         len(line)>2 and line.split()[0]=='Bottom_ACD.Position'][0]
    # top_acd_line = [line for line in lines if
    #         len(line)>2 and line.split()[0]=='Top_ACD.Position'][0]
    cell_position_lines = [line for line in lines if (
        len(line)>2
        and (line.split()[0][:5] == 'Cell_')
        and (line.split()[0][8:] == '.Position')
        )]
    cell_rotation_lines = [line for line in lines if (
        len(line)>2
        and (line.split()[0][:5] == 'Cell_')
        and (line.split()[0][8:] == '.Rotation')
        )]

    # #   TPC outer dimension from x distance between cells
    # xd = np.diff(np.array([float(line.split()[1]) for line in cell_position_lines]))
    # cell_width_outer = np.abs(xd[xd>1e-4]).min() * 2

    # top_acd_z = float(top_acd_line.split()[3])
    # top_acd_dz = float(base_acd_line.split()[4])
    # cell_z = float(cell_position_lines[0].split()[3])
    # vessel_z = float(vessel_line.split()[4])

    cells = {}

    #   Cell geometry should be based on number of corners of polygon,
    #   which is in tpc_line.  But just assume is hexagonal
    if int(tpc_line.split()[4])==6:
        cells['geometry'] = 'hexagonal'
    elif int(tpc_line.split()[4])==4:
        cells['geometry'] = 'rectangular'
    else:
        sys.exit('Error - unrecognized cell geometry')

    #   TPC height and inner dimension are in TPC cell line
    cells['height'] = float(tpc_line.split()[6]) * 2.0 / 100.0
    cells['flat_to_flat'] = float(tpc_line.split()[8]) * 2.0 / 100.0
    cells['corner_to_corner'] = cells['flat_to_flat'] * 2 / math.sqrt(3)
    cells['area'] = 2 * math.sqrt(3) * (cells['flat_to_flat'] /2.0)**2
    cells['centers'] = np.zeros((3, len(cell_position_lines)), dtype=float)
    for n, line in enumerate(cell_position_lines):
        cells['centers'][:, n] \
            = np.array(cell_position_lines[n].split()[-3:], dtype=float) \
                / 100.0
    cells['rotation'] \
        = np.array(cell_rotation_lines[0].split()[-1], dtype=float)

    #   z location of surface of anode for each cell
    cells['z_anode'] = (
        cells['centers'][2, :]
        + (cells['height'] / 2.0) * (cells['centers'][2, :]>0)
        - (cells['height'] / 2.0) * ~(cells['centers'][2, :]>0)
        ) / 100.0

    return cells