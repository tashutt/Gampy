#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helper routines for TPC cells, including parameters definition.

TODO: Move charge_drift_tools into here.  Deal with materials.
TODO: Add versioning?
TODO: Finishing changing cells indexing: x,y -> 0, 1
TODO: Overhaul hexagonal indexing.  Was previous worrying over
    linear_index, and fussing over row-column indexing needed?
TODO: Move display code out?  Use pythonic patch, etc to simplify.
TODO: move out remaining performance calcs - capacitance
TODO[ts]: see
    https://confluence.slac.stanford.edu/display/GTPC/Detector+Response

Overhaul: implement .yaml inputs, split geometry and rename this tpc_cells,
    clean up, including revised parameters names.  8/24

Signficant rewrite of original Matlab routines, iniial port 8/10/2020
Major repackaging 3/21

@author: tshutt
"""

class Params:
    """
    Create parameters object for TPC cells according to settings, and
        cell geometry, which is supplied or generated separately.

        Currently all cells are identical, if more than one.

    Inputs:
        settings_file_name - full file name for inputs .yaml file.
            'default' is standard file in Gampy/settings
        charge_readout_name - name of charge readout architecture, which
            of supplied overides values in settings file
        cells - supplied from detector geometry of simulation.  Not used
            if simulation is simple uniform medium.
        cell_bounds - if cells not provided, these define simple rectangular
            cell geometry.  Array of lower and upper cell bounds in
            all three dimensions, shape [3, 2].  Default is 1 m^3 cube
            at origin.
    """

    def __init__(self,
                 settings_file_name='default',
                 charge_readout_name=None,
                 cells=None,
                 cell_bounds=None,
                 ):
        """
        Loads tpc cell parameter inputs, with input choices governed by
        charge_readout_name.  Calculates parameters from inputs and
        cell geometry.
        """

        import numpy as np

        #   Load inputs
        self.inputs, self.charge_readout_name = get_params_inputs(
            settings_file_name,
            charge_readout_name,
            )

        #%%  If cells not provided, create simple cell
        if not cells:

            #   If cell_bounds not supplied, then make 1 m^3 box
            if cell_bounds is None:
                cell_bounds = np.array(
                    [[-1, -1, -1], [1, 1, 1]],
                    dtype=float
                    ).T / 2.0

            #   Add fixed buffer, and pitch of largest electrodes
            buffer = 0.01
            if 'coarse_tiles' in self.inputs:
                buffer += self.inputs['coarse_tiles']['pitch']
            elif 'coarse_grids' in self.inputs:
                buffer += self.inputs['coarse_grids']['pitch']
            else:
                if 'pixels' in self.inputs:
                    buffer += self.inputs['pixels']['pitch']
                elif 'anode_grid' in self.inputs:
                    buffer += self.inputs['anode_grid']['pitch']

            #   Add buffer
            cell_bounds[:, 0] -= buffer
            cell_bounds[:, 1] += buffer

            #   initialize cells
            cells = {}

            #   Location is center of cells bounds in x-y, upper edge in z
            cells['positions'] = np.zeros((3,1), dtype=float)
            cells['thetas'] = np.zeros(1, dtype=float)

            #   Minimal cell description
            cells['geometry'] = 'rectangular'
            cells['num_cells'] = 1
            cells['width_x'] = 2 * np.max(np.abs(cell_bounds[0, :]))
            cells['width_y'] = 2 * np.max(np.abs(cell_bounds[1, :]))
            cells['height'] = np.diff(cell_bounds[2, :])[0]

        #   Save cells and charge readout to params
        self.cells = cells

        #   Calculate
        self.calculate()

    def calculate(self):
        """
        Calculates parameters based on input constants
        """

        import math
        import numpy as np

        import charge_drift_tools

        #   Dielectric constants
        eps = define_eps()

        #%%   Assign all inputs to attributes of params
        for key in self.inputs.keys():
            attribute_dictionary = {}
            for sub_key in self.inputs[key]:
                attribute_dictionary[sub_key] \
                    = self.inputs[key][sub_key]
            setattr(self, key, attribute_dictionary)

        #%%   Drift properties

        drift_properties = charge_drift_tools.properties(
            self.charge_drift['drift_field'],
            self.material
            )

        self.charge_drift['velocity'] = drift_properties['velocity']

        self.charge_drift['drift_length'] = self.charge_drift['velocity'] \
            * self.charge_drift['electron_lifetime']

        self.charge_drift['max_drift_drift_time'] = \
            self.cells['height'] \
            / self.charge_drift['velocity']

        #%%   Coarse grids
        if hasattr(self, 'coarse_grids'):

            #   shorthand
            coarse_grids = self.coarse_grids

            #   Coarse grid wire centers.  Start with convenient variables
            pitch = self.coarse_grids['pitch']
            if self.cells['geometry']=='rectangular':
                span = [self.cells['width_x'], self.cells['width_y']]
            elif self.cells['geometry']=='hexagonal':
                span = [self.cells['flat_to_flat'],
                        self.cells['corner_to_corner']]

            #   Save pitch as coarse_pitch
            self.coarse_pitch = pitch

            #   Find edges and centers. Centers are wire locations.
            edges = []
            self.coarse_grids['centers'] = []
            for n in range(2):
                num_edges = math.floor(span[n] / pitch)
                edges.append(
                    np.arange(-num_edges / 2, num_edges / 2 + 1) * pitch
                    )
                self.coarse_grids['centers'].append(
                    edges[n][0:-1] + pitch / 2
                    )

            #   Locations of wire ends.  Useful for plotting.
            #   Dimensions are [x-y, end, wire]
            coarse_grids['x_wire_ends'] = {}
            coarse_grids['y_wire_ends'] = {}

            #   Rectangular cells are fairly simple
            if self.cells['geometry']=='rectangular':
                xe = []
                ye = []
                for y in coarse_grids['centers'][1]:
                    xe.append(np.array([-span[0] / 2, span[0] / 2]))
                    ye.append(np.array([y, y]))
                    coarse_grids['x_wire_ends']['x'] = xe
                    coarse_grids['x_wire_ends']['y'] = ye
                for x in coarse_grids['centers'][0]:
                    xe.append(np.array([x, x]))
                    ye.append(np.array([-span[1] / 2, span[1] / 2]))
                    coarse_grids['y_wire_ends']['x'] = xe
                    coarse_grids['y_wire_ends']['y'] = ye

            #   Hexagonal cells take work
            elif self.cells['geometry']=='hexagonal':

                # xe = []
                # ye = []
                coarse_grids['x_wire_ends'] = []
                for y in coarse_grids['centers'][1]:
                    ends = np.zeros((2, 2), dtype=float)
                    ends[0, :] = np.array([
                        - span[0] / 2 + abs(y) / math.sqrt(3),
                        span[0] / 2 - abs(y) / math.sqrt(3)
                        ])
                    ends[1, :] = np.array([y, y])
                    # xe.append(np.array([
                    #     - span[0] / 2 + abs(y) / math.sqrt(3),
                    #     span[0] / 2 - abs(y) / math.sqrt(3)
                    #     ]))
                    # ye.append(np.array([y, y]))
                    coarse_grids['x_wire_ends'].append(ends)
                # coarse_grids['x_wire_ends']['x'] = xe
                # coarse_grids['x_wire_ends']['y'] = ye

                coarse_grids['y_wire_ends'] = []
                for x in coarse_grids['centers'][0]:
                    ends = np.zeros((2, 2), dtype=float)
                    if x < - span[0] / 4:
                        ends[0, :] = np.array([x, x])
                        ends[1, :] = np.array([
                            - (span[0] / 2 + x) * math.sqrt(3),
                            (span[0] / 2 + x) * math.sqrt(3)
                            ])
                        # xe.append(np.array([x, x]))
                        # ye.append(np.array([
                        #     - (span[0] / 2 + x) * math.sqrt(3),
                        #     (span[0] / 2 + x) * math.sqrt(3)
                        #     ]))
                    elif x < span[0] / 4:
                        ends[0, :] = np.array([x, x])
                        ends[1, :] = np.array([-span[1] / 2, span[1] / 2])
                        # xe.append(np.array([x, x]))
                        # ye.append(np.array([-span[1] / 2, span[1] / 2]))
                    else:
                        ends[0, :] = np.array([x, x])
                        ends[1, :] = np.array([
                            - (span[0] / 2 - x) * math.sqrt(3),
                            (span[0] / 2 - x) * math.sqrt(3),
                            ])
                        # xe.append(np.array([x, x]))
                        # ye.append(np.array([
                        #     - (span[0] / 2 - x) * math.sqrt(3),
                        #     (span[0] / 2 - x) * math.sqrt(3),
                        #     ]))
                    coarse_grids['y_wire_ends'].append(ends)
                # coarse_grids['y_wire_ends']['x'] = xe
                # coarse_grids['y_wire_ends']['y'] = ye

            #   Sampling time and pitch - equal to gap
            coarse_grids['sampling_pitch'] = self.coarse_grids['gap']
            coarse_grids['sampling_time'] = \
                coarse_grids['sampling_pitch'] \
                    / self.charge_drift['velocity']

            # #   TODO: Move this elsehwere
            # #   Capacitances is approximate per length for
            # #   plane of grid wires between grounded plates,
            # #   following Erskine.  See
            # #   Van Esch criticism that this
            # #   isn't correct for noise estimates.
            # coarse_grids['capacitance'] = (
            #     2 * math.pi * eps['lar'] * eps['not']
            #     / (math.pi
            #        * self.coarse_grids['gap']
            #     / self.coarse_grids['pitch']
            #     - math.log(2 * math.pi
            #        * self.coarse_grids['wire_radius']
            #     / self.coarse_grids['pitch']))
            #     * self.cells['width_x']
            #     )

        #%%   Anode grid - 1D wires oriented along y
        if hasattr(self, 'anode_grid'):

            #   Convenient variable
            pitch = self.anode_grid['pitch']

            #   Save pitch as coarse_pitch
            self.coarse_pitch = pitch

            #   Wires orienteted along y, so span based on cell x width
            span = self.cells['width_x']

            #   Find edges and centers. Centers are wire locations.
            num_edges = math.floor(span / pitch)
            self.anode_grid['edges'] \
                = np.arange(-num_edges / 2, num_edges / 2 + 1) * pitch
            self.anode_grid['centers'] \
                = self.anode_grid['edges'][0:-1] + pitch / 2

            #   Sampling is set by wire pitch
            self.anode_grid['sampling_pitch'] = pitch
            self.anode_grid['sampling_time'] = \
                self.anode_grid['sampling_pitch'] \
                    / self.charge_drift['velocity']

        #%%   Chip array for GAMPixG defined based on coarse grids.
        #   Note - there are no input params for chip_array.
        if self.charge_readout_name=='GAMPixG':

            self.chip_array = {}

            #   Sensor type
            self.chip_array['type'] = 'chip_array'

            #   Array pitch is same as coarse grids
            self.chip_array['pitch'] = self.coarse_grids['pitch']

            #   Array pitch is same as coarse grids
            self.chip_array['sampling_pitch'] = self.coarse_grids['gap']
            self.chip_array['sampling_time'] = (
                self.chip_array['sampling_pitch']
                / self.charge_drift['velocity']
                )

            #   Rectangular cells are fairly simple
            if self.cells['geometry']=='rectangular':

                pitch = np.diff(self.coarse_grids['centers'][0][:2])

                self.chip_array['centers'] = []
                self.chip_array['edges'] = []

                #   Loop over y, x
                for n in range(2):

                    #   Edges are at coarse grid wires.
                    edges = self.coarse_grids['centers'][n]

                    #   Centers are between edges
                    centers = edges[0:-1] + pitch / 2

                    #   Assign
                    self.chip_array['centers'].append(centers)
                    self.chip_array['edges'].append(edges)

            #   Hexagonal calculations. The cell is oriented with an
            #   edge on the x axis.  tiles are arranged in rows.
            elif self.cells['geometry']=='hexagonal':

                #   convenient variables
                w =  self.chip_array['pitch']
                num_rows = math.floor(self.cells['corner_to_corner'] / w)
                slope = 1 / math.sqrt(3)
                xp = self.cells['corner_to_corner'] * slope

                #   ys is y location of tile centers, nx is the number
                #   of tiles in each row of ys
                #   Cacluation is different for num_rows odd and even

                #   Odd number of rows
                if num_rows % 2:
                    m = (num_rows - 1) / 2
                    ys = np.arange(0, m + 1) * w
                    xlim = xp - (ys + w / 2) * slope
                    nx = np.fix(2 * xlim / w).astype(int)
                    ys = np.insert(ys, 0, -np.flip(ys[1:]))
                    nx = np.insert(nx, 0, np.flip(nx[1:]))

                #   Even number of rows
                else:
                    m = num_rows / 2
                    ys = (np.arange(1, m+1) - 0.5) * w
                    xlim = xp - (ys + w) * slope
                    nx = np.fix(2 * xlim / w).astype(int)
                    ys = np.insert(ys, 0, -np.flip(ys))
                    nx = np.insert(nx, 0, np.flip(nx))

                #   Now centers - x must be list of lists: [row][column]
                self.chip_array['centers'] = [[], []]
                self.chip_array['centers'][0] = []
                for n in range(len(ys)):
                    m = (nx[n]-1) / 2
                    self.chip_array['centers'][0].append(
                        np.arange(-m, m+1) * w
                        )
                #   y is simple array
                self.chip_array['centers'][1] = ys

            # #   Numbers
            # num_tiles = 0
            # num_rows =len(self.chip_array['centers']['x'])
            # num_columns = []
            # for xs in self.chip_array['centers']['x']:
            #     num_columns.append(len(xs))
            #     num_tiles += len(xs)
            # self.pixels['num_tiles'] = num_tiles
            # self.pixels['num_tile_columns'] \
            #     = np.array(num_columns, dtype=int)
            # self.pixels['num_tile_rows'] = num_rows

            # #   Linear index has strucure:
            # #   ...[row][column]
            # linear_index = []
            # i_o = 0
            # n = 0
            # for xs in self.chip_array['centers']['x']:
            #     # linear_index.append((np.arange(0, len(xs)) + i_o).tolist())
            #     linear_index.append(np.arange(0, len(xs)) + i_o)
            #     i_o = linear_index[n][-1] + 1
            #     n += 1
            # self.chip_array['linear_index'] = linear_index

            # #   Row  and column indices have structures:
            # #   ...['row_index'][tile_number]
            # #   ...['column_index'][tile_number]
            # row_index = []
            # column_index = []
            # n = 0
            # for row in range(num_rows):
            #     for column in range(num_columns[row]):
            #         row_index.append(row)
            #         column_index.append(column)
            # self.chip_array['row_index'] \
            #     = np.array(row_index, dtype=int)
            # self.chip_array['column_index'] \
            #     = np.array(column_index, dtype=int)

            # #   Tile edges in x and y
            # x_edges = []
            # for xs in self.chip_array['centers']['x']:
            #     x_edges.append(np.append(xs - w / 2, xs[-1] + w / 2))
            # y_edges = np.append(ys - w / 2, ys[-1] + w / 2)
            # self.chip_array['edges'] = {}
            # self.chip_array['edges']['x'] = x_edges
            # self.chip_array['edges']['y'] = y_edges

            # #   Sampling time and pitch - equal to lateral pitch
            # self.chip_array['sampling_pitch'] = self.pixels['pitch']
            # self.chip_array['sampling_time'] = \
            #     self.chip_array['sampling_pitch'] \
            #         / self.charge_drift['velocity']

            # #   Tile outlines for plotting
            # self.chip_array['plotting'] = {}
            # self.chip_array['plotting']['outline'] = {}
            # self.chip_array['plotting']['outline_rc'] = {}
            # self.chip_array['plotting']['outline']['x'] = []
            # self.chip_array['plotting']['outline']['y'] = []
            # self.chip_array['plotting']['outline_rc']['x'] = []
            # self.chip_array['plotting']['outline_rc']['y'] = []
            # x = []
            # y = []
            # for row in range(num_rows):
            #     x_outline = []
            #     y_outline = []
            #     for xs in self.chip_array['centers']['x'][row]:
            #         xo = (
            #             xs
            #             + np.array([-1, 1, 1, -1, -1]) * w / 2
            #             )
            #         yo = (
            #             self.chip_array['centers']['y'][row]
            #             + np.array([-1, -1, 1, 1, -1]) * w / 2
            #             )
            #         x.append(xo)
            #         y.append(yo)
            #         x_outline.append(xo)
            #         y_outline.append(yo)
            #     self.chip_array['plotting']['outline_rc']['x'] \
            #         .append(x_outline)
            #     self.chip_array['plotting']['outline_rc']['y'] \
            #         .append(y_outline)
            # self.chip_array['plotting']['outline']['x'] = x
            # self.chip_array['plotting']['outline']['y'] = y

        #%%   Coarse tiles
        if hasattr(self, 'coarse_tiles'):

            #   Pitch - convenient variable
            pitch = self.coarse_tiles['pitch']

            #   Save pitch as coarse_pitch
            self.coarse_pitch = pitch

            #   Calculate maximum number of tiles in span, and their edges
            #   and centers

            self.coarse_tiles['centers'] = []
            self.coarse_tiles['edges'] = []

            #   Span is cell width.
            span = [self.cells['width_x'], self.cells['width_y']]

            #   Loop over y, x
            for n in range(2):

                #   centers of tile edges and centers
                num_edges = math.floor(span[n] / pitch)
                edges = np.arange(-num_edges / 2, num_edges / 2 + 1) * pitch
                centers = edges[0:-1] + pitch / 2

                #   Assign
                self.coarse_tiles['centers'].append(centers)
                self.coarse_tiles['edges'].append(edges)

            #   Sampling time and pitch, equal to gap .  But don't create
            #   for LArPix since there this sensor is virtual
            if not self.charge_readout_name=='LArPix':
                self.coarse_tiles['sampling_pitch'] \
                    = self.coarse_tiles['gap']
                self.coarse_tiles['sampling_time'] \
                    = self.coarse_tiles['sampling_pitch'] \
                        / self.charge_drift['velocity']

        #%%   Pixels - this smallest read out unit, and should match to
        #   an actual or logical chip
        if hasattr(self, 'pixels'):

            #   Convenient variable.  Note we assume that pitch is equal in
            #   both directions
            pitch = self.pixels['pitch']

            #   Span depends on architecture
            if self.charge_readout_name=='GAMPixG':
                span = [self.chip_array['pitch']] * 2
            elif self.charge_readout_name=='GAMPixD':
                span = [self.coarse_tiles['pitch']] * 2
            elif self.charge_readout_name=='LArPix':
                span = [self.cells['width_x'], self.cells['width_y']]


            #   Edges and centers
            self.pixels['centers'] = []
            self.pixels['edges'] = []
            for n in range(2):
                num_edges = math.floor(span[n] / pitch)
                self.pixels['edges'].append(
                    np.arange(-num_edges / 2, num_edges / 2 + 1) * pitch)
                self.pixels['centers'].append(
                    self.pixels['edges'][n][0:-1] + pitch / 2)

            # #   Focus factor given that pixels may not fill span.
            #   TODO: implement this fully, and separately in x, y for
            #   LArPix?   Also, should also apply to coarse_tiles
            # focus_factor = (
            #     self.pixels['edges'][0][-1]
            #     - self.pixels['edges'][0][0]
            #     ) / span
            # self.pixels['focus_factor'] = focus_factor

            #   Sampling time and pitch - equal to lateral pitch
            self.pixels['sampling_pitch'] = self.pixels['pitch']
            self.pixels['sampling_time'] = \
                self.pixels['sampling_pitch'] \
                    / self.charge_drift['velocity']

    def apply_study_case(self, study, case):
        """ Apply case of study to inputs, and calculate params
        """

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

def get_params_inputs(
             settings_file_name='default',
             charge_readout_name=None,
             ):
    """Load default or supplied params inputs, with charge_readout_name
    used to choose amongst inputs
    """

    import os
    import yaml

    #   Set file to default if not supplied
    if settings_file_name == 'default':
        settings_file_name = os.path.join(
            os.path.dirname(os.path.split(__file__)[0]),
            'settings',
            'default_readout_settings.yaml'
            )

    #   Load inputs
    with open(settings_file_name, 'r') as file:
        inputs = yaml.safe_load(file)

    #   Take charge readout from inputs if not supplied, and remove
    #   from inputs in any case
    if not charge_readout_name:
        charge_readout_name = inputs['charge_readout_name']
    inputs.pop('charge_readout_name')

    #   Assign appropriate charge readout inputs
    if charge_readout_name == 'GAMPixG':
        inputs['coarse_grids'] \
            = inputs['charge_readout']['GAMPixG']['coarse_grids']
        inputs['pixels'] \
            = inputs['charge_readout']['GAMPixG']['pixels']
    elif charge_readout_name == 'GAMPixD':
        inputs['coarse_tiles'] \
            = inputs['charge_readout']['GAMPixD']['coarse_tiles']
        inputs['pixels'] \
            = inputs['charge_readout']['GAMPixD']['pixels']
    elif charge_readout_name == 'LArPix':
        inputs['pixels'] \
            = inputs['charge_readout']['LArPix']['pixels']
    elif charge_readout_name == 'AnodeGridG':
        inputs['anode_grid'] \
            = inputs['charge_readout']['AnodeGridG']['anode_grid']
    elif charge_readout_name == 'AnodeGridD':
        inputs['anode_grid'] \
            = inputs['charge_readout']['AnodeGridD']['anode_grid']

    #   Remove charge_readout from inputs
    inputs.pop('charge_readout')

    return inputs, charge_readout_name

def cell_to_chip_coordinates(
        r_in,
        chip_indices,
        chip_locations,
        reverse=False
        ):
    """
    Translates r_in to r_out, between cell coordinates and
        pixel chip coordinates.

    Default is from cell to pixel_chip; if reverse then is pixel_chip to cell

    Inputs:
        r_in - locations in cell coordates. Dimension [3, :] or [2, :].
            The z coordinate, if present, is not affected.
        chip_indices - array of indices of chips for each entry in r_in
        chip_locations -
    pixel_chips is from params

    r_in is currenty on 2 dimensions [space, element]

    TODO[ts]: implement for hits - model on global to cell,
        including comments.??   Had opposit idea in struck chips below
    """

    import numpy as np
    import copy

    r_out = copy.copy(r_in)

    #   If one chip, transpose it
    if chip_indices.ndim==1: chip_indices = chip_indices[:, None]

    #   Create array of length of long dimension of r_in with
    #   centers of chips for each entry in r_in
    x_mesh, y_mesh = np.meshgrid(chip_locations[0], chip_locations[1])
    chip_centers = np.zeros_like(chip_indices[0:2, :], dtype=float)
    chip_centers[0, :] = x_mesh[chip_indices[1, :], chip_indices[0, :]]
    chip_centers[1, :] = y_mesh[chip_indices[1, :], chip_indices[0, :]]

    #   Translate chip to cell: add the center(s)
    if reverse:
        r_out[0:2, :] += chip_centers

    #   Translate cell to chip: subtract the center(s)
    else:
        r_out[0:2, :] -= chip_centers

    return r_out

def find_struck_chips_hits(hits, params):
    """
    Using r_cell in hits, finds chips that are struck for each hit,
    returns hits['chip'], the index of chips struck.

    TODO:  Use more general input than hits - r perhaps?
    TODO[ts]: reconcile this with find_span in charge_readout_tools
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

def define_eps():
    """ inputs for capacitance calculation.  (In principle at
    least the LAr and LXe values should derive from
    material definition detector
    defition """
    # TODO - kill this.

    eps = {}
    eps = {
        'not': 8.8854e-12,
        'si': 12,
        'sio2': 3.6,
        'lxe': 1.7,
        'lar': 1.505,
        'ptfe': 2,  #  Used to be sold by Arlon, but can't find now
        'polyimide': 4.2,  #  see e.g., http://www.arlonemd.com
        'g10': 4.7
        }

    return eps
