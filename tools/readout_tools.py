#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collection of routines used to define and apply detector params.

TODO: Add versioning?
TODO: Where should materials be?
TODO: Finishing changing cells indexing: x,y -> 0, 1
TODO: Overhaul hexagonal indexing.  Was previous worrying over
    linear_index, and fussing over row-column indexing needed?
TODO: Move display code out.  Use pythonic patch, etc to simplify.
TODO: move out remaining performance calcs - capacitance
TODO[ts]: see
    https://confluence.slac.stanford.edu/display/GTPC/Detector+Response

Signficant rewrite of original Matlab routines, iniial port 8/10/2020
Major repackaging 3/21

@author: tshutt
"""

def default_geomega_inputs(cell_geometry='rectangular'):
    """
    Default values of inputs defining the detector geometry for
    geomega.

    TODO: Add versioning.   Call versions in geometry_tools?
    """

    inputs = {}

    inputs['vessel'] = {}
    inputs['vessel']['r_outer'] = 1.85
    inputs['vessel']['wall_thickness'] = 0.004
    inputs['vessel']['lar_min_radial_gap'] = 0.05

    inputs['cells'] = {}
    inputs['cells']['geometry'] = cell_geometry
    inputs['cells']['height'] = 0.175
    inputs['cells']['flat_to_flat'] = 0.175
    inputs['cells']['wall_thickness'] = 0.3e-3

    inputs['acd'] = {'thickness': 0.007}

    inputs['calorimeter'] = {}
    inputs['calorimeter']['thickness'] = 0.0

    inputs['shield'] = {}
    inputs['shield']['thickness'] = 0.0
    inputs['shield']['material'] = 'Lead'

    inputs['anode_planes'] = {'thickness': 0.005}
    inputs['cathode_plane'] = {'thickness': 0.005}

    return inputs

def default_response_inputs(charge_readout):
    """"
    Default values of inputs defining the detector response,
    including the definition of the charge readout, which is not part
    of the geometry

    Needs versioning?
    """

    import sys

    #%% Define inputs
    inputs = {}

    # Material
    inputs['material'] = {}
    inputs['material']['name'] = 'Ar'
    inputs['material']['long_name'] = 'Argon'
    inputs['material']['temperature'] = 110.0
    inputs['material']['w'] = 0.0220
    inputs['material']['recombination'] = 0.3000
    inputs['material']['fano'] = 0.1160
    inputs['material']['sigma_p'] = 0.0600

    #   Charge Drfit
    inputs['charge_drift'] = {}
    inputs['charge_drift']['electron_lifetime'] = 10.0e-3
    inputs['charge_drift']['drift_field'] = 0.5e5   #  V/m
    inputs['charge_drift']['diffusion_fraction'] = 1.0

    #  Light readout
    inputs['light'] = {}
    inputs['light']['spe_threshold'] = 3.0
    inputs['light']['spe_noise'] = 0.2   #   About right for PMT???
    inputs['light']['collection'] = 0.1

    #   Simple single number for now
    inputs['spatial_resolution'] = {}
    inputs['spatial_resolution']['sigma_xy'] = 0.5e-3
    inputs['spatial_resolution']['sigma_z'] = 0.5e-3

    #   Simple energy resolution - should retire.  Currently
    #   I think just
    #   used for error estimate for .evta file.  Should do this
    #   differently
    """ Simplified treatment of energy, used to estimate energy
    resolution as input to Revan  Resolution has
    rt(E) term and a constant. rt(E) is sigma_o at energy_o,
    constant is sigma_f
    Might want to add a simple estimate of threshold here
    """
    inputs['simple_energy_resolution'] = {}
    inputs['simple_energy_resolution']['sigma_o'] = 0.015
    inputs['simple_energy_resolution']['energy_o'] = 1000
    inputs['simple_energy_resolution']['sigma_f'] = 5

    #%%  Charge readout
    """
    Charge readout is independent of detector geometry.  The readout
    defined here extends over a cell. The cells are
    defined in detector geometery.  The simplest geometry has only
    a single cell.

    Note 'type' which defines sensor type for charge_readout_tools
    """

    if charge_readout=='GAMPixG':
        """ GAMPix for GammaTPC """

        #  Coarse grids
        inputs['coarse_grids'] = {}
        inputs['coarse_grids']['type'] = 'coarse_grids'
        inputs['coarse_grids']['pitch'] = 10*1e-3
        inputs['coarse_grids']['wire_radius'] = 100e-6
        inputs['coarse_grids']['gap'] = 10*1e-3
        inputs['coarse_grids']['power_per_wire'] = 10e-3
        inputs['coarse_grids']['noise'] = 20
        inputs['coarse_grids']['signal_fraction'] = 0.6
        inputs['coarse_grids']['signal_sharing'] = 4  # of wires measured
        inputs['coarse_grids']['threshold_sigma'] = 4.0

        #  Pixels
        inputs['pixels'] = {}
        inputs['pixels']['type'] = 'pixels'
        inputs['pixels']['pitch'] = 500e-6
        inputs['pixels']['power_per_pixel'] = 1e-3
        inputs['pixels']['noise'] = 25
        inputs['pixels']['threshold_sigma'] = 4.0
        inputs['pixels']['pad_width'] = 20e-6
        inputs['pixels']['sio2_thickenss'] = 1e-6
        inputs['pixels']['stray_capacitance'] = 0.01e-12

    elif charge_readout=='GAMPixD':
        """ GAMPix for DUNE """

        #  Coarse tiles
        inputs['coarse_tiles'] = {}
        inputs['coarse_tiles']['type'] = 'coarse_tiles'
        inputs['coarse_tiles']['pitch'] = 0.1
        inputs['coarse_tiles']['gap'] = 0.01
        inputs['coarse_tiles']['power_per_tile'] = 1e-3
        inputs['coarse_tiles']['noise'] = 50
        inputs['coarse_tiles']['signal_fraction'] = 1
        inputs['coarse_tiles']['threshold_sigma'] = 5.0

        #  Pixels
        inputs['pixels'] = {}
        inputs['pixels']['type'] = 'pixels'
        inputs['pixels']['pitch'] = 0.005
        inputs['pixels']['power_per_pixel'] = 1e-3
        inputs['pixels']['noise'] = 50
        inputs['pixels']['threshold_sigma'] = 4.0

    elif charge_readout=='LArPix':
        """ LArPix for DUNE """

        #   These are "virtual" coarse tiles - used for pixel readout
        #   computation only
        inputs['coarse_tiles'] = {}
        inputs['coarse_tiles']['pitch'] = 0.1

        #  Pixels
        inputs['pixels'] = {}
        inputs['pixels']['type'] = 'pixels'
        inputs['pixels']['pitch'] = {}
        inputs['pixels']['pitch'] = 0.005
        inputs['pixels']['power_per_pixel'] = 1e-3
        inputs['pixels']['noise'] = 1000
        inputs['pixels']['threshold_sigma'] = 5.0

    elif charge_readout=='AnodeGridG':
        """ Single direction of anode wires for GammaTPC """

        #  Anode wires
        inputs['anode_grid'] = {}
        inputs['anode_grid']['type'] = 'anode_grid'
        inputs['anode_grid']['pitch'] = 0.001
        inputs['anode_grid']['power_per_wire'] = 1e-3
        inputs['anode_grid']['noise'] = 50
        inputs['anode_grid']['threshold_sigma'] = 5.0

    elif charge_readout=='AnodeGridD':
        """ Single direction of anode wires for DUNE """

        #  Anode wires
        inputs['anode_grid'] = {}
        inputs['anode_grid']['type'] = 'anode_grid'
        inputs['anode_grid']['pitch'] = 0.004
        inputs['anode_grid']['power_per_wire'] = 1e-3
        inputs['anode_grid']['noise'] = 800
        inputs['anode_grid']['threshold_sigma'] = 5.0

    else:
        sys.exit('*** Error in default_inputs: unrecognized charge readout')

    return inputs

class GeoParams:
    """
    Params object for geometry
    """

    def __init__(self,
                 detector_geometry='simple',
                 cell_geometry='rectangular',
                 bounding_box=None,
                 ):
        """
        Initializes geometry parameters:
            + defines geometry topology, and cell geometry
            + loads default input constants
            + calculates parameters.
        """

        import sys

        #   Set cell and detector geometries in params
        #   Set versions in params here in future
        self.detector_geometry = detector_geometry
        self.cell_geometry = cell_geometry

        #   For geomega geometry, get default inputs
        #   In the future, will need versioning here
        if detector_geometry=='geomega':
            self.inputs = default_geomega_inputs(cell_geometry)

        #   For simple detetor geometry, cell geometry is rectangular
        elif detector_geometry=='simple':
            if cell_geometry!='rectangular':
                print("*** Warning: cell_gometry set to 'rectangular'")
            cell_geometry='rectangular'

        #   Else - bad geometry
        else:
            sys.exit('Error in GeoParam - bad detector geometry')

        #   This flag used to turn off calculations when used after
        #   Cosima has run.   Perhaps better done with a decorator?
        self.live = True

        #   Calculate geometry
        self.calculate(bounding_box)

    def calculate(self,
                  bounding_box=None,
                  ):
        """
        Calculates parameters based on input constants
        """

        import sys
        import numpy as np

        import geometry_tools

        if not self.live:
            sys.exit('Error: GeoParams calculate not allowed in this state')

        #   Simple geometry
        if self.detector_geometry=='simple':

            #   Set default bounding box if not supplied
            if bounding_box is None:
                bounding_box = np.array(
                    [[-1, -1, -1], [1, 1, 1]],
                    dtype=float
                    ).T

            #   All that is calculated is cells
            self.cells = geometry_tools.init_simple_geometry(
                bounding_box,
                )

        #   Geomega geometry
        elif self.detector_geometry=='geomega':

            #   Assign all inputs to attributes of params
            for key in self.inputs.keys():
                attribute_dictionary = {}
                for subkey in self.inputs[key]:
                    attribute_dictionary[subkey] \
                        = self.inputs[key][subkey]
                setattr(self, key, attribute_dictionary)

            #   Calcaulate geomaga geometry.
            #   Different versions will be handled here.
            self = geometry_tools.calculate_geomega_geometry_v1(self)

class ResponseParams:
    """
    Parameter describing the detector readout.  There are a set of
    "input" parameters, from which all else is derived via the "recalculate"
    method.
    To change parameters, modify input parameters and recalculate - do not
    directly modify derived parameters.
    """

    def __init__(self,
                 charge_readout='GAMPixG',
                 geo_params=None,
                 ):
        """
        Initializes response parameters:
            + defines charge readout
            + loads default input constants
            + calculates parameters.

        Inputs:
            geo_params - geometry parameters.  If not supplied, then
                uses GeoParams default
            charge_readout - Options defined in charge_readout_tools
        """

        #   Default simple geometry if geo_params not supplied
        if not geo_params:
            geo_params = GeoParams()

        #   Save cells from geo_params, and charge readout to params
        self.cells = geo_params.cells
        self.cell_geometry = geo_params.cell_geometry
        self.charge_readout = charge_readout

        #   Default inputs
        self.inputs = default_response_inputs(charge_readout)

        #   Calculate
        self.calculate()

    def calculate(self):
        """
        Calculates parameters based on input constants

        TODO revisit charge readout decision making?
        """

        import charge_drift_tools

        import math
        import numpy as np

        #   Dielectric constants
        eps = define_eps()

        #%%   Assign all inputs to attributes of params
        for key in self.inputs.keys():
            attribute_dictionary = {}
            for subkey in self.inputs[key]:
                attribute_dictionary[subkey] \
                    = self.inputs[key][subkey]
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
            span = [self.cells['width_xi'],
                    self.cells['width_yi']]

            #   Save pitch as coarse_pitch
            self.coarse_pitch = pitch

            #   Find edges and centers. Centers are wire locations.
            self.coarse_grids['edges'] = []
            self.coarse_grids['centers'] = []
            for n in range(2):
                num_edges = math.floor(span[n] / pitch)
                self.coarse_grids['edges'].append(
                    np.arange(-num_edges / 2, num_edges / 2 + 1) * pitch
                    )
                self.coarse_grids['centers'].append(
                    self.coarse_grids['edges'][n][0:-1] + pitch / 2
                    )

            #   Description for plotting.   Wires described by their ends
            coarse_grids['plotting'] = {}
            coarse_grids['plotting']['wire_ends_x'] = {}
            coarse_grids['plotting']['wire_ends_y'] = {}

            #   Rectangular cells are fairly simple
            if self.cell_geometry=='rectangular':
                xe = []
                ye = []
                for y in coarse_grids['centers'][1]:
                    xe.append(np.array([-span[0] / 2, span[0] / 2]))
                    ye.append(np.array([y, y]))
                    coarse_grids['plotting']['wire_ends_x']['x'] = xe
                    coarse_grids['plotting']['wire_ends_x']['y'] = ye
                for x in coarse_grids['centers'][0]:
                    xe.append(np.array([x, x]))
                    ye.append(np.array([-span[1] / 2, span[1] / 2]))
                    coarse_grids['plotting']['wire_ends_y']['x'] = xe
                    coarse_grids['plotting']['wire_ends_y']['y'] = ye

            #   Hexagonal cells take work
            elif self.cell_geometry=='hexagonal':

                xe = []
                ye = []
                for y in coarse_grids['centers'][1]:
                    xe.append(np.array([
                        - span[0] / 2 + abs(y) / math.sqrt(3),
                        span[0] / 2 - abs(y) / math.sqrt(3)
                        ]))
                    ye.append(np.array([y, y]))
                coarse_grids['plotting']['wire_ends_x']['x'] = xe
                coarse_grids['plotting']['wire_ends_x']['y'] = ye

                xe = []
                ye = []
                for x in coarse_grids['centers'][0]:
                    if x < - span[0] / 4:
                        xe.append(np.array([x, x]))
                        ye.append(np.array([
                            - (span[0] / 2 + x) * math.sqrt(3),
                            (span[0] / 2 + x) * math.sqrt(3)
                            ]))
                    elif x < span[0] / 4:
                        xe.append(np.array([x, x]))
                        ye.append(np.array([-span[1] / 2, span[1] / 2]))
                    else:
                        xe.append(np.array([x, x]))
                        ye.append(np.array([
                            - (span[0] / 2 - x) * math.sqrt(3),
                            (span[0] / 2 - x) * math.sqrt(3),
                            ]))
                coarse_grids['plotting']['wire_ends_y']['x'] = xe
                coarse_grids['plotting']['wire_ends_y']['y'] = ye

            #   Sampling time and pitch - equal to gap
            coarse_grids['sampling_pitch'] = self.coarse_grids['gap']
            coarse_grids['sampling_time'] = \
                coarse_grids['sampling_pitch'] \
                    / self.charge_drift['velocity']

            #   TODO: Move this elsehwere
            #   Capacitances is approximate per length for
            #   plane of grid wires between grounded plates,
            #   following Erskine.  See
            #   Van Esch criticism that this
            #   isn't correct for noise estimates.
            coarse_grids['capacitance'] = (
                2 * math.pi * eps['lar'] * eps['not']
                / (math.pi
                   * self.coarse_grids['gap']
                / self.coarse_grids['pitch']
                - math.log(2 * math.pi
                   * self.coarse_grids['wire_radius']
                / self.coarse_grids['pitch']))
                * self.cells['width_xi']
                )

        #%%   Anode grid
        if hasattr(self, 'anode_grid'):

            #   Convenient variable
            pitch = self.anode_grid['pitch']

            #   Save pitch as coarse_pitch
            self.coarse_pitch = pitch

            #   Wires orienteted along y, so span based on cell x width
            span = self.cells['width_xi']

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
        if self.charge_readout=='GAMPixG':

            self.chip_array = {}

            #   Sensor type
            self.chip_array['type'] = 'chip_array'

            #   Array pitch is same as coarse grids
            self.chip_array['pitch'] = self.coarse_grids['pitch']

            #   Array pitch is same as coarse grids
            self.chip_array['sampling_pitch'] = self.coarse_grids['gap']
            self.chip_array['sampling_time'] = \
                self.chip_array['sampling_pitch'] \
                    / self.charge_drift['velocity']

            #   Rectangular cells are fairly simple
            if self.cell_geometry=='rectangular':

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
            elif self.cell_geometry=='hexagonal':

                #   convenient variables
                w =  self.chip_array['pitch']
                num_rows = math.floor(self.cells['width_yi'] / w)
                slope = 1 / math.sqrt(3)
                xp = self.cells['width_yi'] * slope

                #   ys is y location of tile centers, nx is the number
                #   of tiles in each row of ys
                #   Cacluation is different for num_rows odd and even

                #   Odd number of rows
                if num_rows % 2:
                    m = (num_rows-1) / 2
                    ys = np.arange(0, m+1) * w
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

            #   Loop over y, x
            for n in range(2):

                #   Span is cell width.
                if n==0: span = self.cells['width_xi']
                else: span = self.cells['width_yi']

                #   centers of tile edges and centers
                num_edges = math.floor(span / pitch)
                edges = np.arange(-num_edges / 2, num_edges / 2 + 1) * pitch
                centers = edges[0:-1] + pitch / 2

                #   Assign
                self.coarse_tiles['centers'].append(centers)
                self.coarse_tiles['edges'].append(edges)

            #   Sampling time and pitch, equal to gap .  But don't create
            #   for LArPix since there this sensor is virtual
            if not self.charge_readout=='LArPix':
                self.coarse_tiles['sampling_pitch'] \
                    = self.coarse_tiles['gap']
                self.coarse_tiles['sampling_time'] \
                    = self.coarse_tiles['sampling_pitch'] \
                        / self.charge_drift['velocity']

        #%%   Pixels - this smallest read out unit, nominally a chip
        if hasattr(self, 'pixels'):

            #   Convenient variable.  Note we assume that pitch is equal in
            #   both directions
            pitch = self.pixels['pitch']

            #   Span depends on architecture
            if self.charge_readout=='GAMPixG':
                span = self.chip_array['pitch']
            elif self.charge_readout=='GAMPixD' \
                or self.charge_readout=='LArPix':
                span = self.coarse_tiles['pitch']

            #   Edges and centers
            self.pixels['centers'] = []
            self.pixels['edges'] = []
            num_edges = math.floor(span / pitch)
            for n in range(2):
                self.pixels['edges'].append(
                    np.arange(-num_edges / 2, num_edges / 2 + 1) * pitch)
                self.pixels['centers'].append(
                    self.pixels['edges'][n][0:-1] + pitch / 2)

            #   Focus factor given that pixels may not fill span.
            focus_factor = (
                self.pixels['edges'][0][-1]
                - self.pixels['edges'][0][0]
                ) / span
            self.pixels['focus_factor'] = focus_factor

            #   Sampling time and pitch - equal to lateral pitch
            self.pixels['sampling_pitch'] = self.pixels['pitch']
            self.pixels['sampling_time'] = \
                self.pixels['sampling_pitch'] \
                    / self.charge_drift['velocity']

    def apply_study_case(self, study, case):
        """ Apply case of study to inputs, then recalculate params"""

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

def define_eps():
    """ inputs for capacitance calculation.  (In principle at
    least the LAr and LXe values should derive from
    material definition detector
    defition """

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
