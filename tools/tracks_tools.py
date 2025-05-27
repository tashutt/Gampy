"""
Created on Mon Aug  3 19:22:53 2020

Tracks object and associated routines.

TODO: Move charge_drift_tools into readout_tools

@author: tshutt
"""

class Tracks():
    """
    Tracks object for charge tracks for a single event, includes readout
    and display, with associated routines.
    """

    def __init__(self, track_file_name, charge_readout_name='GAMPixG',
                 readout_inputs_file_name='default'):
        """
        Start by reading raw track file, and define parameters
        with a specified charge readout in a simple cell.

        readout_inputs_file_name help: readout_tools
        """
        from . import sims_tools
        from . import readout_tools

        #   Read track
        raw, truth, meta  = load_track(track_file_name)

        #   Assign everything in track
        self.raw = raw
        self.truth = truth
        self.meta = meta

        #   Get cell bounds
        cell_bounds = get_cell_bounds(self, charge_readout_name,
                                      readout_inputs_file_name)

        #   Load simple cell geometry parameters
        self.sims_params = sims_tools.Params(
            inputs_source='simple_cell',
            cell_bounds=cell_bounds,
            )

        #   Load readout parameters
        self.read_params = readout_tools.Params(
            charge_readout_name=charge_readout_name,
            cells=self.sims_params.cells,
            )

    def reset_params(self, charge_readout_name='GAMPixG',
                     readout_inputs_file_name='default'):
        """ reset params, which allows change of charge readout,
        removes any samples that have already been generated """
        from . import sims_tools
        from . import readout_tools

        #   Get cell bounds
        cell_bounds = get_cell_bounds(self, charge_readout_name,
                                      readout_inputs_file_name)

        #   Load simple cell geometry parameters
        self.sims_params = sims_tools.Params(
            inputs_source='simple_cell',
            cell_bounds=cell_bounds,
            )

        #   Load readout parameters
        self.read_params = readout_tools.Params(
            charge_readout_name=charge_readout_name,
            cells=self.sims_params.cells,
            )

        #   Strip out any previous samples, and drifted track
        stale_keys = []
        for key in self.__dict__.keys():
            if key.find('samples')>0:
                stale_keys.append(key)
        for key in stale_keys:
            delattr(self, key)
        if hasattr(self, 'drifted'):
            delattr(self, 'drifted')

    def compress(self, scale=200e-6):
        """  Compresses raw to compressed_track, by hierarchical
        3d binning, to a bin size algorithmically determined based on scale.
        The bin size used is save in self.truth"""

        self.raw['r'], self.raw['num_e'], bin_size = compress_track(
            self.raw['r'],
            self.raw['num_e'],
            scale=scale
            )

        self.truth['compressed_bin_size'] = bin_size

    def apply_drift(self, depth=0, decompress=True):
        """
        Drifts track, finding charge loss to electronegative capture, and
        adds diffusion.  Only z value of each entry which are negative are
            read, unless 'depth' is supplied.

        depth: drift distance for each element with z is (z - depth).

        decompress - if True, then decompresses track to accurately treat
            loss to diffusion, based on sensor noise

        creates track.drifted, with fields
            r - same as raw, but note can have fewer entries due to
                charge loss
            num_e - number of charges, after charge loss
            depth - record of input
        """

        import numpy as np
        import sys
        import copy

        from . import charge_drift_tools

        #   Recalculate params, and diffusion constants
        self.read_params.calculate()
        drift_properties = charge_drift_tools.properties(
            self.read_params.charge_drift['drift_field'],
            self.read_params.material
            )

        #   Make a copy of charge and position
        num_e  = copy.copy(self.raw['num_e'])
        r  = copy.copy(self.raw['r'])

        #   Decompress track so num_e is smaller than noise by some factor
        if decompress:

            #   Find noise for the fine grained readout
            if (self.read_params.charge_readout_name=='GAMPixG'
                or self.read_params.charge_readout_name=='GAMPixD'
                or self.read_params.charge_readout_name=='LArPix'):
                noise = self.read_params.pixels['noise']
            elif self.read_params.charge_readout_name=='AnodeGridD':
                noise = self.read_params.anode_grid['noise']

            #   This is maximum size of decompressed num_e
            max_num_e = noise / 2.0

            num_e, r = decompress_track(
                max_num_e,
                num_e,
                r,
                )

        #   Drift distance to anode, a positive quanity.  If depth supplied,
        #   add it.
        drift_distance = self.sims_params.cells['z_anode'] - r[2, :] + depth

        #   Mask for valid drift distance
        drift_mask = drift_distance>0
        self.truth['num_e_inbounds'] = num_e[drift_mask].sum()

        #   If none with positve drift distance, barf out.
        if drift_mask.sum()==0:
            sys.exit('All electrons have negative drift distances')

        #    Survival fraction to trapping.
        survive = np.exp(-drift_distance
                      / self.read_params.charge_drift['drift_length'])
        survive[~drift_mask] = 0
        num_e = np.random.binomial(num_e, survive)
        survival_mask = num_e>0

        #   Combine drift and survival masks
        mask = drift_mask & survival_mask

        #   Find dispersion in both directions
        sigma = charge_drift_tools.get_sigma(
            drift_distance[mask],
            drift_properties
            )

        #   Initialize drifted track, assign surviving charge and ids
        drifted = {}
        drifted['num_e'] = num_e[mask]

        #   Add diffusion to r
        new_num = mask.sum()
        drifted['r'] = np.zeros((3, new_num), dtype=float)
        drifted['r'][0, :] = \
            r[0, mask] \
            + np.random.randn(new_num) \
            * sigma['transverse']
        drifted['r'][1, :] = \
            r[1, mask] \
            + np.random.randn(new_num) \
            * sigma['transverse']
        drifted['r'][2, :] = \
            r[2, mask] \
            + np.random.randn(new_num) \
            * sigma['longitudinal']

        #   Keep depth
        drifted['depth'] = depth

        #   Assign to track
        self.drifted = drifted

    def readout_charge(self, depth=0, stats_output=False):
        """
        Updates params, drifts track, and reads track out with
            configured charge readout

        depth: An added distance to that between each track element and
            the aonde, which is defined per cell.
        stats_output - compute additional statistics

        Output is added to track, and depends on charge readout.  See
            help in charge_readout_tools for decoumentation

        TODO:  Add coarse readout based triggering GAMPixG and GAMPixD

        4/23   TS  consolidates previously separate methods
        """

        import sys

        from . import charge_readout_tools

        #   Recalculate params
        self.read_params.calculate()

        #   Apply drift
        self.apply_drift(depth=depth)

        #   GAMPix for GammaTPC
        if self.read_params.charge_readout_name=='GAMPixG':

            #   Readout coarse grids
            self.coarse_grids_samples \
                = charge_readout_tools.readout_coarse_grids(
                    self.drifted['r'],
                    self.drifted['num_e'],
                    self.read_params.coarse_grids,
                    stats_output=stats_output,
                    )

            #   Readout pixels - as dual scale pixels
            self.pixel_samples \
                = charge_readout_tools.readout_dual_scale_pixels(
                    self.drifted['r'],
                    self.drifted['num_e'],
                    self.read_params.chip_array,
                    self.read_params.pixels,
                    stats_output=stats_output,
                    )

        #   GAMPix for DUNE
        elif self.read_params.charge_readout_name=='GAMPixD':

            #   Readout coarse tiles - as pixels
            self.coarse_tiles_samples \
                = charge_readout_tools.readout_pixels(
                    self.drifted['r'],
                    self.drifted['num_e'],
                    self.read_params.coarse_tiles,
                    stats_output=stats_output,
                    )

            #   Readout pixels - as dual scale pixels
            self.pixel_samples \
                = charge_readout_tools.readout_dual_scale_pixels(
                    self.drifted['r'],
                    self.drifted['num_e'],
                    self.read_params.coarse_tiles,
                    self.read_params.pixels,
                    stats_output=stats_output,
                    )

        #   LArPix
        elif self.read_params.charge_readout_name=='LArPix':

            #   Readout pixels
            self.pixel_samples \
                = charge_readout_tools.readout_pixels(
                    self.drifted['r'],
                    self.drifted['num_e'],
                    self.read_params.pixels,
                    stats_output=stats_output,
                    )

        #   Anode grid
        elif self.read_params.charge_readout_name=='AnodeGridD':

            self.anode_grid_samples \
                = charge_readout_tools.readout_anode_grid(
                    self.drifted['r'],
                    self.drifted['num_e'],
                    self.read_params.anode_grid,
                    stats_output=stats_output,
                    )

        else:
            sys.exit('ERROR in tracks_tools: ' +
                     'charge readout architecture not recognized')

    def display(self, **kwargs):
        """
        Displays track - see help in display_tools for arguments

        returns: fig, ax, plot_lims
        """
        import display_tools

        fig, ax, plot_lims = display_tools.display_track(self, **kwargs)

        return fig, ax, plot_lims

def save_track(full_file_name, track, write_raw=True):
    """
    Saves tracks to npz and pickle files with full_file_name.
    """

    import pickle
    import os
    import numpy as np

    #   Save raw
    np.savez_compressed(
        os.path.join(full_file_name + '.npz'),
        r = track.raw['r'],
        num_e = track.raw['num_e'],
        )

    #   truth and meta data saved as pickle
    track_info = {}
    track_info['truth'] = track.truth
    track_info['meta'] = track.meta
    with open(os.path.join(full_file_name + '.pickle'), 'wb') as f:
        pickle.dump(track_info, f)

def load_track(full_file_name, read_raw=True):
    """
    Loads .npz and/or c.npz tracks, and.pickle

    full_file_name - path + file name, not including extensions

    returns raw, truth and meta

    """

    import os
    import pickle
    import numpy as np

    #   If file ends with .c, strip this off
    if '.c' in full_file_name:
        full_file_name = full_file_name.split('.c')[0]

    #   Read raw track

    if os.path.isfile(os.path.join(full_file_name + '.npz')):

        #   Load
        guts = np.load(os.path.join(full_file_name + '.npz'))

        #   Construct track
        raw = {}
        raw['r'] = guts['r']
        raw['num_e'] = guts['num_e']

    #   load track_info from .pickle
    with open(os.path.join(full_file_name + '.pickle'), 'rb') as f:
        track_info = pickle.load(f)
    truth = track_info['truth']
    meta = track_info['meta']

    return raw, truth, meta

def get_cell_bounds(track, charge_readout_name, readout_inputs_file_name):
    """ Finds cell_bounds that contain raw track, buffering as
    needed to accomodate coarse sensors """

    import readout_tools

    #   Find bounding dimensions that contains track
    cell_bounds = find_bounding_box(track.raw['r'])

    #   Buffer size to allow minimal coarse sensors
    coarse_pitch = readout_tools.Params(
        inputs_file_name = readout_inputs_file_name,
        charge_readout_name=charge_readout_name,
        ).coarse_pitch
    cell_bounds[:, 0] -= coarse_pitch
    cell_bounds[:, 1] += coarse_pitch

    return cell_bounds

def find_bounding_box(r, buffer=0.0):
    """
    Finds box that spans r, with an added buffer
    """

    import numpy as np

    #   First set equal to extremes of r
    bounding_box = np.zeros((3,2))
    bounding_box[:, 0] = r.min(axis=1)
    bounding_box[:, 1] = r.max(axis=1)

    #   Add buffer
    bounding_box[:, 0] -= buffer
    bounding_box[:, 1] += buffer

    return bounding_box

def find_bounding_cube(r, buffer=0.0):
    """
    Finds cube that spans r, with an added buffer.
    """

    import numpy as np

    #   Start with bounding box.
    bounding_box = find_bounding_box(r, buffer=buffer)

    #   Now make all spans equal to largest
    bounding_cube = bounding_box.mean(axis=1)[:, None] \
        +  0.5 * np.diff(bounding_box).max() * np.array([-1., 1.])

    return bounding_cube

def compress_track(r, num_e, scale=200e-6, voxel_cube=None,
                   first=True):
    """
    Recursive cubic binning track compression, to bin size set, roughly
        by  scale

    Input:
        r - locations of charge, dimension [3, number_of_entries]
        num_e - charge at each location
        scale - size of bin withing which r is
            averaged and num_e is summed
        voxel_cube - used internally
        first - used internally

    Returned:
        bin_size - the size of bins
        r_out - charge-averaged value of r within bins_size cubes
        num_e - summed charge in scale cubes

    4/9/23 - TS
    """

    import numpy as np
    from scipy import stats
    import sys

    #   Binning per iteration - optimum must depends on track structure,
    #   possibly in an energy dependent way.  A value around 10
    #   is good at 1 GeV.
    max_num_bins = 10

    #   Must have 2 or more bins
    if max_num_bins<2:
        sys.exit('Error: max_num_bins < 2')

    #   First time, find enclosing cube
    if first:
        voxel_cube = find_bounding_cube(r)

    #   Check if final bin size is reached
    final_step = False
    this_bin_size = np.diff(voxel_cube).max() / max_num_bins
    if this_bin_size < scale:
        this_bin_size = scale
        final_step = True

    #   Bin edges based on bin size and bounding box
    num_bins = np.ceil(np.diff(voxel_cube) / this_bin_size)
    bin_edges = [
        voxel_cube[n, 0] + np.arange(0, num_bins[n]+1) * this_bin_size
        for n in range(3)
        ]

    #   Find voxel indices for each element of r (and num_e)
    indices = np.zeros_like(r, dtype=int)
    indices[0, :] = np.digitize(
        r[0, :],
        bins=bin_edges[0],
        )
    indices[1, :] = np.digitize(
        r[1, :],
        bins=bin_edges[1]
        )
    indices[2, :] = np.digitize(
        r[2, :],
        bins=bin_edges[2]
        )

    #   Need to hitogram to find indices of occupied voxels
    counts, _ = np.histogramdd(
        r.T,
        bins=bin_edges,
        )
    occupied_voxels = np.argwhere(counts>0)

    num_voxels = occupied_voxels.shape[0]

    #   If at final level of iteration, find weighted location within each
    #   bin, and charge at that point
    if final_step:

        r_q_mean = stats.binned_statistic_dd(
            r.transpose(),
            [r[0, :] * num_e, r[1, :] * num_e, r[2, :] * num_e],
            statistic='mean',
            bins=bin_edges,
            )[0]

        r_out = np.zeros((3, num_voxels), dtype=float)
        num_e_out = np.zeros(num_voxels, dtype=int)
        for voxel, n in zip(occupied_voxels, range(num_voxels)):

            in_voxel = np.all(voxel[:, None]==indices-1, axis=0)

            num_e_out[n] = num_e[in_voxel].sum()

            #   This requires finesse: we found the mean of r * num_e,
            #   which assumed weight = 1, thus to normalize the voxel
            #   we not only divide by charges in voxel, but need to multiply
            #   by counts in voxel to remove assumed weight = 1.
            r_out[:, n] = r_q_mean[:, voxel[0], voxel[1], voxel[2]] \
                / num_e_out[n] * in_voxel.sum()

        return r_out, num_e_out, this_bin_size

    #   If not final step, then recursively proceed over voxels
    for voxel, nv in zip(occupied_voxels, range(num_voxels)):

        #   These elements are in this voxel
        in_voxel_mask = np.all(voxel[:, None]==indices-1, axis=0)

        #   Bounding box for this voxel
        voxel_cube = np.array(
            [bin_edges[ns][voxel[ns]:voxel[ns]+2] for ns in range(3)]
            )

        #   Recursively bin
        r_voxel, num_e_voxel, this_bin_size \
            = compress_track(
                r[:, in_voxel_mask],
                num_e[in_voxel_mask],
                voxel_cube = voxel_cube,
                first = False,
                scale=scale,
                )

        #   Append output, with different first step
        if nv==0:
            r_out = r_voxel
            num_e_out = num_e_voxel
        else:
            r_out = np.append(r_out, r_voxel, axis=1)
            num_e_out = np.append(num_e_out, num_e_voxel)

    return r_out, num_e_out, this_bin_size

def decompress_track(max_electrons_per_bin, num_e, r):
    """
    Take in sample_data (shape k,n with n the number of samples
    and k the number of datatypes, e.g. r, trackID, pID) and n_electrons
    (shape n) and splits these up into subsamples such that each
    subsample has less than max_electrons_per_bin electrons

    7/23   H. Purcell
    10/24 TS - simplify to work only with num_e and r
    """

    import numpy as np
    import math

    #check all input samples have at least one electron
    num_e = np.array(num_e)
    assert np.all(num_e > 0), \
        "ValueError: all input samples must have at least one electron"

    #  find the number of subsamples needed for each sample, and in total
    num_bins = np.array(
        [math.ceil(n_elec/max_electrons_per_bin) for n_elec in num_e]
        )
    num_bins_total = np.sum(num_bins)

    #  compute the number of electrons per subsample in each sample,
    #  and the remainder
    num_e_per_bin       = np.array(num_e) // num_bins
    remainder_electrons_per_bin = num_e % num_bins

    #  output electron array
    num_e_out = np.zeros(num_bins_total, dtype = int)
    r_out = np.zeros((3, num_bins_total), dtype = float)

    #loop through samples
    for n in range(num_bins.size):

        #   bins alread filled in previous samples
        bins_filled = np.sum(num_bins[:n])

        #   for sample n, fill num_bins[n] subsamples in the output array
        num_e_out[bins_filled : bins_filled + num_bins[n]] \
            = num_e_per_bin[n]
        r_out[:, bins_filled : bins_filled + num_bins[n]] = r[:, n][:, None]

        #   for the first remainder_electrons_per_bin[n] subsamples,
        #   add one electron to the output electron count to spread
        #   out the remainder evenly
        num_e_out[
            bins_filled: bins_filled + remainder_electrons_per_bin[n]
            ] += 1


    return num_e_out, r_out

