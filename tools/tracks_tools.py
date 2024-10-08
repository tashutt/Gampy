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
        import sims_tools
        import readout_tools

        #   Read track
        raw_track, truth, meta = load_track(track_file_name)

        #   Assign everything in track
        self.raw_track = raw_track
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
        import sims_tools
        import readout_tools

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
        if hasattr(self, 'drifted_track'):
            delattr(self, 'drifted_track')

    def compress(self, compression_bin_size=200e-6):
        """  Compresses raw_track using iterative hierarchical 3d binning,
        to a scale set by compression_bin_size """

        #   Do not compress if already more compressed
        if ('compression_bin_size' in self.meta) \
            and (self.meta['compression_bin_size']>=compression_bin_size):
                print('Warning: requested compression does not exceed'
                      + ' existing, no further compression applied.')

        else:

            #   Compress
            r_out, num_e_out = compress_track(
                self.raw_track['r'],
                self.raw_track['num_e'],
                compression_bin_size=compression_bin_size
                )
            self.raw_track['r'] = r_out
            self.raw_track['num_e'] = num_e_out

            #   Several "meta" things are no longer valid
            self.raw_track.pop('particle', None)
            self.raw_track.pop('generation', None)
            self.raw_track.pop('parent', None)
            self.raw_track.pop('interaction', None)
            self.raw_track.pop('birth_step', None)

            #   Save compression size in meta data
            self.meta['compression_bin_size'] = compression_bin_size

    def apply_drift(self, depth=0, decompress=True):
        """
        Drifts track, finding charge loss to electronegative capture, and
        adds diffusion

        depth: The z value of each entry in the track is assumed negative,
            and the drift distance for each is (z - depth).
        decompress - if true, expands clumpled electrons (expanding r, num_e)
            to properly treat diffusion and threshold.

        creates track.drifted_track, with fields
            r - same as raw_track, but note can have fewer entries due to
                charge loss
            num_e - number of charges, after charge loss
            depth - record of input
        """

        import numpy as np
        import sys

        import charge_drift_tools

        #   Recalculate params
        self.read_params.calculate()

        drift_properties = charge_drift_tools.properties(
            self.read_params.charge_drift['drift_field'],
            self.read_params.material
            )

        #   Decompress track so num_e is smaller than readout noise

        #   Find noise for the fine grained readout
        if (self.read_params.charge_readout_name=='GAMPixG'
            or self.read_params.charge_readout_name=='GAMPixD'
            or self.read_params.charge_readout_name=='LArPix'):
            noise = self.read_params.pixels['noise']
        elif self.read_params.charge_readout_name=='AnodeGridD':
            noise = self.read_params.anode_grid['noise']

        #   This is maximum size of deompressed num_e
        max_num_e = noise / 2.0

        if decompress:

            #   De-compress track so num_e is smaller than noise by some factor
            #   TODO: handle general case of different keys in raw_track
            in_args = [self.raw_track['r']]
            if 'track_id' in self.raw_track:
                in_args.append(self.raw_track['track_id'])
                in_args.append(self.raw_track['particle_id'])

            num_e, out_data = decompress_track(
                max_num_e,
                self.raw_track['num_e'],
                in_args,
                )

            #   r in out_data
            r = out_data[0]

        else:
            num_e = self.raw_track['num_e']
            r = self.raw_track['r']

        #   Drift distance to anode, a positive quanity.  If depth supplied,
        #   add it.
        drift_distance = self.sims_params.cells['z_anode'] - r[2, :] + depth

        #   Mask for valid drift distance
        drift_mask = drift_distance>0
        self.truth['num_electrons_inbounds'] = num_e[drift_mask].sum()

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
        drifted_track = {}
        drifted_track['num_e'] = num_e[mask]
        if 'track_id' in self.raw_track:
            drifted_track['track_id'] = out_data[1][mask]
            drifted_track['particle_id'] = out_data[2][mask]

        #   Add diffusion to r
        new_num = mask.sum()
        drifted_track['r'] = np.zeros((3, new_num), dtype=float)
        drifted_track['r'][0, :] = \
            r[0, mask] \
            + np.random.randn(new_num) \
            * sigma['transverse']
        drifted_track['r'][1, :] = \
            r[1, mask] \
            + np.random.randn(new_num) \
            * sigma['transverse']
        drifted_track['r'][2, :] = \
            r[2, mask] \
            + np.random.randn(new_num) \
            * sigma['longitudinal']

        #   Keep depth
        drifted_track['depth'] = depth

        #   Assign to track
        self.drifted_track = drifted_track

    def readout_charge(self, depth=0, stats_output=False, voxel_output=False):
        """
        Updates params, drifts track, and reads track out with
            configured charge readout

        depth: An added distance to that between each track element and
            the aonde, which is defined per cell.

        Output is added to track, and depends on charge readout.  See
            help in charge_readout_tools for decoumentation

        TODO:  Add coarse readout based triggering GAMPixG and GAMPixD

        4/23   TS  consolidates previously separate methods
        """

        import sys

        import charge_readout_tools

        #   Recalculate params
        self.read_params.calculate()

        #   Apply drift
        if depth<0:
            sys.exit('ERROR: depth must be >= 0')
        self.apply_drift(depth=depth)

        #   GAMPix for GammaTPC
        if self.read_params.charge_readout_name=='GAMPixG':

            #   Readout coarse grids
            self.coarse_grids_samples \
                = charge_readout_tools.readout_coarse_grids(
                    self.drifted_track['r'],
                    self.drifted_track['num_e'],
                    self.read_params.coarse_grids,
                    stats_output=stats_output,
                    )

            #   Readout pixels - as dual scale pixels
            self.pixel_samples \
                = charge_readout_tools.readout_dual_scale_pixels(
                    self.drifted_track['r'],
                    self.drifted_track['num_e'],
                    self.read_params.chip_array,
                    self.read_params.pixels,
                    stats_output=stats_output,
                    )

        #   GAMPix for DUNE
        elif self.read_params.charge_readout_name=='GAMPixD':

            #   Readout coarse tiles - as pixels
            self.coarse_tiles_samples \
                = charge_readout_tools.readout_pixels(
                    self.drifted_track['r'],
                    self.drifted_track['num_e'],
                    self.read_params.coarse_tiles,
                    stats_output=stats_output,
                    )

            #   Readout pixels - as dual scale pixels
            self.pixel_samples \
                = charge_readout_tools.readout_dual_scale_pixels(
                    self.drifted_track['r'],
                    self.drifted_track['num_e'],
                    self.read_params.coarse_tiles,
                    self.read_params.pixels,
                    stats_output=stats_output,
                    )

        #   LArPix
        elif self.read_params.charge_readout_name=='LArPix':

            #   Readout pixels
            self.pixel_samples \
                = charge_readout_tools.readout_pixels(
                    self.drifted_track['r'],
                    self.drifted_track['num_e'],
                    self.read_params.pixels,
                    stats_output=stats_output,
                    )

        #   Anode grid
        elif self.read_params.charge_readout_name=='AnodeGridD':

            self.anode_grid_samples \
                = charge_readout_tools.readout_anode_grid(
                    self.drifted_track['r'],
                    self.drifted_track['num_e'],
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

def save_track(full_file_name, track):
    """
    Saves track to npz and pickle files with full_file_name.

    Here track is either a dictionary containing a raw penelope track
    (or track from G4 or some other source), or the standard track object.

    TODO: This is all a bit ugly, and should be revisited.  Probably:
        + Move penelop parts moved to penelope tools.
        + This becomes method; only current use case is following compression.
    TODO: save/read "raw" penelope tracks in normal format, with raw_track
    """

    import pickle
    import os
    import numpy as np

    #   A "Penelope" track is a dictionary.  Save
    if isinstance(track, dict):
        r = track['r']
        num_e = track['num_e']
        if 'generation' in track:
            particle = track['particle']
            generation = track['generation']
            parent = track['parent']
            interaction = track['interaction']
            birth_step = track['birth_step']
        truth = track['truth']
        meta = track['meta']

    #   Otherwise assume normal track
    else:
        r = track.raw_track['r']
        num_e = track.raw_track['num_e']
        if 'generation' in track.raw_track:
            particle = track.raw_track['particle']
            generation = track.raw_track['generation']
            parent = track.raw_track['parent']
            interaction = track.raw_track['interaction']
            birth_step = track.raw_track['birth_step']
        truth = track.truth
        meta = track.meta

    #   r, ne and generation saved as npz file
    if 'generation' in locals():
        np.savez_compressed(
            os.path.join(full_file_name + '.npz'),
            r = r,
            num_e = num_e,
            particle = particle,
            generation = generation,
            parent = parent,
            interaction = interaction,
            birth_step = birth_step,
            )
    else:
        np.savez_compressed(
            os.path.join(full_file_name + '.npz'),
            r = r,
            num_e = num_e,
            )

    #   truth and meta data saved as pickle
    track_info = {}
    track_info['truth'] = truth
    #   Backwards compatability
    #   TODO: remove
    if not 'file_name' in meta:
        meta['file_name'] = full_file_name.split(os.path.sep)[-1]
    track_info['meta'] = meta
    with open(os.path.join(full_file_name + '.pickle'), 'wb') as f:
        pickle.dump(track_info, f)

def load_track(full_file_name):
    """
    Loads .npz + .pickel track in full_file_name

    TODO: save/read "raw" penelope tracks in normal format, with raw_track
    11/5/21 TS
    """

    import pickle, os
    import numpy as np

    #   load track guts from .npz: r, num_e, and possibly generation
    track_guts = np.load(os.path.join(full_file_name + '.npz'))

    #   load track_info from .pickle
    with open(os.path.join(full_file_name + '.pickle'), 'rb') as f:
        track_info = pickle.load(f)

    raw_track = {}
    raw_track['r'] = track_guts['r']
    raw_track['num_e'] = track_guts['num_e']
    if 'generation' in track_guts:
        raw_track['generation'] = track_guts['generation']
    if 'particle' in track_guts:
        raw_track['particle'] = track_guts['particle']
        raw_track['parent'] = track_guts['parent']
        raw_track['interaction'] = track_guts['interaction']
        raw_track['birth_step'] = track_guts['birth_step']

    truth = track_info['truth']
    #   Backwards compatability
    #   TODO: remove this
    if 'meta_data' in track_info:
        meta = track_info['meta_data']
    else:
        meta = track_info['meta']
    #   Backwards compatability
    if not 'file_name' in meta:
        meta['file_name'] \
            = full_file_name.split(os.path.sep)[-1]

    return raw_track, truth, meta

def get_cell_bounds(track, charge_readout_name, readout_inputs_file_name):
    """ Finds cell_bounds that contain raw track, buffering as
    needed to accomodate coarse sensors """

    import readout_tools

    #   Find bounding dimensions that contains track
    cell_bounds = find_bounding_box(track.raw_track['r'])

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

def compress_track(r, num_e, compression_bin_size=200e-6, voxel_cube=None,
                   first=True):
    """
    Recursive cubic binning track compression.

    Input:
        r - locations of charge, dimension [3, number_of_entries]
        num_e - charge at each location
        compression_bin_size - size of bin withing which r is
            averaged and num_e is summed
        voxel_cube - used internally
        first - used internally

    Returned:
        r_out - charge-averaged value of r within compression_bin_size cubes
        num_e - summed charge in compression_bin_size cubes

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
    if this_bin_size < compression_bin_size:
        this_bin_size = compression_bin_size
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

        r_q_mean, _, _ = stats.binned_statistic_dd(
            r.transpose(),
            [r[0, :] * num_e, r[1, :] * num_e, r[2, :] * num_e],
            statistic='mean',
            bins=bin_edges,
            )

        r_out = np.zeros((3, num_voxels), dtype=float)
        num_e_out = np.zeros(num_voxels, dtype=float)
        for voxel, n in zip(occupied_voxels, range(num_voxels)):

            in_voxel = np.all(voxel[:, None]==indices-1, axis=0)

            num_e_out[n] = num_e[in_voxel].sum()

            #   This requires finesse: we found the mean of r * num_e,
            #   which assumed weight = 1, thus to normalize the voxel
            #   we not only divide by charges in voxel, but need to multiply
            #   by counts in voxel to remove assumed weight = 1.
            r_out[:, n] = r_q_mean[:, voxel[0], voxel[1], voxel[2]] \
                / num_e_out[n] * in_voxel.sum()

        return r_out, num_e_out

    #   If not final step, then recursively proceed over voxels
    for voxel, nv in zip(occupied_voxels, range(num_voxels)):

        #   These elements are in this voxel
        in_voxel_mask = np.all(voxel[:, None]==indices-1, axis=0)

        #   Bounding box for this voxel
        voxel_cube = np.array(
            [bin_edges[ns][voxel[ns]:voxel[ns]+2] for ns in range(3)]
            )

        #   Recursively bin
        r_voxel, num_e_voxel \
            = compress_track(
                r[:, in_voxel_mask],
                num_e[in_voxel_mask],
                voxel_cube = voxel_cube,
                first = False,
                compression_bin_size=compression_bin_size,
                )

        #   Append output, with different first step
        if nv==0:
            r_out = r_voxel
            num_e_out = num_e_voxel
        else:
            r_out = np.append(r_out, r_voxel, axis=1)
            num_e_out = np.append(num_e_out, num_e_voxel)

    return r_out, num_e_out

def decompress_track(max_electrons_per_bin, num_electrons, args):
    """
    Take in sample_data (shape k,n with n the number of samples
    and k the number of datatypes, e.g. r, trackID, pID) and n_electrons
    (shape n) and splits these up into subsamples such that each
    subsample has less than max_electrons_per_bin electrons

    7/23   H. Purcell
    """

    import numpy as np
    import math

    args = [np.transpose(arg) for arg in args]

    #check all input samples have at least one electron
    num_electrons = np.array(num_electrons)
    assert np.all(num_electrons > 0), \
        "ValueError: all input samples must have at least one electron"

    #  find the number of subsamples needed for each sample, and in total
    num_bins = np.array(
        [math.ceil(n_elec/max_electrons_per_bin) for n_elec in num_electrons]
        )
    num_bins_total = np.sum(num_bins)

    #  compute the number of electrons per subsample in each sample,
    #  and the remainder
    num_electrons_per_bin       = np.array(num_electrons) // num_bins
    remainder_electrons_per_bin = np.array(num_electrons) %  num_bins

    #   outputs
    output_arrays = []

    #  output electron array
    output_num_electrons = np.zeros(num_bins_total, dtype = int)

    # Loop over each input array
    for arg in args:
        arg = np.array(arg)
        # Initialize output array for this input array
        output_array = np.zeros(
            (num_bins_total,) + arg.shape[1:],
            dtype=arg.dtype
            )

        #loop through samples
        for n in range(num_bins.size):

            #   bins alread filled in previous samples
            bins_filled = np.sum(num_bins[:n])

            #   for sample n, fill num_bins[n] subsamples in the output array
            output_array[bins_filled: bins_filled + num_bins[n]] = arg[n]
            output_num_electrons[bins_filled: bins_filled + num_bins[n]] \
                = num_electrons_per_bin[n]

            #   for the first remainder_electrons_per_bin[n] subsamples,
            #   add one electron to the output electron count to spread
            #   out the remainder evenly
            output_num_electrons[
                bins_filled: bins_filled + remainder_electrons_per_bin[n]
                ] += 1

        #  transpose again
        output_array = np.transpose(output_array)

        #  append to output list
        output_arrays.append(output_array)

    #output the subsampled data
    return output_num_electrons, output_arrays

