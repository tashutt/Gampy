#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Set of functions that work on "hits" structures - truth_hits or measured_hits.

Note that these are called by the methods in events_tools, but are also
can be used in a stand-alone way.

Port of routines from matlab, started Sat Jul 17, 2021
@author: tshutt
"""

def calculate_compton(edep, etot):
    """ Calculation of compton angle from total energy - etot, and
        Compton scatter energy - edep

        Returned:
            angle - energy calculated scattering angle
            success - true if energy calculation successful
    """

    import numpy as np

    #   Electron mass in keV
    me = 511.000

    success =  np.zeros(edep.shape, dtype=bool)
    angle =  np.zeros(edep.shape, dtype=float)

    energy_cut = (etot!=edep) & (etot!=0)
    success[energy_cut] = np.abs(1 + me * (1 / etot[energy_cut]
        - 1 / (etot[energy_cut] - edep[energy_cut]))) <= 1

    angle[success] = np.arccos(1 + me * (1 / etot[success]
        - 1 / (etot[success] - edep[success])))

    return angle, success

def find_theta(hits, hit_indices, input_cut=None):
    """
    Finds theta for hits, given order specified in hit_indices.
        Length of returned variables are not all full length of events

    There is no checking internally that the requested number of
        hits exist - this should be supplied in input_cut.

    hits - standard hits dictionary with (at least) fields:
        r, alive, energy.

    hit_indices - dict with indices of hits used in calcuations. Must have
        one of singlet, doublet or triplet.  Keys:
        singlet - single hit for which theta energy alone is
            calculated.  This can be an array (from order).
        prior - indices of all hits prior to that for which theta is
            calculated - used to exclude energy from energy calculation
            (prior[-1] must be equal to triplet[0])

    input_cut - optional - true for events to include

    theta  - returned.  Dictionary with keys:
        success - input cut + successful energy reconstruction
        energy  - angle, calculated from energy.  equal to zero when
            success is false

     To fix: add checks on interaction type?
     """

    import numpy as np

    if input_cut is None:
        cut = np.ones(hits['energy'][0, :].size, dtype=bool)
    else:
        cut = np.ndarray.copy(input_cut)

    #   Total energy
    etot = hits['total_energy'][cut]

    #   If prior hits, subtract that energy from total.  Syntax differs
    #   for number of prior hits greater or equal to one.  Single prior
    #   must be zero
    if 'prior' in hit_indices:
        if len(hit_indices['prior']) > 1:
            prior_indices = np.tile(
                hit_indices['prior'],
                (cut.size, 1)
                ).transpose()
            etot = etot \
                - np.sum(
                    hits['energy'][prior_indices,
                                   np.arange(0, cut.size)][:, cut],
                    axis=0)
        else:
            etot = etot - hits['energy'][hit_indices['prior'], cut]

    #   Compton scatter energy
    if isinstance(hit_indices['singlet'], np.ndarray) and \
        len(hit_indices['singlet'])>1:
        edep = hits['energy'][hit_indices['singlet'][cut], cut]
    else:
        edep = hits['energy'][hit_indices['singlet'], cut]

    #   Create theta dictionary

    theta = {}

    theta['energy'] = np.zeros(cut.shape)
    theta['success'] = np.zeros(cut.shape, dtype=bool)

    #   Now calcuate
    theta['energy'][cut], theta['success'][cut]  \
        = calculate_compton(edep, etot)

    return theta

def find_triplet_theta(hits, hit_indices, input_cut=None):
    """
    Finds theta for hits, given ordering specified in hit_indices.
        Length of returned variables are not all full length of events

    There is no checking internally that the requested number of
        hits exist - this should be supplied in input_cut.

    hits - standard hits dictionary with (at least) fields:
        r, alive, energy.

    hit_indices - dict with indices of hits used in calcuations. Must have
        one of singlet, doublet or triplet.  Keys:
        triplet - list.  Sequence of 3 sequential hits, with theta of 2nd
            calculated both energetically and geometrically
        prior - indices of all hits prior to that for which theta is
            calculated - used to exclude energy from energy calculation
            (prior[-1] must be equal to triplet[0])

    input_cut - optional - true for events to include

    theta  - returned.  Dictionary with keys:
        success - combines input cut, and successful energy
            reconstruction. Size is all events.
        energy  - calculated from energy of compton scatter, and total
            energy of all scatters.  length = sum(theta['success'])
        geometry - calculated from geometry.  Not present for 2 hits
            unless initial_ray supplied.  length = sum(theta['success'])
        truth - calculated from s_primary and s_secondary, if they
            are present in hits (these are in events.truth_hits)

     To fix: add checks on interaction type?
     """

    import numpy as np
    from math_tools import dot

    if input_cut is None:
        cut = np.ones(hits['energy'][0, :].size, dtype=bool)
    else:
        cut = np.ndarray.copy(input_cut)

    theta = {}

    #%%  Geometry reconstruction

    #   For 3 hits, find both rays, and angle between
    ray1 = hits['r'][:, hit_indices['triplet'][1], cut] \
        - hits['r'][:, hit_indices['triplet'][0], cut]
    ray2 = hits['r'][:, hit_indices['triplet'][2], cut] \
        - hits['r'][:, hit_indices['triplet'][1], cut]
    theta['geometry'] = np.arccos(
        dot(ray1, ray2) \
            /  np.sqrt(dot(ray1, ray1) * dot(ray2, ray2))
        )

    #   If hits are truth_hits, then s_primary allows true theta to be
    #   calculated - do this
    if 's_primary' in hits:
        ray1 = hits['s_primary'] \
            [:, hit_indices['triplet'][0], cut]
        ray2 = hits['s_primary'] \
            [:, hit_indices['triplet'][1], cut]
        theta['truth'] = np.arccos(
            dot(ray1, ray2) \
                /  np.sqrt(dot(ray1, ray1) * dot(ray2, ray2))
            )

    #%%   Energy reconstruction

    #   Total energy
    etot = hits['total_energy'][cut]

    #   If prior hits, subtract that energy from total.  Syntax differs
    #   for number of prior hits greater or equal to one.  Single prior
    #   must be zero
    if 'prior' in hit_indices:
        if len(hit_indices['prior']) > 1:
            prior_indices = np.tile(
                hit_indices['prior'],
                (cut.size, 1)
                ).transpose()
            etot = etot \
                - np.sum(
                    hits['energy'][prior_indices,
                                   np.arange(0, cut.size)][:, cut],
                    axis=0)
        else:
            etot = etot - hits['energy'][hit_indices['prior'], cut]

    #   Energy deposited in compton scatter
    edep = hits['energy'][hit_indices['triplet'][1], cut]

    #   Now calcuate
    theta['energy'], theta['success'] = calculate_compton(edep, etot)

    #   Restrict geometry angles to those for which energy angle
    #   successfully calculated
    theta['energy'] = theta['energy'][theta['success']]
    theta['geometry'] = theta['geometry'][theta['success']]
    if 'truth' in theta:
        theta['truth'] = theta['truth'][theta['success']]

    #   theta['success'] is currently size of cut.  Redefine as full
    #   length of events
    cut[cut] = theta['success']
    theta['success'] = cut

    return theta

def calculate_ckd(hits, num_hits, input_cut=None):
    """
    hits - standard hits dictionary with (at least) fields:
        r, alive, energy.
    num_hits - ckd calculated only for events with num_hits
    input_cut - optional, events to include

    ckd - returned
        num_hits - same as input
        has_num_hits - true for events with exactly num_hits
        included - has_num_hits and supplied cut
        permutation - best premutation for each event
        value - value of best ckd sum for each event
        failed_theta_count - # of failed theta calcultions
        permutations - list of all permutations for num_hits
        Note:
            .has_num_hits - length is # of events
            .included - length is # of events
            .value - length is sum of cut and events with num_hits

    POSSIBLE TO DO: KEEP TRACK OF UNSUCCESSFUL THETA CALCULATIONS.  Check
        for no valid permutation found

    9/18 TS - first version
    11/19 - switch to sum test
    """
    from sympy.utilities.iterables import multiset_permutations
    import numpy as np
    from math import pi
    import math

    ckd = {}

    #   Select only events with num_hits
    ckd['num_hits'] = num_hits
    ckd['has_num_hits'] = hits['num_hits']==num_hits

    #   Add external cut, if any.  This is full cut used.
    if input_cut is None:
        cut = ckd['has_num_hits']
    else:
        cut = input_cut & ckd['has_num_hits']

    ckd['included'] = cut

    ckd['num_permutations'] = math.factorial(num_hits)
    ckd['permutations'] = np.zeros((num_hits, ckd['num_permutations']))
    ckd['permutation'] = np.zeros(np.sum(cut), dtype=int)

    min_sum = np.zeros(np.sum(cut))

    hit_indices = {}

    # hit_indices['num_hits'] = num_hits

    #   Loop over permutations
    failed_theta_count = np.zeros(np.sum(cut), dtype=int)
    p_num = -1
    for p_set in multiset_permutations(np.arange(0, num_hits)):

        p_num += 1
        ckd['permutations'][:, p_num] = p_set

        #   Loop over all sets of triple hits within a permutation
        this_sum = np.zeros(np.sum(cut))
        for nh in range(len(p_set) - 2):

            hit_indices['triplet'] = p_set[nh : nh+3]
            hit_indices['prior'] = p_set[0 : nh+1]

            theta = find_triplet_theta(hits, hit_indices, cut)

            #   This boolean is length of cuts, and is true for successful
            #   energy construction
            good_theta = theta['success'][cut]

            #   Difference in angles - size is sum(theta.success)
            delta = theta['geometry'] - theta['energy']

            #   Add to sums - failed construction is assigned angle of pi
            this_sum[good_theta] = this_sum[good_theta] + delta**2
            this_sum[~good_theta] = this_sum[~good_theta] + pi**2

            #   Keep trakck of failed theta
            failed_theta_count[~theta['success'][cut]] +=1

        this_sum = np.sqrt(this_sum)

        #   For all events, if this is minimum ckd, then asign this
        #   ordering
        if p_num==0:
            new_min_sum = np.ones(min_sum.size, dtype=bool)
        else:
            new_min_sum = this_sum < min_sum

        min_sum[new_min_sum] = this_sum[new_min_sum]
        ckd['permutation'][new_min_sum] = p_num

    # value is minimum sum
    ckd['value'] = min_sum

    ckd['num_permutations'] = len(ckd['permutations'])
    ckd['failed_theta_count'] = failed_theta_count

    return ckd

def calculate_order(hits, num_hits_list, cut=None):
    """
    Calculates and returns order, for all combinations in num_hits_list

    num_hits_list - list of number of hits to evaluate

    order - dict with fields
        has_num_hits - true for events that have this many hits
        tried - true for all has_num_hits and external cut
        ordered - tried and first two hits correct
        fully_ordered - tried and all hits correct
        disordered - tried, and found wrong order
        permutation - which permutation found, for each event
        ckd_value - value of ckd discriminant found
        initial_hits - indices of first two hits found
        num_hits_list - same as input
        permutations - list (length of num_hits_list) of all possible
            permutations

    Matlab version 2018 - 2020
    python port Aug 2021
    """

    import numpy as np
    import copy

    #   If cut not supplied, take everything
    if cut is None:
        cut = np.ones(hits['num_hits'].size, dtype=bool)

    #   Make num_hits_list a list if it is not
    if (not isinstance(num_hits_list, list)) \
        and (not isinstance(num_hits_list, tuple)):
        num_hits_list = [num_hits_list]

    #   Initialize arrays

    order = {}

    order['tried'] = np.zeros(hits['num_hits'].size, dtype=bool)
    order['ordered'] = np.zeros(hits['num_hits'].size, dtype=bool)
    order['fully_ordered'] = np.zeros(hits['num_hits'].size, dtype=bool)
    order['disordered'] = np.zeros(hits['num_hits'].size, dtype=bool)
    order['permutation'] = np.zeros(hits['num_hits'].size, dtype=int)
    order['ckd_value'] = np.zeros(hits['num_hits'].size)
    order['initial_hits'] = np.zeros((2, hits['num_hits'].size), dtype=int)
    order['has_num_hits'] \
        = np.zeros((len(num_hits_list), hits['num_hits'].size), dtype=bool)

    #   num_hits_list is just copy of input
    order['num_hits_list'] = copy.copy(num_hits_list)

    order['permutations'] = []

    #  Loop over num_hits
    nh = 0
    for num_hits in num_hits_list:

        #   Calculate ckd for this number of hits
        ckd = calculate_ckd(
            hits,
            num_hits,
            cut
            )

        #   Now assign things ot orering dictionary

        #   Events with this number of hits
        order['has_num_hits'][nh, :] = ckd['has_num_hits']

        #   Useful variable - number of hit and cut supplied to ckd
        this_set = ckd['included']

        #   These have order calculated - i.e., num_hits
        #   is in num_hits_list
        order['tried'][this_set] = True

        #   These are ordered: the first two hits are correct.  Note
        #   that the permutatations start with correct orders
        nl = ckd['num_permutations'] \
            / (num_hits_list[nh] * (num_hits_list[nh] - 1))
        order['ordered'][this_set] = ckd['permutation'] < nl

        #   These are fully ordered, a subset of ordered
        order['fully_ordered'][this_set] \
            = ckd['permutation']==0

        #   Value of ckd and permutatio for chosen permutation for all
        #   tried events
        order['ckd_value'][this_set] = ckd['value']
        order['permutation'][this_set] = ckd['permutation']

        #  Indices of first two hits
        order['initial_hits'][0, order['has_num_hits'][nh, :]] \
            = ckd['permutations'] \
                [0, order['permutation'][order['has_num_hits'][nh, :]]]
        order['initial_hits'][1, order['has_num_hits'][nh, :]] \
            = ckd['permutations'] \
                [1, order['permutation'][order['has_num_hits'][nh, :]]]

        #
        order['permutations'].append(ckd['permutations'])

        nh += 1

    #   Disoreded were calculalted (because have correct num hits),
    #   but are not ordered
    order['disordered'] = order['tried'] & ~order['ordered']

    return order

def construct_pointing_truth_order(hits, input_cut=None):
    """
    Finds theta and pointing ray from the first two vertices.  If no
        threshold has been applied, then this is always the truth ordering
        If a threshold has been applied, then the order will be wrong
        if either of these vertices was below threshold

    cut is optional

    Adds the following to hits:
        theta_truth_order - calculated from energy of compton scatter and
            total energy of all scatters
        theta_success_truth_order - success at energy reconstruction of theta,
            and have at least two scatters to calculate ray
        ray_truth_order - ray between compton scatter and second vertices
        rayhat_truth_order - normalized pointing['ray

      Length of fields except .failed and .at_least_two_scatters is
        sum(cut&at_least_two_scatters)

    11/19   TS - original matlab
    3/22    TS - python port
    """

    import numpy as np
    from math_tools import dot

    #   Now go
    at_least_two_scatters = hits['num_hits']>= 2

    if not input_cut:
        cut = at_least_two_scatters
    else:
        cut = input_cut & at_least_two_scatters

    #   Energies

    #   Find Theta
    theta = find_theta(hits, {'singlet': 0}, cut)

    hits['theta_truth_order'] = theta['energy']
    hits['theta_truth_order_success'] = theta['success']

    #   Ray reconstruction
    hits['ray_truth_order'] = np.zeros((3, hits['num_hits'].size))
    hits['ray_hat_truth_order'] = np.zeros((3, hits['num_hits'].size))
    hits['ray_truth_order'][:, theta['success']] \
        = hits['r'][:, 1, theta['success']] \
                 - hits['r'][:, 0, theta['success']]
    hits['ray_hat_truth_order'][:, theta['success']] \
        = hits['ray_truth_order'][:, theta['success']] \
        / np.sqrt(dot(
            hits['ray_truth_order'][:, theta['success']],
            hits['ray_truth_order'][:, theta['success']]
            ))

def construct_pointing_measured_order(hits, order, input_cut=None):
    """
    Finds theta and pointing ray from two vertices ordered on an
        event-by-event basis

    Applies an internal cut on successful energy-based construction of
        theta, though this should not fail if detector response has
        not changed.

    Adds the following to hits:
        theta_measured_order - calculated from energy of compton scatter
            and total energy of all scatters`xxx`x`x`
        theta_success_measured_order - success at energy reconstruction
            of theta
        ray_measured_order - ray between compton scatter and second vertices
        rayhat_measured_order - normalized pointing['ray

    Length of fields except .failed and .goodnumscatters is
      sum(cut & order['succeeded'] & pointing['failed'])

    11/19   TS
    """

    import numpy as np
    from math_tools import dot

    if not input_cut:
        cut = order['tried']
    else:
        cut = input_cut & order['tried']

    #   Find energy theta
    theta = find_theta(hits, {'singlet': order['initialhits'][0, :]}, cut)

    hits['theta_measured_order'] = theta['energy']
    hits['theta_measured_order_success'] = theta['success']

    #    Ray between first two points
    hits['ray_measured_order'] = np.zeros((3, hits['num_hits'].size))
    hits['ray_hat_measured_order'] = np.zeros((3, hits['num_hits'].size))
    hits['ray_measured_order'][:, theta['success']] \
        = hits['r'][
            :,
            order['initial_hits'][1, theta['success']],
            theta['success']
            ] \
        - hits['r'][
            :,
            order['initial_hits'][0, theta['success']],
            theta['success']
            ]
    hits['ray_hat_measured_order'][:, theta['success']] \
        = hits['ray_measured_order'][:, theta['success']] \
        / np.sqrt(dot(
            hits['ray_measured_order'][:, theta['success']],
            hits['ray_measured_order'][:, theta['success']]
            ))

def calculate_pointing_error(events):
    """
    Computes pointing from vertices and response, and compares these to
    truth from vertices.  Excludes events with missing energy.

    cones.
        dtheta.geometry - opening angle between true and pointing.ray, the
            measured ray
        dtheta.energy - difference between pointing.theta, the
            energy determined cone angle, and the true angle
        dtheta.combined - quadrature sum of the se
        ray_length - length of measured ray
        theta - energy measured cone angle
        energy - compton scatter measured energy

    Note of length of fields: pointing.theta, etc., have length
    sum(pointing.success).  The length of cones fields is smaller,
    because of the additional requirement of no missing energy

    3/15/20     TS
    3/22 - python port, TS
    """

    import numpy as np

    import reconstruction_tools
    from math_tools import dot

    #   Exclude missing energy event
    cut = ~events.truth['missing_energy']

    #   Measured pointing
    measured_theta = np.zeros(cut.shape)
    measured_ray = np.zeros((3, cut.size))
    measured_energy = np.zeros(cut.shape)
    measured_total_energy = np.zeros(cut.shape)
    measured_pointing \
        = reconstruction_tools.construct_pointing_truth_order(
        events.measured_hits,
        initial_ray = events.truth['s_incident']
        )
    measured_theta[measured_pointing['success']] \
        = measured_pointing['theta']
    measured_ray[:, measured_pointing['success']] \
        = measured_pointing['ray']
    measured_energy[measured_pointing['success']] \
        = measured_pointing['energy']['compton']
    measured_total_energy[measured_pointing['success']] \
        = measured_pointing['energy']['total']

    #   True ray and theta
    true_theta = np.zeros(cut.shape)
    true_ray = np.zeros((3, cut.size))
    true_pointing = reconstruction_tools.construct_pointing_truth_order(
        events.truth_hits,
        initial_ray = events.truth['s_incident']
        )
    true_theta[true_pointing['success']] = true_pointing['theta']
    true_ray[:, true_pointing['success']] = true_pointing['ray']

    #   Update cut to include successful measure and truth pointing
    cut = cut & true_pointing['success'] & measured_pointing['success']

    #   Calculate cones
    cones = {}
    cones['dtheta'] = {}

    #   Geometry error is opening angle between true and measured rays,
    #   measured ray is pointing.ray
    cones['dtheta']['geometry'] = np.arccos(
        dot(measured_ray[:, cut], true_ray[:, cut]) \
            / np.sqrt(dot(measured_ray[:, cut], measured_ray[:, cut]))
            / np.sqrt(dot(true_ray[:, cut], true_ray[:, cut]))
            )

    #   Energy error is difference between true and measured theta,
    #   measured is in pointing, while true is in vertices
    cones['dtheta']['energy'] =  np.abs(
        true_theta[cut] - measured_theta[cut]
        )

    #   Combined cones is quadrature sum
    cones['dtheta']['combined']= np.sqrt(
        cones['dtheta']['geometry']**2 \
            + cones['dtheta']['energy']**2
        )

    cones['ray_length'] = np.sqrt(
        dot(measured_ray[:, cut], measured_ray[:, cut])
        )
    cones['theta'] = measured_theta[cut]
    cones['energy'] = measured_energy[cut]
    cones['total_energy'] = measured_total_energy[cut]
    cones['theta_true'] = true_theta[cut]

    return cones


