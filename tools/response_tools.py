#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collection of routines used to apply detector response to vectorized
hit-level data.  Parameters used here are defined and manipulated
in params_tools.

See also:
Electron track level response: electron_track_tools
File io: sim_file_tools

Signficant rewrite of original Matlab routines, initial port 8/10/2020
Major repackaging 3/21

@author: tshutt
"""

def apply_detector_response(events):
    """
    Apply detector response to MC truth
        params - settings
        events - vectorized event data

    events returned.  The primary result is creating events.measured_hits,
    but also adds trigger information to events.truth_hits

    Generates raw photons and electrons, applies simplified treatment
    of their propogation and measurement including fluctuations at all
    branching steps, adds readout noise, sums charge and light energy
    in cells and estimates energy per hit in multiple hit cells via
    scaling of charge signals from total energy in cell, and applies
    trigger to both charge and light measurements

    Numerous improvements/fixes to do:
        + Recombination should be function of energy and field, based
            on LAr data which now does exist
        + Recombination fluctuations are approximate and from Xe, should
            revise for LAr when this becomes known
        + Add coarse + pixel readout, revise triggering appropriately
        + Add some treatment of electron optics and induced signals
        + Base spatial resolution on studies of electron track reconstruction,
            adding energy, drift distance and any other dependences
        + Spatially varying light response, based on MC
        + Calculate expected energy resolution per hit, to feed into
            .evta file. Currently a kludge in evta file writer, which
            also should be tested, and needs correct charge noise
        + Event times should get error based on light signal
        + This list surely incompletes
        + "triggered" has comment saying it should be removed?  Why? Should
            this go in truth?

        More complete treatment is documented in a slide deck

    """

    import numpy as np
    from numpy.random import randn
    import copy

    import geometry_tools

    print('Applying detector response')

    #   This accumulates signals in cells
    def add_values_to_cells(values, cell_values, hits, nh, event_cut):
        """
        For all struck cells, adds values from hit nh to cell_values.

        values - length is sum(event_cut).

        cell_values - size is (num_cells, num_events)

        Note that values is the nh hit slice of a larger array,
            but nh must be separely
            supplied to index information in hits
        """

        for nc in range(hits['num_cells'].max()):
            hit_in_cell \
                = hits['cell'][nh, :] == hits['cells_list'][nc, :]
            cell_values[nc, hit_in_cell & event_cut]  \
                += values[hit_in_cell[event_cut]]

        return

    #   Calculate detector params
    events.params.calculate()

    #   Measured hits is a dictionary
    measured_hits = {}

    #   convenient variable
    max_num_hits = events.truth['num_hits'].max()

    #  Add fluctuations, find measured signal, apply threshold

    #   Initialize variables

    measured_hits['energy'] = np.zeros(events.truth_hits['energy'].shape)
    measured_hits['r'] = np.zeros(events.truth_hits['r'].shape)
    measured_hits['total_energy'] \
        = np.zeros(events.truth['num_hits'].shape)

    triggered_q = np.zeros(events.truth_hits['energy'].shape, dtype=bool)
    triggered_p = np.zeros(events.truth_hits['energy'].shape, dtype=bool)

    charge_fraction = np.ones(events.truth_hits['energy'].shape)
    quanta_q = np.zeros(events.truth_hits['energy'].shape)

    cell_energy = np.zeros(events.truth_hits['cells_list'].shape)
    cell_q = np.zeros(events.truth_hits['cells_list'].shape)
    cell_p = np.zeros(events.truth_hits['cells_list'].shape)

    #   Loop over hits, find quanta of charge and light per event,
    #   sum hits into cells, find charge trigger
    for nh in range(max_num_hits):

        #   convenient variables
        alive = events.truth_hits['alive'][nh, :]
        # cell_hit_count = events.truth_hits['cell_hit_count'][nh, alive]

        #   Charge collection - due to charge loss
        #   Note: values of z are negative
        charge_fraction[nh, alive] = np.exp(
            - events.truth_hits['z_drift'][nh, alive]
            / events.params.charge_drift['drift_length']
            )

        #   Find quanta, for "alive" hits
        quanta = find_hit_quanta(
            events.truth_hits['energy'][nh, alive],
            charge_fraction[nh, alive],
            events.params
            )

        #   Measured charge signal.  First, reduce signal according to
        #   fraction that appears on coarse grids.  Then add noise of
        #   total set of grids with signal.
        quanta['q']['measured'] \
            = quanta['q']['collected'] \
            * events.params.coarse_grids['signal_fraction'] \
            + events.params.coarse_grids['noise'] \
                * np.sqrt(events.params.coarse_grids['signal_sharing']) \
                * randn(quanta['q']['collected'].size) \

        #   Now trigger on this signal, compared to noise in sum wires
        triggered_q[nh, alive] = (
            quanta['q']['measured'] > events.params.coarse_grids['noise']
            * np.sqrt(events.params.coarse_grids['signal_sharing'])
            * events.params.coarse_grids['threshold_sigma']
            )

        #   Now correct signal for fraction on wires, and charge drift
        quanta['q']['measured'] = quanta['q']['measured'] \
            / events.params.coarse_grids['signal_fraction'] \
            / charge_fraction[nh, alive]

        #   Measured light adds readout spe noise
        quanta['p']['measured'] = quanta['p']['collected'] \
            + events.params.light['spe_noise'] \
                * randn(quanta['p']['collected'].size)

        #   Add triggered, loss corrected charge to
        #   summed charge in cells
        add_values_to_cells(
            quanta['q']['measured'][triggered_q[nh, alive]],
            cell_q,
            events.truth_hits,
            nh,
            (triggered_q[nh, :] & alive)
            )

        #   Add untriggered loss corrected light to summed light
        #   in cells
        add_values_to_cells(
            quanta['p']['measured'] \
                / events.params.light['collection'],
            cell_p,
            events.truth_hits,
            nh,
            alive
            )

        #   Loss-corrected, measured charge signal per hit - needed below
        quanta_q[nh, alive] = quanta['q']['measured']


        #   Smear space
        measured_hits['r'][:, nh, alive] = \
            smear_space(events.truth_hits['r'][:, nh, alive], events.params)

    #   Summed energy in cells comes from sum of charge and light
    cell_energy = (cell_p + cell_q) * events.params.material['w']

    #   With cell sums generated, now loop on hits again
    #   and find meaured hit energy and light trigger
    for nh in range(max_num_hits):

        #   convenient variables
        alive = events.truth_hits['alive'][nh, :]
        i_cells = events.truth_hits['cell_index'][nh, :]
        i_events = np.arange(0, len(alive))

        #   Measured hit energy is based on charge of this hit, scaled
        #   by summed energy and sumnmed charge in this cell.  Trigger
        #   not explicitly applied  here (but already used to create
        #   summed charge in cell).
        #   Note that for single hit in cell this is the exact treatment
        measured_hits['energy'][nh, alive] \
            = cell_energy[i_cells, i_events][alive] \
                * quanta_q[nh, alive] \
                    / cell_q[i_cells, i_events][alive]

        #   Find triggered light signals, per hit, but based
        #   on total light signal in cell
        triggered_p[nh, alive] = cell_p[i_cells, i_events][alive] \
            > events.params.light['spe_threshold']

    #   Hits trigger is and of q and p triggers
    triggered = triggered_q & triggered_p

    #   Event energy is sum of cell charge and light, applying
    #   threhshold to light signal
    measured_hits['total_energy'] = np.sum((
        cell_p * (cell_p > events.params.light['spe_threshold'])
        + cell_q) * events.params.material['w'], axis=0)

    #   Charge signal
    measured_hits['quanta_q'] = copy.copy(quanta_q)

    #   Save trigger information - in truth hits, since this is
    #   effectively truth data that is not known as a measurement
    events.truth_hits['triggered_q'] = triggered_q
    events.truth_hits['triggered_p'] = triggered_p
    events.truth_hits['triggered'] = triggered

    #   alive and cell info copied from measured_hits.
    measured_hits['alive'] = copy.copy(events.truth_hits['alive'])
    measured_hits['cell'] = copy.copy(events.truth_hits['cell'])
    measured_hits['cell_index'] \
        = copy.copy(events.truth_hits['cell_index'])

    #  good = alive & triggerered
    measured_hits['alive'] = copy.copy(events.truth_hits['alive'])

    #   Now remove untriggered hits.  Some messiness
    #   needed to cut out hits from matrix of (hits, events)
    #   This is still probably slowet part of calculation.  Might be
    #   worth trying to create new arrays for each num_hits instead of
    #   slicing out of full arrays?

    #   convenient variable
    max_num_hits = events.truth['num_hits'].max()

    measured_hits['num_hits'] = \
        np.sum(events.truth_hits['alive'] * triggered, 0)
    measured_hits['missing_hits'] = np.sum(~triggered, 0)

    for num_hits in range(1, max_num_hits+1):

        this_num_hits = events.truth['num_hits']==num_hits

        skips =copy.copy(~triggered & this_num_hits)
        shifted_skips = copy.copy(skips)
        # for n1 in range(max_num_hits-1):
        for n1 in range(num_hits):
            while 1:
                shift = copy.copy(shifted_skips[n1,:])
                if np.sum(shift)>0:
                    #   Shift through end of steps
                    for n2 in range(n1,max_num_hits-1):
                        shifted_skips[n2, shift] = shifted_skips[n2+1, shift]
                        measured_hits['energy'][n2, shift] = \
                            measured_hits['energy'][n2+1, shift]
                        measured_hits['r'][:, n2, shift] = \
                            measured_hits['r'][:, n2+1, shift]
                        measured_hits['alive'][n2, shift] = \
                            measured_hits['alive'][n2+1, shift]
                        measured_hits['cell'][n2, shift] = \
                            measured_hits['cell'][n2+1, shift]
                        measured_hits['cell_index'][n2, shift] = \
                            measured_hits['cell_index'][n2+1, shift]
                        measured_hits['quanta_q'][n2, shift] = \
                            measured_hits['quanta_q'][n2+1, shift]

                    shifted_skips[n2+1, shift] = False
                    measured_hits['energy'][n2+1, shift] = 0.0
                    measured_hits['r'][:, n2+1, shift] = 0.0
                    measured_hits['alive'][n2+1, shift] = False
                    measured_hits['cell'][n2+1, shift] = 0
                    measured_hits['cell_index'][n2+1, shift] = 0
                    measured_hits['quanta_q'][n2+1, shift] = 0

                else:
                    break

                shift = shifted_skips[n1, :]
                if np.sum(shift)==0:
                    break

    #   Generate locations in cell coordinates
    if events.meta['geo_params'].detector_geometry=='geomega':
        measured_hits['r_cell'] = geometry_tools.global_to_cell_coordinates(
            measured_hits['r'],
            measured_hits['cell'],
            events.meta['geo_params'],
            alive = measured_hits['alive']
            )

    #   If max num hits now reduced, cut out the empty end of the arrays
    max_hits = np.max(measured_hits['num_hits'])
    if measured_hits['alive'].shape[0] > max_hits:
        measured_hits['energy'] = measured_hits['energy'][0:max_hits, :]
        measured_hits['r'] = measured_hits['r'][:, 0:max_hits, :]
        measured_hits['alive'] = measured_hits['alive'][0:max_hits, :]
        measured_hits['cell'] = measured_hits['cell'][0:max_hits, :]
        measured_hits['cell_index'] \
            = measured_hits['cell_index'][0:max_hits, :]
        measured_hits['quanta_q'] = measured_hits['quanta_q'][0:max_hits, :]

    #   Time of events is directly from truth.  Probably should add an error
    #   here based on light readout timing
    measured_hits['time'] = events.truth['time']

    #   Assign to events
    events.measured_hits = measured_hits

    return events

def find_hit_quanta(energy, charge_fraction, params):
    """
    Given the energy in a hit, generates quanta of initial and
    measured charge and light, using detector parameters in
    params, and the charge collection fraction
    at each hit depth.  This treatent needs improvement in two key
    respects: currently it has a fixed recombination fraction (should be
    field, energy and particle dependent), and recombination
    fluctuations uses a treatment from Xe data from LUX
    (as of 8/20 I am unaware of such measuremnts in LAr).

    returns dictionary 'quanta', with 'raw' and 'measured' fields.
    raw is quanta created at the hit site, after recombination.
    meeasured is the "raw" measured quanitities, which includes light
    and charge collection losses, but not yet noise
    """

    import numpy as np
    from numpy import sqrt
    from numpy.random import randn
    from numpy import round_

    q = {}
    p = {}
    quanta = {}

    #   Total raw quanta, with fano factor rt(n) term
    raw_summed_quanta = round_(
        energy / params.material['w']
        + sqrt(energy / params.material['w']
               * params.material['fano'])
        * randn(len(energy))
        )

    #   This can lead to negative quanta.  NEED BETTER TREATMENT, but
    #   for now set any such to zero.  Below, set such q and p to
    #   zero.
    if np.sum(raw_summed_quanta<0) > 0:
        # disp(['*** ' num2str(sum(raw_summed_quanta<0)) ...s
        #     ' events with negative total excitation num *** ']);
        raw_summed_quanta[raw_summed_quanta<0] = 0

    #   Add recombination fluctuations
    #   Here I (nearly) follow Dobi - but am not dealing with
    #   DIRECT EXCITATION.  ALSO, mabye delta_r should be applied before
    #   initial fano flucutations
    #   Also - this stinks in Dobi - should have r*(1-r) behavior
    #   NEED to implement energy dependent recombination fraction here
    delta_r = \
        params.material['sigma_p'] \
        * raw_summed_quanta \
        * randn(energy.size)
    q['raw'] = \
        raw_summed_quanta \
        * (1 - params.material['recombination'])
    p['raw'] = \
        raw_summed_quanta \
        * params.material['recombination']
    q['raw'] = \
        round_(q['raw']) \
        + round_(delta_r)
    p['raw'] = \
        round_(p['raw']) \
        - round_(delta_r)
    if np.sum(q['raw']<0)>0:
        # disp(['*** ' num2str(sum(q['raw']<0)) ...
        #     ' events with negative electron num *** ']);
        q['raw'][q['raw']<0]=0;

    if np.sum(p['raw']<0)>0:
        # disp(['*** ' num2str(sum(p['raw']<0)) ...
        #     ' events with negative photon num *** ']);
        p['raw'][p['raw']<0]=0;

    #   Charge: drift loss, and flucutations from that
    q['collected'] = (
        q['raw'] * charge_fraction
        + sqrt(
        q['raw']
        * (1 - charge_fraction)
        * charge_fraction)
        * randn(energy.size)
        )

    #   Light: collection fraction and flucutations due to that
    p['collected'] = (
        p['raw'] * params.light['collection']
        + sqrt(
        p['raw']
        * (1 - params.light['collection'])
        * params.light['collection'])
        * randn(energy.size)
        )

    # #   Measured signals.  Charge gets coarse grid noise, light adds
    # #   readout spe noise
    # q['measured'] = q['collected'] \
    #     + params.coarse_grids['noise'] * randn(energy.size)
    # p['measured'] = p['collected'] \
    #     + params.light']['spe_noise'] * randn(energy.size)

    quanta['p'] = p
    quanta['q'] = q

    return quanta

def smear_space(r, params):
    """
    Currently a simple quadrature sum of fixed resolution (described in
    parms), and diffusion.  Needs  better treatment of diffusion term
    - currently probably an overesimate

    5/20    TS
    """

    # import charge_drift_tools

    import numpy as np
    from numpy.random import randn
    from numpy import sqrt

    # drift_properties = charge_drift_tools.properties(
    #     params.charge_drift']['drift_field'],
    #     params.material']
    #     )
    # sigma = charge_drift_tools.get_sigma(-r[2, :], drift_properties)
    sigma = {}
    sigma['transverse'] = np.zeros(r[2, :].shape)
    sigma['longitudinal'] = np.zeros(r[2, :].shape)

    num=len(r[0, :])

    r[0, :] = \
        r[0, :] \
        + randn(num) * sqrt(
        params.spatial_resolution['sigma_xy']**2
        + (sigma['transverse']
           * params.charge_drift['diffusion_fraction'])**2
        )
    r[1, :] = \
        r[1, :] \
        + randn(num) * sqrt(
        params.spatial_resolution['sigma_xy']**2
        + (sigma['transverse']
           * params.charge_drift['diffusion_fraction'])**2
        )
    r[2, :] = \
        r[2, :] \
        + randn(num) * sqrt(
        params.spatial_resolution['sigma_z']**2
        + (sigma['longitudinal']
           * params.charge_drift['diffusion_fraction'])**2
        )

    return r
