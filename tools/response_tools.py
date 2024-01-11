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
Rewrite to awkward arrays 9/23 BT
implemented calorimeter readout, 100x spc noise
@author: tshutt
"""

def apply_detector_response(events):
    """
    Apply detector response to MC truth
        params - settings
        events - vectorized event data

    events returned.  The primary result is creating events.measured_hits,
    but also adds trigger information to events.truth_hits

    events.measured_hits contains:
        energy, total_energy, quanta_q, r, 
        cell, cell_index, time, r_cell
    events.truth_hits:
        triggered_p, triggered_q, triggered,
        
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

    AWKWARD UPDATE
    -- is there a need to remove hits that did not trigger? 

    """

    import awkward as ak
    import numpy as np
    from numpy.random import randn
    import copy
    import geometry_tools

    print('Applying detector response')


    #   Calculate detector params
    events.params.calculate()

    #   Measured hits is a dictionary
    measured_hits = {}

    #  Add fluctuations, find measured signal, apply threshold
    # charge_fraction = np.exp(-events.truth_hits['z_drift'] /
    #                          events.params.charge_drift['drift_length'])
    charge_fraction = 1
    
    quanta = find_hit_quanta(events.truth_hits['energy'], charge_fraction,
                             events.params)
    
    quanta['q']['measured'] \
                = quanta['q']['collected'] \
                 * events.params.coarse_grids['signal_fraction'] \
                 + events.params.coarse_grids['noise'] \
                    * np.sqrt(events.params.coarse_grids['signal_sharing']) \
                     * ak.Array([randn(L) for L in ak.num(quanta['q']['collected'])])
    
    triggered_q = (quanta['q']['measured'] > events.params.coarse_grids['noise'] *
                   np.sqrt(events.params.coarse_grids['signal_sharing']) *
                   events.params.coarse_grids['threshold_sigma'])
    
    quanta['q']['measured'] = quanta['q']['measured'] \
        / events.params.coarse_grids['signal_fraction'] \
        / charge_fraction
    
    #   Measured light adds readout spe noise
    quanta['p']['measured'] = quanta['p']['collected'] \
        + events.params.light['spe_noise'] \
            * ak.Array([randn(L) for L in ak.num(quanta['q']['collected'])])
    
    quanta_q = quanta['q']['measured']
    
    cell_q, cell_light = [], []
    for event_num in range(len(triggered_q)):
        cells_in_this_event = events.truth['num_cells'][event_num]
    
        m_q = quanta['q']['measured'][event_num]
        m_light = quanta['p']['measured'][event_num] / events.params.light[
            'collection']
    
        temp_cq, temp_clight = [], []
        for nc in range(cells_in_this_event):
            this_cells = events.truth_hits['cells_list'][event_num, nc]
            same_cell = events.truth_hits['cell'][event_num] == this_cells
    
            sum_of_charges = np.sum(m_q[same_cell & triggered_q[event_num]])
            sum_of_charges = 0 if sum_of_charges < 0 else sum_of_charges
            temp_cq.append(sum_of_charges)
    
            sum_of_light = np.sum(m_light[same_cell])
            sum_of_light = 0 if sum_of_light < 0 else sum_of_light
            temp_clight.append(sum_of_light)
    
        cell_q.append(temp_cq)
        cell_light.append(temp_clight)
    
    cell_q = ak.Array(cell_q)
    cell_p = ak.Array(cell_light)
    
    #   Summed energy in cells comes from sum of charge and light
    cell_energy = (cell_p + cell_q) * events.params.material['w']
    
    i_cells = events.truth_hits['cell_index']
    
    # Define a small constant to avoid division by very small numbers
    epsilon = 0.45

    safe_denominator = cell_q[i_cells] + epsilon

    # Apply the scaling factor to the energy calculation
    measured_hits['energy'] = (cell_energy[i_cells] / safe_denominator) * quanta_q 
        
    
    triggered_p = cell_p[i_cells] > events.params.light['spe_threshold']
    
    #   Hits trigger is and of q and p triggers
    triggered = triggered_q & triggered_p
    
    #   Event energy is sum of cell charge and light, applying
    #   threhshold to light signal
    measured_hits['total_energy'] = np.sum(
        (cell_p * (cell_p > events.params.light['spe_threshold']) + cell_q) *
        events.params.material['w'],
        axis=1)
    
    #   Charge signal
    measured_hits['quanta_q'] = copy.copy(quanta_q)
    measured_hits['r'] = smear_space(events.truth_hits['r'], events.params)
    
    #   Save trigger information - in truth hits, since this is
    #   effectively truth data that is not known as a measurement
    events.truth_hits['triggered_q'] = triggered_q
    events.truth_hits['triggered_p'] = triggered_p
    events.truth_hits['triggered'] = triggered
    
    #   alive and cell info copied from measured_hits.
    measured_hits['cell'] = copy.copy(events.truth_hits['cell'])
    measured_hits['cell_index'] = copy.copy(events.truth_hits['cell_index'])
    
    from tools import geometry_tools
    #   Generate locations in cell coordinates
    if events.meta['geo_params'].detector_geometry=='geomega':
        measured_hits['r_cell'] = geometry_tools.global_to_cell_coordinates(
            measured_hits['r'],
            measured_hits['cell'],
            events.meta['geo_params'],
            )
        
    #   Measured light from ACD
    measured_acd_energy = events.truth['front_acd_energy'] + events.truth['back_acd_energy'] \
        + events.params.light['spe_noise'] \
            * ak.Array(randn(len(events.truth['front_acd_energy'])) )

    acd_activated = measured_acd_energy > events.params.light['spe_threshold']


    #  measure calorimenter stuff
    spe_calorimeter_noise_multiplier = 100
    CALORIMETER_ACTIVATION_ENERGY = 1000 # 1MeV, a guess

    measured_calorimeter_energy = events.truth['calorimeter_energy'] \
        + spe_calorimeter_noise_multiplier * events.params.light['spe_noise'] \
            * ak.Array(randn(len(events.truth['calorimeter_energy'])) )
    
    measured_hits['calorimeter_energy'] = measured_calorimeter_energy
    calorimeter_activated = measured_calorimeter_energy > CALORIMETER_ACTIVATION_ENERGY

    #   Time of events is directly from truth.  Probably should add an error
    #   here based on light readout timing
    measured_hits['time'] = events.truth['time']
    
    good_mask = ak.num(triggered) > 0   
    measured_hits['energy'] = measured_hits['energy'][good_mask]
    measured_hits['r']      = measured_hits['r'][good_mask]
    measured_hits['r_cell'] = measured_hits['r_cell'][good_mask]
    measured_hits['time']   = events.truth['time'][good_mask]
    measured_hits['cell']   = measured_hits['cell'][good_mask]
    measured_hits['quanta_q']     = measured_hits['quanta_q'][good_mask]
    measured_hits['total_energy'] = measured_hits['total_energy'][good_mask]
    measured_hits['cell_index']   = measured_hits['cell_index'][good_mask]
    measured_hits['triggered']    = events.truth_hits['triggered'][good_mask]
    measured_hits['ACD_activated'] = acd_activated[good_mask]
    measured_hits['calorimeter_energy'] = measured_hits['calorimeter_energy'][good_mask]
    measured_hits['calorimeter_activated'] = calorimeter_activated[good_mask]
    
    measured_hits['_good_mask']   = good_mask
    measured_hits['_bad_mask']    = ~good_mask

    events.measured_hits = measured_hits
    return events

    return events

### HELPER FUNCTIONS
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
    import awkward as ak

    def round_(a):
        return ak.values_astype(a, "int64")

    def replace_negative_elements(a):
        new_arrays = []
        for sub_array in a:
            sum_sub_array = np.sum(sub_array)
            if sum_sub_array < 0:
                try:
                    sub_array = ak.with_field(
                        sub_array, np.where(sub_array < 0, 0, sub_array), 0)
                except:
                    print("ERROR in replace_negative_elements")
                    sub_array = np.where(sub_array < 0, 0, sub_array)
            new_arrays.append(sub_array)
        return ak.Array(new_arrays)

    q = {}
    p = {}
    quanta = {}

    #   Total raw quanta, with fano factor rt(n) term
    raw_summed_quanta = round_(
        energy / params.material['w'] +
        np.sqrt(energy / params.material['w'] * params.material['fano']) *
        randn(len(energy)))

    #   This can lead to negative quanta.  NEED BETTER TREATMENT, but
    #   for now set any such to zero.  Below, set such q and p to
    #   zero.
    raw_summed_quanta = replace_negative_elements(raw_summed_quanta)

    #   Add recombination fluctuations
    #   Here I (nearly) follow Dobi - but am not dealing with
    #   DIRECT EXCITATION.  ALSO, mabye delta_r should be applied before
    #   initial fano flucutations
    #   Also - this stinks in Dobi - should have r*(1-r) behavior
    #   NEED to implement energy dependent recombination fraction here
    delta_r = \
        params.material['sigma_p'] \
        * raw_summed_quanta \
        * ak.Array([randn(len(sub_array)) for sub_array in energy])
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

    q['raw'] = replace_negative_elements(q['raw'])
    p['raw'] = replace_negative_elements(p['raw'])

    #   Charge: drift loss, and flucutations from that
    q['collected'] = (q['raw'] * charge_fraction +
                      sqrt(q['raw'] *
                           (1 - charge_fraction) * charge_fraction) *
                      ak.Array([randn(len(sub_array))
                                for sub_array in energy]))

    #   Light: collection fraction and flucutations due to that
    p['collected'] = (
        p['raw'] * params.light['collection'] +
        sqrt(p['raw'] *
             (1 - params.light['collection']) * params.light['collection']) *
        ak.Array([randn(len(sub_array)) for sub_array in energy]))

    # #   Measured signals.  Charge gets coarse grid noise, light adds
    # #   readout spe noise
    # q['measured'] = q['collected'] \
    #     + params.coarse_grids['noise'] * randn(energy.size)
    # p['measured'] = p['collected'] \
    #     + params.light']['spe_noise'] * randn(energy.size)

    p['collected'] = replace_negative_elements(p['collected'])
    q['collected'] = replace_negative_elements(q['collected'])
    print("Replaced negative elements in q and p")

    quanta['p'] = p
    quanta['q'] = q

    return quanta



def smear_space(r, params):
    """
    Adapted for awkward array input, constructing a new array.
    """
    from numpy.random import randn
    import awkward as ak
    import numpy as np

    #TODO
    sigma = {}
    sigma['transverse'] = 0
    sigma['longitudinal'] = 0

    transverse_part = (params.spatial_resolution['sigma_xy']**2 +
                       (sigma['transverse'] *
                        params.charge_drift['diffusion_fraction'])**2)**0.5
    longitudinal_part = (params.spatial_resolution['sigma_z']**2 +
                         (sigma['longitudinal'] *
                          params.charge_drift['diffusion_fraction'])**2)**0.5

    rx, ry, rz = r[:, 0], r[:, 1], r[:, 2]
    num_elements = ak.num(rx)

    new_rx = ak.Array([[subarr + randn(n) * transverse_part]
                       for subarr, n in zip(rx, num_elements)])
    new_ry = ak.Array([[subarr + randn(n) * transverse_part]
                       for subarr, n in zip(ry, num_elements)])
    new_rz = ak.Array([[subarr + randn(n) * longitudinal_part]
                       for subarr, n in zip(rz, num_elements)])

    # Concatenating the arrays along the new axis
    combined_array = ak.concatenate([new_rx, new_ry, new_rz], axis=1)

    return combined_array
    
