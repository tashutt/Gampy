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

    print('Applying detector response', events.params.coarse_grids['signal_fraction'], "Signal Fraction")


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
                = quanta['q']['collected'] * events.params.coarse_grids['signal_fraction'] \
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
    epsilon = 0.00000000000045

    safe_denominator = cell_q[i_cells] + epsilon
    bad_q_in_cell = (cell_q[i_cells] < events.params.coarse_grids['noise'] *
                   np.sqrt(events.params.coarse_grids['signal_sharing']) *
                   events.params.coarse_grids['threshold_sigma']) 

    # Apply the scaling factor to the energy calculation
    measured_hits['energy'] = np.where(bad_q_in_cell, 
                                   0,  # Energy is 0 when bad_q_in_cell is True
                                   (cell_energy[i_cells] / safe_denominator) * quanta_q)
    
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
    measured_hits['r'] = smear_space(events.truth_hits['r'], events.params, events.truth_hits['energy'])
    measured_hits['a'] = smear_angle(events.truth_hits['s_secondary'], 
                                     measured_hits['r'],
                                     measured_hits['energy'])

    #   Save trigger information - in truth hits, since this is
    #   effectively truth data that is not known as a measurement
    events.truth_hits['triggered_q'] = triggered_q
    events.truth_hits['triggered_p'] = triggered_p
    events.truth_hits['triggered'] = triggered
    
    #   alive and cell info copied from measured_hits.
    measured_hits['cell'] = copy.copy(events.truth_hits['cell'][triggered])
    measured_hits['cell_index'] = copy.copy(events.truth_hits['cell_index'][triggered])
            
    #   Measured light from ACD
    measured_acd_energy = events.truth['front_acd_energy'] + events.truth['back_acd_energy'] \
        + events.params.light['spe_noise'] \
            * ak.Array(randn(len(events.truth['front_acd_energy'])) )

    acd_activated = measured_acd_energy > 4*events.params.light['spe_threshold']


    measured_calorimeter_energy = calculate_measured_calorimeter_energy(events.truth['calorimeter_energy'],
                                                                        events.params)
    cal_activated = measured_calorimeter_energy > 4*events.params.light['spe_threshold']
    measured_hits['calorimeter_energy'] = np.where(cal_activated,
                                                   measured_calorimeter_energy,
                                                   0)
    
    good_mask  = ak.num(triggered) > 0
    good_maskT = good_mask & triggered
    r_mask     = ak.concatenate([triggered[:,np.newaxis],]*3,axis=1)  
    measured_hits['energy'] = measured_hits['energy'][good_maskT]
    measured_hits['r']      = measured_hits['r'][r_mask][good_mask]
    measured_hits['s_secondary'] = measured_hits['a'][r_mask][good_mask]
    
    measured_hits['time']   = events.truth['time'][good_mask]
    measured_hits['triggered_id'] = events.truth['triggered_id'][good_mask]
    measured_hits['cell']   = measured_hits['cell'][good_mask]
    measured_hits['quanta_q']     = measured_hits['quanta_q'][good_maskT]
    measured_hits['total_energy'] = measured_hits['total_energy'][good_mask]
    measured_hits['cell_index']   = measured_hits['cell_index'][good_mask]
    measured_hits['triggered']    = events.truth_hits['triggered'][good_mask]
    measured_hits['ACD_activated'] = acd_activated[good_mask]
    measured_hits['calorimeter_energy'] = measured_hits['calorimeter_energy'][good_mask]
    
    measured_hits['_good_mask']   = good_mask
    measured_hits['_bad_mask']    = ~good_mask

    from tools import geometry_tools
    if events.meta['geo_params'].detector_geometry=='geomega':
        measured_hits['r_cell'] = geometry_tools.global_to_cell_coordinates(
            measured_hits['r'],
            measured_hits['cell'],
            events.meta['geo_params'],
            )[good_mask]
        measured_hits['r_cell'] = measured_hits['r_cell']

    events.measured_hits = measured_hits

    return events

    return events

### HELPER FUNCTIONS
def replace_negative_elements(a):
    import awkward as ak
    import numpy as np

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


def calculate_measured_calorimeter_energy(cal_ene, params):
    import numpy as np
    import awkward as ak
    from numpy.random import randn

    ph_collected = cal_ene * params.light['collection'] + \
                   0.1* np.sqrt(cal_ene* (1 - params.light['collection']) * params.light['collection']) * \
                   ak.Array(randn(len(cal_ene)))
    
    ph_measured = ph_collected + params.light['spe_noise'] * ak.Array(randn( len(cal_ene) ) )

    return ph_measured / params.light['collection'] 


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



def smear_space(r, params, energy=[]):
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
    
    def rndm(rand_len,std=1):
        return np.random.normal(0,std,size=rand_len)

    ENERGY = [50,300,300,750,1000]
    Rrms   = [0.02*1e-3, 
              0.09*1e-3, 
              0.05*1e-3,                            
              0.17*1e-3, 
              0.08*1e-3]

    def rrms_quadratic_fit(energy):
        a = -3.47527666e-10
        b = 4.64690829e-07
        c = -1e-5

        rrms = a * energy**2 + b * energy + c
        return np.array(rrms) 

    rx, ry, rz = r[:, 0], r[:, 1], r[:, 2]
    toShape = ak.num(rx)
    rand_len = len(np.ravel(rx))

    # if there is 'spatial_resolution_multiplier' in params.spatial_resolution then set SRM to that value otherwise set it to 1
    SRM = params.spatial_resolution.get('spatial_resolution_multiplier', 1)

    if len(energy) > 0:
        temp_qf = rrms_quadratic_fit(ak.flatten(energy)) * SRM

        new_rx = rx +  ak.unflatten(list(rndm(rand_len)*temp_qf),
                                    toShape, axis=0)
        new_ry = ry +  ak.unflatten(list(rndm(rand_len)*temp_qf),
                            toShape, axis=0)
        new_rz = rz +  ak.unflatten(list(rndm(rand_len)*temp_qf),
                            toShape, axis=0)
        
    else:
        new_rx = rx +  ak.unflatten(rndm(rand_len,transverse_part),
                                    toShape, axis=0)
        new_ry = ry +  ak.unflatten(rndm(rand_len,transverse_part),
                            toShape, axis=0)
        new_rz = rz +  ak.unflatten(rndm(rand_len,longitudinal_part),
                            toShape, axis=0)

    # Concatenating the arrays along the new axis
    combined_array = ak.concatenate([new_rx[:,np.newaxis],
                                     new_ry[:,np.newaxis], 
                                     new_rz[:,np.newaxis]], axis=1)

    return combined_array

def smear_angle(s, Z=[],E=[]):
    """
    Smears the angle of the hit
    """
    from numpy.random import randn
    import numpy as np
    import awkward as ak

    toShape = ak.num(s[:,0])
    rand_len = len(np.ravel(s[:,0]))
    # std of 0.01 gives 0.4 degrees for 1 sigma

    # hardcoding the z value for now
    if len(Z) != 0:
        z = ak.ravel(Z[:,2])
        up   = z > 0.36 / 2
        down = z < 0.36 / 2
        z_cell = np.zeros(len(z))
        z_cell[up] = 0.36 - z[up]
        z_cell[down] = z[down]

        e = ak.ravel(E)

        STD = np.ones(len(z)) * 0.08
        # for energies less than 500, assign full uncertainty
        STD[e < 500] = 2.5 
        STD[e > 300] = 0.7*z[e > 300]/0.17
        STD[e > 500] = 0.2*z[e > 500]/0.17  


    else:
        # all the recoil errors are 3 degrees
        STD = 0.08

    def rndm(rand_len,std=STD):
        return list(np.random.exponential(1,size=rand_len)*std)

    sx = s[:,0] + ak.unflatten(rndm(rand_len),toShape, axis=0)
    sy = s[:,1] + ak.unflatten(rndm(rand_len),toShape, axis=0)
    sz = s[:,2] + ak.unflatten(rndm(rand_len),toShape, axis=0)

    normal = (sx**2 + sy**2 + sz**2)**0.5
    sx = sx / normal
    sy = sy / normal
    sz = sz / normal

    sn = ak.concatenate([sx[:,np.newaxis],
                        sy[:,np.newaxis],
                        sz[:,np.newaxis]],axis=1)

    return sn


