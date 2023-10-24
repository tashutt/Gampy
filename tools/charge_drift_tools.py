#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
These routines:
    + find drift propeties (velocity and diffusion constant)
        for given detector response conditions
    + find sigma, the diffusion-based rms spread for an array of
        drift lengths, using drift properties

To do:
    + Fully compute transverse diffusion constant
    + Add Xe response back in
    + Consider wheter some of the material specifc routines should be put
        into the corresponding materials propertis routines

Completed first python port from matlab on Mon Aug 10 2020

@author: tshutt
"""

def properties(field, material):
    """
    Finds velocity and diffusion constaant as function of field, in V/m.

    "material" is not from standard material system.
    Only valid: "Si", "Xe", "Ar";  Ar/Xe in liquid phase.
    Xe currently disabled in python

    Needs clean-up, and thought: Some/much of this might go in
    relevant materials properties routines
    """

    import sys

    #   Diffusion constant, D

    # This from Chen et. al. - French paper,. cm^2`
    # sigma.t=0.16*sqrt(driftlengthcm);

    diffusion_constant = {}

    if material['name'] == 'Si':

        #   http://www.ioffe.ru/SVA/NSM/Semicond/Si/electric.html
        # ****NEED TO TREAT E/HOLES SEPARATELY***********
        diffusion_constant['transverse'] =  36 * 1e-4
        diffusion_constant['longitudinal ']= \
            (2 / 3)** 2 * diffusion_constant['transverse']
        mobiilty = 1400/1e4
        velocity = mobiilty * field

    # elif strcmp(material,'Xe')

    #     #   New exo paper - 1609.04467v1
    #     diffusion_constant['transverse'] =  55 * 1e-4
    #     diffusion_constant['longitudinal'] = (
    #         (2 / 3)** 2 * diffusion_constant['transverse']
    #         )
    #     velocity=ElectronVelocity(field);

    elif material['name'] == 'Ar':

        #   For now, using constant transverse constant.
        #   from 1508.07059v2,  at 500 V/cm.
        #   Longitudinal is found as a function of
        #   drift field - would be good to code that up
        #   diffusion_constant['longitudinal=7.2*1e-4;
        diffusion_constant['transverse'] =  12 * 1e-4

        #   Longitudinal drift constant, from from 1508.07059v2,
        #   takes more work

        #   Change input field to kV/cm - following paper
        field=field/1000/100

        a0 = 551.6
        a1 = 7953.7
        a2 = 4440.43
        a3 = 4.29
        a4 = 43.63
        a5 = 0.2053
        temperaturenot = 89

        b0 = 0.0075
        b1 = 742.9
        b2 = 3269.6
        b3 = 31678.2
        temperatureone = 87

        #   mu units are cm^2/V/sec - this is standard.
        mu =  (
            (a0 + a1 * field + a2 * field**(3/2) + a3 * field**(5/2))
            / (1 + (a1 / a0) * field + a4 * field**2 + a5 * field**3)
            * (material['temperature'] / temperaturenot)**(-3/2)
            )

        #   electron energy in eV
        el = (
            (b0 + b1 * field + b2 * field**2)
            / (1 + (b1 / b0) * field + b3 * field**2)
            * (material['temperature'] / temperatureone)
            )

        #   Velocity - note need field in V/cm
        velocity = mu * field * 1000
        #   longitudinal diffusion - cm^2/sec
        dl = mu * el

        #   now for ouput, make MKS
        velocity = velocity / 100
        longitudinal_diffusion_constant =  dl / 1e4

        diffusion_constant['longitudinal'] = \
            longitudinal_diffusion_constant

    #   Material not recognized
    else:
        sys.exit('ERROR in charge_drift_tools - unrecognized material')

    drift_properties = {}
    drift_properties['diffusion_constant'] = diffusion_constant
    drift_properties['velocity'] = velocity

    return drift_properties

def get_sigma(drift_distance, drift_properties):
    """ Finds sigma, the spread due to diffusion, in longitudinal
    and transverse directions

    Warning: drift_distance and the drift_field used to generate
    drift_properties must be same length, or will crash

    """

    from numpy import sqrt

    sigma = {}
    sigma['transverse'] = sqrt(2 *
          drift_properties['diffusion_constant']['transverse']
          * drift_distance / drift_properties['velocity'])
    sigma['longitudinal'] = sqrt(2 *
          drift_properties['diffusion_constant']['longitudinal']
          * drift_distance / drift_properties['velocity'])

    return sigma