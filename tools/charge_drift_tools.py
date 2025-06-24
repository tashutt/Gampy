#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
These routines:
    + find drift propeties (velocity and diffusion constant)
        for given detector response conditions
    + find sigma, the diffusion-based rms spread for an array of
        drift lengths, using drift properties

    TODO: Rethink how this is related to material properties.
    TODO: update Xe, Ar transvese data?
    TODO: Re-implement old Xe gas treatment, if robust enough

First python port from matlab on 8/10/20
@author: tshutt
"""

def properties(field, material):
    """
    Finds velocity and diffusion constaant as function of field, in V/m.

    from material dictionary, use 'name' and 'temperature.
    Only valid: Si, Xe, Ar, with Xe and Ar only in liquid phase
    """

    import sys

    #   Diffusion constant, D

    # This from Chen et. al. - French paper,. cm^2`
    # sigma.t=0.16*sqrt(driftlengthcm)

    diffusion_constant = {}

    if material['name'] == 'Si':

        #   http://www.ioffe.ru/SVA/NSM/Semicond/Si/electric.html
        # ****NEED TO TREAT E/HOLES SEPARATELY***********
        diffusion_constant['transverse'] =  36 * 1e-4
        diffusion_constant['longitudinal']= \
            (2 / 3)** 2 * diffusion_constant['transverse']
        mobiilty = 1400/1e4
        velocity = mobiilty * field

    elif material['name'] == 'Xe':

        #   New exo paper - 1609.04467v1
        diffusion_constant['transverse'] =  55 * 1e-4
        diffusion_constant['longitudinal'] = (
            (2 / 3)** 2 * diffusion_constant['transverse']
            )

        velocity = velocity_in_xe(field)

    elif material['name'] == 'Ar':

        #   For now, using constant transverse constant.
        #   from 1508.07059v2,  at 500 V/cm.
        #   Longitudinal is found as a function of
        #   drift field - would be good to code that up
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

        diffusion_constant['longitudinal'] = dl / 1e4

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
    sigma['mean'] = 0.5 * (sigma['longitudinal'] + sigma['transverse'])

    return sigma

def velocity_in_xe(field, phase='liquid', gas_density=5.5249):
    """
    Returns electron drift velocity in Xe in m/s

    Inputs
        field - electric field, V/m
        not yet implemented:
           phase - 'liquid'  or 'gas'
           gas_density - kg/m^3

    Liquid data from plot in L.S.Miller, S.Howe, W.E.Spear,
    Phys. Rev. 166 (1968), 871)

    Gas data is from routine XeGasDriftData, based currently on 3 papers

    8.29.2005, John Kwong
    1.22.2007, JK
    9/16/2016 - TS change to standard units.  Havent don't gas yet
    7/4/2018 - fix up gas treatment, add density input.
    12/23/23 - python port
    """

    import sys
    import numpy as np
    from scipy.interpolate import CubicSpline

    if phase == 'liquid':

        # This data is V/cm, and mm/microsec
        # electric field
        e_array = np.array([33.598,45.276,61.776,78.233,100.31,156.92,
                           211.46,413.75,608.24,
                           829.9,1038,2030.9,4023.4,6063.5,9969,20499,
                           40610,61203,28.232,19.934,13.56])
        # drift velocity
        v_array = np.array([0.61464,0.78889,0.96325,1.1329,1.2676,1.5096,
                            1.6474,1.9376,
                            2.0883,2.1951,2.2789,2.5182,2.714,2.8175,2.8887,
                            2.925,2.925,2.8529,0.54934,0.39711,0.28])

        #   Sort, and change units to mks
        idx = np.argsort(e_array)
        e_array = e_array[idx] * 100
        v_array = v_array[idx] / 1000 * 1e6

        vs = CubicSpline(e_array, v_array)
        v = vs(field)

        return v

    elif  phase == 'gas':

        sys.exit('*** Error: velocity in gas not yet implemented')

        #     driftdata=XeGasDriftData

        #     usepack=driftdata.pack300k.eovern.data<0.07
        #     usepack(13)=false
        #     usebowe=true(size(driftdata.bowefig9.eovern.data))
        #     usepatrick=driftdata.patrickfig2.eovern.data>
        #       driftdata.bowefig9.eovern.data(end)

        #     #   Kludge: Patrick and Bowe don't match.  Bowe and Pack
        #     #     roughly match.
        #     #   So shift Patrick to match Bowe
        #     patrickcorrection=1.1

        #     #   Now extedn the range to higher field by adding a line
        #     #   of points determined by eye
        #     highpoints.eovern=logspace(log10(23),2,10)
        #     highpoint=4.85e4
        #     lowpoint=11050
        #     m=(highpoint-lowpoint)/(100-23)
        #     highpoints.v=(highpoints.eovern-23)*m+lowpoint

        #     eovernfitdata=[ ...
        #         driftdata.pack300k.eovern.data(usepack)' ...
        #         driftdata.bowefig9.eovern.data(usebowe)' ...
        #         driftdata.patrickfig2.eovern.data(usepatrick)' ...
        #         highpoints.eovern ...
        #         ]
        #     vfitdata=[ ...
        #         driftdata.pack300k.v.data(usepack)' ...
        #         driftdata.bowefig9.v.data(usebowe)' ...
        #         patrickcorrection*driftdata.patrickfig2.v.data(usepatrick)'
        #         highpoints.v ...
        #         ]

        #     #   Assign stp density
        #     if nargin<3
        #         density=5.5249
        #     end

        #     #   #/m^3
        #     xe.mw=131.29
        #     numberdensity=density*1000*6.02214e23/xe.mw

        #     #   Now in Td = 10^-21 V * m^2
        #     eovern=field/numberdensity*1e21

        #     #   Use spline to find interpolated values
        #     v=spline(eovernfitdata,vfitdata,eovern)
