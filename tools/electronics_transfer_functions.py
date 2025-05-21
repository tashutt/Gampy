#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Response of the readout electronics to the drifted electrons
Cosnists of values derived from using COMSOL to calculate the overall field
and weighting fields. Garfield++ is used to derive the signals.
The signal chain is as follows:
    pixel/wire response -> shaping function -> digital sampling

The 'analog' sampling rate used to create these signals is 10MHz
"""

import numpy as np

def get_pixel_response():
	"""
	Created by Bahrudin
	COMSOL and Garfield settings:
		1 pixel, 0.5mm X 0.5mm
		4 guiding wires at 1mm above the pixel
		drifted from (0.25, 0.25, 5)mm in LAr
        sampeled at 20 MHz
		E = 50 kV/cm
		v = #### cm/us
        done at 110K
	"""
	pixel_response =\
	np.array([-5.97299753e+02, -6.01022279e+02, -6.33818141e+02,
        -6.40143776e+02, -6.52980602e+02, -7.88570759e+02,
        -7.88570759e+02, -7.88570759e+02, -7.88570759e+02,
        -7.88570759e+02, -7.88570759e+02, -7.88570759e+02,
        -7.88570759e+02, -8.88897255e+02, -1.03916802e+03,
        -1.12576687e+03, -1.12576687e+03, -1.25306763e+03,
        -1.29548337e+03, -1.29548337e+03, -1.34681184e+03,
        -1.43677077e+03, -1.60927637e+03, -1.76747375e+03,
        -1.76747375e+03, -1.76747375e+03, -1.76747375e+03,
        -1.76747375e+03, -1.76747375e+03, -1.82109716e+03,
        -2.32429696e+03, -2.59784210e+03, -2.62040484e+03,
        -2.62040484e+03, -2.62040484e+03, -2.62040484e+03,
        -2.83342601e+03, -3.29008382e+03, -3.29008382e+03,
        -3.29008382e+03, -3.29008382e+03, -3.29008382e+03,
        -3.64489244e+03, -4.17884662e+03, -4.17884662e+03,
        -4.24946270e+03, -4.64960540e+03, -4.78437804e+03,
        -4.82046644e+03, -4.87830404e+03, -4.90165397e+03,
        -5.80043990e+03, -5.91642128e+03, -6.14134502e+03,
        -7.70527069e+03, -7.98905613e+03, -7.98905613e+03,
        -1.02294372e+04, -1.10738679e+04, -1.67726823e+04,
        -2.12699080e+04, -2.44768650e+04, -3.36522540e+04,
        -4.49825510e+04, -6.79032003e+04, -9.83428821e+04,
        -1.32493362e+05, -1.75353241e+05, -2.44350986e+05,
        -3.28977396e+05, -4.32538080e+05, -5.59387749e+05,
        -7.11568288e+05, -9.02018826e+05, -1.06412649e+06,
        -1.56642845e+06, -1.86375269e+06, -2.13110727e+06,
        -5.17101663e+06, -4.24551688e+06,  0.00000000e+00])
	return pixel_response




# wire grid has a x,y dependance on the response