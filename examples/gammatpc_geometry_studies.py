#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 15:47:44 2023

@author: tshutt
"""

def CellSize():
    """
    Vary the width (defined as flat-to-flat distance for
        hexagonal cells), and height of square GammaTPC cells.

    Index order is [width][height]

    6/23  TS
    """

    import numpy as np

    study = {}

    study['kit'] = {}

    #   Values to scan over.
    widths = np.array([15, 17.5, 20]) / 100
    heights = np.array([10, 15, 20]) / 100

    #   Add values to kit
    study['kit']['widths'] = widths
    study['kit']['heights'] = heights

    #  Indices and such in kit
    study['kit']['width_case_index'] = []
    study['kit']['height_case_index'] = []
    study['kit']['num_widths'] = len(widths)
    study['kit']['num_heights'] = len(heights)
    study['kit']['num_cases'] = len(widths) * len(heights)
    study['kit']['case'] = [[0     # note here we reverse index order
        for j in range(len(heights))]
        for i in range(len(widths))]
    study['kit']['width_indices'] = range(len(widths))
    study['kit']['height_indices'] = range(len(heights))

    #   Labels
    study['labels'] = {}

    study['labels']['study_name'] = 'GammaTPC cell size'
    study['labels']['study_tag'] = 'GammaTPCCellSize'

    #   Variable labels
    study['labels']['width'] = []
    for nw in range(len(widths)):
        study['labels']['width'].append(
            f'{widths[nw]*1e2:3.1f} cm width'
            )
    study['labels']['height'] = []
    for nd in range(len(heights)):
        study['labels']['height'].append(
            f'{heights[nw]*1e2:3.1f} cm height'
            )

    #   Assign values, looping over both variables
    study['fields'] = []
    study['sub_fields'] = []
    study['values'] = []
    study['labels']['case'] = []
    study['labels']['case_tag'] = []
    nc = 0
    for nw in range(len(widths)):
        for nd in range(len(heights)):

            #   Indices
            study['kit']['case'][nw][nd] = nc
            study['kit']['width_case_index'].append(nw)
            study['kit']['height_case_index'].append(nd)
            nc += 1

            #   labels and tags
            study['labels']['case'].append(
                study['labels']['width'][nw]
                + ', '
                + study['labels']['height'][nd]
                )
            study['labels']['case_tag'].append(
                'W' + str(nw)
                + 'D' + str(nd)
                )

            #   For params inputs, put fields, sub-fields,
            #   and values into list of arrays
            study['fields'].append(['cells', 'cells'])
            study['sub_fields'].append(['width', 'height'])
            study['values'].append([widths[nw], heights[nd]])
