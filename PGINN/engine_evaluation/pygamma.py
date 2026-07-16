#!/usr/bin/env python
# coding: utf-8

import os
from pymedphys import gamma
import numpy as np
from mpi4py import MPI

def calc_MSE(ground_truth,prediction):
    return np.mean(np.square(ground_truth - prediction))

def calc_gamma(ground_truth,
               prediction,
               prediction_scaling_factor=1.0,
               matrix_size=(96, 96, 64),
               voxel_dimensions=(2.0, 2.0, 2.5),
               interp_fraction=15,
               random_subset_div=1,
               dose_percent_threshold=3,
               distance_mm_threshold=3,
               lower_percent_dose_cutoff=10,
               local_gamma=False,
               quiet=True
               ):

    reference = ground_truth
    evaluation = prediction * prediction_scaling_factor

    # Define dose grid
    x_axis = np.arange(0, matrix_size[0]) * voxel_dimensions[0]
    y_axis = np.arange(0, matrix_size[1]) * voxel_dimensions[1]
    z_axis = np.arange(0, matrix_size[2]) * voxel_dimensions[2]
    axes = (x_axis, y_axis, z_axis)

    gamma_cal = gamma(
        axes_reference=axes,
        dose_reference=reference,
        axes_evaluation=axes,
        dose_evaluation=evaluation,
        dose_percent_threshold=dose_percent_threshold,
        distance_mm_threshold=distance_mm_threshold,
        lower_percent_dose_cutoff=lower_percent_dose_cutoff,
        interp_fraction=interp_fraction,
        local_gamma=local_gamma,
        ram_available=2**31,
        quiet=quiet,
        max_gamma=1.05,
        random_subset=int(len(reference.flat) // random_subset_div)
    )

    valid_gamma = gamma_cal[~np.isnan(gamma_cal)]
    gamma_pass_rate = np.mean(valid_gamma <= 1) * 100

    if not quiet:
        print(f"\nReference dose: {np.max(reference)}")
        print(f"Evaluation dose: {np.max(evaluation)}")
        print(f"Normalization factor : {prediction_scaling_factor}")
        print('Number of reference points with a valid gamma: {0}'.format( len(valid_gamma)) )
        print(f"Gamma Passing Rate: {gamma_pass_rate:.2f}%")

    return gamma_pass_rate

def calc_gamma_map(ground_truth,
               prediction,
               prediction_scaling_factor=1,
               matrix_size=(96, 96, 64),
               voxel_dimensions=(2.0, 2.0, 2.5),
               interp_fraction=15,
               random_subset_div=1,
               dose_percent_threshold=3,
               distance_mm_threshold=3,
               lower_percent_dose_cutoff=10,
               local_gamma=False,
               quiet=True
               ):

    reference = ground_truth
    evaluation = prediction * prediction_scaling_factor

    x_axis = np.arange(0, matrix_size[0]) * voxel_dimensions[0]
    y_axis = np.arange(0, matrix_size[1]) * voxel_dimensions[1]
    z_axis = np.arange(0, matrix_size[2]) * voxel_dimensions[2]
    axes = (x_axis, y_axis, z_axis)

    gamma_values = gamma(
        axes_reference=axes,
        dose_reference=reference,
        axes_evaluation=axes,
        dose_evaluation=evaluation,
        dose_percent_threshold=dose_percent_threshold,
        distance_mm_threshold=distance_mm_threshold,
        lower_percent_dose_cutoff=lower_percent_dose_cutoff,
        interp_fraction=interp_fraction,
        local_gamma=local_gamma,
        ram_available=2**31,
        quiet=quiet,
        max_gamma=1.05,
        random_subset=int(len(reference.flat) // random_subset_div)
    )

    gamma_values = np.nan_to_num(gamma_values, 0)

    return gamma_values
