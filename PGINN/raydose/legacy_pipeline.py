#!/usr/bin/env python
'''
Aperture Input Composer Aperature Raytracer
---------------------------------------------------------------------------------------------------------------------
Takes CT Images (nrrd), RT Dose/Plan/Structure files to generate input files for networks to learn from/evaluate
CT Image is used for defining the simulation geometry and calculating depth for %DD. Uses nrrd file from 3D Slicer
RT Dose file provides the dose array information, physical coordinate/orientation 
RT Plan file provides the isocenter position, MLC/Gantry/Coll/Table position
RT Structure file is used to define body mask, so only voxels inside the body are active

All these input files are used to generate control point specific MLC Aperture's BEV mapping in the 
patient's body/phantom. Voxels inside the BEV are assigned a value correspoding to hand calculation
and are in units of cGy/MU. Voxels out of the body and those outside are assigned value equal to 0.

Code is written with HFS/Couch-0 orientation in mind. Geometry used for calculation is same as 3D slicer's:
    X : Is patient lateral direction and + X is towards patient left
    Y : Is patient anterior-posterior direction and +Y is towards patient posterior
    Z : Is patient superior-inferior direction and +Z is towards patient superior

May need additional testing/modifications to make use with for non-standard orientation and
add a couch-rotation mechanism to account for non-zero couch rotations to various functions ()
    
Author  :   Ankit Pant
Date    :   Sep 27, 2024
Version :   2.0
'''

import os
from copy import deepcopy
from scipy.interpolate import RegularGridInterpolator
import numpy as np
import numba
from numba import njit, prange

### Axes Rotator
def rotate_axis_legacy(vox_pos, ang_deg, axes):
    '''
    Description
    -----------
    Rotates the specified point relative to isocenter instead of the specified axes (e.g., gantry, or collimator)

    Parameters
    ----------
    vox_pos :   List of float values with (3,) or (2,) dimensions
                Voxel coordinate (3D-coordinate for gantry and 2D for collimator).
    ang_deg :   Float value
                Gantry/Collimator Rotation in Degree
    axes :      Boolean (int)
                Axes value, 0 is gantry and 1 is collimator

    Returns
    -------
    vox_pos :   List of float values with (3,) or (2,) dimensions
                New position of rotated voxel 
    '''
 
    if axes == 0:
        ang_rad  = np.deg2rad(360 - ang_deg)  
        
        rot_mat = np.round(np.single([
            [np.cos(ang_rad), -np.sin(ang_rad), 0], 
            [np.sin(ang_rad), np.cos(ang_rad), 0], [0, 0, 1]
        ]),decimals=5) 
        
        vox_pos = np.matmul(rot_mat, vox_pos) 
        
    elif axes == 1:
        ang_rad  = np.deg2rad(360-ang_deg)  
        
        rot_mat = np.round(np.single([
            [np.cos(ang_rad), -np.sin(ang_rad)],
            [np.sin(ang_rad), np.cos(ang_rad)]
        ]),decimals=5) 
        
        vox_pos = np.matmul(rot_mat, vox_pos)
        
    else: 
        print('Invalid rotation axes specified')
    
    return vox_pos

### Voxel Collimation Check
def check_vox_coll_legacy(cp_config, x, z):
    '''
    Description
    -----------
    Check if the voxel is collimated by jaws and MLC leaves on the way to source through isoplane.
    Locates the MLC leaf pair(s) close to point on the isocenter plane and check if collimators 
    are in the BEV as defined by Jaws and MLC

    Parameters
    ----------
    cp_config : Structured array (vector) 
                Contains the control point configuration parameters (see cce function) 
    x_pos :     float
                Position of x coordinate in isocenter plane
    z_pos :     float
                Position of z coordinate in isocenter plane

    Returns
    -------
    VB : Boolean (int)
         Return 0 if not in BEV and 1 if in BEV

    '''
    
    # Jaws/MLC Positions
    jaw_xp = np.array(cp_config['x_jaw'], 'single')    # X Jaw Positions
    jaw_yp = np.array(cp_config['y_jaw'], 'single')    # Y Jaw Position
    mlc_xp = np.array(cp_config['x_mlc'], 'single')    # MLC X Positions
    mlc_zp = np.array(cp_config['z_mlc'], 'single')    # MLC Pair Z Positions

    # Jaws Check
    jaw_xc = np.logical_and(jaw_xp[0] < x, x < jaw_xp[1]) * 1
    jaw_yc = np.logical_and(jaw_yp[0] < z, z < jaw_yp[1]) * 1
    jaw_check = np.logical_and(jaw_xc, jaw_yc)
    
    # MLC Check
    z_min = ((mlc_zp <= z[:, None]) * 1).sum(axis=1) - 1                   # Find the lower edge of the MLC leaf-pair given the z-value
    z_out = np.abs(z) <= 110                                               # Check within max. MLC field
    z_edg = mlc_zp[z_min] == z                                             # Check if z-value on the edge
    
    cz_min = np.clip(z_min, 0, len(mlc_zp) - 2)
    x_opn = np.logical_and(mlc_xp[cz_min] < x, x < mlc_xp[cz_min + 60])        # Check if in open field in x-dir
    x_edg = np.logical_and(mlc_xp[z_min - 1] < x, x < mlc_xp[z_min + 60 - 1])  # Check on x-dir if on edge
    
    mlc_xc = np.where(z_out & z_edg, np.logical_and(x_edg, x_opn), x_opn)
    mlc_check = np.where(np.abs(z) > 110, 0, mlc_xc)
    
    vox_bool = np.logical_and(jaw_check, mlc_check).astype(int)
       
    return vox_bool

### Voxel Iso-Plane Raytracer (aperture mask generator)
def create_aperture_mask_legacy(cp_config, proc_data):
    '''
    Take a Control point Aperture (in form CCC), and identifes voxels that are in 
    BEV. RDF is used initialize dose array with Body Mask used for select body voxels.
    
    Create an aperture mask

    Parameters
    ----------
    cp_config :      List 
                     Contains the control point configuration parameters (see cce function) 
    proc_data :      Dictionary
                     RT Dose DICOM  loaded using pydicom pkg
    body_mask :      3D Binary Integer Array
                     Contains the Body Mask

    Returns
    -------
    aperture_mask : 3D Array (float value)
                    Contain the BEV Mask with active voxel have Equivalent square area value, otherwise 0.

    '''
    
    # Array Position and Spacing
    iso_ipp =  proc_data['img_pos_pat'] - cp_config['isocenter']
        # iso_ipp      :  Isocenter zero-ed img_pos_pat coordinates. Sets the origin to Tx. isocenter instead of DICOM origin
        
    # Mask Initializers + ES Calc.    
    x_indices = np.arange(proc_data['body_mask'].shape[0])
    y_indices = np.arange(proc_data['body_mask'].shape[1])
    z_indices = np.arange(proc_data['body_mask'].shape[2])
    
    x_mesh, y_mesh, z_mesh = np.meshgrid(x_indices, y_indices, z_indices, indexing='ij')
    
    # Voxel Position
    vox_pos = np.array([
        iso_ipp[0] + x_mesh * proc_data['dose_grid'][0],
        iso_ipp[1] + y_mesh * proc_data['dose_grid'][1], 
        iso_ipp[2] + z_mesh * proc_data['dose_grid'][2]
    ]).reshape(3, -1)    
    
    # Gantry Rotated Voxel Position
    gantry_rot = rotate_axis_legacy(vox_pos, cp_config['gantry_angle'], 0)

    # Plane-Line Intersection
    # src_vector = 1000 * [0, -1, 0]                # Source Vector 
    # point_vec = gantry_rot - src_vec              # Path Vector
    # line = src_vec + t_param * point_vec          # Line Vector, where t is a parameter
    # -y = 0                                        # Equation of plane (fixed gantry but point rotated)
    t = 1000 / (gantry_rot[1,:] + 1000)             # Solution to parameter by plugging line (parametric) equation into plane equation

    # Intersection Point Coordinate
    #IP = [0, 0, 0]      # Initialized list
    #IP[0] = GRvP[0] * t  # x-value of IP
    #IP[1] = 0           # y-value = 0
    #IP[2] = GRvP[2] * t  # z-value of IP                                                                           
    
    # Voxel Iso-Plane Position
    isoplane_pos = (np.c_[gantry_rot[0,:] * t, gantry_rot[2,:] * t]).T

    # Rotate Point for Collimator Rotation 
    collimator_rot = rotate_axis_legacy(isoplane_pos, cp_config['collimator_angle'], 1)

    active_voxel = check_vox_coll_legacy(cp_config, collimator_rot[0,:], collimator_rot[1,:]).reshape(proc_data['body_mask'].shape)
    aperture_mask = (active_voxel * proc_data['body_mask']).astype('int8')
    
    return aperture_mask


#%% SSD Calculator
def calculate_ssd_legacy(cp_config, proc_data):
    '''
    Description
    -----------
    Calculates SSD for a given control point.
    
    Parameters
    ----------
    cp_config :     Structured Array 
                    Contains the control point configuration parameters (see control_p function) 

    rt_dose :       DICOM Dataset
                    RT Dose DICOM metadata loaded with pydicom 

    body_mask :     3D Binary Integer Array
                    Contains the Body Mask

    Returns
    -------
    cp_ssd :        Float
                    Calculate the SSD along the path from gantry to isocenter point for given control point


    '''
    # Array Position and Spacing
    iso_ipp = proc_data['img_pos_pat'] - cp_config['isocenter']
        # iso_ipp      :  Isocenter zero-ed img_pos_pat coordinates. Sets the origin to Tx. isocenter instead of DICOM origin
    
    ang_rad  = np.deg2rad(cp_config['gantry_angle'])
    src_path_vector = 1000 * np.round(np.single([np.sin(ang_rad),  -np.cos(ang_rad),  0]), decimals=5)
    
    
    x_grid = np.arange(proc_data['body_mask'].shape[0])
    y_grid = np.arange(proc_data['body_mask'].shape[1])
    z_grid = np.arange(proc_data['body_mask'].shape[2])
    interpolator = RegularGridInterpolator((x_grid, y_grid, z_grid), proc_data['body_mask'], bounds_error=False, fill_value=0)
    

    vox_pos = np.array([0, 0, 0])                                  # SSD wrt isocenter point and gantry 
    
    vox_src_vec = src_path_vector - vox_pos                        # Voxel to Gantry Vector
    unit_vec = vox_src_vec / np.linalg.norm(vox_src_vec)           # Unit Path Vector (Voxel to Gantry)
    
    t = 0                                                          # Parameter t (used for moving along uPV)                
    in_body = 1                                                    # Initialize INB. INB >  0.3 if PVt inside/on Body     
    
    while in_body > 0.3:   
        path_vec = vox_pos + t * unit_vec                          # Path Vector to Gantry/Source
        grid_pos = (path_vec - iso_ipp) / proc_data['dose_grid']   # Grid equivalent position
        #grid_pos_int = np.array(grid_pos, "int32")
        
        in_body = interpolator(grid_pos)                           # Interpolate for GEP on body_mask array
        #in_body = body_mask[int(grid_pos[0]), int(grid_pos[1]), int(grid_pos[2])]
        
        t += 1                                                     # Advance to next position (1mm increment)
    
    path_vec_final = path_vec - unit_vec                           # Goes back to position where still inside/on body
    cp_ssd = np.linalg.norm(src_path_vector - path_vec_final)      # Calculate SSD at the current positon
    
    return cp_ssd

#%% Apeture Specific Dose Function and Voxel Dose Calculator
@njit(parallel=True)
def compute_voxel_dose_legacy(
    voxel_positions, unit_path_vectors, source_vector, 
    iso_ipp, dose_grid, 
    aperture_mask, body_mask, density_array, 
    ssd, ssd_ref, ref_depth, setup_factor, max_pdd_depth,
    pdd_depths, pdd_values, oar_values):
    '''
    Description
    -----------
    Calculate various parameters (e.g., PDD, Meyneord's Factor) to do dose calculation for active voxels.
                                  
    Parameters
    ----------
    voxel_positions :   Vector Array
                        Contains the isocenter point zeroed voxel position coordinates (vector)
    
    unit_path_vectors : Vector Array
                        Contains the unit vectors that from voxel of interest (VPA) to Source
    
    source_vector :     Vector Array
                        Contains the unit vectors that from voxel of interest (VPA) to Source
    
    iso_ipp :           Vector (numpy list)
                        Contains the isocenter point zeroed Image Position Coordinate
    
    dose_grid :         Vector (numpy list)
                        Dose grid size vector
    
    aperture_mask :     3D Array
                        Mask array defining Active Voxel (from vipr)
    
    body_mask :         3D Array
                        Body Mask array defining where the body is in
    
    density_array :     3D Array (float)
                        Mapped Density Array
    
    ssd :               Float
                        Source to Surface Distance
    
    ssd_ref :           Float
                        Reference source to Surface Distance
    
    ref_depth :         Float
                        Reference depth for calibration
    
    setup_factor :      Float
                        SSD/SAD setup factor
    
    max_pdd_depth :     Float
                        Max depth with defined PDD value
    
    pdd_depths :        Vector (numpy list)
                        Depth value associated with PDD 
    
    pdd_values :        Vector (numpy list)
                        Percent Depth Dose values
   
    oar_values :        Vector (numpy list)
                        Computed off-axis ratio values for active voxel in aperture mask 
                        
    Returns
    -------
    aperture_dose :     3D Array (float value)
                        Contains the dose value for the apeture of interest (cGy/MU)       
    '''
    
    aperture_mask_shape = aperture_mask.shape
        
    aperture_dose = np.zeros(aperture_mask_shape, dtype=np.float32)
    eps = np.float32(1e-15)
    
    for i in prange(len(voxel_positions)):
        vox_pos = voxel_positions[i]
        unit_vec = unit_path_vectors[i]
        
        t = np.float32(0.0)
        water_equivalent_depth = 0
        
        while True:
            path_vect_1 = vox_pos + (t + np.float32(1.0)) * unit_vec
            path_vect_2 = vox_pos +  t * unit_vec
            
            grid_pos1 = (path_vect_1 - iso_ipp) / dose_grid
            grid_pos2 = np.rint((path_vect_2 - iso_ipp) / dose_grid + eps)
            
            water_equivalent_depth = water_equivalent_depth + density_array[int(grid_pos2[0]), int(grid_pos2[1]), int(grid_pos2[2])]
            
            if not (0 <= grid_pos1[0] < aperture_mask_shape[0] and 0 <= grid_pos1[1] < aperture_mask_shape[1] and 0 <= grid_pos1[2] < aperture_mask_shape[2]):
                break
            if body_mask[int(grid_pos1[0]), int(grid_pos1[1]), int(grid_pos1[2])] <= 0:
                break
            t += np.float32(1.0)
        # Meyneord's Factor
        mayneord_factor1 = (((ssd + ref_depth)/(ssd_ref + ref_depth)) ** 2)
        mayneord_factor2 = (((ssd_ref + water_equivalent_depth) / (ssd + water_equivalent_depth)) **2)   
        mayneord_factor = mayneord_factor1 * mayneord_factor2
        
        if 0.0 <= water_equivalent_depth <= max_pdd_depth:
            pdd_eval = np.interp(water_equivalent_depth, pdd_depths, pdd_values) * mayneord_factor * setup_factor
        else:
            pdd_eval = 0.0
        
        oar_eval = oar_values[i]
        
        voxel_dose = pdd_eval * oar_eval

        vox_xidx = int((vox_pos[0] - iso_ipp[0]) / dose_grid[0])
        vox_yidx = int((vox_pos[1] - iso_ipp[1]) / dose_grid[1])
        vox_zidx = int((vox_pos[2] - iso_ipp[2]) / dose_grid[2])
        
        aperture_dose[vox_xidx, vox_yidx, vox_zidx] = voxel_dose

    return aperture_dose

def compute_aperture_dose_legacy(aperture_mask, cp_config, proc_data, ssd):
    '''
    Description
    -----------
    Initalizes the compute_voxel_dose() function by calculating and storing various parameters in memory.
    Parameters details provided unique to compute_aperture_dose() otherwise see compute_voxel_dose()
    
    Parameters
    ----------
    cp_config :     Structured Array 
                    Contains the control point configuration parameters (see configuration_extractor function) 
                    
    rt_dose :       DICOM Dataset
                    RT Dose DICOM metadata loaded with pydicom 
    
    pdd_list :      Numpy Array
                    Contains depth (1st col), 6x/6fff/10x/10fff PDDs (2/3/4/5 cols)
    
    oar_list :      Numpy Array
                    Contains distance (1st col), 6x/6fff/10x/10fff OARs (2/3/4/5 cols)
    Returns
    -------
    aperture_dose : 3D Array (float value)
                    Contains the dose value for the apeture of interest (cGy/MU)   
    '''
    # Array Position and Spacing
    iso_ipp = proc_data['img_pos_pat'] - cp_config['isocenter']
        # iso_ipp      :  Isocenter zero-ed img_pos_pat coordinates. Sets the origin to Tx. isocenter instead of DICOM origin
    print(f'isoipp: {iso_ipp}')

    # Calculate Source Position Vector
    ang_rad = np.deg2rad(cp_config['gantry_angle'])
    source_vector = 1000.0 * np.round(np.array([np.sin(ang_rad), -np.cos(ang_rad), 0], dtype=np.float32), decimals=5)     # Source Position Vector

    # Get active voxel indices within or on the body/aperture
    active_x, active_y, active_z = np.where(aperture_mask)

    # Compute voxel positions and unit path vectors
    dose_grid = proc_data['dose_grid']
    voxel_positions = np.array(
        [iso_ipp[0] + active_x * dose_grid[0], 
         iso_ipp[1] + active_y * dose_grid[1], 
         iso_ipp[2] + active_z * dose_grid[2]]).T
    voxel_to_source_vectors = source_vector - voxel_positions
    unit_path_vectors = voxel_to_source_vectors / np.linalg.norm(voxel_to_source_vectors, axis=1, keepdims=True)
    
    # Provide parameters for dose calc
    ref_depth = proc_data['ref_depth']
    ssd_ref = proc_data['ssd_ref']                                                  # PDD measured at SSD = 100cm)
    setup_factor = np.float32(((ssd_ref + ref_depth)/(ssd + ref_depth)) ** 2)       # SSD calibration > SAD treatment
    
    # Store PDD
    max_pdd_depth = proc_data['max_pdd_depth']
    pdd_values = proc_data['pdd_values']
    pdd_depths = proc_data['pdd_depths']
       
    # Compute OAR
    oar_dist = proc_data['oar_dist']
    oar_values = proc_data['oar_values']
    
    off_axix_dist = np.linalg.norm(np.cross(voxel_positions, source_vector), axis=1) / np.linalg.norm(source_vector)
    
    vox_oar_pos = np.interp(off_axix_dist, oar_dist, oar_values, left=0, right=0)
    vox_oar_neg = np.interp(-off_axix_dist, oar_dist, oar_values, left=0, right=0)
    vox_oar_values = np.float32(((vox_oar_pos + vox_oar_neg) / 2))
    
    # Store variable
    body_mask = proc_data['body_mask']
    density_array = proc_data['density_array']
    
    # Compute SSD
    aperture_dose = compute_voxel_dose_legacy(
        voxel_positions, unit_path_vectors, source_vector, 
        iso_ipp, dose_grid, 
        aperture_mask, body_mask, density_array, 
        ssd, ssd_ref, ref_depth, setup_factor, max_pdd_depth,
        pdd_depths, pdd_values, vox_oar_values)

    return aperture_dose


def crop_matrix_legacy(cp_config, proc_data, inp_array, crop_dim=(96, 96, 64)):
    """
    Crops a 3D array (matrix) around the isocenter based on the given crop dimensions.

    The cropping is done relative to the isocenter position in the patient coordinate system, ensuring that the output 
    matrix is centered on the isocenter. If the crop dimensions extend beyond the boundaries of the input array, 
    adjustments are made to ensure the cropped array remains within bounds.

    Parameters
    ----------
    cp_config : structured array/dict
        A dictionary containing the control point configuration. Must contain the key `'isocenter'`, which is the 3D position 
        of the isocenter in the patient coordinate system.
    
    proc_data : dict
        A dictionary containing patient and image data. Must contain the keys:
        - `'img_pos_pat'`: 3D position of the image in the patient coordinate system.
        - `'dose_grid'`: Voxel size in the dose grid (3D resolution).
        - `'body_mask'`: A 3D array representing the body mask used to determine the array shape.
    
    inp_array : numpy.ndarray
        A 3D input array (matrix) to be cropped. This could be a dose distribution or any other 3D medical imaging data.
    
    crop_dim : tuple of int
        A tuple specifying the size of the crop window in each of the three dimensions.
        iDoTA uses (96, 96, 64), for Depth (y), Width (x,), and Height (z)
    
    Returns
    -------
    crop_array : numpy.ndarray
        A 3D array that is cropped around the isocenter. The size of the cropped array is defined by the `crop_dim` parameter, 
        with adjustments made to ensure the cropped region stays within the boundaries of the original array.

    Notes
    -----
    - The function handles out-of-bounds cropping by adjusting the start and end indices if the crop window exceeds the 
      input array's dimensions.
    - The cropping is done relative to the isocenter position, which is computed in voxel coordinates based on the patient 
      position and dose grid spacing.
    
    Example
    -------
    >>> crop_dim = (64, 64, 64)
    >>> cropped_matrix = crop_matrix(cp_config, proc_data, dose_array, crop_dim)
    """
    
    eps = 1e-15
    iso_ipp =  proc_data['img_pos_pat'] - cp_config['isocenter']
    iso_vox =  np.rint((-1 * iso_ipp) / proc_data['dose_grid'] + eps)
    arr_shape = np.array(np.shape(proc_data['body_mask']))
    
    crop_dim = np.array(crop_dim, dtype='i4')
    
    idx_start = iso_vox - (crop_dim // 2)
    idx_end = iso_vox + (crop_dim // 2)
    
    oob_start = idx_start * (1 * idx_start < 0)
    idx_start = idx_start - oob_start
    idx_end = idx_end - oob_start
    
    oob_end = (arr_shape - idx_end ) * ( ((arr_shape - idx_end) < 0 ) * 1)
    idx_start = ((idx_start + oob_end).astype('i4')).tolist()
    idx_end = ((idx_end + oob_end).astype('i4')).tolist()
    
    crop_array = inp_array[
        idx_start[0]:idx_end[0],
        idx_start[1]:idx_end[1],
        idx_start[2]:idx_end[2]]
    
    return crop_array

#%% Array/Matrix Rotator Code
def rotate_matrix_legacy(cp_config, inp_array):
    """
    Rotates a 2D or 3D array (matrix) based on the gantry angle of the control point configuration.

    The rotation is performed using 90-degree steps, with the number of 90-degree rotations determined by the gantry angle.
    This function is typically used to adjust medical imaging data or dose distributions to account for different gantry angles.

    Parameters
    ----------
    cp_config : structured array/dict
        A dictionary containing the control point configuration. Must contain the key `'gantry_angle'`, which specifies 
        the gantry angle in degrees.
    
    inp_array : numpy.ndarray
        A 2D or 3D input array to be rotated. This could be a dose distribution or any other 2D/3D medical imaging data.
    
    Returns
    -------
    rot_array : numpy.ndarray
        A rotated version of the input array. The rotation is done in multiples of 90 degrees, depending on the gantry angle.
    
    Notes
    -----
    - The gantry angle determines the number of 90-degree rotations:
        - No rotation if `0 <= gantry_angle < 45` or `315 <= gantry_angle <= 360`.
        - Rotate +90 degrees if `45 <= gantry_angle < 135`.
        - Rotate 180 degrees if `135 <= gantry_angle < 225`.
        - Rotate -90 degrees if `225 <= gantry_angle < 315`.
    
    Example
    -------
    >>> rotated_matrix = rotate_matrix(cp_config, dose_array)
    """
    
    g_pos = cp_config['gantry_angle']
    
    num_rot90_nr = 0
    num_rot90_n1 = ((45 <= g_pos) and (g_pos < 135)) * 1
    num_rot90_r2 = ((135 <= g_pos) and (g_pos < 225)) *  -2
    num_rot90_p1 = ((225 <= g_pos) and (g_pos < 315)) *  -1
    
    num_rot_90 = num_rot90_nr + num_rot90_n1 + num_rot90_p1 + num_rot90_r2
    
    rot_array = np.rot90(inp_array, num_rot_90)
    
    return rot_array

#%% Data Stitcher
def stitch_network_data_legacy(acp_config, proc_data, density_array, order=(1,0,2)):
    """
    Combines the raytracing (aperture specific dose), CT volume (mapped HU dose array), and ground truth dose (via MC)
    into one file.
    
    Additionally processes the input arrays so they are cropped and/or rotated (to be between ~+/- 45 deg)

    The rotation is performed using 90-degree steps, with the number of 90-degree rotations determined by the gantry angle.
    This function is typically used to adjust medical imaging data or dose distributions to account for different gantry angles.

    Parameters
    ----------
    cp_config : structured array/dict
        A dictionary containing the control point configuration. Must contain the key `'gantry_angle'`, which specifies 
        the gantry angle in degrees.
    
    proc_data : dictionary
        A dictionary containing processed patient/treatement geometry data, see input_data_loader.py
    
    order : tuple
        Specifiy the order of a rows, column, slices, changes from 3D slicer orientation. Default value set to (1,0,2). 
        
    Returns
    -------
    None
    """
    
    den_vol = density_array
    cp_indices = range(1, len(acp_config))

    use_cp_indices = proc_data['control_points'] or cp_indices

    for cp in use_cp_indices:
        
        cp_config_a = deepcopy(acp_config[cp - 1])
        cp_config = deepcopy(acp_config[cp])
        
        if cp_config['cp_idx'] == 0:
            continue

        cp_config['isocenter'] = cp_config_a['isocenter']
        
        #arc_name_corr = str(int(cp_config['arc_name'].astype('U20')) - 2)

        ray_name = proc_data['pat_name'] + '_arc' + cp_config['arc_name'].astype('U20') + '_CP' + str(cp_config['cp_idx']) + '_RTD'
        
        dose_name = proc_data['pat_name'] + '_arc' + cp_config['arc_name'].astype('U20') + '_CP' + str(cp_config['cp_idx']) + '_DTD'
        #arc_name_corr = str(int(cp_config['arc_name'].astype('U20')) - 2)
        #dose_name = proc_data['pat_name'] + '_arc' + arc_name_corr + '_CP' + str(cp_config['cp_idx']) + '_DTD'

        train_name = proc_data['pat_name'] + '_arc' + cp_config['arc_name'].astype('U20') + '_CP' + str(cp_config['cp_idx']) + '_STD'
        
        save_dir = proc_data['save_dir']
        ray_file_name = os.path.join(save_dir, ray_name + '.npz')
        train_data_file = os.path.join(save_dir, train_name + '.npz')

        load_dir = proc_data['dtd_load_dir']
        dose_file_name = os.path.join(load_dir, dose_name + '.npz')

        if os.path.exists(dose_file_name) == False:
            print(f"{dose_file_name}: Dose not found")
            continue

        vol_p = den_vol
        ray_p  = np.load(ray_file_name)['ray']
        dose_p = np.load(dose_file_name)['dose']

        # Crop
        x_res, y_res, z_res = proc_data['crop_vol']

        if proc_data['crop']:
            vol_p  = crop_matrix_legacy(cp_config, proc_data, den_vol, crop_dim=(x_res, y_res, z_res))
            ray_p  = crop_matrix_legacy(cp_config, proc_data, ray_p, crop_dim=(x_res, y_res, z_res))
            dose_p = crop_matrix_legacy(cp_config, proc_data, dose_p, crop_dim=(x_res, y_res, z_res))
            
        # Rotate Array
        # vol_p  = rotate_matrix_legacy(cp_config, vol_p)
        # ray_p  = rotate_matrix_legacy(cp_config, ray_p)
        # dose_p = rotate_matrix_legacy(cp_config, dose_p)
        
        # # Transpose
        # vol_p  = np.transpose(vol_p, order)
        # ray_p  = np.transpose(ray_p, order)
        # dose_p = np.transpose(dose_p, order)
        
        
        print(f"\nStitching data for {train_name}")
        print(f"Max dose in ground truth: {np.max(dose_p)}")
        print(f"Max dose in ray trace data: {np.max(ray_p)}")
        
        np.savez_compressed(train_data_file, 
                            vol=vol_p, 
                            ray=ray_p, 
                            dose=dose_p,
                            bid=train_name,
                            gap=cp_config['gantry_angle'])
        
    return None


def compose_aperture_inputs_legacy(acp_config, proc_data, density_array):
    '''
    Parallelized version of the dose calculator for various apertures using mpi4py.
    Combines two aperture dose calculations to create a control point dose distribution.
    Dose distributions for all control points are saved in individual compressed files (.npz).

    Parameters
    ----------
    acp_config :     Structured Array
                    Contains control point information.
    proc_data :     Dictionary
                    Processed data dictionary, see input_data_loader.py

    Returns
    -------
    None.
    '''

    # Divide the control points among the available processes
    total_cps = len(acp_config)
    cp_indices = list(range(1, total_cps))  # Control point indices (skip first one for pair processing)

    use_cp_indices = proc_data['control_points'] or cp_indices

    for cp in use_cp_indices:

        cp_config_a = deepcopy(acp_config[cp - 1])
        cp_config_b = deepcopy(acp_config[cp])

        cp_config_b['isocenter'] = cp_config_a['isocenter']
        cp_config_b['collimator_angle'] = cp_config_a['collimator_angle']
        
        if cp_config_b['cp_idx'] == 0:
            continue
        
        cp_name_a = proc_data['pat_name'] + '_arc' + cp_config_a['arc_name'].astype('U20') + '_CP' + str(cp_config_a['cp_idx'])

        dens = proc_data['density_array']

        ap_mask_a = create_aperture_mask_legacy(cp_config_a, proc_data)
        ap_mask_b = create_aperture_mask_legacy(cp_config_b, proc_data)

        ssd_a = calculate_ssd_legacy(cp_config_a, proc_data)
        ssd_b = calculate_ssd_legacy(cp_config_b, proc_data)

        dose_a = compute_aperture_dose_legacy(ap_mask_a, cp_config_a, proc_data, ssd_a)
        dose_b = compute_aperture_dose_legacy(ap_mask_b, cp_config_b, proc_data, ssd_b)

        dose = (dose_a + dose_b) / 2

        # dose = ap_mask_a

        print(f"\nGenerating input for control point number {cp_config_b['cp_idx']}")
        
        print(f"Isocenter of Arc {cp_config_a['arc_name']} and CP {cp_config_a['cp_idx']}: {cp_config_a['isocenter']}")
        print(f"Isocenter of Arc {cp_config_a['arc_name']} and CP {cp_config_b['cp_idx']}: {cp_config_b['isocenter']}")
        
        print(f"Collimator Angle of Arc {cp_config_a['arc_name']} and CP {cp_config_a['cp_idx']}: {cp_config_a['collimator_angle'] }")
        print(f"Collimator Angle of Arc {cp_config_a['arc_name']} and CP {cp_config_b['cp_idx']}: {cp_config_b['collimator_angle']}")
        
        print(f"Max dose in Arc {cp_config_a['arc_name']} and CP {cp_config_a['cp_idx']}: {np.max(dose_a)}")
        print(f"Max dose in Arc {cp_config_b['arc_name']} and CP {cp_config_b['cp_idx']}: {np.max(dose_b)}")
        print(f"Max dose in CP Beam {cp_config_b['cp_idx']}: {np.max(dose)}")
        print(f"Max ray-dose: {np.max(dose)}")
        #sys.exit()
    
        # Construct beam file name
        beam_name = proc_data['pat_name'] + '_arc' + cp_config_b['arc_name'].astype('U20') + '_CP' + str(cp_config_b['cp_idx'])
        save_dir = proc_data['save_dir']
        save_file_name = os.path.join(save_dir, beam_name + '_RTD.npz')
        
        # Save dose distribution to file (this happens in parallel)
        np.savez_compressed(save_file_name, 
                            ray=dose, 
                            bid=cp_config_b['beam_idx'])
        
        
    
    print("Stitching data together for the control point")
    stitch_network_data_legacy(acp_config, proc_data, density_array, (1,0,2))
