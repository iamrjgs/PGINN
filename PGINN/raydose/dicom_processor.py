import pydicom
import numpy as np
import os
from skimage.draw import polygon2mask
import SimpleITK as sitk
import cv2

def load_ct_as_sitk(file_or_folder_path):
    # Load CT as NRRD file
    if file_or_folder_path.endswith('.nrrd') and not os.path.isdir(file_or_folder_path):
        return sitk.ReadImage(file_or_folder_path)

    # Load CT DICOM series fallback
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(file_or_folder_path)
    if not series_ids:
        raise ValueError(f"No DICOM series found in {file_or_folder_path}")
    for sid in series_ids:
        file_names = reader.GetGDCMSeriesFileNames(file_or_folder_path, sid)
        test_reader = sitk.ImageFileReader()
        test_reader.SetFileName(file_names[0])
        test_reader.ReadImageInformation()
        modality = test_reader.GetMetaData("0008|0060")
        if modality == "CT":
            reader.SetFileNames(file_names)
            image = reader.Execute()
            return image
    raise ValueError("No CT series found in folder.")

def get_dose_array_from_dicom_dose(rt_dose):
    if isinstance(rt_dose, str):
        rt_dose = pydicom.dcmread(rt_dose)
    
    dose_array = rt_dose.pixel_array.astype(np.float32) * rt_dose.DoseGridScaling
    
    return np.transpose(dose_array, (2,1,0))

def substitute_dose_array_in_dicom_dose(rt_dose,
                                        new_dose_array,
                                        generate_new_uid=False
                                        ):
    """
    Replace the dose array in an RT Dose DICOM dataset with a new one.
    
    Parameters
    ----------
    rt_dose : pydicom.dataset.FileDataset
        The RT Dose DICOM dataset
    new_dose : np.ndarray
        New dose array in Gy, same shape as the existing dose grid.
    
    Returns
    -------
    ds : pydicom.dataset.FileDataset
        The modified dataset with updated PixelData and DoseGridScaling.
    
    Raises
    ------
    ValueError
        If the new array shape does not match the existing dose grid.
    """
    if isinstance(rt_dose, str):
        rt_dose = pydicom.dcmread(rt_dose)

    old_dose_array = get_dose_array_from_dicom_dose(rt_dose)
    old_shape = old_dose_array.shape

    new_rt_dose = rt_dose.copy()
    bits = rt_dose.BitsAllocated
    rows = new_rt_dose.Rows
    cols = new_rt_dose.Columns
    frames = int(getattr(new_rt_dose, "NumberOfFrames", 1))
    samples = new_rt_dose.SamplesPerPixel
    bytes_per_sample = new_rt_dose.BitsAllocated // 8
    dtype = np.uint16 if bits == 16 else np.uint32
    expected_shape = (cols, rows, frames) if frames > 1 else (cols, rows)

    if new_dose_array.shape != expected_shape:
        raise ValueError(
            f"Shape mismatch: new {new_dose_array.shape}, expected {expected_shape}"
        )

    max_val = float(np.max(new_dose_array))
    scaling = 1.0 if max_val == 0 else max_val / np.iinfo(dtype).max

    max_int = np.iinfo(dtype).max
    effective_max = max_int * 0.95
    scaling = max(max_val / effective_max, 1e-12)
    scaled = new_dose_array / scaling
    scaled = np.clip(scaled, 0, max_int)

    new_pixel_array = np.round(scaled).astype(dtype)

    new_pixel_array = np.transpose(new_pixel_array, (2,1,0))
    
    expected_bytes = rows * cols * frames * samples * bytes_per_sample
    actual_bytes = new_pixel_array.nbytes
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"PixelData length mismatch: expected {expected_bytes}, got {actual_bytes}. "
            f"Check dtype and shape."
        )
    
    new_rt_dose.PixelData = np.ascontiguousarray(new_pixel_array).tobytes()
    new_rt_dose.DoseGridScaling = float(scaling)

    if generate_new_uid:
        series_instance_uid = pydicom.uid.generate_uid()

        new_rt_dose.SOPInstanceUID = sop_instance_uid
        new_rt_dose.SeriesInstanceUID = series_instance_uid
        new_rt_dose.InstanceNumber = 1
    
    return new_rt_dose

def get_geometry_from_dicom_dose(rt_dose):
    if isinstance(rt_dose, str):
        rt_dose = pydicom.dcmread(rt_dose)

    origin = np.array(rt_dose.ImagePositionPatient, dtype=float)

    row_spacing, col_spacing = map(np.float32, rt_dose.PixelSpacing)

    if hasattr(rt_dose, "GridFrameOffsetVector"):
        slice_spacing = np.float32(abs(np.mean(np.diff(rt_dose.GridFrameOffsetVector))))
    else:
        # Fallback: use SliceThickness if available
        slice_spacing = np.float32(getattr(rt_dose, "SliceThickness", 1.0))

    iop = np.array(rt_dose.ImageOrientationPatient, dtype=float).reshape(2, 3)
    row_dir = iop[0]
    col_dir = iop[1]
    slice_dir = np.cross(row_dir, col_dir)

    dirs = np.vstack([row_dir, col_dir, slice_dir])

    spacing = np.array([col_spacing, row_spacing, slice_spacing], dtype=float)

    num_cols = int(rt_dose.Columns)
    num_rows = int(rt_dose.Rows)
    num_slices = int(getattr(rt_dose, "NumberOfFrames", 1))
    shape = (num_cols, num_rows, num_slices)

    return origin, spacing, shape, dirs

def get_beam_indexes_types(rt_plan):
    """
    Extracts and returns beam information from a radiotherapy (RT) plan.

    This function processes a DICOM RT Plan dataset to extract specific details about each beam
    in the plan, including the beam index, control point index, treatment machine name, and 
    number of control points.

    Parameters:
    -----------
    rt_plan : pydicom.Dataset
        A DICOM dataset representing the RT Plan, containing beam-related information.

    Returns:
    --------
    beam_info : list of lists
        A list where each element is a list containing the following information for each beam:
        - Beam index (int)
        - Control point index (int)
        - Treatment machine name (str)
        - Number of control points (int)

    Example:
    --------
    >>> beam_info = get_beam_indexes_types(rt_plan)
    >>> print(beam_info)
    [[0, 1, 'LINAC', 90], [1, 2, 'LINAC', 95], ...]
    """

    beam_info = []
    num_beams = len(rt_plan[0x300a,0xb0].value)
    for beam in range(num_beams):
        beam_info.append([
            beam,                                                      # Beam Number/Python Index
            str(rt_plan[0x300a,0xb0][beam][0x300a,0xc2].value),        # Beam Name
            str(rt_plan[0x300a,0xb0][beam][0x300a,0xce].value),        # Treatment?
            int(rt_plan[0x300a,0xb0][beam][0x300a,0x0110].value)])     # Number of Control Points
    return beam_info

def get_metersets(rt_plan, arc_index):
    """
    Extracts and returns the meterset weights for a specified arc from a radiotherapy (RT) plan.

    This function processes a DICOM RT Plan dataset to extract the meterset weight for each control 
    point within a specified arc. Meterset weights are typically used to define the amount of radiation 
    delivered at each control point during treatment.

    Parameters:
    -----------
    rt_plan : pydicom.Dataset
        A DICOM dataset representing the RT Plan, containing information about beams and control points.
    arc_index : int
        The index of the arc (beam) for which meterset weights need to be extracted. 
        This corresponds to the index of the beam in the RT Plan dataset.

    Returns:
    --------
    metersets : list of float
        A list of meterset weights for each control point in the specified arc.

    Example:
    --------
    >>> metersets = get_metersets(rt_plan, 0)
    >>> print(metersets)
    [0.0, 1.25, 2.5, ..., 100.0]

    Notes:
    ------
    - The function uses the DICOM tag (300A, 00B0) to access the beam sequence, 
      (300A, 0110) to get the number of control points, 
      (300A, 0111) for the control point sequence, and 
      (300A, 0134) for the meterset weight.
    """
    if isinstance(rt_plan, str):
        rt_plan = pydicom.dcmread(rt_plan)

    beam_sequence = rt_plan.BeamSequence
    beam = [b for b in beam_sequence if str(b.BeamName) == str(arc_index)][0]
    n_cps = int(beam[0x300a,0x110].value)
    n_cps_i = range(1,n_cps+1)

    metersets = []
    for con_pts in n_cps_i:
        metersets.append(float(
                beam[0x300a,0x111][con_pts-1][0x300a,0x134].value
            ))
    return metersets

def get_active_cps(rt_plan):
    """
    Extracts and returns a list of active control points for treatment beams in a radiotherapy (RT) plan.

    This function processes a DICOM RT Plan dataset to identify active control points for each treatment 
    beam. It retrieves relevant information including beam indices, control point indices, 
    the accumulated monitor units (MU) for the arc, and the total number of fractions. 
    Setup beams are ignored in this process.

    Parameters:
    -----------
    rt_plan : pydicom.Dataset
        A DICOM dataset representing the RT Plan, containing information about beams, control points, 
        and treatment fractions.

    Returns:
    --------
    arc_cp_list : list of lists
        A list where each element is a list containing the following information for each treatment beam:
        - Beam index (int)
        - Beam number (int)
        - Control point indices (list of int)
        - Accumulated monitor units (MU) for the arc (float)
        - Total number of fractions (float)

    Example:
    --------
    >>> arc_cp_list = get_active_cps(rt_plan)
    >>> print(arc_cp_list)
    [[0, 1, [1, 2, 3, ...], 200.0, 30.0], ...]

    Notes:
    ------
    - The function utilizes the `get_beam_indexes_types` function to extract initial beam information.
    - It assumes a single fraction group when extracting the monitor units (MU).
    - The DICOM tags used include (300A, 00B0) for the beam sequence, (300A, 0110) for the control points, 
      (300A, 0086) for the MU, and (300A, 0078) for the total number of fractions.
    """

    beam_info = get_beam_indexes_types(rt_plan) #[index,beamNum,beamType]
    arc_cp_list = []    # Active Arc and Control Point list

    tot_fx = float(rt_plan[0x300a,0x0070][0][0x300a, 0x0078].value)      # Total frx
    for b_num, beam in enumerate(beam_info):
        if beam[2] == 'TREATMENT': #ignores setup beams
            cp_idx = list(range(1, beam[3]))     # Control Point Indicies
            arc_mu = float(rt_plan[0x300a,0x0070][0][0x300c,0x0004][b_num][0x300a,0x0086].value)     # Note!!! -> Assumes Single Fraction Group!!
            arc_cp_list.append([beam[0], beam[1], cp_idx, arc_mu, tot_fx])
    return arc_cp_list

def get_control_point_weights(rt_plan, arc_name):
    # Compute weights for each control point as the successive difference of metersets
    metersets = get_metersets(rt_plan, arc_name)
    cp_weights = np.diff(metersets, prepend=0.0)
    return cp_weights / np.sum(cp_weights)

def extract_cp_configuration(rt_plan, energy, save_path, pat_name, randomize_geometry=False, save_result=True):    
    ''' 
    Description
    -----------
    Extracts various control point parameters and store in structured array
    
    Parameters
    ----------
    rt_plan :       DICOM Dataset 
                    RT Plan DICOM loaded using pydicom pkg

    energy :        Integer/String
                    PDD and OAR selector parameter:
                        > For 6MV  ==> 1/"6x"/"6X",    For 6FFF  ==> 2/"6fff"/"6FFF"
                        > For 10MV ==> 3/"10x"/"10X"   For 10FFF ==> 4/"10fff"/"10FFF"

    pat_name :      String
                    Patient name (aka SIMID)

    Returns
    -------
    acp_config :    Structured Array
                    Contains various extracted parameters for all the control points (see data_type)
                    Stored with beam index into a .npz file

    '''

    if isinstance(rt_plan, str):
        rt_plan = pydicom.dcmread(rt_plan)

    if randomize_geometry:
        np.random.seed(42)
   
    # Configuration Array
    data_type = np.dtype([
        ('name' , 'S20'),
        ('beam_idx', 'i4'),
        ('arc_name', 'S10'),
        ('cp_idx', 'i4'),
        ('col_idx', 'i4'),
        ('gantry_angle', 'f4'),
        ('collimator_angle', 'f4'),
        ('table_angle', 'f4'),
        ('isocenter', 'f4', (3,)),
        ('x_jaw', 'f4', (2,)),
        ('y_jaw', 'f4', (2,)),
        ('x_mlc', 'f4', (120,)),
        ('z_mlc', 'f2', (61,)),
        ('cp_weight' , 'f8'), # Control point weight
        ('CMU' , 'f8'), # Control point MU,
    ])
    acp_config = np.array([], dtype=data_type)
    arc_acp_list = get_active_cps(rt_plan)

    # Energy/Column Selection
    col_id1 = (energy == "6x" or energy == "6X" or energy == 1) * 1
    col_id2 = (energy == "6fff" or energy == "6FFF" or energy == 2) * 2
    col_id3 = (energy == "10x" or energy == "10X" or energy == 3) * 3
    col_id4 = (energy == "10fff" or energy == "10FFF" or energy == 4) * 4
    col_id = col_id1 + col_id2 + col_id3 + col_id4
    
    beam_idx = 0
    beam_sequence = rt_plan.BeamSequence._list
    beam_sequence = [b for b in beam_sequence if any(ch.isdigit() for ch in str(b.BeamName))]
    
    for _, beam_data in enumerate(beam_sequence):
        
        arc_name = str(beam_data.BeamName)  

        arc_acp_list_element = [e for e in arc_acp_list if str(e[1]) == str(arc_name)][0]
        arc_mu = arc_acp_list_element[3]
        tot_frx = arc_acp_list_element[4]

        z_mlc = np.array(
            list(np.single(beam_data.BeamLimitingDeviceSequence._list[2].LeafPositionBoundaries._list)),
            dtype='f2'
        )
        
        cp_sequence = beam_data.ControlPointSequence._list
        cp_weights = get_control_point_weights(rt_plan, arc_name)

        for c, cp_data in enumerate(cp_sequence):

            cp_idx = np.int16(cp_data.ControlPointIndex)
            cp_weight = cp_weights[c]
            cp_mu = arc_mu * cp_weight * tot_frx

            gantry_angle = np.float32(cp_data.GantryAngle) 
            name = f'{pat_name}_arc{arc_name}_CP{cp_idx}'

            jaw_data = cp_data.BeamLimitingDevicePositionSequence._list
             
            if c == 0:
                collimator_angle = np.single(cp_data.BeamLimitingDeviceAngle)
                table_angle = np.single(cp_data.PatientSupportAngle)
                isocenter = np.array(list(np.single(cp_data.IsocenterPosition)), 'f4')
                x_jaw = np.array(list(np.single(jaw_data[0].LeafJawPositions._list)), 'f4')
                y_jaw = np.array(list(np.single(jaw_data[1].LeafJawPositions._list)), 'f4')
                x_mlc = np.array(list(np.single(jaw_data[2].LeafJawPositions._list)), 'f4')
            else:
                x_mlc = np.array(jaw_data[0].LeafJawPositions._list, 'f4')

    
            if cp_idx == 0:
                nbeam_idx = 0
                
                data = [(
                    name, nbeam_idx, arc_name, cp_idx, col_id,
                    gantry_angle, collimator_angle, table_angle,
                    isocenter, x_jaw, y_jaw, x_mlc, z_mlc, cp_weight, cp_mu
                )]
            
            else:
                beam_idx = beam_idx + 1
                collimator_angle_r = collimator_angle
                isocenter_r = isocenter

                # Used to generate random geometry for training data augmentation
                if randomize_geometry:
                    
                    # Randomly choose new collimator angle
                    collimator_angle_r = round(np.random.uniform(270,525) % 360)
                    
                    # Randomly sample new isocenter from Gaussian distribution centered on true isocenter
                    sigma = 10
                    cov_matrix = np.eye(np.array(isocenter).shape[0]) * sigma**2
                    isocenter_r = np.round(np.random.multivariate_normal(mean=isocenter, cov=cov_matrix), 2)
    
                data = [(
                    name, beam_idx, arc_name, cp_idx, col_id,
                    gantry_angle, collimator_angle_r, table_angle,
                    isocenter_r, x_jaw, y_jaw, x_mlc, z_mlc, cp_weight, cp_mu,
                )]
            
            data = np.array(data, dtype=data_type)
            acp_config = np.append(acp_config, data)
                
            if beam_data.BeamType == "STATIC":
                break
    
    if save_result:
        complete_name = os.path.join(save_path, pat_name + "_ACPConfig.npz")
        np.savez(complete_name, ACPConfig=acp_config)
    
    return acp_config

def get_structure_number_from_name(rt_struct, struct_name):
    struct_num = None
    num_of_structs = len(rt_struct.StructureSetROISequence)
    for i in range(num_of_structs):
        curr_struct_name = str(rt_struct[0x3006,0x20][i].ROIName)
        if curr_struct_name.upper() == struct_name.upper():
            struct_num = int(rt_struct[0x3006,0x20][i].ROINumber)
    return struct_num

def generate_structure_mask_fast(rt_dose, rt_struct, str_num):
    ds_pos = rt_dose.ImagePositionPatient
    grid_size = rt_dose.PixelSpacing
    grid_frame = rt_dose.GridFrameOffsetVector
    
    nz, ny, nx = int(rt_dose.NumberOfFrames), int(rt_dose.Rows), int(rt_dose.Columns)
    z_grid = np.array([ds_pos[2] + offset for offset in grid_frame])
    
    scale_x, scale_y = 1.0 / grid_size[0], 1.0 / grid_size[1]
    off_x, off_y = ds_pos[0], ds_pos[1]

    roi_sequence = {s.ROINumber: s for s in rt_struct.StructureSetROISequence}
    if str_num not in roi_sequence:
        raise ValueError(f"ROI {str_num} not found.")

    contour_sequence = next(c for c in rt_struct.ROIContourSequence 
                            if c.ReferencedROINumber == str_num)
    
    mask = np.zeros((nz, ny, nx), dtype=np.int8)
    
    for contour in getattr(contour_sequence, "ContourSequence", []):
        vertices = np.array(contour.ContourData).reshape((-1, 3))
        z_idx = np.argmin(np.abs(z_grid - vertices[0, 2]))
        
        pixel_coords = np.zeros((vertices.shape[0], 2), dtype=np.int32)
        pixel_coords[:, 0] = np.round((vertices[:, 0] - off_x) * scale_x)
        pixel_coords[:, 1] = np.round((vertices[:, 1] - off_y) * scale_y)

        cv2.fillPoly(mask[z_idx], [pixel_coords], 1)

    z_filled = np.where(mask.any(axis=(1, 2)))[0]
    for i in range(len(z_filled) - 1):
        z1, z2 = z_filled[i], z_filled[i+1]
        if z2 - z1 > 1:
            mask[z1+1 : z2] = mask[z1]

    return mask.transpose(2, 1, 0).astype('int8')

def generate_structure_mask(rt_dose, rt_struct, str_num=None):
    '''
    Description
    -----------
    Takes RT DICOM Dose and structure files and structure number (optional) and creates a
    binary mask, outlining the structure in the dose array

    Parameters
    ----------
    rt_dose :   DICOM Dataset
                RT Dose DICOM  loaded using pydicom pkg 

    rt_struct : DICOM Dataset 
                RT Structure DICOM loaded using pydicom pkg

    str_num :   Integer 
                DICOM structure number of the structure of interest (i.e., for body)

    Returns
    -------
    mask_exp :  3D Array (float value)
                Binary array mask of 0, 1 outline the structure in the dose array matrix

    '''

    #-S1: Generate voxel coordinates from DICOM dose file and mask shape

    nx, ny, nz = [int(rt_dose[tag].value) for tag in [[0x28,0x11], [0x28,0x10], [0x28,0x08]]]
    image_pos_pat, grid_size, grid_frame = [rt_dose[tag].value for tag in [[0x20,0x32], [0x28,0x30], [0x3004,0xc]]]
    
    x_grid = np.linspace(image_pos_pat[0], image_pos_pat[0] + nx * grid_size[0], nx + 1)
    y_grid = np.linspace(image_pos_pat[1], image_pos_pat[1] + ny * grid_size[1], ny + 1)
    z_grid = np.array([image_pos_pat[2] + offset for offset in grid_frame])

    mask_shape = (len(z_grid), len(y_grid)-1, len(x_grid)-1)

    #-S2: Locate the structure of interest if None specified (i.e., roi_dictGenerator)
    if str_num is None:
        roi_dict = {}  # initialize empty dictionary
        num_of_structs = rt_struct[0x3006,0x20].VM  # VM is value multiplicity
        for struct in range(num_of_structs):
            roi_dict[int(rt_struct[0x3006,0x20][struct].ROINumber)] = rt_struct[0x3006,0x20][struct].ROIName
    
        keys_temp = sorted(roi_dict.keys())
    
        for i in range(len(keys_temp)):
            print(keys_temp[i], roi_dict[keys_temp[i]])
        
        str_num = eval(input("Input structure's ROI number, e.g., 29: "))
    
    for roi_contour_sequence in rt_struct.ROIContourSequence:
        if roi_contour_sequence.ReferencedROINumber == str_num:
            contour_data = roi_contour_sequence.ContourSequence
            break
    else:
        raise ValueError(f"structure number {str_num} not found.")
    
    #-S3: Rasterize DICOM contours onto a mask array using skimage.

    mask = np.zeros(mask_shape, dtype=bool)
    z_filled = []
    
    for contour in contour_data:
        vertices = np.array(contour.ContourData).reshape((-1, 3))
        z_coords = vertices[:, 2]
        
        # Interpolate contour data to the nearest z-slice
        z_indices = np.digitize(z_coords, z_grid) - 1
        z_indices = np.clip(z_indices, 0, mask_shape[0] - 1)
        
        for z in np.unique(z_indices):
            slice_vertices = vertices[z_indices == z]
            
            # Use subpixel precision for x and y coordinates
            x_coords = (slice_vertices[:, 0] - x_grid[0]) / (x_grid[1] - x_grid[0])
            y_coords = (slice_vertices[:, 1] - y_grid[0]) / (y_grid[1] - y_grid[0])
            
            # Generate a mask using subpixel precision
            if len(x_coords) > 2 and len(y_coords) > 2:  # Avoid degenerate polygons
                poly_mask = polygon2mask(mask_shape[1:], np.vstack((y_coords, x_coords)).T)
                mask[z] |= poly_mask
                z_filled.append(z)

    #-S4: Interpolate between filled slices to prevent blanks

    z_filled = sorted(set(z_filled))
    
    for i in range(1, len(z_filled)):
        z1, z2 = z_filled[i-1], z_filled[i]
        if z2 - z1 > 1:
            for z in range(z1 + 1, z2):
                # Perform linear interpolation between slices
                alpha = (z - z1) / (z2 - z1)
                mask[z] = (1 - alpha) * mask[z1] + alpha * mask[z2]

    #-S5: Binary Mask Return
    mask_exp = np.transpose(mask, (2, 1, 0)) * 1
    return mask_exp.astype('int8')

def isVMAT(rtplan):
    GantryRotation = []
    for i in range(len(rtplan[0x300a, 0x00b0].value)):
        GantryRotation.append(rtplan[0x300a, 0x00b0].value[i][0x300a, 0x0111].value[0][0x300a, 0x011f].value)
    isVMAT = len(set(['CC', 'CW']) & set(GantryRotation)) > 0
    return isVMAT