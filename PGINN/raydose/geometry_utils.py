import numpy as np
import SimpleITK as sitk

try:
    from numba import cuda, float32
    import cupy as cp
    _CUDA_AVAILABLE = cuda.is_available()
    _CUDA_DEVICE_NAME = cuda.get_current_device()
    from .general_utils import is_on_gpu
except Exception:
    _CUDA_AVAILABLE = False

########################## GPU FUNCTIONS ####################################

@cuda.jit
def pad_and_crop_kernel(inp, out,
                        pad_x, pad_y, pad_z,
                        start_x, start_y, start_z,
                        Nx, Ny, Nz,
                        pad_value):
    i, j, k = cuda.grid(3)
    if i < Nx and j < Ny and k < Nz:
        src_x = start_x + i - pad_x
        src_y = start_y + j - pad_y
        src_z = start_z + k - pad_z
        if (0 <= src_x < inp.shape[0] and
            0 <= src_y < inp.shape[1] and
            0 <= src_z < inp.shape[2]):
            out[i, j, k] = inp[src_x, src_y, src_z]
        else:
            out[i, j, k] = pad_value

def crop_or_pad_matrices(inp_arrays, isocenter_voxel,
                        crop_dim=(96,96,64), pad_value=0.0,
                        block=(8,8,8)):
    if not inp_arrays:
        raise ValueError("inp_arrays must be a non-empty list")

    arr_shape = np.array(inp_arrays[0].shape, dtype=np.int32)
    crop_dim  = np.array(crop_dim, dtype=np.int32)

    deficit   = np.maximum(crop_dim - arr_shape, 0)
    pad_width = np.stack([deficit // 2, deficit - deficit // 2], axis=1)
    arr_shape_padded = arr_shape + deficit

    isocenter_voxel = np.array(np.round(isocenter_voxel), dtype=np.int32) + pad_width[:,0]

    idx_start = isocenter_voxel - crop_dim // 2
    idx_end   = idx_start + crop_dim

    for d in range(3):
        if idx_start[d] < 0:
            idx_end[d] -= idx_start[d]
            idx_start[d] = 0
        if idx_end[d] > arr_shape_padded[d]:
            shift = idx_end[d] - arr_shape_padded[d]
            idx_start[d] -= shift
            idx_end[d]   -= shift
        idx_start[d] = max(idx_start[d], 0)
        idx_end[d]   = min(idx_end[d], arr_shape_padded[d])

    slices = tuple(slice(s, e) for s, e in zip(idx_start, idx_end))
    Nx, Ny, Nz = map(int, crop_dim)

    cropped_arrays = []
    for arr in inp_arrays:
        if arr is None:
            continue
        if is_on_gpu(arr):
            out_dev = cuda.device_array((Nx, Ny, Nz), dtype=arr.dtype)
            threadsperblock = block
            blockspergrid = (
                (Nx + threadsperblock[0] - 1) // threadsperblock[0],
                (Ny + threadsperblock[1] - 1) // threadsperblock[1],
                (Nz + threadsperblock[2] - 1) // threadsperblock[2],
            )
            pad_and_crop_kernel[blockspergrid, threadsperblock](
                arr, out_dev,
                pad_width[0,0], pad_width[1,0], pad_width[2,0],
                idx_start[0], idx_start[1], idx_start[2],
                Nx, Ny, Nz,
                arr.dtype.type(pad_value)
            )
            cropped_arrays.append(out_dev)
        else:
            pad_arr = np.pad(arr, pad_width=pad_width,
                            mode='constant', constant_values=pad_value)
            cropped_arrays.append(pad_arr[slices])

    dest_start = idx_start - pad_width[:,0]
    dest_end   = idx_end   - pad_width[:,0]

    metadata = {
        "pad_width": pad_width,
        "idx_start": idx_start,
        "idx_end": idx_end,
        "orig_shape": arr_shape,
        "dest_start": dest_start,
        "dest_end": dest_end,
    }
    return cropped_arrays, metadata

@cuda.jit
def pad_and_crop_kernel_batched(inp, out,
                                pad_x, pad_y, pad_z,
                                start_x_arr, start_y_arr, start_z_arr,
                                Nx, Ny, Nz,
                                pad_value):
    """
    inp: (batch_size, nx, ny, nz)
    out: (batch_size, Nx, Ny, Nz)
    start_*_arr: arrays of length batch_size giving crop start indices
    """

    i, j, k = cuda.grid(3)

    if i >= Nx or j >= Ny or k >= Nz:
        return

    batch_size = inp.shape[0]
    
    # Loop over batch dimension inside kernel
    for b in range(batch_size):

        start_x = start_x_arr[b]
        start_y = start_y_arr[b]
        start_z = start_z_arr[b]

        src_x = start_x + i - pad_x
        src_y = start_y + j - pad_y
        src_z = start_z + k - pad_z

        if (0 <= src_x < inp.shape[1] and
            0 <= src_y < inp.shape[2] and
            0 <= src_z < inp.shape[3]):
            out[b, i, j, k] = inp[b, src_x, src_y, src_z]
        else:
            out[b, i, j, k] = pad_value

def crop_or_pad_batched_matrices(inp_arrays, isocenter_voxels,
                                    crop_dim=(96,96,64), pad_value=0.0,
                                    block=(8,8,8)):
    if not inp_arrays:
        raise ValueError("inp_arrays must be a non-empty list")

    batch_size = inp_arrays[0].shape[0]
    nx, ny, nz = inp_arrays[0].shape[-3:]
    crop_dim = np.array(crop_dim, dtype=np.int32)

    arr_shape = np.array([nx, ny, nz], dtype=np.int32)
    deficit   = np.maximum(crop_dim - arr_shape, 0)
    pad_width = np.stack([deficit // 2, deficit - deficit // 2], axis=1)
    arr_shape_padded = arr_shape + deficit

    start_x_list = []
    start_y_list = []
    start_z_list = []
    metadata_list = []

    for iso in isocenter_voxels:
        iso = np.array(np.round(iso), dtype=np.int32) + pad_width[:,0]

        idx_start = iso - crop_dim // 2
        idx_end   = idx_start + crop_dim

        for d in range(3):
            if idx_start[d] < 0:
                idx_end[d] -= idx_start[d]
                idx_start[d] = 0
            if idx_end[d] > arr_shape_padded[d]:
                shift = idx_end[d] - arr_shape_padded[d]
                idx_start[d] -= shift
                idx_end[d]   -= shift

        start_x_list.append(idx_start[0])
        start_y_list.append(idx_start[1])
        start_z_list.append(idx_start[2])

        dest_start = idx_start - pad_width[:,0]
        dest_end   = idx_end   - pad_width[:,0]

        metadata_list.append({
            "pad_width": pad_width.copy(),
            "idx_start": idx_start.copy(),
            "idx_end": idx_end.copy(),
            "orig_shape": arr_shape.copy(),
            "dest_start": dest_start.copy(),
            "dest_end": dest_end.copy(),
        })

    start_x_dev = cuda.to_device(np.array(start_x_list, dtype=np.int32))
    start_y_dev = cuda.to_device(np.array(start_y_list, dtype=np.int32))
    start_z_dev = cuda.to_device(np.array(start_z_list, dtype=np.int32))

    Nx, Ny, Nz = map(int, crop_dim)

    threadsperblock = block
    blockspergrid = (
        (Nx + block[0] - 1) // block[0],
        (Ny + block[1] - 1) // block[1],
        (Nz + block[2] - 1) // block[2],
    )

    cropped_list = []

    for arr in inp_arrays:
        if len(arr.shape) == 4:
            out_dev = cuda.device_array((batch_size, Nx, Ny, Nz),
                                        dtype=arr.dtype)

            pad_and_crop_kernel_batched[blockspergrid, threadsperblock](
                arr, out_dev,
                pad_width[0,0], pad_width[1,0], pad_width[2,0],
                start_x_dev, start_y_dev, start_z_dev,
                Nx, Ny, Nz,
                arr.dtype.type(pad_value)
            )
        else:
            raypoints = arr.shape[1]
            out_dev = cuda.device_array((batch_size, raypoints, Nx, Ny, Nz),
                                        dtype=arr.dtype)

            for i in range(raypoints):
                pad_and_crop_kernel_batched[blockspergrid, threadsperblock](
                    arr[:,i,:,:,:], out_dev[:,i,:,:,:],
                    pad_width[0,0], pad_width[1,0], pad_width[2,0],
                    start_x_dev, start_y_dev, start_z_dev,
                    Nx, Ny, Nz,
                    arr.dtype.type(pad_value)
                )

        cropped_list.append(out_dev)

    return cropped_list, metadata_list

@cuda.jit
def uncrop_kernel(crop_array, out_array, offset_h, offset_w, offset_d):
    h, w, d = cuda.grid(3)
    
    if h < crop_array.shape[0] and w < crop_array.shape[1] and d < crop_array.shape[2]:
        out_h = h + offset_h
        out_w = w + offset_w
        out_d = d + offset_d
        
        if (0 <= out_h < out_array.shape[0] and 
            0 <= out_w < out_array.shape[1] and 
            0 <= out_d < out_array.shape[2]):
            out_array[out_h, out_w, out_d] = crop_array[h, w, d]

def uncrop_matrix_gpu(crop_array, metadata, pad_value=0):
    orig_shape = metadata["orig_shape"]
    idx_start = metadata["idx_start"]
    pad_width = metadata["pad_width"]

    offsets = tuple(idx_start[i] - pad_width[i][0] for i in range(3))

    if pad_value == 0:
        out_device = cuda.device_array(orig_shape, dtype=crop_array.dtype)
        cp.asarray(out_device).fill(pad_value)
    else:
        out_device = cuda.to_device(np.full(orig_shape, pad_value, dtype=crop_array.dtype))

    threadsperblock = (8, 8, 8)
    blockspergrid_x = int(np.ceil(crop_array.shape[0] / threadsperblock[0]))
    blockspergrid_y = int(np.ceil(crop_array.shape[1] / threadsperblock[1]))
    blockspergrid_z = int(np.ceil(crop_array.shape[2] / threadsperblock[2]))
    blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)

    uncrop_kernel[blockspergrid, threadsperblock](
        crop_array, out_device, offsets[0], offsets[1], offsets[2]
    )
    
    return out_device

@cuda.jit
def uncrop_kernel_batched(crop_array, out_array,
                          offset_h_arr, offset_w_arr, offset_d_arr):
    """
    crop_array: (batch_size, Nx, Ny, Nz)
    out_array:  (batch_size, orig_x, orig_y, orig_z)
    offset_*_arr: arrays of length batch_size
    """

    h, w, d = cuda.grid(3)

    Nx = crop_array.shape[1]
    Ny = crop_array.shape[2]
    Nz = crop_array.shape[3]

    if h >= Nx or w >= Ny or d >= Nz:
        return

    batch_size = crop_array.shape[0]

    for b in range(batch_size):
        out_h = h + offset_h_arr[b]
        out_w = w + offset_w_arr[b]
        out_d = d + offset_d_arr[b]

        if (0 <= out_h < out_array.shape[1] and
            0 <= out_w < out_array.shape[2] and
            0 <= out_d < out_array.shape[3]):
            out_array[b, out_h, out_w, out_d] = crop_array[b, h, w, d]

def uncrop_matrices_gpu_batched(crop_arrays, metadata_list, pad_value=0):
    """
    crop_arrays: list of cuda.device_array, each shaped (batch_size, Nx, Ny, Nz)
    metadata_list: list of metadata dicts, length = batch_size
    returns: list of uncropped cuda.device_array
    """

    if not crop_arrays:
        raise ValueError("crop_arrays must be a non-empty list")

    batch_size = crop_arrays[0].shape[0]

    offset_h = np.zeros(batch_size, dtype=np.int32)
    offset_w = np.zeros(batch_size, dtype=np.int32)
    offset_d = np.zeros(batch_size, dtype=np.int32)

    orig_shape = metadata_list[0]["orig_shape"]

    for b, meta in enumerate(metadata_list):
        idx_start = meta["idx_start"]
        pad_width = meta["pad_width"]
        offset_h[b] = idx_start[0] - pad_width[0][0]
        offset_w[b] = idx_start[1] - pad_width[1][0]
        offset_d[b] = idx_start[2] - pad_width[2][0]

    offset_h_dev = cuda.to_device(offset_h)
    offset_w_dev = cuda.to_device(offset_w)
    offset_d_dev = cuda.to_device(offset_d)

    out_list = []

    Nx, Ny, Nz = crop_arrays[0].shape[1:]
    threads = (8, 8, 8)
    blocks = (
        (Nx + threads[0] - 1) // threads[0],
        (Ny + threads[1] - 1) // threads[1],
        (Nz + threads[2] - 1) // threads[2],
    )

    for crop_array in crop_arrays:

        if pad_value == 0:
            out_dev = cuda.device_array((batch_size, *orig_shape),
                                        dtype=crop_array.dtype)
            cp.asarray(out_dev).fill(0)
        else:
            out_dev = cuda.to_device(
                np.full((batch_size, *orig_shape),
                        pad_value,
                        dtype=crop_array.dtype)
            )

        uncrop_kernel_batched[blocks, threads](
            crop_array, out_dev,
            offset_h_dev, offset_w_dev, offset_d_dev
        )

        out_list.append(out_dev)

    return out_list

########################## CPU FUNCTIONS ####################################

def index_to_physical(index, origin, spacing, direction_matrix):
    index = np.asarray(index, dtype=float)
    origin = np.asarray(origin, dtype=float)
    spacing = np.asarray(spacing, dtype=float)
    D = np.asarray(direction_matrix, dtype=float).reshape(3,3)
    return origin + D @ (spacing * index)

def physical_to_index(physical, origin, spacing, direction_matrix):
    physical = np.asarray(physical, dtype=float)
    origin = np.asarray(origin, dtype=float)
    spacing = np.asarray(spacing, dtype=float)
    D = np.asarray(direction_matrix, dtype=float).reshape(3,3)
    return (np.linalg.inv(D) @ (physical - origin)) / spacing

def resample_ct_to_new_geometry(ct_sitk_image, origin, spacing, shape, dirs, return_as_array=False):
    ref = sitk.Image(shape, sitk.sitkFloat32)
    ref.SetSpacing(tuple(spacing))
    ref.SetOrigin(tuple(origin))
    ref.SetDirection(list(dirs.flatten()))

    resampled_image = sitk.Resample(ct_sitk_image, ref, sitk.Transform(),
                         sitk.sitkLinear, -1050, ct_sitk_image.GetPixelID())

    if return_as_array:
        return np.transpose(sitk.GetArrayFromImage(resampled_image), (2,1,0))
    return resampled_image

def crop_or_pad_matrix(inp_array, isocenter_voxel, crop_dim=(96,96,64), pad_value=0):
    crop_dim = np.array(crop_dim, dtype=np.int32)
    arr_shape = np.array(inp_array.shape, dtype=np.int32)

    # --- Step 1: Pad if needed ---
    deficit = np.maximum(crop_dim - arr_shape, 0)
    pad_width = np.stack([deficit // 2, deficit - deficit // 2], axis=1)
    pad_arr = np.pad(inp_array, pad_width=pad_width, mode='constant', constant_values=pad_value)
    arr_shape = np.array(pad_arr.shape, dtype=np.int32)

    # --- Step 2: Crop indices ---
    isocenter_voxel = np.array(np.round(isocenter_voxel), dtype=np.int32) + pad_width[:,0]
    idx_start = isocenter_voxel - crop_dim // 2
    idx_end   = idx_start + crop_dim

    # --- Step 3: Clamp to bounds ---
    # If start < 0, shift forward; if end > shape, shift backward
    for i in range(3):
        if idx_start[i] < 0:
            idx_end[i] -= idx_start[i]   # shift end forward
            idx_start[i] = 0
        if idx_end[i] > arr_shape[i]:
            shift = idx_end[i] - arr_shape[i]
            idx_start[i] -= shift
            idx_end[i]   -= shift
        # Final safety clamp
        idx_start[i] = max(idx_start[i], 0)
        idx_end[i]   = min(idx_end[i], arr_shape[i])

    slices = tuple(slice(s, e) for s, e in zip(idx_start, idx_end))
    crop_array = pad_arr[slices]

    metadata = {
        "pad_width": pad_width,
        "idx_start": idx_start,
        "idx_end": idx_end,
        "orig_shape": inp_array.shape
    }
    return crop_array, metadata

def uncrop_matrix(crop_array, metadata, pad_value=0):
    pad_width = metadata["pad_width"]
    idx_start = metadata["idx_start"]
    idx_end   = metadata["idx_end"]
    orig_shape = metadata["orig_shape"]

    # Recreate padded canvas
    padded_shape = tuple(orig_shape[i] + pad_width[i].sum() for i in range(3))
    out = np.full(padded_shape, pad_value, dtype=crop_array.dtype)

    # Place crop back
    slices = tuple(slice(s, e) for s, e in zip(idx_start, idx_end))
    out[slices] = crop_array

    # Remove padding to recover original shape
    unpad_slices = tuple(slice(pad_width[i,0], padded_shape[i]-pad_width[i,1]) for i in range(3))
    return out[unpad_slices]

def rotate_matrix(gantry_angle, inp_array):
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
    num_rot90_nr = 0
    num_rot90_n1 = ((45 <= gantry_angle) and (gantry_angle < 135)) * 1
    num_rot90_r2 = ((135 <= gantry_angle) and (gantry_angle < 225)) *  -2
    num_rot90_p1 = ((225 <= gantry_angle) and (gantry_angle < 315)) *  -1
    
    num_rot_90 = num_rot90_nr + num_rot90_n1 + num_rot90_p1 + num_rot90_r2
    
    rot_array = np.rot90(inp_array, num_rot_90)
    
    return rot_array
