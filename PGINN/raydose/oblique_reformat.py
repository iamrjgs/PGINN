import numpy as np
import SimpleITK as sitk
import time
import math
import cupy as cp

from .geometry_utils import index_to_physical, physical_to_index

from PGINN.utils.timing import TimeProfiler
from PGINN.utils.cuda_array_operations import is_on_gpu

try:
    from numba import cuda, float32, int32, int64
    _CUDA_AVAILABLE = cuda.is_available()
except Exception:
    _CUDA_AVAILABLE = False


def oblique_reformat_batched(
        input_arrays,
        input_geometry,    # list of length B, each a dict
        beam_vector,       # list/array of length B, per-batch beam vector
        interpolator=sitk.sitkLinear,
        match_size=False,
        up=None,
        beam_axis=0,
        use_GPU=True,
        block=(8, 8, 8),
        cast_float32=True,
        preserve_dtype=True,
        inverse=False,
        initial_geometry=None,   # if inverse=True, list of length B
        keep_arrays_in_GPU=True
    ):
    """
    Batched oblique reformat.

    input_geometry: list of geometry dicts, length B
    beam_vector:    list/array of beam vectors, length B
    input_arrays:   list of CUDA arrays, each (B, nx, ny, nz)
    """

    B = len(input_geometry) # number of batch elements
    assert len(beam_vector) == B, "beam_vector must be length B to match input_geometry"

    if input_arrays[0].shape[0] > B:
        input_arrays = [copy_first_k(a, B) for a in input_arrays]

    s_in_list   = []
    D_in_list   = []
    O_in_list   = []
    size_in_list = []

    s_out_list  = []
    D_out_list  = []
    O_out_list  = []
    size_out_list = []

    new_geometry_list = []

    if not inverse:
        for b in range(B):
            geom_b = input_geometry[b]
            beam_b = beam_vector[b]

            D_out_b, spacing_order = get_rotation_matrix_from_beam_vector(
                beam_b, up=up, beam_axis=beam_axis
            )

            O_in_b = np.array(geom_b['origin'], dtype=float)
            D_in_b = np.array(geom_b['direction_matrix'], dtype=float)
            s_in_b = np.array(geom_b['spacing'], dtype=float)
            size_in_b = np.array(geom_b['size'], dtype=int)

            s_out_b = s_in_b[spacing_order]

            corners_idx = np.array(
                np.meshgrid(
                    [0, size_in_b[0] - 1],
                    [0, size_in_b[1] - 1],
                    [0, size_in_b[2] - 1],
                    indexing="ij",
                )
            ).reshape(3, -1).T
            P_in = O_in_b[:, None] + D_in_b @ (s_in_b[:, None] * corners_idx.T)
            inv_s_out = 1.0 / s_out_b[:, None]
            coords_unshifted = inv_s_out * (D_out_b.T @ P_in)
            mins = coords_unshifted.min(axis=1)
            maxs = coords_unshifted.max(axis=1)

            if match_size:
                size_out_b = size_in_b
                center_in_phys = O_in_b + D_in_b @ (s_in_b * (size_in_b - 1) / 2.0)
                center_out_coords = D_out_b.T @ center_in_phys
                O_out_axis = center_out_coords - (size_out_b - 1) * s_out_b / 2.0
                O_out_b = D_out_b @ O_out_axis
            else:
                size_out_b = np.ceil(maxs - mins).astype(int) + 1
                size_out_b = np.maximum(size_out_b, 1)
                O_out_b = D_out_b @ (s_out_b * mins)

            s_in_list.append(s_in_b)
            D_in_list.append(D_in_b)
            O_in_list.append(O_in_b)
            size_in_list.append(size_in_b)

            s_out_list.append(s_out_b)
            D_out_list.append(D_out_b)
            O_out_list.append(O_out_b)
            size_out_list.append(size_out_b)

            geom_dict_b = {
                'original_origin': O_in_b.tolist(),
                'original_spacing': s_in_b.tolist(),
                'original_direction_matrix': D_in_b.tolist(),
                'original_size': size_in_b.tolist(),
                'origin': O_out_b.tolist(),
                'spacing': s_out_b.tolist(),
                'direction_matrix': D_out_b.tolist(),
                'size': size_out_b.tolist()
            }

            if 'isocenter' in geom_b.keys():
                iso_phys_in = np.asarray(geom_b['isocenter'], dtype=float)
                iso_idx_in = physical_to_index(iso_phys_in, O_in_b, s_in_b, D_in_b)
                i_out = np.diag(1.0 / s_out_b) @ (D_out_b.T @ (iso_phys_in - O_out_b))
                iso_idx_out = tuple(i_out.tolist())
                iso_phys_out = index_to_physical(iso_idx_out, O_out_b, s_out_b, D_out_b)

                geom_dict_b['original_isocenter'] = tuple(iso_phys_in.tolist())
                geom_dict_b['original_isocenter_index'] = tuple(iso_idx_in.tolist())
                geom_dict_b['isocenter'] = tuple(iso_phys_out.tolist())
                geom_dict_b['isocenter_index'] = iso_idx_out

            new_geometry_list.append(geom_dict_b)

    else:
        # Inverse: map from beam-aligned back to initial geometry
        assert initial_geometry is not None, "initial_geometry must be provided when inverse=True"
        assert len(initial_geometry) == B, "initial_geometry must be length B"

        for b in range(B):
            geom_beam = input_geometry[b]
            geom_init = initial_geometry[b]

            O_in_b = np.array(geom_beam['origin'], dtype=float)
            D_in_b = np.array(geom_beam['direction_matrix'], dtype=float)
            s_in_b = np.array(geom_beam['spacing'], dtype=float)
            size_in_b = np.array(geom_beam['size'], dtype=int)

            O_out_b = np.array(geom_init['origin'], dtype=float)
            D_out_b = np.array(geom_init['direction_matrix'], dtype=float)
            s_out_b = np.array(geom_init['spacing'], dtype=float)
            size_out_b = np.array(geom_init['size'], dtype=int)

            s_in_list.append(s_in_b)
            D_in_list.append(D_in_b)
            O_in_list.append(O_in_b)
            size_in_list.append(size_in_b)

            s_out_list.append(s_out_b)
            D_out_list.append(D_out_b)
            O_out_list.append(O_out_b)
            size_out_list.append(size_out_b)

            geom_dict_b = {
                'original_origin': O_in_b.tolist(),
                'original_spacing': s_in_b.tolist(),
                'original_direction_matrix': D_in_b.tolist(),
                'original_size': size_in_b.tolist(),
                'origin': O_out_b.tolist(),
                'spacing': s_out_b.tolist(),
                'direction_matrix': D_out_b.tolist(),
                'size': size_out_b.tolist()
            }
            new_geometry_list.append(geom_dict_b)

    results = []
    if use_GPU:
        try:
            results = run_oblique_reformat_gpu_batched(
                input_arrays,
                s_in_list, D_in_list, O_in_list, size_in_list,
                s_out_list, D_out_list, O_out_list, size_out_list,
                interpolator=interpolator,
                threads_per_block=block[0] * block[1] * block[2],
                preserve_dtype=preserve_dtype,
                cast_float32=cast_float32,
                default_val=0.0,
                keep_arrays_in_GPU=keep_arrays_in_GPU
            )
        except Exception as e:
            print(f'GPU error | Defaulting to CPU | Error: {e}')
            use_GPU = False

    if not use_GPU:
        results = run_oblique_reformat_cpu(
            input_arrays,
            s_in_list, D_in_list, O_in_list, size_in_list,
            s_out_list, D_out_list, O_out_list, size_out_list,
            interpolator=interpolator
        )

    return results, new_geometry_list

def run_oblique_reformat_gpu_batched(
    input_arrays,          # list of CUDA arrays, each (B, nx, ny, nz)
    s_in_list, D_in_list, O_in_list, size_in_list, 
    s_out_list, D_out_list, O_out_list, size_out_list, 
    interpolator=None,
    threads_per_block=256,
    preserve_dtype=True,
    cast_float32=True,
    default_val=None,
    keep_arrays_in_GPU=True
):
    results = []

    for vol_dev in input_arrays:
        input_shape_length = len(vol_dev.shape)
        
        if input_shape_length == 4:
            r = 1
            B, nx, ny, nz = map(int, vol_dev.shape)
        else:
            B, r, nx, ny, nz = map(int, vol_dev.shape)

        s_in_arr  = np.asarray(s_in_list,  dtype=np.float32)      
        D_in_arr  = np.asarray(D_in_list,  dtype=np.float32)      
        O_in_arr  = np.asarray(O_in_list,  dtype=np.float32)      
        s_out_arr = np.asarray(s_out_list, dtype=np.float32)      
        D_out_arr = np.asarray(D_out_list, dtype=np.float32)      
        O_out_arr = np.asarray(O_out_list, dtype=np.float32)      
        size_out_arr = np.asarray(size_out_list, dtype=np.int32)  

        Nx = int(size_out_arr[0, 0])
        Ny = int(size_out_arr[0, 1])
        Nz = int(size_out_arr[0, 2])

        invS_in = np.zeros((B, 3, 3), dtype=np.float32)
        M_all   = np.zeros((B, 3, 3), dtype=np.float32)
        O_shift_all = np.zeros((B, 3), dtype=np.float32)

        for b in range(B):
            invS_in[b] = np.diag(1.0 / s_in_arr[b])
            M = invS_in[b] @ D_in_arr[b].T
            M_all[b] = M
            O_shift_all[b] = -M @ O_in_arr[b]

        R_all = D_out_arr.reshape(B, 9).astype(np.float32)
        M_all_flat = M_all.reshape(B, 9).astype(np.float32)

        R_dev        = cuda.to_device(np.ascontiguousarray(R_all))
        M_dev        = cuda.to_device(np.ascontiguousarray(M_all_flat))
        s_out_dev    = cuda.to_device(np.ascontiguousarray(s_out_arr))
        O_out_dev    = cuda.to_device(np.ascontiguousarray(O_out_arr))
        O_shift_dev  = cuda.to_device(np.ascontiguousarray(O_shift_all))

        out_dtype = np.float32

        total = B * Nx * Ny * Nz
        blocks_per_grid = (total + threads_per_block - 1) // threads_per_block

        default_val_host = np.float32(0.0) if default_val is None else np.float32(default_val)

        if len(vol_dev.shape) == 4:
            out_dev = cuda.device_array((B, Nx, Ny, Nz), dtype=out_dtype)
            resample_linear_kernel_batched[blocks_per_grid, threads_per_block](
                vol_dev, B, nx, ny, nz,
                R_dev, s_out_dev, O_out_dev, M_dev, O_shift_dev,
                default_val_host,
                out_dev, Nx, Ny, Nz
            )
        else:
            out_dev = cuda.device_array((B, r, Nx, Ny, Nz), dtype=out_dtype)
            for i in range(r):
                resample_linear_kernel_batched[blocks_per_grid, threads_per_block](
                    vol_dev[:,i,:,:,:], B, nx, ny, nz,
                    R_dev, s_out_dev, O_out_dev, M_dev, O_shift_dev,
                    default_val_host,
                    out_dev[:,i,:,:,:], Nx, Ny, Nz
                )

        out = out_dev if keep_arrays_in_GPU else out_dev.copy_to_host()

        if (not keep_arrays_in_GPU) and preserve_dtype:
            in_dtype = vol_dev.dtype
            if out.dtype != in_dtype:
                out = out.astype(in_dtype)

        results.append(out)

    return results


@cuda.jit
def resample_linear_kernel_batched(
    vol,         
    B, nx, ny, nz,
    R_all,        
    s_out_all, 
    O_out_all, 
    M_all, 
    O_shift_all,    
    default_val,
    out,        
    Nx, Ny, Nz
):
    idx = cuda.grid(1)
    total = B * Nx * Ny * Nz
    if idx >= total:
        return

    # Decode batch, x, y, z from flat index
    voxels_per_vol = Nx * Ny * Nz
    b = idx // voxels_per_vol
    rem = idx % voxels_per_vol
    x = rem // (Ny * Nz)
    rem = rem % (Ny * Nz)
    y = rem // Nz
    z = rem % Nz

    sx_out = s_out_all[b, 0]
    sy_out = s_out_all[b, 1]
    sz_out = s_out_all[b, 2]

    O_out_x = O_out_all[b, 0]
    O_out_y = O_out_all[b, 1]
    O_out_z = O_out_all[b, 2]

    R = cuda.local.array(9, float32)
    for i in range(9):
        R[i] = R_all[b, i]

    M = cuda.local.array(9, float32)
    for i in range(9):
        M[i] = M_all[b, i]

    O_shift_x = O_shift_all[b, 0]
    O_shift_y = O_shift_all[b, 1]
    O_shift_z = O_shift_all[b, 2]

    sx = sx_out * x
    sy = sy_out * y
    sz = sz_out * z

    # World coords: P_out = O_out + R @ [sx, sy, sz]
    wx = O_out_x + R[0]*sx + R[1]*sy + R[2]*sz
    wy = O_out_y + R[3]*sx + R[4]*sy + R[5]*sz
    wz = O_out_z + R[6]*sx + R[7]*sy + R[8]*sz

    # Input fractional index: idx_in = M @ P_out + O_shift
    fx = M[0]*wx + M[1]*wy + M[2]*wz + O_shift_x
    fy = M[3]*wx + M[4]*wy + M[5]*wz + O_shift_y
    fz = M[6]*wx + M[7]*wy + M[8]*wz + O_shift_z

    x0 = int(math.floor(fx)); x1 = x0 + 1
    y0 = int(math.floor(fy)); y1 = y0 + 1
    z0 = int(math.floor(fz)); z1 = z0 + 1

    dx = float32(fx - x0)
    dy = float32(fy - y0)
    dz = float32(fz - z0)

    c000 = _sample_default_batched_4D(vol, B, nx, ny, nz, b, x0, y0, z0, default_val)
    c100 = _sample_default_batched_4D(vol, B, nx, ny, nz, b, x1, y0, z0, default_val)
    c010 = _sample_default_batched_4D(vol, B, nx, ny, nz, b, x0, y1, z0, default_val)
    c110 = _sample_default_batched_4D(vol, B, nx, ny, nz, b, x1, y1, z0, default_val)
    c001 = _sample_default_batched_4D(vol, B, nx, ny, nz, b, x0, y0, z1, default_val)
    c101 = _sample_default_batched_4D(vol, B, nx, ny, nz, b, x1, y0, z1, default_val)
    c011 = _sample_default_batched_4D(vol, B, nx, ny, nz, b, x0, y1, z1, default_val)
    c111 = _sample_default_batched_4D(vol, B, nx, ny, nz, b, x1, y1, z1, default_val)

    c00 = c000 * (1.0 - dx) + c100 * dx
    c10 = c010 * (1.0 - dx) + c110 * dx
    c01 = c001 * (1.0 - dx) + c101 * dx
    c11 = c011 * (1.0 - dx) + c111 * dx

    c0 = c00 * (1.0 - dy) + c10 * dy
    c1 = c01 * (1.0 - dy) + c11 * dy

    out[b, x, y, z] = c0 * (1.0 - dz) + c1 * dz


@cuda.jit(device=True)
def _sample_default_batched_4D(vol, B, nx, ny, nz, b, x, y, z, default_val):
    if x < 0 or x >= nx or y < 0 or y >= ny or z < 0 or z >= nz:
        return default_val
    return vol[b, x, y, z]

@cuda.jit
def copy_first_k_kernel(src, dst, B, nx, ny, nz):
    idx = cuda.grid(1)
    total = B * nx * ny * nz
    if idx >= total:
        return

    b = idx // (nx*ny*nz)
    rem = idx % (nx*ny*nz)
    x = rem // (ny*nz)
    rem = rem % (ny*nz)
    y = rem // nz
    z = rem % nz

    dst[b, x, y, z] = src[b, x, y, z]

def copy_first_k(arr, k):
    B, nx, ny, nz = arr.shape
    new_arr = cuda.device_array((k, nx, ny, nz), dtype=arr.dtype)

    threads = 256
    blocks = (k*nx*ny*nz + threads - 1) // threads

    copy_first_k_kernel[blocks, threads](arr, new_arr, B, nx, ny, nz)
    cuda.synchronize()

    return new_arr

def get_rotation_matrix_from_beam_vector(beam_vector,
                                        beam_axis=2,
                                        up=None,
                                        ):
    b = np.asarray(beam_vector, dtype=float)
    nb = np.linalg.norm(b)
    if not np.isfinite(nb) or nb == 0:
        raise ValueError("beam_vector must be non-zero and finite.")
    b /= nb

    if up is None:
        ref = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(ref, b)) > 0.99:
            ref = np.array([0.0, 1.0, 0.0])
        up_vec = ref
    else:
        up_vec = np.asarray(up, dtype=float)
        nu = np.linalg.norm(up_vec)
        if not np.isfinite(nu) or nu == 0:
            raise ValueError("up must be non-zero and finite.")
        up_vec /= nu

    u_raw = up_vec - np.dot(up_vec, b) * b
    nu_raw = np.linalg.norm(u_raw)
    if not np.isfinite(nu_raw) or nu_raw == 0:
        ref = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(ref, b)) > 0.9:
            ref = np.array([0.0, 1.0, 0.0])
        u_raw = ref - np.dot(ref, b) * b
        nu_raw = np.linalg.norm(u_raw)
    u = u_raw / nu_raw
    v = np.cross(b, u)
    v /= np.linalg.norm(v)

    if beam_axis == 2:
        R = np.column_stack([u, v, b])
        spacing_order = [0, 1, 2]
    elif beam_axis == 1:
        R = np.column_stack([v, b, u])
        spacing_order = [2, 0, 1]
    else:
        R = np.column_stack([b, u, v])
        spacing_order = [1, 2, 0]

    return R, spacing_order


def run_oblique_reformat_cpu(input_arrays,
                            s_in, D_in, O_in,size_in,
                            s_out, D_out, O_out, size_out,
                            interpolator=None
                            ): 
    results = []

    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(interpolator if interpolator is not None else sitk.sitkLinear)
    resampler.SetSize(tuple(int(x) for x in size_out))
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetOutputOrigin(tuple(O_out.tolist()))
    resampler.SetOutputSpacing(tuple(s_out.tolist()))
    resampler.SetOutputDirection(tuple(D_out.ravel(order="C")))

    for vol_np in input_arrays:
        if is_on_gpu(vol_np):
            vol_np = vol_np.copy_to_host()
        img = sitk.GetImageFromArray(np.transpose(vol_np, (2,1,0)))
        img.SetOrigin(tuple(O_in.tolist()))
        img.SetSpacing(tuple(s_in.tolist()))
        img.SetDirection(tuple(D_in.ravel(order="C")))

        resampler.SetDefaultPixelValue(float(np.min(vol_np)))
        resampler.SetOutputPixelType(img.GetPixelID())
        out_img = resampler.Execute(img)

        out_np = np.transpose(sitk.GetArrayFromImage(out_img), (2,1,0))

        results.append(out_np)
        
    return results