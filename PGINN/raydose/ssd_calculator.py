import numpy as np
from numba import njit, prange, cuda, float32, int32
from scipy.interpolate import RegularGridInterpolator
import math

def calculate_ssd_gpu_batched(
    gantry_angles,
    iso_ipps,
    spacing,
    body_mask,
    threshold,
):
    m = len(gantry_angles)
    gantry_dev  = cuda.to_device(gantry_angles.astype(np.float32))
    iso_dev     = cuda.to_device(iso_ipps.astype(np.float32))
    spacing_dev = cuda.to_device(spacing.astype(np.float32))

    ssd_dev = cuda.device_array(m, dtype=np.float32)
    threads = 256
    blocks = 256

    calculate_ssd_gpu_kernel[blocks, threads](
        gantry_dev, iso_dev, spacing_dev, body_mask,
        np.float32(0.3), ssd_dev
    )
    cuda.synchronize()

    return ssd_dev

@cuda.jit
def calculate_ssd_gpu_kernel(
    gantry_angles, 
    iso_ipps,    
    spacing,    
    body_mask, 
    threshold,
    ssd_out  
):
    idx = cuda.grid(1)
    m = gantry_angles.shape[0]
    if idx >= m:
        return

    gantry_angle = gantry_angles[idx]
    iso_x = iso_ipps[idx, 0]
    iso_y = iso_ipps[idx, 1]
    iso_z = iso_ipps[idx, 2]

    sx_dim = body_mask.shape[0]
    sy_dim = body_mask.shape[1]
    sz_dim = body_mask.shape[2]

    sx = spacing[0]
    sy = spacing[1]
    sz = spacing[2]
    inv_sx = 1.0 / sx
    inv_sy = 1.0 / sy
    inv_sz = 1.0 / sz

    ang_rad = math.radians(gantry_angle)
    vx = math.sin(ang_rad)
    vy = -math.cos(ang_rad)
    vz = 0.0

    vx = math.floor(vx * 1e5 + 0.5) / 1e5
    vy = math.floor(vy * 1e5 + 0.5) / 1e5
    src_path_vector = cuda.local.array(3, dtype=float32)
    src_path_vector[0] = 1000.0 * vx
    src_path_vector[1] = 1000.0 * vy
    src_path_vector[2] = 0.0

    pos = cuda.local.array(3, dtype=float32)
    pos[0] = 0.0
    pos[1] = 0.0
    pos[2] = 0.0

    norm = math.sqrt(
        src_path_vector[0] * src_path_vector[0] +
        src_path_vector[1] * src_path_vector[1] +
        src_path_vector[2] * src_path_vector[2]
    )
    dir_vec = cuda.local.array(3, dtype=float32)
    dir_vec[0] = src_path_vector[0] / norm
    dir_vec[1] = src_path_vector[1] / norm
    dir_vec[2] = src_path_vector[2] / norm

    grid = cuda.local.array(3, dtype=int32)
    grid[0] = int(math.floor((pos[0] - iso_x) * inv_sx))
    grid[1] = int(math.floor((pos[1] - iso_y) * inv_sy))
    grid[2] = int(math.floor((pos[2] - iso_z) * inv_sz))

    step = cuda.local.array(3, dtype=int32)
    for i in range(3):
        v = dir_vec[i]
        if v > 0.0:
            step[i] = 1
        elif v < 0.0:
            step[i] = -1
        else:
            step[i] = 1

    tDelta = cuda.local.array(3, dtype=float32)
    for i in range(3):
        v = dir_vec[i]
        if v != 0.0:
            if i == 0:
                tDelta[i] = abs(sx / v)
            elif i == 1:
                tDelta[i] = abs(sy / v)
            else:
                tDelta[i] = abs(sz / v)
        else:
            tDelta[i] = 1e9

    tMax = cuda.local.array(3, dtype=float32)
    # axis 0
    if dir_vec[0] > 0.0:
        next_boundary_x = (grid[0] + 1) * sx + iso_x
    else:
        next_boundary_x = grid[0] * sx + iso_x
    if dir_vec[0] != 0.0:
        tMax[0] = (next_boundary_x - pos[0]) / dir_vec[0]
    else:
        tMax[0] = 1e9
    # axis 1
    if dir_vec[1] > 0.0:
        next_boundary_y = (grid[1] + 1) * sy + iso_y
    else:
        next_boundary_y = grid[1] * sy + iso_y
    if dir_vec[1] != 0.0:
        tMax[1] = (next_boundary_y - pos[1]) / dir_vec[1]
    else:
        tMax[1] = 1e9
    # axis 2
    if dir_vec[2] > 0.0:
        next_boundary_z = (grid[2] + 1) * sz + iso_z
    else:
        next_boundary_z = grid[2] * sz + iso_z
    if dir_vec[2] != 0.0:
        tMax[2] = (next_boundary_z - pos[2]) / dir_vec[2]
    else:
        tMax[2] = 1e9

    left_body = False
    last_axis = 0
    while (
        0 <= grid[0] < sx_dim and
        0 <= grid[1] < sy_dim and
        0 <= grid[2] < sz_dim and
        body_mask[grid[0], grid[1], grid[2]] > 0
    ):
        axis = 0
        if tMax[1] < tMax[axis]:
            axis = 1
        if tMax[2] < tMax[axis]:
            axis = 2

        grid[axis] += step[axis]
        tMax[axis] += tDelta[axis]
        last_axis = axis

        if not (0 <= grid[0] < sx_dim and 0 <= grid[1] < sy_dim and 0 <= grid[2] < sz_dim):
            left_body = True
            break

        if body_mask[grid[0], grid[1], grid[2]] <= 0:
            left_body = True
            break

    if not left_body:
        t_exit_coarse = tMax[0]
        if tMax[1] < t_exit_coarse:
            t_exit_coarse = tMax[1]
        if tMax[2] < t_exit_coarse:
            t_exit_coarse = tMax[2]
    else:
        t_exit_coarse = tMax[last_axis] - tDelta[last_axis]

    step_size = 1.0
    t_ref = t_exit_coarse - step_size
    if t_ref < 0.0:
        t_ref = 0.0

    in_body_val = 1.0
    path_vec = cuda.local.array(3, dtype=float32)

    while in_body_val > threshold:
        path_vec[0] = pos[0] + t_ref * dir_vec[0]
        path_vec[1] = pos[1] + t_ref * dir_vec[1]
        path_vec[2] = pos[2] + t_ref * dir_vec[2]

        gx = (path_vec[0] - iso_x) * inv_sx
        gy = (path_vec[1] - iso_y) * inv_sy
        gz = (path_vec[2] - iso_z) * inv_sz

        if (
            gx < 0.0 or gx >= sx_dim - 1 or
            gy < 0.0 or gy >= sy_dim - 1 or
            gz < 0.0 or gz >= sz_dim - 1
        ):
            in_body_val = 0.0
        else:
            grid_pos = cuda.local.array(3, dtype=float32)
            grid_pos[0] = gx
            grid_pos[1] = gy
            grid_pos[2] = gz
            in_body_val = _trilinear_interp_body_gpu(grid_pos, body_mask, sx_dim, sy_dim, sz_dim)

        t_ref += step_size
        if t_ref > 4000.0:
            break

    path_vec_final0 = path_vec[0] - dir_vec[0] * step_size
    path_vec_final1 = path_vec[1] - dir_vec[1] * step_size
    path_vec_final2 = path_vec[2] - dir_vec[2] * step_size

    dx = src_path_vector[0] - path_vec_final0
    dy = src_path_vector[1] - path_vec_final1
    dz = src_path_vector[2] - path_vec_final2
    cp_ssd = math.sqrt(dx * dx + dy * dy + dz * dz)

    ssd_out[idx] = cp_ssd


@cuda.jit(device=True)
def _trilinear_interp_body_gpu(grid_pos, body_mask, sx, sy, sz):
    x = grid_pos[0]
    y = grid_pos[1]
    z = grid_pos[2]

    i0 = int(math.floor(x))
    j0 = int(math.floor(y))
    k0 = int(math.floor(z))

    fx = x - i0
    fy = y - j0
    fz = z - k0

    if i0 < 0:
        i0 = 0
        fx = 0.0
    if j0 < 0:
        j0 = 0
        fy = 0.0
    if k0 < 0:
        k0 = 0
        fz = 0.0

    if i0 >= sx - 1:
        i0 = sx - 2
        fx = 1.0
    if j0 >= sy - 1:
        j0 = sy - 2
        fy = 1.0
    if k0 >= sz - 1:
        k0 = sz - 2
        fz = 1.0

    i1 = i0 + 1
    j1 = j0 + 1
    k1 = k0 + 1

    c000 = body_mask[i0, j0, k0]
    c100 = body_mask[i1, j0, k0]
    c010 = body_mask[i0, j1, k0]
    c110 = body_mask[i1, j1, k0]
    c001 = body_mask[i0, j0, k1]
    c101 = body_mask[i1, j0, k1]
    c011 = body_mask[i0, j1, k1]
    c111 = body_mask[i1, j1, k1]

    c00 = c000 * (1.0 - fx) + c100 * fx
    c01 = c001 * (1.0 - fx) + c101 * fx
    c10 = c010 * (1.0 - fx) + c110 * fx
    c11 = c011 * (1.0 - fx) + c111 * fx

    c0 = c00 * (1.0 - fy) + c10 * fy
    c1 = c01 * (1.0 - fy) + c11 * fy

    return c0 * (1.0 - fz) + c1 * fz

@njit
def _trilinear_interp_body(grid_pos, body_mask):
    x = grid_pos[0]
    y = grid_pos[1]
    z = grid_pos[2]

    sx, sy, sz = body_mask.shape

    i0 = int(np.floor(x))
    j0 = int(np.floor(y))
    k0 = int(np.floor(z))

    fx = x - i0
    fy = y - j0
    fz = z - k0

    if i0 < 0: i0 = 0; fx = 0.0
    if j0 < 0: j0 = 0; fy = 0.0
    if k0 < 0: k0 = 0; fz = 0.0

    if i0 >= sx - 1: i0 = sx - 2; fx = 1.0
    if j0 >= sy - 1: j0 = sy - 2; fy = 1.0
    if k0 >= sz - 1: k0 = sz - 2; fz = 1.0

    i1 = i0 + 1
    j1 = j0 + 1
    k1 = k0 + 1

    c000 = body_mask[i0, j0, k0]
    c100 = body_mask[i1, j0, k0]
    c010 = body_mask[i0, j1, k0]
    c110 = body_mask[i1, j1, k0]
    c001 = body_mask[i0, j0, k1]
    c101 = body_mask[i1, j0, k1]
    c011 = body_mask[i0, j1, k1]
    c111 = body_mask[i1, j1, k1]

    c00 = c000 * (1 - fx) + c100 * fx
    c01 = c001 * (1 - fx) + c101 * fx
    c10 = c010 * (1 - fx) + c110 * fx
    c11 = c011 * (1 - fx) + c111 * fx

    c0 = c00 * (1 - fy) + c10 * fy
    c1 = c01 * (1 - fy) + c11 * fy

    c = c0 * (1 - fz) + c1 * fz
    return c


@njit(cache=True)
def calculate_ssd_numba(gantry_angle, iso_ipp, spacing, body_mask, threshold=0.3):
    ang_rad = np.deg2rad(gantry_angle)
    src_path_vector = 1000.0 * np.round(
        np.array([np.sin(ang_rad), -np.cos(ang_rad), 0.0], dtype=np.float32), 5
    )

    pos = np.zeros(3, dtype=np.float32)
    norm = np.float32(np.sqrt((src_path_vector ** 2).sum()))
    dir_vec = (src_path_vector / norm).astype(np.float32)

    inv_spacing = 1.0 / spacing.astype(np.float32)
    shape = np.array(body_mask.shape, dtype=np.int32)

    grid = np.floor((pos - iso_ipp) * inv_spacing).astype(np.int32)

    step = np.empty(3, dtype=np.int32)
    for i in range(3):
        if dir_vec[i] > 0:
            step[i] = 1
        elif dir_vec[i] < 0:
            step[i] = -1
        else:
            step[i] = 1 

    tDelta = np.empty(3, dtype=np.float32)
    for i in range(3):
        if dir_vec[i] != 0.0:
            tDelta[i] = abs(spacing[i] / dir_vec[i])
        else:
            tDelta[i] = 1e9 

    tMax = np.empty(3, dtype=np.float32)
    for i in range(3):
        if dir_vec[i] > 0:
            next_boundary = (grid[i] + 1) * spacing[i] + iso_ipp[i]
        else:
            next_boundary = grid[i] * spacing[i] + iso_ipp[i]
        if dir_vec[i] != 0.0:
            tMax[i] = (next_boundary - pos[i]) / dir_vec[i]
        else:
            tMax[i] = 1e9

    left_body = False
    last_axis = 0
    while (
        0 <= grid[0] < shape[0]
        and 0 <= grid[1] < shape[1]
        and 0 <= grid[2] < shape[2]
        and body_mask[grid[0], grid[1], grid[2]] > 0
    ):
        axis = 0
        if tMax[1] < tMax[axis]:
            axis = 1
        if tMax[2] < tMax[axis]:
            axis = 2

        grid[axis] += step[axis]
        tMax[axis] += tDelta[axis]
        last_axis = axis

        if not (0 <= grid[0] < shape[0] and 0 <= grid[1] < shape[1] and 0 <= grid[2] < shape[2]):
            left_body = True
            break

        if body_mask[grid[0], grid[1], grid[2]] <= 0:
            left_body = True
            break

    if not left_body:
        t_exit_coarse = tMax[0]
        if tMax[1] < t_exit_coarse:
            t_exit_coarse = tMax[1]
        if tMax[2] < t_exit_coarse:
            t_exit_coarse = tMax[2]
    else:
        t_exit_coarse = tMax[last_axis] - tDelta[last_axis]

    step_size = np.float32(1.0)  # mm
    t_ref = np.float32(max(0.0, t_exit_coarse - step_size))

    in_body_val = np.float32(1.0)
    path_vec = pos.copy()

    while in_body_val > threshold:
        path_vec = pos + t_ref * dir_vec
        grid_pos = (path_vec - iso_ipp) / spacing.astype(np.float32)

        if (
            grid_pos[0] < 0 or grid_pos[0] >= shape[0] - 1 or
            grid_pos[1] < 0 or grid_pos[1] >= shape[1] - 1 or
            grid_pos[2] < 0 or grid_pos[2] >= shape[2] - 1
        ):
            in_body_val = 0.0
        else:
            in_body_val = _trilinear_interp_body(grid_pos, body_mask)

        t_ref += step_size

        if t_ref > 4000.0:
            break

    path_vec_final = path_vec - dir_vec * step_size

    diff = src_path_vector - path_vec_final
    cp_ssd = np.sqrt((diff ** 2).sum())
    return float(cp_ssd)


def calculate_ssd(gantry_angle, iso_ipp, spacing, body_mask):
    ang_rad  = np.deg2rad(gantry_angle)
    src_path_vector = 1000 * np.round(np.single([np.sin(ang_rad),  -np.cos(ang_rad),  0]), decimals=5)
    
    x_grid = np.arange(body_mask.shape[0])
    y_grid = np.arange(body_mask.shape[1])
    z_grid = np.arange(body_mask.shape[2])
    interpolator = RegularGridInterpolator((x_grid, y_grid, z_grid), body_mask, bounds_error=False, fill_value=0)
    
    vox_pos = np.array([0, 0, 0])                                  # SSD wrt isocenter point and gantry 
    
    vox_src_vec = src_path_vector - vox_pos                        # Voxel to Gantry Vector
    unit_vec = vox_src_vec / np.linalg.norm(vox_src_vec)           # Unit Path Vector (Voxel to Gantry)
    
    t = 0                                                          # Parameter t (used for moving along uPV)                
    in_body = 1                                                    # Initialize INB. INB >  0.3 if PVt inside/on Body     
    
    while in_body > 0.3:   
        path_vec = vox_pos + t * unit_vec                          # Path Vector to Gantry/Source
        grid_pos = (path_vec - iso_ipp) / spacing   # Grid equivalent position
        
        in_body = interpolator(grid_pos)                           # Interpolate for GEP on body_mask array
        
        t += 1                                                     # Advance to next position (1mm increment)
    
    path_vec_final = path_vec - unit_vec                           # Goes back to position where still inside/on body
    cp_ssd = np.linalg.norm(src_path_vector - path_vec_final)      # Calculate SSD at the current positon
    
    return cp_ssd

