import time
import math
import numpy as np
from numba import njit, prange, float32
from PGINN.utils.timing import TimeProfiler

try:
    # Import cuda only when needed (keeps environments without CUDA happy)
    from numba import cuda, float32, int32
    _CUDA_AVAILABLE = cuda.is_available()
    _CUDA_DEVICE_NAME = cuda.get_current_device()
except Exception as e:
    _CUDA_AVAILABLE = False

@cuda.jit(device=True, inline=True)
def upper_bound(arr, v):
    low = 0
    high = arr.size
    while low < high:
        mid = (low + high) >> 1
        if v < arr[mid]:
            high = mid
        else:
            low = mid + 1
    return low

@cuda.jit(device=True)
def linear_interp(x, xs, ys, n, left=None, right=None):
    if x < xs[0]:
        if left is not None:
            return left
        return ys[0]

    if x > xs[n-1]:
        if right is not None:
            return right
        return ys[n-1]

    lo = 0
    hi = n - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if x < xs[mid]:
            hi = mid
        else:
            lo = mid

    x0 = xs[lo]
    x1 = xs[hi]
    y0 = ys[lo]
    y1 = ys[hi]

    denom = x1 - x0
    if denom == 0.0:
        return y0
    t = (x - x0) / denom
    return y0 + t * (y1 - y0)

def compute_aperture_dose_batched(
        cp_list,
        n_voxels,
        body_mask,
        body_mask_shape,
        dose_grid, density_array,
        ssd_ref, ref_depth, max_pdd_depth,
        pdd_depths, pdd_values, oar_dist, oar_values,
        case_params,
        ssd_arr,
        mlc_x_block,
        mlc_z_block,
        points_per_subarc=2,
        num_mlc_pairs=60, max_leaf_boundary=110.0,
        use_GPU=False, keep_arrays_in_GPU=True,
        reuse_first_subarc_point_dose=False,
        last_cp_endpoint_dose=None,
        aperture_doses=None 
        ):
    if not use_GPU or not _CUDA_AVAILABLE:
        raise NotImplementedError("CPU fallback not updated for batched mode.")
    try:
        nx, ny, nz = body_mask_shape
        dx, dy, dz = dose_grid
        skip_first_point = reuse_first_subarc_point_dose and last_cp_endpoint_dose is not None

        if isinstance(cp_list, (int, np.integer, str)):
            cp_list = [cp_list]

        batch_size = len(cp_list)
        cp_list = cuda.to_device(np.asarray(cp_list, dtype=np.int32))
        n_points = points_per_subarc

        if (
            aperture_doses is None
            or aperture_doses.shape[0] != batch_size
            or aperture_doses.shape[1] != n_points
            or aperture_doses.shape[2] != n_voxels
        ):
            aperture_doses = cuda.device_array((batch_size, n_points, n_voxels), dtype=np.float32)

        total = batch_size * n_points * n_voxels
        threads_per_block = 512
        blocks_per_grid = (total + threads_per_block - 1) // threads_per_block

        aperture_calculation_gpu_batched[blocks_per_grid, threads_per_block](
            cp_list, batch_size, n_points, n_voxels,
            body_mask, density_array,
            dx, dy, dz,
            nx, ny, nz,
            ssd_ref, ref_depth, max_pdd_depth,
            pdd_depths, pdd_values, oar_dist, oar_values,
            case_params, ssd_arr, mlc_x_block, mlc_z_block,
            max_leaf_boundary,
            skip_first_point,
            aperture_doses
        )

        if skip_first_point: 
            # Because the first point in the subarc of control point N coincides with the last point in subarc of
            # control point N-1, we can skip the calculation for the first point if the dose will be the same
            # (i.e. if the geometry did not change from one CP to the next). This block copies the last subarc-point
            # doses of N-1 to the first subarc point dose of N.
            if not isinstance(last_cp_endpoint_dose, cuda.cudadrv.devicearray.DeviceNDArray): 
                last_cp_endpoint_dose = cuda.to_device(np.asarray(last_cp_endpoint_dose, dtype=np.float32)) 
            else:
                threads_2d = (16, 16) 
                blocks_2d = ( 
                    (n_voxels + threads_2d[0] - 1) // threads_2d[0], 
                    (batch_size + threads_2d[1] - 1) // threads_2d[1], 
                )
                fill_first_subarc_from_prev_cp[blocks_2d, threads_2d](
                    aperture_doses, batch_size, n_points, n_voxels, last_cp_endpoint_dose
                    )

        if keep_arrays_in_GPU:
            return aperture_doses.reshape((batch_size, n_points, nx, ny, nz))

        aperture_host = aperture_doses.copy_to_host()
        return aperture_host.reshape((batch_size, n_points,  nx, ny, nz))

    except Exception as e:
        print("GPU error:", e)
        raise


@cuda.jit(fastmath=True)
def aperture_calculation_gpu_batched(
    cp_list, batch_size, n_points, n_voxels,
    body_mask, density_array,
    gx, gy, gz,
    nx, ny, nz,
    ssd_ref, ref_depth, max_pdd_depth,
    pdd_depths, pdd_values, oar_dist, oar_values,
    case_params, ssd_arr, mlc_x_block, mlc_zp_arr,
    max_leaf_boundary,
    skip_first_point,
    aperture_dose
):
    idx = cuda.grid(1)
    total = batch_size * n_points * n_voxels

    if idx >= total:
        return

    # Decode batch, subarc point, voxel indices
    case_size = n_points * n_voxels
    batch_idx = idx // case_size
    rem       = idx %  case_size
    subarc_point_id = rem // n_voxels
    i               = rem %  n_voxels

    # If enabled, skip dose computation for first subarc control point
    if skip_first_point and subarc_point_id == 0:
        aperture_dose[batch_idx, subarc_point_id, i] = 0.0 
        return

    # Map batch -> cp_idx
    cp = cp_list[batch_idx]
    cp_idx = cp - 1

    # Per-case beam parameters
    p = case_params[cp_idx, subarc_point_id]
    iso_x = p.ox
    iso_y = p.oy
    iso_z = p.oz
    sx = p.sx
    sy = p.sy
    sz = p.sz
    source_norm = p.source_norm
    cos_gantry = p.cg
    sin_gantry = p.sg
    cos_ca     = p.ca
    sin_ca     = p.sa
    jx_min = p.jx_min
    jx_max = p.jx_max
    jy_min = p.jy_min
    jy_max = p.jy_max

    # MLC arrays
    mlc_left_xp  = mlc_x_block[cp_idx, subarc_point_id, 0]
    mlc_right_xp = mlc_x_block[cp_idx, subarc_point_id, 1]
    mlc_zp       = mlc_zp_arr[cp_idx, subarc_point_id]

    # Decode voxel indices
    vx = i // (ny * nz)
    vy = (i // nz) % ny
    vz = i % nz

    # Body mask check
    if body_mask[vx, vy, vz] <= 0:
        aperture_dose[batch_idx, subarc_point_id, i] = 0.0
        return

    # Physical voxel coordinates
    vpx = float32(iso_x + vx * gx)
    vpy = float32(iso_y + vy * gy)
    vpz = float32(iso_z + vz * gz)

    # Gantry rotation
    gantry_rot_x = cos_gantry * vpx - sin_gantry * vpy
    gantry_rot_y = sin_gantry * vpx + cos_gantry * vpy
    gantry_rot_z = vpz

    # Perspective scaling
    scale = 1000.0 / (gantry_rot_y + 1000.0)
    rot_x = gantry_rot_x * scale
    rot_z = gantry_rot_z * scale

    # Collimator rotation
    xi = cos_ca * rot_x - sin_ca * rot_z
    zi = sin_ca * rot_x + cos_ca * rot_z

    # Collimation checks
    if not (jx_min < xi < jx_max):
        aperture_dose[batch_idx, subarc_point_id, i] = 0.0
        return
    if not (jy_min < zi < jy_max):
        aperture_dose[batch_idx, subarc_point_id, i] = 0.0
        return
    if abs(zi) > max_leaf_boundary:
        aperture_dose[batch_idx, subarc_point_id, i] = 0.0
        return

    # Leaf index
    ub = upper_bound(mlc_zp, zi)
    leaf_idx = max(0, min(ub - 1, mlc_zp.size - 2))

    left_edge  = mlc_left_xp[leaf_idx]
    right_edge = mlc_right_xp[leaf_idx]

    if not (left_edge < xi < right_edge):
        aperture_dose[batch_idx, subarc_point_id, i] = 0.0
        return

    # Dose ray-tracing
    dx = sx - vpx
    dy = sy - vpy
    dz = sz - vpz
    diff_norm = math.sqrt(dx*dx + dy*dy + dz*dz)

    ux = dx / diff_norm
    uy = dy / diff_norm
    uz = dz / diff_norm

    t = 0.0
    wed = 0.0

    while True:
        p1x = vpx + (t + 1.0) * ux
        p1y = vpy + (t + 1.0) * uy
        p1z = vpz + (t + 1.0) * uz

        p2x = vpx + t * ux
        p2y = vpy + t * uy
        p2z = vpz + t * uz

        g1x = (p1x - iso_x) / gx
        g1y = (p1y - iso_y) / gy
        g1z = (p1z - iso_z) / gz

        g2x = (p2x - iso_x) / gx
        g2y = (p2y - iso_y) / gy
        g2z = (p2z - iso_z) / gz

        ix2 = int(g2x)
        iy2 = int(g2y)
        iz2 = int(g2z)

        if not (0 <= ix2 < nx and 0 <= iy2 < ny and 0 <= iz2 < nz):
            break

        wed += density_array[ix2, iy2, iz2]

        if not (0 <= g1x < nx and 0 <= g1y < ny and 0 <= g1z < nz):
            break
        if body_mask[int(g1x), int(g1y), int(g1z)] <= 0:
            break

        t += 1.0

    # Mayneord factor
    ssd = ssd_arr[cp_idx]
    setup_factor = ((ssd_ref + ref_depth) / (ssd + ref_depth)) ** 2
    mf1 = ((ssd + ref_depth) / (ssd_ref + ref_depth)) ** 2
    mf2 = ((ssd_ref + wed) / (ssd + wed)) ** 2
    mayneord = mf1 * mf2

    # Percent depth dose correction
    pdd_eval = 0.0
    if 0.0 <= wed <= max_pdd_depth:
        pdd = linear_interp(wed, pdd_depths, pdd_values, pdd_depths.shape[0])
        pdd_eval = pdd * mayneord * setup_factor

    # Off-axis ratio correction
    cross_x = vpy * sz - vpz * sy
    cross_y = vpz * sx - vpx * sz
    cross_z = vpx * sy - vpy * sx
    cross_norm = math.sqrt(cross_x*cross_x + cross_y*cross_y + cross_z*cross_z)
    off_axis_dist = cross_norm / source_norm
    vox_oar_pos = linear_interp(off_axis_dist,  oar_dist, oar_values, oar_dist.shape[0], left=0, right=0)
    vox_oar_neg = linear_interp(-off_axis_dist, oar_dist, oar_values, oar_dist.shape[0], left=0, right=0)
    vox_oar_value = 0.5 * (vox_oar_pos + vox_oar_neg)

    aperture_dose[batch_idx, subarc_point_id, i] = pdd_eval * vox_oar_value


@cuda.jit
def fill_first_subarc_from_prev_cp(
    aperture_dose,
    batch_size, n_points, n_voxels,
    last_cp_endpoint_dose
):
    vox_idx, batch_idx = cuda.grid(2)
    if batch_idx >= batch_size or vox_idx >= n_voxels:
        return

    if batch_idx == 0:
        # For the first CP in batch: use precomputed last CP endpoint dose
        aperture_dose[0, 0, vox_idx] = last_cp_endpoint_dose[vox_idx]
    else:
        # For CP k: copy from last subarc point of CP k-1
        aperture_dose[batch_idx, 0, vox_idx] = aperture_dose[batch_idx - 1,
                                                             n_points - 1,
                                                             vox_idx]


@cuda.jit(fastmath=True)
def aperture_calculation_and_averaging_gpu_batched(
    cp_list, batch_size, n_points, n_voxels,
    body_mask, density_array,
    iso_x, iso_y, iso_z,
    gx, gy, gz,
    nx, ny, nz,
    ssd_arr, ssd_ref, ref_depth, max_pdd_depth,
    pdd_depths, pdd_values, oar_dist, oar_values,
    case_params, mlc_x_block, mlc_zp_arr,
    max_leaf_boundary,
    skip_first_control_point,
    dose_out
):
    idx = cuda.grid(1)
    total = batch_size * n_voxels

    if idx >= total:
        return

    batch_idx = idx // n_voxels
    voxel_id  = idx %  n_voxels

    cp = cp_list[batch_idx]
    cp_idx = cp - 1

    vx = voxel_id // (ny * nz)
    vy = (voxel_id // nz) % ny
    vz = voxel_id % nz

    if body_mask[vx, vy, vz] <= 0:
        dose_out[batch_idx, voxel_id] = 0.0
        return

    vpx = float32(iso_x + vx * gx)
    vpy = float32(iso_y + vy * gy)
    vpz = float32(iso_z + vz * gz)

    accum = 0.0

    for p_id in range(n_points):

        # skip-first-control-point must still contribute a zero
        if skip_first_control_point == 1 and p_id == 0:
            accum += 0.0
            continue

        p = case_params[cp_idx, p_id]
        ssd = ssd_arr[cp_idx]
        sx = p.sx
        sy = p.sy
        sz = p.sz
        source_norm = p.source_norm
        cos_gantry = p.cg
        sin_gantry = p.sg
        cos_ca     = p.ca
        sin_ca     = p.sa
        jx_min = p.jx_min
        jx_max = p.jx_max
        jy_min = p.jy_min
        jy_max = p.jy_max

        mlc_left_xp  = mlc_x_block[cp_idx, p_id, 0]
        mlc_right_xp = mlc_x_block[cp_idx, p_id, 1]
        mlc_zp       = mlc_zp_arr[cp_idx, p_id]

        gantry_rot_x = cos_gantry * vpx - sin_gantry * vpy
        gantry_rot_y = sin_gantry * vpx + cos_gantry * vpy
        gantry_rot_z = vpz

        scale = 1000.0 / (gantry_rot_y + 1000.0)
        rot_x = gantry_rot_x * scale
        rot_z = gantry_rot_z * scale

        xi = cos_ca * rot_x - sin_ca * rot_z
        zi = sin_ca * rot_x + cos_ca * rot_z

        # blocked → contributes zero
        if not (jx_min < xi < jx_max):
            accum += 0.0
            continue
        if not (jy_min < zi < jy_max):
            accum += 0.0
            continue
        if abs(zi) > max_leaf_boundary:
            accum += 0.0
            continue

        ub = upper_bound(mlc_zp, zi)
        leaf_idx = max(0, min(ub - 1, mlc_zp.size - 2))

        left_edge  = mlc_left_xp[leaf_idx]
        right_edge = mlc_right_xp[leaf_idx]

        if not (left_edge < xi < right_edge):
            accum += 0.0
            continue

        dx = sx - vpx
        dy = sy - vpy
        dz = sz - vpz
        diff_norm = math.sqrt(dx*dx + dy*dy + dz*dz)

        ux = dx / diff_norm
        uy = dy / diff_norm
        uz = dz / diff_norm

        t = 0.0
        wed = 0.0

        while True:
            p1x = vpx + (t + 1.0) * ux
            p1y = vpy + (t + 1.0) * uy
            p1z = vpz + (t + 1.0) * uz

            p2x = vpx + t * ux
            p2y = vpy + t * uy
            p2z = vpz + t * uz

            g1x = (p1x - iso_x) / gx
            g1y = (p1y - iso_y) / gy
            g1z = (p1z - iso_z) / gz

            g2x = (p2x - iso_x) / gx
            g2y = (p2y - iso_y) / gy
            g2z = (p2z - iso_z) / gz

            ix2 = int(g2x)
            iy2 = int(g2y)
            iz2 = int(g2z)

            if not (0 <= ix2 < nx and 0 <= iy2 < ny and 0 <= iz2 < nz):
                break

            wed += density_array[ix2, iy2, iz2]

            if not (0 <= g1x < nx and 0 <= g1y < ny and 0 <= g1z < nz):
                break
            if body_mask[int(g1x), int(g1y), int(g1z)] <= 0:
                break

            t += 1.0

        setup_factor = ((ssd_ref + ref_depth) / (ssd + ref_depth)) ** 2
        mf1 = ((ssd + ref_depth) / (ssd_ref + ref_depth)) ** 2
        mf2 = ((ssd_ref + wed) / (ssd + wed)) ** 2
        mayneord = mf1 * mf2

        pdd_eval = 0.0
        if 0.0 <= wed <= max_pdd_depth:
            pdd = linear_interp(wed, pdd_depths, pdd_values, pdd_depths.shape[0])
            pdd_eval = pdd * mayneord * setup_factor

        cross_x = vpy * sz - vpz * sy
        cross_y = vpz * sx - vpx * sz
        cross_z = vpx * sy - vpy * sx
        cross_norm = math.sqrt(cross_x*cross_x + cross_y*cross_y + cross_z*cross_z)

        off_axis_dist = cross_norm / source_norm

        vox_oar_pos = linear_interp(off_axis_dist,  oar_dist, oar_values, oar_dist.shape[0], left=0, right=0)
        vox_oar_neg = linear_interp(-off_axis_dist, oar_dist, oar_values, oar_dist.shape[0], left=0, right=0)
        vox_oar_value = 0.5 * (vox_oar_pos + vox_oar_neg)

        voxel_dose = pdd_eval * vox_oar_value

        accum += voxel_dose

    dose_out[batch_idx, voxel_id] = accum / n_points