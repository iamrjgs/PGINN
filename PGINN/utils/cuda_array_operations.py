import os
import numpy as np
import math
from numba import cuda

def is_on_gpu(x):
    return isinstance(x, cuda.cudadrv.devicearray.DeviceNDArray)

def to_gpu(x):
    return cuda.to_device(x)

def sum_cuda_arrays(arr_input, index_list=None):
    if index_list is None:
        dims = len(arr_input) if isinstance(arr_input, list) else arr_input.shape[0]
        index_list = list(range(dims))

    if isinstance(arr_input, list):
        if len(arr_input) == 0:
            raise ValueError("arr_list must contain at least one array")

        dst = arr_input[index_list[0]]
        nx, ny, nz = dst.shape

        threads = (8, 8, 8)
        blocks = (
            math.ceil(nx / threads[0]),
            math.ceil(ny / threads[1]),
            math.ceil(nz / threads[2]),
        )

        for idx in index_list[1:]:
            src = arr_input[idx]
            add_3d_inplace[blocks, threads](dst, src)

        cuda.synchronize()
        return dst

    b, nx, ny, nz = arr_input.shape

    idx_dev = cuda.to_device(np.asarray(index_list, dtype=np.int32))
    n_indices = len(index_list)

    out_dev = cuda.device_array((nx, ny, nz), dtype=arr_input.dtype)

    threads = (8, 8, 8)
    blocks = (
        math.ceil(nx / threads[0]),
        math.ceil(ny / threads[1]),
        math.ceil(nz / threads[2]),
    )

    sum_4d_kernel[blocks, threads](arr_input, idx_dev, n_indices, out_dev)
    cuda.synchronize()

    return out_dev


@cuda.jit
def sum_4d_kernel(arr, indices, n_indices, out):
    i, j, k = cuda.grid(3)
    nx, ny, nz = out.shape

    if i >= nx or j >= ny or k >= nz:
        return

    acc = 0.0
    for idx in range(n_indices):
        b_idx = indices[idx]
        acc += arr[b_idx, i, j, k]

    out[i, j, k] = acc

@cuda.jit
def add_3d_inplace(dst, src):
    i, j, k = cuda.grid(3)
    nx, ny, nz = dst.shape

    if i < nx and j < ny and k < nz:
        dst[i, j, k] += src[i, j, k]

@cuda.jit
def scale_kernel_batched(doses, cmu_per_batch):
    i, j, k = cuda.grid(3)

    batch_size = doses.shape[0]
    nx = doses.shape[1]
    ny = doses.shape[2]
    nz = doses.shape[3]

    if i < nx and j < ny and k < nz:
        for b in range(batch_size):
            scale = cmu_per_batch[b]
            doses[b, i, j, k] *= scale


def tile_volume_to_batch(orig_array, batch_size):
    nx, ny, nz = orig_array.shape

    out = cuda.device_array((batch_size, nx, ny, nz),
                            dtype=orig_array.dtype)

    threadsperblock = (8, 8, 8)
    blockspergrid = (
        math.ceil(nx / threadsperblock[0]),
        math.ceil(ny / threadsperblock[1]),
        math.ceil(nz / threadsperblock[2]),
    )

    tile_3d_kernel[blockspergrid, threadsperblock](orig_array, out)

    return out

@cuda.jit
def tile_3d_kernel(orig, out):
    i, j, k = cuda.grid(3)

    nx = orig.shape[0]
    ny = orig.shape[1]
    nz = orig.shape[2]
    batch_size = out.shape[0]

    if i < nx and j < ny and k < nz:
        val = orig[i, j, k]
        for b in range(batch_size):
            out[b, i, j, k] = val


def average_3D_numba_arrays(cuda_batched_arrays):
    batch_size, n_points, nx, ny, nz = cuda_batched_arrays.shape

    out_dev = cuda.device_array((batch_size, 1, nx, ny, nz),
                                dtype=cuda_batched_arrays.dtype)

    threadsperblock = (8, 8, 8)
    blockspergrid = (
        math.ceil(nx / threadsperblock[0]),
        math.ceil(ny / threadsperblock[1]),
        math.ceil(nz / threadsperblock[2]),
    )

    average_batched_kernel[blockspergrid, threadsperblock](
        cuda_batched_arrays, n_points, out_dev[:,0,:,:,:]
    )

    return out_dev

@cuda.jit
def average_batched_kernel(batch, n_points, out):
    """
    batch: (batch_size, n_points, nx, ny, nz)
    out:   (batch_size, nx, ny, nz)
    """
    i, j, k = cuda.grid(3)

    batch_size, nx, ny, nz = out.shape

    if i < nx and j < ny and k < nz:
        for b in range(batch_size):
            total = 0.0
            for p in range(n_points):
                total += batch[b, p, i, j, k]
            out[b, i, j, k] = total / n_points

    