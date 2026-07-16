import time
import math
import numpy as np
import cupy as cp

try:
    from numba import cuda, float32
    _CUDA_AVAILABLE = cuda.is_available()
    _CUDA_DEVICE_NAME = cuda.get_current_device()
except Exception:
    _CUDA_AVAILABLE = False

class TimeProfiler:
    def __init__(self, prefix, on=True):
        self.last = time.perf_counter()
        self.prefix = prefix
        self.on = on

    def mark(self, label=""):
        if self.on:
            # Ensure all GPU kernels have completed
            cuda.synchronize()

            now = time.perf_counter()
            elapsed = now - self.last
            print(f"{self.prefix} | {label} elapsed: {elapsed:.6f} sec")
            self.last = now

def split_into_batches(lst, k):
    return [lst[i:i+k] for i in range(0, len(lst), k)]

if _CUDA_AVAILABLE:

    def is_on_gpu(x):
        return isinstance(x, cuda.cudadrv.devicearray.DeviceNDArray)

    def to_gpu(x):
        return cuda.to_device(x)

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

    def multiply_3D_numba_arrays(d_a, d_b):
        cp_a = cp.asarray(d_a)
        cp_b = cp.asarray(d_b)
        cp_out = cp_a * cp_b
        return cuda.as_cuda_array(cp_out)

    def cast_device_array_int8(d_arr):
        d_out = cuda.device_array(d_arr.shape, dtype=np.int8)
        threads_per_block = 128
        blocks_per_grid = (d_arr.size + threads_per_block - 1) // threads_per_block
        cast_to_int8_kernel[blocks_per_grid, threads_per_block](d_arr, d_out)
        return d_out

    ################ KERNELS ##########################################

    @cuda.jit
    def cast_to_int8_kernel(src, dst):
        i = cuda.grid(1)
        if i < src.size:
            dst[i] = src[i]

    @cuda.jit
    def average_3d_kernel(arr_list, n_arrays, out):
        i, j, k = cuda.grid(3)

        if i < out.shape[0] and j < out.shape[1] and k < out.shape[2]:
            total = 0.0

            for idx in range(n_arrays):
                total += arr_list[idx][i, j, k]

            out[i, j, k] = total / n_arrays

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

    @cuda.jit
    def average_batched_kernel(batch, n_points, out):
        """
        batch: (batch_size, n_points, nx, ny, nz)
        out:   (batch_size, nx, ny, nz)
        """

        # 3D CUDA grid → (b, i, j)
        i, j, k = cuda.grid(3)

        batch_size, nx, ny, nz = out.shape

        if i < nx and j < ny and k < nz:
            for b in range(batch_size):
                total = 0.0
                for p in range(n_points):
                    total += batch[b, p, i, j, k]
                out[b, i, j, k] = total / n_points

