from abc import ABC, abstractmethod
import json
import time
from functools import wraps
import os

import cupy
from numba import cuda
import numpy as np

from PGINN.utils.timing import timed_method

class ModelFactory:
    registry = {}

    @classmethod
    def register(cls, key):
        def decorator(model_cls):
            cls.registry[key] = model_cls
            return model_cls
        return decorator

    @classmethod
    def create(cls, key, **kwargs):
        return cls.registry[key](**kwargs)

class DosePredictionModel(ABC):

    TIMED_METHODS = {
        "preprocess",
        "infer",
        "postprocess"
        }

    def __init__(self,
                model_data_dirpath=None,
                model_weights_path=None,
                hyperparams_path=None,
                scale_path=None,
                batch_size=1,
                allocate_persistent_buffer=True,
                preload=True,
                **kwargs
                ):
        self._model = None
        self._scale = None
        self._hyperparams = None
        self._architechture_name = None
        self._device = 'cuda'
        self._platform_ = 'generic'
        self._timings = {}

        if model_data_dirpath is not None:
            self.set_model_paths_from_dirpath(model_data_dirpath)
        elif model_weights_filepath is not None and hyperparams_path is not None and scale_path is not None:
            self.model_weights_filepath = model_weights_path
            self.model_hyperparams_filepath = hyperparams_path
            self.scale_constants_filepath = scale_path
        else:
            raise ValueError(
                "You must provide either `model_data_dirpath` OR all three of "
                "`model_weights_filepath`, `hyperparams_path`, and `scale_path`."
            )

        self.load_hyperparams(self.model_hyperparams_filepath)
        self.load_scale(self.scale_constants_filepath)

        self.inshape = tuple(self.hyperparams['inshape'])
        self.input_matrix_size = self.inshape[0:3]
        self.input_channels = self.hyperparams.get('num_channels', 2)
        self.batch_size = batch_size
        self.input_batch_shape = (self.batch_size, *self.input_matrix_size, self.input_channels)

        # Allocate persistent buffer cuda array and pytorch/tensorflow view on which inference will always be run
        if allocate_persistent_buffer:
            self.batch_cuda_buffer = cuda.device_array(self.input_batch_shape, dtype=np.float32)
            self.set_persistent_view(self.batch_cuda_buffer)
        else:
            self.batch_cuda_buffer = None
            self.input_persistent_view = None

        if preload:
            self.load(self.model_weights_filepath)
    
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        for name in cls.TIMED_METHODS:
            if name in cls.__dict__:  
                fn = cls.__dict__[name]
                setattr(cls, name, timed_method(fn))

    @abstractmethod
    def set_model_paths_from_dirpath(self, model_data_dirpath):
        pass

    @abstractmethod
    def load(self, path, **kwargs):
        pass

    @abstractmethod
    def preprocess(self, **kwargs):
        pass

    @abstractmethod
    def infer(self, prepared_input, **kwargs):
        pass

    @abstractmethod
    def postprocess(self, inference_output, **kwargs):
        pass

    @abstractmethod
    def preprocess_infer_postprocess(self, **kwargs):
        pass

    @abstractmethod
    def print_summary(self):
        pass
        
    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self._model = value

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, value):
        self._scale = value

    def load_scale(self, scale_path):
        with open(scale_path, 'r') as sfile:
            self.scale = json.load(sfile)

    @property
    def hyperparams(self):
        return self._hyperparams

    @hyperparams.setter
    def hyperparams(self, value):
        self._hyperparams = value

    def load_hyperparams(self, hyperparams_path):
        with open(hyperparams_path, 'r') as sfile:
            self.hyperparams = json.load(sfile)

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, value):
        self._device = value

    @property
    def platform(self):
        return self._platform

    @platform.setter
    def platform(self, value):
        self._platform = value

    @property
    def architechture_name(self):
        return self._architechture_name

    @architechture_name.setter
    def architechture_name(self, value):
        self._architechture_name = value

    @property
    def model_input_persistent_view(self):
        return self._model_input_persistent_view

    @model_input_persistent_view.setter
    def model_input_persistent_view(self, value):
        self._model_input_persistent_view = value

    @property
    def timings(self):
        return self._timings

    @timings.setter
    def timings(self, value):
        self._timings = value

try:
    import torch

    class PyTorchModel(DosePredictionModel, ABC):
        def __init__(self, autocast_bf16=True, **kwargs):
            super().__init__(preload=False, **kwargs)
            self.autocast_bf16 = autocast_bf16
            self.autocast_context = torch.autocast(device_type='cuda', dtype=torch.bfloat16)
            self.load(self.model_weights_filepath)

        def save(self, path):
            torch.save(self.model.state_dict(), path)

        def load(self, path):
            self.model = torch.load(path, map_location=self.device)
            self.model.eval()

        def preprocess(self, geometry_dev, raytrace_dev, output_cuda_array=None, output_view=None, **kwargs):
            if output_cuda_array is None:
                output_cuda_array = self.batch_cuda_buffer
                output_view = self.input_persistent_view

            inv_x_range = 1.0 / (self.scale['x_max'] - self.scale['x_min'])
            inv_r_range = 1.0 / (self.scale['r_max'] - self.scale['r_min'])

            threads = (8, 8, 8)
            grid_depth = geometry_dev.shape[0] * self.input_matrix_size[0]
            blocks = (
                int(np.ceil(grid_depth / threads[0])),
                int(np.ceil(self.input_matrix_size[1] / threads[1])),
                int(np.ceil(self.input_matrix_size[2] / threads[2]))
            )

            self.preprocess_kernel[blocks, threads](
                geometry_dev, 
                raytrace_dev, 
                self.scale['x_min'], 
                inv_x_range, 
                self.scale['r_min'], 
                inv_r_range, 
                output_cuda_array
            )

            return output_view

        @staticmethod
        @cuda.jit
        def preprocess_kernel(geometry, raytrace, x_min, inv_x_range, r_min, inv_r_range, output):
            idx_bd, h, w = cuda.grid(3)
            
            batch_size = output.shape[0]
            depth = output.shape[1]
            height = output.shape[2]
            width = output.shape[3]
            num_rays = raytrace.shape[1]

            b = idx_bd // depth
            d = idx_bd % depth

            if b < batch_size and d < depth and h < height and w < width:
                output[b, d, h, w, 0] = (geometry[b, d, h, w] - x_min) * inv_x_range
                for r in range(num_rays):
                    output[b, d, h, w, r + 1] = (raytrace[b, r, d, h, w] - r_min) * inv_r_range

        def infer(self, input_data=None, **kwargs):
            if input_data is None:
                input_data = self.input_persistent_view
            return self.run_inference(self.model, input_data, self.autocast_bf16, self.autocast_context)

        @staticmethod
        @torch.inference_mode()
        def run_inference(model, tensor, autocast_bf16=True, autocast_context=None):
            if not autocast_bf16:
                return model(tensor)
            if autocast_context is not None:
                with autocast_context:
                    return model(tensor)
            else:
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    return model(tensor)

        def postprocess(self, inference_output, cutoff=0.5, **kwargs):
            preds = inference_output.to(torch.float32)
            preds = preds.squeeze(1)
            y_range = self.scale["y_max"] - self.scale["y_min"]
            preds = (preds * y_range) + self.scale["y_min"]
            cutoff_val = (cutoff / 100.0) * self.scale["y_max"]
            preds = torch.where(preds < cutoff_val, 0, preds)
            preds_cp = cupy.asarray(preds.detach().contiguous())
            return cuda.as_cuda_array(preds_cp)

        def preprocess_infer_postprocess(self, geometry_dev, raytrace_dev, **kwargs):
            return self.postprocess(
                self.infer(
                    self.preprocess(
                        geometry_dev, raytrace_dev, 
                        output_cuda_array=self.batch_cuda_buffer,
                        output_view=self.input_persistent_view,
                    **kwargs),
                **kwargs),
            **kwargs)

        def print_summary(self):
            try:
                from torchinfo import summary
                summary(self.model, input_size=(1, self.input_channels, *self.input_matrix_size), mode='eval')
            except ImportError:
                print('torchinfo not installed, model summary cannot be printed.')

        def set_persistent_view(self, buffer_cuda_array):
            raw_view = torch.from_dlpack(cupy.asarray(buffer_cuda_array).toDlpack())
            self.input_persistent_view = raw_view.permute(0, 4, 1, 2, 3).contiguous(memory_format=torch.channels_last_3d)

        def set_model_paths_from_dirpath(self, model_data_dirpath):
            model_data = os.listdir(model_data_dirpath)
            self.model_weights_filepath = os.path.join(model_data_dirpath, [f for f in model_data if 'best' in f and (f.endswith('pt'))][0])
            self.model_hyperparams_filepath = os.path.join(model_data_dirpath, [f for f in model_data if 'hyperparam' in f][0])
            self.scale_constants_filepath = os.path.join(model_data_dirpath, [f for f in model_data if 'scale' in f][0])

except ImportError:
    pass

try:
    import tensorflow as tf

    class TensorflowModel(DosePredictionModel, ABC):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.platform = 'tensorflow'

        def save(self, path, weights_only=False):
            if weights_only:
                try:
                    self.model.save_weights(path)
                except:
                    self.model.save_weights(path.replace('keras', 'weights.h5'))
            else:
                self.model.save(path)

        def load(self, path: str):
            self.model = tf.keras.models.load_model(path)

        def preprocess(self, geometry_dev, raytrace_dev, output_cuda_array=None, output_view=None, target_batch_size=None, **kwargs):
            b = geometry_dev.shape[0]
            nx, ny, nz = self.input_matrix_size
        
            if output_cuda_array is None:
                alloc_size = target_batch_size if target_batch_size else b
                output_cuda_array = cuda.device_array((alloc_size, nx, ny, nz, 2), dtype=np.float32)
                output_view = tf.experimental.dlpack.from_dlpack(cupy.asarray(output_cuda_array).toDlpack())

            inv_geom_range = 1.0 / (self.scale['x_max'] - self.scale['x_min'])
            inv_ray_range  = 1.0 / (self.scale['r_max'] - self.scale['r_min'])

            threads = (8, 8, 8)
            blocks = ((nx + 7) // 8, (ny + 7) // 8, (nz + 7) // 8)

            for i in range(b):
                self.rescale_and_stack_kernel[blocks, threads](
                    geometry_dev[i,:,:,:], self.scale['x_min'], inv_geom_range, output_cuda_array[i], 0
                )
                self.rescale_and_stack_kernel[blocks, threads](
                    raytrace_dev[i,0,:,:,:], self.scale['r_min'], inv_ray_range, output_cuda_array[i], 1
                )

            cuda.synchronize()
            
            return output_view

        def infer(self, input_data=None, **kwargs):
            if input_data is None:
                input_data = self.input_persistent_view
            return self.optimized_model_call(self.model, input_data)

        def postprocess(self, inference_output, cutoff=0.5, **kwargs):
            preds_cp = cupy.from_dlpack(tf.experimental.dlpack.to_dlpack(inference_output))
            y_range = self.scale['y_max'] - self.scale['y_min']
            preds_cp = preds_cp * y_range + self.scale['y_min']
            threshold = (cutoff / 100.0) * self.scale['y_max']
            preds_cp[preds_cp < threshold] = 0
            preds_cp = cupy.squeeze(preds_cp, axis=-1)
            return cuda.as_cuda_array(preds_cp)

        def preprocess_infer_postprocess(self, geometry_dev, raytrace_dev, **kwargs):
            return self.postprocess(
                self.infer(
                    self.preprocess(
                        geometry_dev, raytrace_dev, 
                        output_cuda_array=self.batch_cuda_buffer,
                        output_view=self.input_persistent_view,
                    **kwargs),
                **kwargs),
            **kwargs)
        
        @staticmethod
        @tf.function(jit_compile=True)
        def optimized_model_call(model, x):
            return model(x, training=False)

        @staticmethod
        @cuda.jit
        def rescale_and_stack_kernel(input_data, x_min, inv_range, output_slice, channel_idx):
            i, j, k = cuda.grid(3)
            if i < input_data.shape[0] and j < input_data.shape[1] and k < input_data.shape[2]:
                val = input_data[i, j, k]
                output_slice[i, j, k, channel_idx] = (val - x_min) * inv_range

        def print_summary(self):
            self.model.summary()

        def set_persistent_view(self, buffer_cuda_array):
            self.input_persistent_view = tf.experimental.dlpack.from_dlpack(cupy.asarray(buffer_cuda_array).toDlpack())

        def set_model_paths_from_dirpath(self, model_data_dirpath):
            model_data = os.listdir(model_data_dirpath)
            self.model_hyperparams_filepath = os.path.join(model_data_dirpath, [f for f in model_data if 'hyperparam' in f][0])
            self.scale_constants_filepath = os.path.join(model_data_dirpath, [f for f in model_data if 'scale' in f][0])
            
            keras_files = [f for f in model_data if 'best' in f and (f.endswith('keras'))]
            if len(keras_files) > 0:
                self.model_weights_filepath = os.path.join(model_data_dirpath, keras_files[0])
            else:
                weights_files = [f for f in model_data if 'best' in f and (f.endswith('weights.h5'))]
                if len(weights_files) > 0:
                    self.model_weights_filepath = os.path.join(model_data_dirpath, weights_files[0])

except ImportError:
    pass

