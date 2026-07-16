import keras
import cupy
from numba import cuda
import numpy as np
import tensorflow as tf

from .base import TensorflowModel, ModelFactory

@ModelFactory.register("iDOTA")
class iDOTAModel(TensorflowModel):
    def __init__(self, **kwargs):
        super().__init__(preload=False, **kwargs)
        
        """
        Before loading, check whether to process input data (density/ray-traced doses) as a single batched input
        (new behavior used in models trained after 03/26/2026).
        If false, the process will feed them as separate inputs to model (original iDOTA behavior).
        """
        self.use_stacked_input_pipeline = self.hyperparams.get('input_type', '') != ''
        self.load(self.model_weights_filepath)
        self.architechture_name = 'IDOTA'

    def create_model_backbone(self, param):
        from .iDOTA.models import dota_residual_original, dota_residual_new
        model_to_use = dota_residual_new if self.use_stacked_input_pipeline else dota_residual_original
        return model_to_use(
            inshape=param["inshape"],
            steps=param["num_levels"],
            enc_feats=param["enc_feats"],
            num_heads=param["num_heads"],
            num_transformers=param["num_transformers"],
            kernel_size=param["kernel_size"]
        )

    def load(self, model_path, loss=None):
        print(f"Loading iDOTA model at {model_path}")
        
        try:
            from .iDOTA.blocks import (
                PosEmbedding, LinearProj, ConvBlock, CatLayer,
                TransformerEncoder, ConvEncoder, ConvDecoder
            )
            from .iDOTA.loss import SharpLoss, CombinedLoss, TopPMSE
            loss_fn = CombinedLoss if loss == "Combined" else SharpLoss
            self.model = keras.saving.load_model(
                model_path,
                custom_objects={
                    "PosEmbedding": PosEmbedding,
                    "LinearProj": LinearProj,
                    "ConvBlock": ConvBlock,
                    "TransformerEncoder": TransformerEncoder,
                    "ConvEncoder": ConvEncoder,
                    "ConvDecoder": ConvDecoder,
                    "CatLayer": CatLayer,
                    "Loss": loss_fn,
                },
                compile=True,
                safe_mode=False
            )

        except Exception as e:
            print(f"Failed to load full model. Falling back to backbone. Error: {e}")
            weights_path = model_path
            if "best_model" in weights_path:
                weights_path = weights_path.replace("best_model", "best_weights")
            if "keras" in weights_path:
                weights_path = weights_path.replace("keras", "weights.h5")
            if "ckpt.index" in weights_path:
                path = weights_path.replace(".index", "")
            self.load_from_weights(weights_path)

        if self.input_persistent_view is not None:
            if self.use_stacked_input_pipeline:
                warmup_input = self.input_persistent_view
            else:
                cp_arr = cupy.from_dlpack(tf.experimental.dlpack.to_dlpack(self.input_persistent_view))
                ct_cp  = cupy.ascontiguousarray(cp_arr[..., 0:1])
                tf_ct  = tf.experimental.dlpack.from_dlpack(ct_cp.toDlpack())
                warmup_input = [
                    tf.random.uniform(tf_ct.shape, dtype=tf_ct.dtype),
                    tf.random.uniform(tf_ct.shape, dtype=tf_ct.dtype)
                ]
            for _ in range(3):
                _ = self.infer(warmup_input)

        return self.model

    def load_from_weights(self, path):
        self.model = self.create_model_backbone(self.hyperparams)
        try:
            self.model.load_weights(path)
        except:
            self.model.load_weights(path).expect_partial()

    def preprocess(self, geometry_dev, raytrace_dev, output_cuda_array=None, output_view=None, target_batch_size=None, **kwargs):
        if self.use_stacked_input_pipeline:
            return super().preprocess(
                geometry_dev, raytrace_dev,
                output_cuda_array=output_cuda_array,
                output_view=output_view,
                target_batch_size=target_batch_size, 
                **kwargs
            )
        else:
            return self.old_unstacked_preprocess(
                geometry_dev, raytrace_dev
            )

    def old_unstacked_preprocess(self, geometry_dev, raytrace_dev, target_batch_size=None, **kwargs):
        """
        Preprocessing implementation required for iDOTA models trained before 03/26/2026.
        """
        stacked_view = super().preprocess(
            geometry_dev,
            raytrace_dev,
            target_batch_size=target_batch_size,
            **kwargs
        )

        cp_arr = cupy.from_dlpack(tf.experimental.dlpack.to_dlpack(stacked_view))
        ct_cp  = cupy.ascontiguousarray(cp_arr[..., 0:1])
        ray_cp = cupy.ascontiguousarray(cp_arr[..., 1:2])

        tf_ct  = tf.experimental.dlpack.from_dlpack(ct_cp.toDlpack())
        tf_ray = tf.experimental.dlpack.from_dlpack(ray_cp.toDlpack())

        return [tf_ct, tf_ray]
