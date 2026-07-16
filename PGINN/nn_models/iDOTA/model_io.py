import keras
import numpy as np
import tensorflow as tf
from PGINN.nn_models.iDOTA.models import dota_residual
from PGINN.nn_models.iDOTA.blocks import (
    PosEmbedding, LinearProj, ConvBlock, CatLayer,
    TransformerEncoder, ConvEncoder, ConvDecoder
)
from PGINN.nn_models.iDOTA.loss import SharpLoss, CombinedLoss
from .evaluation import stack_batch_input, optimized_model_call

def create_model_backbone(param):
    return dota_residual(
        inshape=param['inshape'],
        steps=param['num_levels'],
        enc_feats=param['enc_feats'],
        num_heads=param['num_heads'],
        num_transformers=param['num_transformers'],
        kernel_size=param['kernel_size']
    )

def save_model(model, path, weights_only=False):
    if weights_only:
        try:
            model.save_weights(path)
        except:
            model.save_weights(path.replace('keras', 'weights.h5'))
    else:
        model.save(path)


def load_model(model_path, param=None, loss=None, input_view=None):
    try:
        loss = CombinedLoss if loss == 'Combined' else SharpLoss
        model = keras.saving.load_model(model_path,
                            custom_objects={
                                "PosEmbedding": PosEmbedding,
                                "LinearProj": LinearProj,
                                "ConvBlock": ConvBlock,
                                "TransformerEncoder": TransformerEncoder,
                                "ConvEncoder": ConvEncoder,
                                "ConvDecoder": ConvDecoder,
                                "CatLayer" : CatLayer,
                                "Loss" : loss,
                            },
                            compile=True,
                            safe_mode=False
                            )
        
    except Exception as e:
        print(f'Error: {e}')
        print('Failed to load full model. Creating backbone and loading weights.')
        
        model = create_model_backbone(param)

        path = model_path
        if 'keras' in model_path:
            path = model_path.replace('keras', 'weights.h5')
        if 'ckpt.index' in model_path:
            path = model_path.replace('.index', '')

        try:
            model.load_weights(path)
        except:
            model.load_weights(path).expect_partial()
        
    if input_view is not None:
        for i in range(3):
            _ = optimized_model_call(model, input_view)
        print("Loaded model warmed up.")

    return model