import keras
import cupy
from numba import cuda
import numpy as np
import torch
from nnunet_mednext.network_architecture.mednextv1.create_mednext_v1 import create_mednext_v1

from .base import PyTorchModel, ModelFactory

@ModelFactory.register("mednext")
class MedNeXtModel(PyTorchModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load(self, weights_path):
        device = torch.device(self.device if torch.cuda.is_available() else "cpu")

        state = torch.load(weights_path, map_location=device)
        if "state_dict" in state:
            state = state["state_dict"]

        self.model = create_mednext_v1(
            num_input_channels=self.input_channels,
            num_classes=1,
            model_id=self.hyperparams.get('model_id', 'S'),
            kernel_size=self.hyperparams.get('kernel_size', 3)
        )
        
        self.model = self.model.to(memory_format=torch.channels_last_3d)
        self.model.load_state_dict(state)
        self.model = self.model.to(device)
        self.model.eval()

        for p in self.model.parameters():
            p.requires_grad_(False)

        torch._inductor.config.triton.cudagraphs = True
        self.model = torch.compile(self.model, mode="max-autotune")
        
        if self.input_persistent_view is not None:
            with torch.no_grad():
                for _ in range(10):
                    _ = self.infer(self.input_persistent_view)

        return self.model


