# MoGe ONNX Support

MoGe-2 is compatible with the ONNX format (opset version ≥ 14). We have exported several models for use in ONNXRuntime or deployment on other compatible inference engines.

> **Important Note:** The `.infer()` method in our PyTorch code includes some post-processing logic (e.g., recovering focal and shift and reprojection) that cannot be exported to ONNX. The ONNX model only includes the raw forward() pass, which outputs intermediate predictions (affine point map, normal map, floating point mask, metric scale). You will need to implement any required post-processing steps separately if replicating the full inference pipeline.

The exported models are in **FP32** precision, with **dynamic input resolution** and **variable-length** token support. You can further optimize these models based on your target deployment platform.

<table>
  <thead>
    <tr>
      <th>Version</th>
      <th>Hugging Face Model</th>
    </tr>
  </thead>
  <tbody>
    <tr>
     <td rowspan="3">MoGe-2</td>
      <td><a href="https://huggingface.co/Ruicheng/moge-2-vitl-normal-onnx" target="_blank"><code>Ruicheng/moge-2-vitl-normal-onnx</code></a></td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Ruicheng/moge-2-vitb-normal-onnx" target="_blank"><code>Ruicheng/moge-2-vitb-normal-onnx</code></a></td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Ruicheng/moge-2-vits-normal-onnx" target="_blank"><code>Ruicheng/moge-2-vits-normal-onnx</code></a></td>
    </tr>
  </tbody>
</table>

## Customized Exportation

### Dynamic Shape & Variable Number of Tokens
```python
import os
os.environ['XFORMERS_DISABLED'] = '1'   # Disable xformers
import numpy as np
import torch
from moge.model.v2 import MoGeModel

PRETRAINED_MODEL = 'Ruicheng/moge-2-vits-normal.pt'
ONNX_FILE = 'moge-2-vits-normal.onnx'

model = MoGeModel.from_pretrained(PRETRAINED_MODEL)
model.onnx_compatible_mode = True  # Enable ONNX compatible mode

torch.onnx.export(
    model, 
    (torch.rand(1, 3, 518, 518), torch.tensor(1800)),
    ONNX_FILE,
    input_names=['image', 'num_tokens'],
    output_names=['points', 'normal', 'mask', 'metric_scale'],
    dynamic_axes={
        'image': {0: 'batch_size', 2: 'height', 3: 'width'},
    },
    opset_version=14
)
```

### Static Shape & Fixed Number of Tokens

```python
import os
os.environ['XFORMERS_DISABLED'] = '1'   # Disable xformers
import numpy as np
import torch
from moge.model.v2 import MoGeModel

class MoGeStatic(MoGeModel):
    def forward(self, image: torch.Tensor):
        return super().forward(image, NUM_TOKENS)

NUM_TOKENS = 1800
FIXED_IMAGE_INPUT = torch.rand(1, 3, 518, 518)
PRETRAINED_MODEL = 'Ruicheng/moge-2-vits-normal.pt'
ONNX_FILE = 'moge-2-vits-normal.onnx'

model = MoGeStatic.from_pretrained(PRETRAINED_MODEL)
model.onnx_compatible_mode = True  # Enable ONNX compatible mode

torch.onnx.export(
    model, 
    (FIXED_IMAGE_INPUT,),
    ONNX_FILE,
    input_names=['image'],
    output_names=['points', 'normal', 'mask', 'metric_scale'],
    dynamic_axes=None,
    opset_version=14
)
```
