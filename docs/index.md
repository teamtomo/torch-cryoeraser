# Overview

*torch-cryoeraser* is a Python package for erasing local regions of cryo-EM images in PyTorch.

<script
  defer
  src="https://unpkg.com/img-comparison-slider@7/dist/index.js"
></script>


<img-comparison-slider tabindex="0">
  <img slot="first" src="https://user-images.githubusercontent.com/7307488/205206563-00944ef6-02b9-4830-9e67-86daed9ffffb.png"/>
  <img slot="second" src="https://user-images.githubusercontent.com/7307488/205206583-c9df5cdb-2034-484b-99d2-ce07827e90e3.png" />
</img-comparison-slider>

Image data in masked regions are replaced with noise matching local image statistics.

## Installation

```python
pip install torch-cryoeraser
```

## Usage

```python
import torch
import tifffile
from torch_cryoeraser import erase_region_2d

# load image and mask
image = tifffile.imread("image.tif")
mask = tifffile.imread("mask.tif")

# to torch tensor
image = torch.tensor(image)
mask = torch.tensor(mask)

# erase masked regions
erased_image = erase_region_2d(image=image, mask=mask)
```
