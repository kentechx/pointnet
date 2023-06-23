<img src="./pointnet.jpg" width="1200px"></img>

[![PyPI version](https://badge.fury.io/py/pointnet.svg)](https://badge.fury.io/py/pointnet)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# pointnet

A pytorch implementation of PointNet and PointNet++.

## Installation

```bash
pip install pointnet
```

If you encounter `No matching distribution found for pointnet` using a mirror source, please install from source:

```bash
pip install pointnet -i https://pypi.org/simple
```

## Usage

### PointNet
Perform classification with inputs xyz coordinates:

```python
import torch
from pointnet import PointNetCls

model = PointNetCls(in_dim=3, out_dim=40)
x = torch.randn(16, 3, 1024)
logits = model(x)
```

If you have other features, you can put them after the xyz coordinates:

```python
import torch
from pointnet import PointNetCls, STN

in_dim = 3 + 10
stn_3d = STN(in_dim=in_dim, out_nd=3)
model = PointNetCls(in_dim=in_dim, out_dim=40, stn_3d=stn_3d)
xyz = torch.randn(16, 3, 1024)
other_feats = torch.randn(16, 10, 1024)
x = torch.cat([xyz, other_feats], dim=1)
logits = model(x)
```

Perform semantic segmentation:

```python
import torch
from pointnet import PointNetSeg

model = PointNetSeg(3, 40)
x = torch.randn(16, 3, 1024)
logits = model(x)
```

### PointNet2
PointNet2 uses [taichi](https://github.com/taichi-dev/taichi) to accelerate the computation of ball query. You need to 
initialize taichi before using PointNet2.

Perform classification with inputs xyz coordinates:

```python
import torch
from pointnet import PointNet2SSGCls

import taichi as ti
ti.init(arch=ti.cuda)

model = PointNet2SSGCls(in_dim=3, out_dim=40).cuda()
x = torch.randn(16, 3, 1024).cuda()
logits = model(x)
```



## Performance
Classification accuracy on ModelNet40 dataset (2048 points, see [modelnet40_experiments](
https://github.com/kentechx/modelnet40_experiments) for details):

| Model                   | Overall Accuracy |
|-------------------------|------------------|
| PointNet                | 89.4%            |

## Other Implementationss
[charlesq34/pointnet](https://github.com/charlesq34/pointnet)

[fxia22/pointnet.pytorch](https://github.com/fxia22/pointnet.pytorch)

[yanx27/Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)


## References

```bibtex
@article{qi2017pointnet,
  title={Pointnet: Deep learning on point sets for 3d classification and segmentation},
  author={Qi, Charles R and Su, Hao and Mo, Kaichun and Guibas, Leonidas J},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  year={2017}
}
```

```bibtex
@article{qi2017pointnet++,
  title={Pointnet++: Deep hierarchical feature learning on point sets in a metric space},
  author={Qi, Charles Ruizhongtai and Yi, Li and Su, Hao and Guibas, Leonidas J},
  journal={Advances in neural information processing systems},
  volume={30},
  year={2017}
}
```
