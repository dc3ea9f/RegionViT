# Region ViT

This repo is the pytorch implementation of ["RegionViT: Regional-to-Local Attention for Vision Transformers"](https://arxiv.org/pdf/2106.02689.pdf) by Chun-Fu Chen, Rameswar Panda, Quanfu Fan.

The code is implemented based on [Microsoft/Swin-Transformer](https://github.com/microsoft/Swin-Transformer).

This is **NOT** the official repository of RegionViT. At the moment in time the official code of the authors 
is not available yet but can be found later at: [https://github.com/IBM/RegionViT](https://github.com/IBM/RegionViT).

## Getting Started
This repo only contain the implementation of the RegionViT for image classification.

Since the implementation is largely based on [Microsoft/Swin-Transformer](https://github.com/microsoft/Swin-Transformer) 
and the code structure is well maintained. You can refer to [this doc](https://github.com/microsoft/Swin-Transformer/blob/main/get_started.md).

Just change the config file to `configs/regionvit/{}.yaml` for RegionViT.

## Some Issues
1. The instance repetition haven't been implemented (Table 11 in paper).
2. `RegionViT-{}+` haven't been implemented due to the limited details.

## Citing RegionViT
```
@article{chen2021regionvit,
  title={RegionViT: Regional-to-Local Attention for Vision Transformers},
  author={Chen, Chun-Fu and Panda, Rameswar and Fan, Quanfu},
  journal={arXiv preprint arXiv:2106.02689},
  year={2021}
}
```

## Acknowledgements
Repo [Microsoft/Swin-Transformer](https://github.com/microsoft/Swin-Transformer) really helps a lot when implement the code.

