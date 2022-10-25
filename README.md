# pstage_01_image_classification

## Getting Started    
### Dependencies
- torch==1.6.0
- torchvision==0.7.0                                                              

### [PyTorch Template](https://github.com/lukemelas/EfficientNet-PyTorch)

# 10월25일 화요일 피어세션

## Baseline 실험 목록

- [TorchVision](https://pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b7.html?highlight=efficientnet#torchvision.models.efficientnet_b7) → EfficientNet-B7 : 1 이강희님
- [mmClassification](https://mmclassification.readthedocs.io/en/latest/papers/efficientnet.html) → EfficientNet-B7 : 2 함수민, 최휘준 님
- [Implemented EfficientNet-B7](https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py) : 2 박용민, 정상헌님

### Hyperparameter

| Learning Rate | Epoch | Batch Size | SE | Scheduler | loss | Transform | label | optimizer | Aug |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.001 | 15 | 32 | OK | step-LR | CE | ToTensor | 18 (Multi-class) | Adam | centercrop(320x256) |