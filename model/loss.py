import torch.nn as nn


def CE_loss(output, target): # CrossEntropy Loss
    loss = nn.CrossEntropyLoss()
    return loss(output, target)
