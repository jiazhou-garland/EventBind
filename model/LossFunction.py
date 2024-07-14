import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch

def symmetric_cross_entropy_loss(logits):
# symmetric loss function
    batch_size = logits.shape[0]
    device = logits.device
    labels = torch.arange(batch_size, device=device, dtype=torch.long)
    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.T, labels)
    loss = (loss_i + loss_t)/2
    return loss