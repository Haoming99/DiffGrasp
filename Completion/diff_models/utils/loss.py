import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.datamodules import mask_op

def get_loss_weight(pts, mask_type, pos_weight, neg_weight):
    mask_op[mask_type]