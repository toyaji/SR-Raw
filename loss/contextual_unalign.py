import torch
import torch.nn as nn

import loss.functional as F

class UnalignmentLoss(nn.Module):
    """
    Create a criterion that measure how much given inpusts are unalinged against its pair.
    
    Parameters:
    tol : int
        tolerance(pixel buffer) for boundary unalingment between x and y.
    stride : int
        stride size for making aligned features by translating input.
    loss_type : str, optional
        a loss type to measure the degree of unlignment of given paris. ['l1', 'l2', 'cosine']
    """
    def __init__(self, 
                 tol: int = 16, 
                 stride: int = 1,
                 loss_type: str = 'l1'):

        super(UnalignmentLoss, self).__init__()

        self.tol = tol
        self.stride = stride
        self.loss_type = loss_type


    def forward(self, x, y):
        assert self.tol >= 0 and self.tol < x.size()[2], \
            'tolarance pixel size could not be negative or more than input size'

        return F.unalign_loss(x, y, self.tol, self.stride, self.loss_type)