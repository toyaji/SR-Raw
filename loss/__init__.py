import torch
import torch.nn as nn

from torch import Tensor
from torch.nn.modules import loss
from loss.contextual import ContextualLoss
from loss.contextual_bilateral import ContextualBilateralLoss
from loss.contextual_unalign import UnalignmentLoss


loss_config = {}

class Loss(nn.modules.loss._WeightedLoss):
    def __init__(self,
                 weight: Tensor,
                 CoBi_Bilatera: bool = True,
                 reduction: str = 'mean') -> None:

        super().__init__(weight=weight)

        self.loss_module = nn.ModuleList([])
        
        if CoBi_Bilatera: 
            self.set_CoBi()
        self.loss_module.cuda()

        # assert all([module.is_cuda() for module in self.loss_module])

    def set_CoBi(self):
        CoBi_RGB = ContextualBilateralLoss(use_vgg=False)
        CoBi_VGG = ContextualBilateralLoss(use_vgg=True)
        self.loss_module.add_module('CoBi_RGB', CoBi_RGB)
        self.loss_module.add_module('CoBi_VGG', CoBi_VGG)

    def forward(self, x, y):
        loss = 0
        for w, f in zip(self.weight, self.loss_module):
            loss += w * f.forward(x, y)
        return  loss

    def start_log(self):
        return NotImplemented

    def display_loss(self):
        return NotImplemented       