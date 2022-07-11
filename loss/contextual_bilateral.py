from os import path
import torch
import torch.nn as nn
import utils
from model import VGG19
import loss.functional as F
from dataset import utils


LOSS_TYPES = ['cosine', 'l1', 'l2']

class ContextualBilateralLoss(nn.Module):
    """
    Creates a criterion that measures the contextual bilateral loss.

    Parameters
    ---
    weight_sp : float, optional
        a balancing weight between spatial and feature loss.
    band_width : int, optional
        a band_width parameter described as :math:`h` in the paper.
    use_vgg : bool, optional
        if you want to use VGG feature, set this `True`.
    vgg_layer : str, optional
        intermidiate layer name for VGG feature.
        Now we support layer names:
            `['relu1_2', 'relu2_2', 'relu3_4', 'relu4_4', 'relu5_4']`
    """

    def __init__(self,
                 weight_sp: float = 0.1,
                 band_width: float = 0.5,
                 loss_type: str = 'cosine',
                 use_vgg: bool = True,
                 vgg_layer: str = 'relu3_4',
                 patch_size: int = 15):

        super(ContextualBilateralLoss, self).__init__()

        assert band_width > 0, 'band_width parameter must be positive.'
        assert loss_type in LOSS_TYPES,\
            f'select a loss type from {LOSS_TYPES}.'
        
        self.weight_sp = weight_sp
        self.band_width = band_width
        self.loss_type = loss_type
        self.patch_szie = patch_size # this is for RGB CoBi when use_vgg is false

        if use_vgg:
            self.vgg_model = VGG19()
            self.vgg_layer = vgg_layer
            self.register_buffer(
                name='vgg_mean',
                tensor=torch.tensor(
                    [[[0.485]], [[0.456]], [[0.406]]], requires_grad=False, )
            )
            self.register_buffer(
                name='vgg_std',
                tensor=torch.tensor(
                    [[[0.229]], [[0.224]], [[0.225]]], requires_grad=False, )
            )

            
    def forward(self, x, y):
        if hasattr(self, 'vgg_model'):
            assert x.shape[1] == 3 and y.shape[1] == 3,\
                'VGG model takes 3 channel images.'

            # normalization
            x = x.sub(self.vgg_mean.detach()).div(self.vgg_std.detach())
            y = y.sub(self.vgg_mean.detach()).div(self.vgg_std.detach())

            # picking up vgg feature maps
            x = getattr(self.vgg_model(x), self.vgg_layer)
            y = getattr(self.vgg_model(y), self.vgg_layer)

        else:
            # get patch for RGB CoBi 
            x, y = utils.get_patch(x, y, patch_size=self.patch_szie)

        return F.contextual_bilateral_loss(x, y, self.weight_sp, self.band_width, self.loss_type)