from fileinput import filename
import cv2
import numpy as np
import rawpy
from pathlib import Path
from .srdata import SRData
from . import common

class ZoomRaw2RGBv2(SRData):
    def __init__(self, dir, scale, name='ZoomRaw2RGBV2', train=True, val=True, patch_size=48, rgb_range=1, augment=True, **kwargs):

        super(ZoomRaw2RGBv2, self).__init__(dir=dir, scale=scale, name=name, train=train, patch_size=patch_size, 
                                    n_colors=3, rgb_range=rgb_range, augment=augment)
        # seperate val set from test set because their patch sizes differ                         
        self.val = val

    def _set_filesystem(self, data_dir):
        self.apath = Path("/data/paul/") / data_dir

        assert self.apath.exists(), "Given base data dir path is wrong!: {}".format(self.apath)

        if self.train:
            self.raw_hr = self.apath / 'train'
        elif self.train and self.val:
            self.raw_hr = self.apath / 'test'
        else:
            self.raw_hr = self.apath / 'test'

        assert self.raw_hr.exists(), "HR input data path does not exist!"

        self.ext = ("ARW", "ARW")

    def __getitem__(self, idx):
        lr, hr, wb, filename = self._load_file(idx)
        pair = common.set_channel(lr, hr, n_channels=self.n_colors)
        pair_t = common.np2Tensor(*pair, rgb_range=self.rgb_range)
        return pair_t[0], pair_t[1], wb, filename

    def _scan(self):
        self.hr_pathes = sorted(list(self.raw_hr.glob("*." + self.ext[0])))

    def _load_file(self, idx):
        f_hr = self.hr_pathes[idx]
        filename = f_hr.name
        iset = np.load(f_hr)
        lr_raw = iset['lr_raw']
        hr_rgb = iset['hr_rgb']
        wb = iset['wb']
        return lr_raw, hr_rgb, wb, filename
   