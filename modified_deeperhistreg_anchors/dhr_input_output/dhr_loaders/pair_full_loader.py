### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import pathlib
from typing import Union, Iterable, Tuple
import logging

### External Imports ###
import numpy as np
import torch as tc

import pyvips

### Internal Imports ###

from loader import WSILoader, LoadMode
from vips_loader import VIPSLoader
from dhr_utils import utils as u

########################


class PairFullLoader():
    """
    TODO - documentation
    """
    def __init__(
        self,
        source_path : Union[str, pathlib.Path],
        target_path : Union[str, pathlib.Path],
        loader : WSILoader = VIPSLoader,
        mode : LoadMode = LoadMode.NUMPY,
        ):
        """
        TODO
        """
        self.source_path = source_path
        self.target_path = target_path
        self.mode = mode
        self.source_loader : WSILoader = loader(self.source_path, mode=self.mode)
        self.target_loader : WSILoader = loader(self.target_path, mode=self.mode)
    
    def pad_to_same_shape(
        self,
        source : Union[tc.Tensor, np.ndarray],
        target : Union[tc.Tensor, np.ndarray],
        pad_value : float) -> Tuple[Union[tc.Tensor, np.ndarray], Union[tc.Tensor, np.ndarray], dict]:
        """
        TODO
        """
        padded_source, padded_target, padding_params = u.pad_to_same_size(source, target, pad_value)
        return padded_source, padded_target, padding_params
    
    def pad_to_same_shape_image(
        self,
        source : pyvips.Image,
        target : pyvips.Image,
        pad_value : float) -> Tuple[pyvips.Image, pyvips.Image, dict]:
        """
        TODO
        """
        padded_source, padded_target, padding_params = u.pad_to_same_size(source, target, pad_value)
        return padded_source, padded_target, padding_params
        
    def load_array(
        self,
        source_resample_ratio : float = 1.0,
        target_resample_ratio : float = 1.0,
        pad_value : float = 255) -> Tuple[Union[tc.Tensor, np.ndarray], Union[tc.Tensor, np.ndarray], dict]:
        """
        TODO
        """
        # print(self.source_loader.image.level_downsamples, self.source_loader.image.level_dimensions)
        # print(self.target_loader.image.level_downsamples, self.target_loader.image.level_dimensions)
        # 从level_0开始load
        source = self.source_loader.resample(source_resample_ratio)
        target = self.target_loader.resample(target_resample_ratio)
        
        padded_source, padded_target, padding_params = self.pad_to_same_shape(source, target, pad_value)
        
        """
        acrobat valid_0  横的长  PIL.Image.size是(width, height)
        # res_level_0 res (17024, 35840, 3) (16128, 48128, 3) 里是 (height, width), 放进res前手动换过顺序的
        Source shape: torch.Size([1, 3, 3405, 9626])
        Target shape: torch.Size([1, 3, 3405, 9626])
        Preprocessed source shape: torch.Size([1, 1, 3405, 9626])
        Preprocessed target shape: torch.Size([1, 1, 3405, 9626])

        sdpc   也是横的长 (说明横竖搞反了) 改了loader.get_best_level里org_height, org_width = self.image.level_dimensions[0]
        现在
        # res_level0 (24192, 40320, 3) (29568, 99456, 3) 里是 (height, width)
        # level0_dim (40320, 24192) (99456, 29568) 里是 (width, height)
        best_level_img w: 10080 h: 6048
        best_level_img w 24864 h 7392 
        (之前问题出在self.resolutions里是h,w,是手动换过顺序的, 不能self.resolutions出来的直接塞进read_region里)

        Source shape: torch.Size([1, 3, 5914, 19891])
        Target shape: torch.Size([1, 3, 5914, 19891])
        Preprocessed source shape: torch.Size([1, 1, 4096, 13776])
        Preprocessed target shape: torch.Size([1, 1, 4096, 13776])
        warped_source.jpg: 4096 * 13776  target.jpg: 4096 * 13776
        """
        return padded_source, padded_target, padding_params, source.shape, target.shape
    
    def load_image(
        self,
        level : int = 0,
        pad_value : float = 255) -> Tuple[Union[pyvips.Image, pyvips.Image], dict]:
        """
        TODO
        """
        self.source_loader.mode = LoadMode.PYVIPS
        self.target_loader.mode = LoadMode.PYVIPS
        source = self.source_loader.load_level(level=level)
        target = self.target_loader.load_level(level=level)
        self.source_loader.mode = self.mode
        self.target_loader.mode = self.mode
        padded_source, padded_target, padding_params = self.pad_to_same_shape_image(source, target, pad_value)
        return padded_source, padded_target, padding_params        