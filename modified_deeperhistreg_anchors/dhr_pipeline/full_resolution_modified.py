### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from typing import Iterable
import time

### External Imports ###
import numpy as np
import torch as tc
import torch.nn.functional as F
### Internal Imports ###
from dhr_preprocessing import preprocessing as pre
from dhr_utils import utils as u
from dhr_utils import warping as w

from dhr_input_output.dhr_loaders import tiff_loader
from dhr_input_output.dhr_loaders import vips_loader
from dhr_input_output.dhr_loaders import pil_loader
from dhr_input_output.dhr_loaders import openslide_loader
from dhr_input_output.dhr_loaders import pair_full_loader
from dhr_input_output.dhr_loaders import opensdpc_loader

from dhr_input_output.dhr_savers import pil_saver
from dhr_input_output.dhr_savers import tiff_saver
from modified_deeperhistreg_anchors.dhr_registration.dhr_initial_alignment.multi_feature_modified import multi_feature_modified
########################

loader_mapper = {
    'tiff' : tiff_loader.TIFFLoader,
    'vips' : vips_loader.VIPSLoader,
    'pil' : pil_loader.PILLoader,
    'openslide' : openslide_loader.OpenSlideLoader,
    'opensdpc' : opensdpc_loader.OpenSdpcLoader
}
    
saver_mapper = {
    'tiff' : tiff_saver.TIFFSaver,
    'pil' : pil_saver.PILSaver,
}

saver_params_mapper = {
    'tiff' : tiff_saver.default_params,
    'pil' : pil_saver.default_params,
}

class DeeperHistReg_FullResolution():
    def __init__(self, registration_parameters : dict):
        self.registration_parameters = registration_parameters
        self.registration_parameters['device'] = self.registration_parameters['device'] if tc.cuda.is_available() else "cpu"
        self.device = self.registration_parameters['device']
        self.echo = self.registration_parameters['echo']
        self.case_name = self.registration_parameters['case_name']

    def load_images(self) -> None:
        loading_params = self.registration_parameters['loading_params']
        pad_value = loading_params['pad_value']
        loader = loader_mapper[loading_params['loader']]
        source_resample_ratio = loading_params['source_resample_ratio']
        target_resample_ratio = loading_params['target_resample_ratio']
        pair_loader = pair_full_loader.PairFullLoader(self.source_path, self.target_path, loader=loader, mode=pair_full_loader.LoadMode.PYTORCH)
        self.source, self.target, self.padding_params, ori_source_shape, ori_target_shape = pair_loader.load_array(source_resample_ratio=source_resample_ratio,
                                                                                target_resample_ratio=target_resample_ratio,
                                                                                pad_value=pad_value)
        
        self.source_shape_before_pad = ori_source_shape
        self.target_shape_before_pad = ori_target_shape

        self.padding_params['source_resample_ratio'] = source_resample_ratio
        self.padding_params['target_resample_ratio'] = target_resample_ratio
        
        self.org_source, self.org_target = self.source.to(tc.float32).to(self.device), self.target.to(tc.float32).to(self.device)
    

    def run_prepreprocessing(self) -> None:
        with tc.set_grad_enabled(False):
            b_t = time.time()
            self.preprocessing_params = self.registration_parameters['preprocessing_params']
            preprocessing_function = pre.get_function(self.preprocessing_params['preprocessing_function'])
            self.pre_source, self.pre_target, _, _, self.postprocessing_params = preprocessing_function(self.org_source, self.org_target, None, None, self.preprocessing_params)
            
            e_t = time.time()
            self.preprocessing_time = e_t - b_t
            self.padding_params['initial_resampling'] = self.postprocessing_params['initial_resampling']
            if self.postprocessing_params['initial_resampling']:
                self.padding_params['initial_resample_ratio'] = self.postprocessing_params['initial_resample_ratio']
            self.current_displacement_field = u.create_identity_displacement_field(self.pre_source)
            tc.cuda.empty_cache()
                
    def preprocessing(self) -> None:
        self.run_prepreprocessing()


    def find_landmarks(
        self, source_path,
        target_path,
        initial_transform=None,
        device='cuda:0',
        landmark_paras=None) -> None:
        
        self.source_path, self.target_path = source_path, target_path
        
        if initial_transform is not None:
            self.initial_transform = initial_transform.to(device)
            grid = F.affine_grid(self.initial_transform, self.pre_source.shape, align_corners=False)
            warped = F.grid_sample(self.pre_source, grid, align_corners=False, mode='bilinear')
            # transformed_tensor = F.grid_sample(tensor, sampling_grid, mode=mode, padding_mode=padding_mode, align_corners=False)
        else:
            warp = None
        
        initial_registration_params = self.registration_parameters['initial_registration_params']
        
        is_warp = landmark_paras['is_warp']
        if is_warp:
            input_source = warped
        else:
            input_source = self.pre_source
        
        best, sift_ransac, superpoint_superglue = multi_feature_modified(input_source, self.pre_target, initial_registration_params, landmark_paras=landmark_paras)
        # print('input_source.dtype', input_source.dtype)
        return best, sift_ransac, superpoint_superglue
