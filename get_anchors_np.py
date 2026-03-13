
from modified_deeperhistreg_anchors.dhr_pipeline.registration_params import default_initial_nonrigid

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), "./modified_deeperhistreg_anchors"))
from PIL import Image
Image.MAX_IMAGE_PIXELS = 2000000000
# ----------------------------------------------------------------
import cv2
import numpy as np
import torch as tc

from dhr_utils import utils as u
from dhr_pipeline import registration_params as rp
from dhr_registration.dhr_initial_alignment.multi_feature_patch import multi_feature_patch
# ----------------------------------------------------------------

def load_anchors_np(source_img, target_img, landmark_paras=None):
    device = landmark_paras['device']

    if isinstance(source_img, np.ndarray):
        source_img = tc.from_numpy(source_img).permute(2, 0, 1).unsqueeze(0)
        target_img = tc.from_numpy(target_img).permute(2, 0, 1).unsqueeze(0)
    
    source = source_img.to(device, dtype=tc.float32)
    target = target_img.to(device, dtype=tc.float32)

    registration_params : dict = default_initial_nonrigid()
    registration_params['loading_params']['loader'] = 'opensdpc'
    registration_params["initial_registration_params"]["device"] = device
    registration_params["device"] = device

    ### Create Config ###
    config = dict()
    config['registration_parameters'] = registration_params
    config['device'] = device
    
    try:
        registration_parameters_path = config['registration_parameters_path']
        registration_parameters = rp.load_parameters(registration_parameters_path)
    except KeyError:
        registration_parameters = config['registration_parameters']
    
    normalization = True
    convert_to_gray = True
    flip_intensity = True
    clahe = True

    if normalization:
        source, target = u.normalize(source), u.normalize(target)
    
    if convert_to_gray:
        if flip_intensity:
            source = 1 - u.convert_to_gray(source)
            target = 1 - u.convert_to_gray(target)
        else:
            source = u.convert_to_gray(source)
            target = u.convert_to_gray(target)

        if clahe:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            src = clahe.apply((source[0, 0].detach().cpu().numpy()*255).astype(np.uint8)) 
            trg = clahe.apply((target[0, 0].detach().cpu().numpy()*255).astype(np.uint8))
            source = tc.from_numpy((src.astype(np.float32) / 255)).to(source.device).unsqueeze(0).unsqueeze(0)
            target = tc.from_numpy((trg.astype(np.float32) / 255)).to(target.device).unsqueeze(0).unsqueeze(0)

    tc.cuda.empty_cache()
    
    initial_registration_params = registration_parameters['initial_registration_params']
    best, sift_ransac, superpoint_superglue = multi_feature_patch(source, target, initial_registration_params, landmark_paras)

    return best, sift_ransac, superpoint_superglue
   
