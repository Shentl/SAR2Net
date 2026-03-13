# from registration_params import default_initial_nonrigid
from modified_deeperhistreg_anchors.dhr_pipeline.registration_params import default_initial_nonrigid
import opensdpc
import openslide

from PIL import Image
Image.MAX_IMAGE_PIXELS = 2000000000
# ----------------------------------------------------------------

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), "./modified_deeperhistreg_anchors"))
import pathlib
import torch as tc
from dhr_pipeline import full_resolution_modified as fr
from dhr_pipeline import registration_params as rp
# ----------------------------------------------------------------


def get_resample_ratio(source_path, target_path):
    if '.sdpc' in str(source_path):
        source = opensdpc.OpenSdpc(str(source_path))  # openslide.OpenSlide(source_path)
        target = opensdpc.OpenSdpc(str(target_path))
    else:
        source = openslide.OpenSlide(str(source_path))  # openslide.OpenSlide(source_path)
        target = openslide.OpenSlide(str(target_path))


    source_dim0, target_dim0 = source.level_dimensions[0], target.level_dimensions[0]
    if (source_dim0[0] * source_dim0[1] > 5e9) or (target_dim0[0] * target_dim0[1] > 5e9): # CUDA OOM
        ratio = 0.1
    elif (source_dim0[0] * source_dim0[1] > 1e9) or (target_dim0[0] * target_dim0[1] > 2e9): 
        ratio = 0.15
    else:
        ratio = 0.2
    
    return ratio

def load_anchors(
        source_path, 
        target_path, 
        output_path,
        slide_id, 
        device=None,  
        initial_resolution=4096, 
        landmark_paras=None
    ):
    source_path = pathlib.Path(source_path)
    target_path = pathlib.Path(target_path)
    output_path = None

    ### Define Params ###
    registration_params : dict = default_initial_nonrigid()
    registration_params['loading_params']['loader'] = 'opensdpc'

    # ----------------------------------------------------------------------------------------------------------------------
    resample_ratio = landmark_paras['resample_ratio']
    if resample_ratio is None:
        resample_ratio =  get_resample_ratio(source_path, target_path)
    else:
        print("Input resample_ratio is: %f" % resample_ratio)
    registration_params['loading_params']['source_resample_ratio'] = resample_ratio # default 0.2
    registration_params['loading_params']['target_resample_ratio'] = resample_ratio
    # -----------------------------------------------------------------------------------------
    
    
    registration_params["initial_registration_params"]["device"] = device
    registration_params["device"] = device
    
    registration_params['preprocessing_params']['initial_resolution'] = initial_resolution
    
    case_name : str = f"{source_path.stem}_{target_path.stem}" # Used only if the temporary_path is important, otherwise - provide whatever

    ### Create Config ###
    config = dict()
    config['source_path'] = source_path
    config['target_path'] = target_path
    config['registration_parameters'] = registration_params
    config['case_name'] = case_name
    config['device'] = device
    
    try:
        registration_parameters_path = config['registration_parameters_path']
        registration_parameters = rp.load_parameters(registration_parameters_path)
    except KeyError:
        registration_parameters = config['registration_parameters']

    source_path = config['source_path']
    target_path = config['target_path']
    experiment_name = config['case_name']


    registration_parameters['case_name'] = experiment_name
    
    pipeline = fr.DeeperHistReg_FullResolution(registration_parameters)
    pipeline.source_path, pipeline.target_path = source_path, target_path
    pipeline.load_images()

    initial_transform = None
    
    pipeline.run_prepreprocessing()
    tc.cuda.empty_cache()
    best, sift_ransac, superpoint_superglue = pipeline.find_landmarks(
        source_path, target_path, initial_transform, device=device, landmark_paras=landmark_paras)

    return best, sift_ransac, superpoint_superglue, initial_transform, pipeline.source, pipeline.target, pipeline.source_shape_before_pad, pipeline.target_shape_before_pad
