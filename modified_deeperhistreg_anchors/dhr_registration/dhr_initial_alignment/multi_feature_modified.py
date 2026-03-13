### Ecosystem Imports ###
import os
import sys
current_file = sys.modules[__name__]
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

### External Imports ###
import numpy as np
import torch as tc
import cv2

### Internal Imports ###
from dhr_utils import utils as u
from dhr_utils import warping as w

import sift_ransac as sr
import superpoint_superglue as sg
import torch
import math
import PIL
from scipy.spatial import cKDTree

########################
import cv2
import torch

def warp_coords_ori(forward_transform, coords, img_shape):  
    H, W = img_shape[0], img_shape[1]
    coords = np.array(coords)
    
    normalized_x = (2.0 * coords[:, 1] + 1) / W - 1.0
    normalized_y = (2.0 * coords[:, 0] + 1) / H - 1.0
    
    homogeneous_coords_norm = torch.stack(
        (torch.from_numpy(normalized_x).float(), 
         torch.from_numpy(normalized_y).float(), 
         torch.ones(coords.shape[0], dtype=torch.float32)), 
        dim=0
    )
    
    warped_points_norm = torch.matmul(forward_transform, homogeneous_coords_norm)
    
    warped_x = (warped_points_norm[0, :].numpy() + 1.0) * W / 2.0 - 0.5
    warped_y = (warped_points_norm[1, :].numpy() + 1.0) * H / 2.0 - 0.5

    warped_coords = np.stack((warped_y, warped_x), axis=1)
    return warped_coords

# ---------------------

def multi_feature_modified(
    source : tc.Tensor,
    target : tc.Tensor,
    params=None,
    landmark_paras=None
    ) -> tc.Tensor:

    transforms = []
    super_transforms = []
    
    device = params['device']
    # run_sift_ransac = params['run_sift_ransac']
    run_sift_ransac=False
    run_superpoint_superglue = params['run_superpoint_superglue']
    run_superpoint_ransac = params['run_superpoint_ransac']
    
    initial_resample = landmark_paras['initial_resample']
    reg_size = landmark_paras['reg_size']
    angle_step = landmark_paras['angle_step']
    angle_range = landmark_paras['angle_range']
    merge = landmark_paras['merge']
    merge_thre = landmark_paras['merge_thre']
    multi_anchor = landmark_paras['multi_anchor']
    print('multi_anchor', multi_anchor)

    if initial_resample:
        resolution = landmark_paras['initial_resample_size']
        print('Using initial_resample %s' % resolution)
        source, target = u.initial_resampling(source, target, resolution)
    else:
        print('Direct resample to %d' % reg_size)
    
    angle_start, angle_stop = -angle_range, angle_range

    sift_ransac = None
    angle_num = {}

    for angle in range(angle_start, angle_stop, angle_step):
        _, _, y_size, x_size = source.shape
        x_origin = x_size // 2 
        y_origin = y_size // 2
        r_transform = w.generate_rigid_matrix(angle, x_origin, y_origin, 0, 0)
        r_transform = w.affine2theta(r_transform, (source.size(2), source.size(3))).to(device).unsqueeze(0)
        # print("r_transform dtype:", r_transform.dtype, " device:", r_transform.device, source.dtype, source.size(2), source.size(3))
        
        current_displacement_field = w.tc_transform_to_tc_df(r_transform, (1, 1, source.size(2), source.size(3)))
        transformed_source = w.warp_tensor(source, current_displacement_field)

        params['echo'] = False
       
        ex_params = {**params, **{'registration_size': reg_size}}
        if run_sift_ransac:
            current_transform, num_matches, source_keypoints, target_keypoints, resampled_source = sr.sift_ransac(transformed_source, target, {**ex_params, **{'return_num_matches': True}})
            current_transform = w.compose_transforms(r_transform[0], current_transform[0]).unsqueeze(0)
            
            # --------------------------------------------------------
            source_keypoints = source_keypoints.squeeze(axis=1) # (N, 1, 2)
            target_keypoints = target_keypoints.squeeze(axis=1)
            # --------------------------------------------------------
            ratio = source.shape[2] / resampled_source.shape[2]
            source_keypoints = np.column_stack((source_keypoints[:, 1], source_keypoints[:, 0]))
            source_keypoints = warp_coords_ori(r_transform[0].cpu(), source_keypoints, resampled_source.squeeze().shape)
            source_keypoints = np.column_stack((source_keypoints[:, 1], source_keypoints[:, 0]))
            # --------------------------------------------------------
            
            sift_ransac = (current_transform, source_keypoints, target_keypoints, reg_size, angle)

            transforms.append((current_transform, num_matches, source_keypoints, target_keypoints, reg_size))
            print('angle: %d, size: %s, sift, num_matches: %d' % (angle, reg_size, num_matches))
            
        if run_superpoint_superglue:
            current_transform, num_matches, source_keypoints, target_keypoints, resampled_source = sg.superpoint_superglue(transformed_source, target, {**ex_params, **{'return_num_matches': True}})
            current_transform = w.compose_transforms(r_transform[0], current_transform[0]).unsqueeze(0)

            ratio = source.shape[2] / resampled_source.shape[2]
            
            source_keypoints = np.column_stack((source_keypoints[:, 1], source_keypoints[:, 0]))
            # source_keypoints_big = ratio * source_keypoints
            source_keypoints = warp_coords_ori(r_transform[0].cpu(), source_keypoints, resampled_source.squeeze().shape)
            # source_keypoints = source_keypoints_big / ratio
            source_keypoints = np.column_stack((source_keypoints[:, 1], source_keypoints[:, 0]))
            
            superpoint_superglue = (current_transform, source_keypoints, target_keypoints, reg_size, angle)

            transforms.append((current_transform, num_matches, source_keypoints, target_keypoints, reg_size))
            super_transforms.append((current_transform, num_matches, source_keypoints, target_keypoints, reg_size))
            # print('angle: %d, size: %s, superpoint_superglue, num_matches: %d' % (angle, reg_size, num_matches))
            angle_num[angle] = num_matches
            # print('in pad, resampled_source', resampled_source.shape)

            
        
        
    best_matches = 0
    best_transform = tc.eye(3, device=source.device)[0:2, :].unsqueeze(0)
    for transform, num_matches, source_keypoints, target_keypoints, reg_size_ in transforms:
        if num_matches > best_matches:
            best_transform = transform
            best_matches = num_matches
            s_k, t_k = source_keypoints, target_keypoints
            reg_size = reg_size_
    print(f"Final matches: {best_matches}")
    print('angle--num_matches:', angle_num)

    
    if multi_anchor and (best_matches < 60):
        print('Try new reg_size')
        angle_step = 30
        best_matches_ori = best_matches
        new_transforms = []
        size_transforms = {400:[], 300:[], 200:[]}
        for angle in range(angle_start, angle_stop, angle_step):
            _, _, y_size, x_size = source.shape
            x_origin = x_size // 2 
            y_origin = y_size // 2
            r_transform = w.generate_rigid_matrix(angle, x_origin, y_origin, 0, 0)
            r_transform = w.affine2theta(r_transform, (source.size(2), source.size(3))).to(device).unsqueeze(0)  
            current_displacement_field = w.tc_transform_to_tc_df(r_transform, (1, 1, source.size(2), source.size(3)))
            transformed_source = w.warp_tensor(source, current_displacement_field)

            # registration_sizes = params['registration_sizes'] [150, 200, 250, 300, 350, 400, 450, 500]
            for new_reg_size in [400, 300, 200]:
                ex_params = {**params, **{'registration_size': new_reg_size}}
                current_transform, num_matches, source_keypoints, target_keypoints, resampled_source_ = sg.superpoint_superglue(transformed_source, target, {**ex_params, **{'return_num_matches': True}})
                current_transform = w.compose_transforms(r_transform[0], current_transform[0]).unsqueeze(0)

                
                source_keypoints = np.column_stack((source_keypoints[:, 1], source_keypoints[:, 0]))
                source_keypoints = warp_coords_ori(r_transform[0].cpu(), source_keypoints, resampled_source_.squeeze().shape)
                source_keypoints = np.column_stack((source_keypoints[:, 1], source_keypoints[:, 0]))
                # print(source_keypoints[0])
                # print(angle, 'resampled_source', resampled_source.shape, 'transformed_source', transformed_source.shape, np.max(source_keypoints[:, 1]), np.max(source_keypoints[:, 0]))
                
                source_keypoints, target_keypoints = (reg_size/new_reg_size) * source_keypoints, (reg_size/new_reg_size) * target_keypoints
                new_transforms.append((current_transform, num_matches, source_keypoints, target_keypoints, new_reg_size))
                size_transforms[new_reg_size].append((current_transform, num_matches, source_keypoints, target_keypoints, new_reg_size))
                # print('angle: %d, size: %s, superpoint_superglue, num_matches: %d' % (angle, reg_size, num_matches))
                angle_num[angle] = num_matches
    
        for transform, num_matches, source_keypoints, target_keypoints, reg_size_ in new_transforms:
            if num_matches > best_matches:
                best_matches = num_matches
                s_k, t_k = source_keypoints, target_keypoints
                new_reg_size = reg_size_

        if best_matches > best_matches_ori:
            print('Use new_transforms, with reg_size %s and num matches %s' % (new_reg_size, best_matches))
            super_transforms = size_transforms[new_reg_size]


    
    merge_points = merge
    if merge_points:
        source_list, target_list = [], []
        for transform, num_matches, source_keypoints, target_keypoints, reg_size_ in super_transforms:
            # print(source_keypoints)
            source_list.append(source_keypoints)
            target_list.append(target_keypoints)
            # print('------------------')
        
        print('start merge')
        new_add_s, new_add_t = merge_point_matches(source_list, target_list, s_k, t_k, threshold=merge_thre) # 只要delta_x > thre & delta_y > thre，就新增点
        return (best_transform, s_k, t_k, new_add_s, new_add_t, reg_size, angle), sift_ransac, superpoint_superglue
    
    # return (best_transform, s_k, t_k, reg_size, angle), sift_ransac, superpoint_superglue
    return (best_transform, s_k, t_k, None, None, reg_size, angle), sift_ransac, superpoint_superglue



def merge_point_matches(source_list, target_list, base_source, base_target, threshold=2.0):
    """
    Merge multiple matching points and remove duplicates.
    For a new match, considered a duplicate if the distance between both the source and target ends is less than a threshold
    """
    assert len(source_list) == len(target_list)

    merged_source = base_source.copy()
    merged_target = base_target.copy()

    newly_added_source_list = []
    newly_added_target_list = []

    pair_num = [len(source_list[i]) for i in range(len(source_list))]
    order = sorted(range(len(pair_num)), key=lambda i: pair_num[i], reverse=True)
    
    add_num = {}
    for i in order:
        src_i = source_list[i]
        tgt_i = target_list[i]     
        # KDTree
        tree_src = cKDTree(merged_source)
        tree_tgt = cKDTree(merged_target)

        # For each new point, query its nearest distance in merged_source/target.
        dist_src, _ = tree_src.query(src_i, k=1)
        dist_tgt, _ = tree_tgt.query(tgt_i, k=1)

        # If both distances are greater than the threshold, it means that this matching point is a new match.
        mask_new = (dist_src > threshold) & (dist_tgt > threshold) # &
        add_num[i] = int(mask_new.sum())

        # Add new points
        if np.any(mask_new):
            merged_source = np.concatenate([merged_source, src_i[mask_new]], axis=0)
            merged_target = np.concatenate([merged_target, tgt_i[mask_new]], axis=0)
            
            newly_added_source_list.append(src_i[mask_new])
            newly_added_target_list.append(tgt_i[mask_new])
        
    if newly_added_source_list:
        newly_added_source = np.concatenate(newly_added_source_list, axis=0)
        newly_added_target = np.concatenate(newly_added_target_list, axis=0)   
    else:
        newly_added_source = np.empty((0, 2))
        newly_added_target = np.empty((0, 2))
       
    print('Adding Points:', add_num)
    return newly_added_source, newly_added_target
    