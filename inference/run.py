import os
import torch
import argparse
import cv2
import skimage
from PIL import Image
from tqdm import tqdm
Image.MAX_IMAGE_PIXELS = 2000000000
import numpy as np

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from modified_deeperhistreg_anchors.dhr_utils import utils as u

from utils_np import *
from models_weight import GeometryFeatureNet_weight_big, GeometryFeatureNet_weight_big_split
from inference.utils_ import *

import opensdpc
from get_anchors_slide import load_anchors
from get_anchors_np import load_anchors_np

# ---------------------------------------------------

def compute_target_sim_map_new(
    ori_feat_ref, source_calc,
    target_feat_ref, target_calc,
    binary_mask, H, W,
    b_s=1000, top_k=1, low_thre=0.5, return_t_mask=False, clip=False
):  # ori_feat_ref @ target_feat_ref.T  CUDA OOM, batching

    target_sim_map = np.zeros((H, W), dtype=np.uint8)
    target_coords = target_calc.squeeze(1).int().cpu().numpy()
    source_coords = source_calc.squeeze(1).int().cpu().numpy()
     
    topk_vals, topk_indices = [], []
    # print('target_feat_ref', target_feat_ref.shape) # [189072, 256]
    for i in range(0, target_feat_ref.shape[0], b_s): # batchig
        t_batch = target_feat_ref[i:i+b_s]  # [B, 128]
        sim = t_batch @ ori_feat_ref.T             # [B, 121213]
        
        if top_k == 1:
            topk_val, topk_indice = sim.max(dim=1) # [t_dim, s_dim]
        else:
            topk_val, topk_indice = sim.topk(k=top_k, dim=1, largest=True, sorted=False) #  [t_dim, top_k]
        topk_vals.append(topk_val)
        topk_indices.append(topk_indice) # N_t values, position in source

    topk_vals = torch.cat(topk_vals, dim=0).detach().cpu()
    topk_indices = torch.cat(topk_indices, dim=0).detach().cpu()
    # print('topk_indices', topk_indices.shape)
    # print(torch.max(topk_vals))

    if top_k == 1:
        keep_mask = (topk_vals >= low_thre)  # [N], bool
        t_coords_kept = target_coords[keep_mask]                     # [N', 2]
        topk_indices_kept = topk_indices[keep_mask].cpu().numpy()      # [N']
        matched_coords = source_coords[topk_indices_kept]             # [N', 2]

        x = matched_coords[:, 0]
        y = matched_coords[:, 1]
        in_mask = binary_mask[x, y].astype(np.uint8)                            # [N']，0/1

        valid_mask = in_mask == 1 
    else:
        keep_mask = (topk_vals >= low_thre).all(dim=1) 
        t_coords_kept = target_coords[keep_mask.cpu().numpy()]        # [N', 2]
        topk_indices_kept = topk_indices[keep_mask].cpu().numpy()     # [N', k]
        source_coords_expanded = source_coords[topk_indices_kept]     # [N', k, 2]
        
        x = source_coords_expanded[:, :, 0]
        y = source_coords_expanded[:, :, 1]
        
        if clip:
            x = np.clip(x, 0, binary_mask.shape[0]-1)
            y = np.clip(y, 0, binary_mask.shape[1]-1)
        
        values = binary_mask[x, y]                                     # [N', k]
        in_mask = values.any(axis=1).astype(np.uint8)                 # [N']
        valid_mask = in_mask == 1 

    t_x = t_coords_kept[:, 0]
    t_y = t_coords_kept[:, 1]
    target_sim_map[t_x, t_y] = in_mask
    if return_t_mask:
        return target_sim_map, valid_mask
    return target_sim_map


def get_dir_name(args=None, slide_id=None, target_stain=None):    
    base_dir = 'visualize_biopsy_multi_new' # align_mask
    model_name = args.model_name
    model_name = '205000_multi2_expands_' + model_name
    if 'center' in args.save_dir:
        model_name = model_name + '/center_rot_%s_site_%s' % (args.r_rot, args.r_site)
    base_dir = os.path.join('/nas/username/code/SAR2Net', base_dir)
    if args.angle_step != 15:
        dir_name = os.path.join(base_dir, 'ihc-he/%s/%s/selectp_%s/%s/patch_%s_stride_%s_angle_%s/%s_%s_low_%s_dist_%d_pratio_%s' % 
        (args.loss_mode, model_name, args.select_point, slide_id, 
         args.patch_size, args.stride, args.angle_step, args.t_f_mode, args.split_mode, args.low_thre, args.dist_thre, args.patch_ratio))
    else:
        dir_name = os.path.join(base_dir, 'ihc-he/%s/%s/selectp_%s/%s/patch_%s_stride_%s/%s_%s_low_%s_dist_%d_pratio_%s' % 
        (args.loss_mode, model_name, args.select_point, slide_id, 
         args.patch_size, args.stride, args.t_f_mode, args.split_mode, args.low_thre, args.dist_thre, args.patch_ratio))
    
    if args.s_as_t:
        dir_name = os.path.join(dir_name, 's_as_t')
     
    if args.s_nomask:
        dir_name = dir_name + '_no_smask'
    if args.set_ori:
        dir_name = dir_name + '_ori'
    
    if args.merge_points:
        dir_name = dir_name + '_merge'
    
    if args.multi_anchor: # multi size for anchor extraction
        dir_name = dir_name + '_multi_anchor'

    if target_stain is not None:
        dir_name = os.path.join(dir_name, target_stain)
    os.makedirs(dir_name, exist_ok=True)

    return dir_name


def get_source_name(args=None, stain=None):
    name = 'source_%s_n%d.jpg' % (stain, args.num_p)
    return name


def get_target_name(args=None, stain=None):
    if args.search_mode == 'thre':
        target_name_base = '%s_step_%d_thre_%.3f_tr%s' % (stain, args.step, args.thre, args.t_ratio)
    elif args.search_mode == 'topk':
        target_name_base = '%s_step_%d_top%s' % (stain, args.step, args.top_k)

    target_name = target_name_base + '_n%d' % (args.num_p)
    
    target_name = target_name + '.jpg'
    # print('args.not_in_wsi', args.not_in_wsi)
    if args.not_in_wsi:
        target_name = 'all_' + target_name

    return target_name


def first_round(source_path, target_path, slide_id, device, names):
    source_stain, source_name, source_save_name, target_stain, target_name, target_save_name, dir_name = names
    
    # output_path = os.path.join(output_base_dir, slide_id)
    output_path = None
    source_slide = opensdpc.OpenSdpc(str(source_path))
    target_slide = opensdpc.OpenSdpc(str(target_path))
    
    mask_base = '/nas/username/code/biopsy_thumb/seg_biopsy_trident' 
    target_foreground_dir = os.path.join(mask_base, slide_id, 'contours_geojson', target_name+'.geojson')
    source_foreground_dir = os.path.join(mask_base, slide_id, 'contours_geojson', source_name+'.geojson')
    
    landmark_paras = {
        'load_mask_align': False,
        'target_foreground_dir': target_foreground_dir,
        'source_foreground_dir': source_foreground_dir,
        'is_warp': False,
        'device': device,
        'resample_ratio': None, 
        'initial_resample': True,
        'initial_resample_size':620,
        'reg_size': 500, 
        'angle_step': args.angle_step,
        'angle_range': 180,
        'merge': args.merge_points,
        'merge_thre': 5, 
        'multi_anchor': args.multi_anchor
    } 
    
    best, sift_ransac, superpoint_superglue, initial_transform, pad_source, pad_target, source_shape_before_pad, target_shape_before_pad = load_anchors(
        source_path, target_path, output_path, slide_id, device=device, landmark_paras=landmark_paras)
      
    transform_new, s_k_base, t_k_base, new_add_s, new_add_t, reg_size, angle = best
    if args.merge_points:
        s_k = np.concatenate([s_k_base, new_add_s], axis=0)
        t_k = np.concatenate([t_k_base, new_add_t], axis=0)
    else:
        s_k = s_k_base
        t_k = t_k_base

    print('s_k', len(s_k))
    
    resampled_source, resampled_target = u.initial_resampling_fix_reg(pad_source, pad_target, reg_size) 
    ratio = pad_source.shape[2] / resampled_source.shape[2]
    
    # print(warped.shape, len(s_k)) # (500, 1031, 3)
    s_k = np.column_stack((s_k[:, 1], s_k[:, 0])) 
    t_k = np.column_stack((t_k[:, 1], t_k[:, 0]))
    h, w = resampled_source.shape[2],  resampled_source.shape[3]
    
    s_in_bound = in_image_bound(s_k, h, w)
    t_in_bound = in_image_bound(t_k, h, w)
    in_bound = s_in_bound & t_in_bound 
    s_k = s_k[in_bound]
    t_k = t_k[in_bound]
    print('Original %d key_points, %d key points in bounds' % (len(in_bound), np.sum(in_bound)))
    print(np.max(s_k[:, 0]), np.max(s_k[:, 1]), np.max(t_k), resampled_source.shape)
    
    source_rescale = resampled_source.squeeze().permute(1, 2, 0).numpy() 
    target_rescale = resampled_target.squeeze().permute(1, 2, 0).numpy()

    # endregion
    # --------------------------------------------------------------------------
    # region Foreground
    pad_value = 0
    t_new_shape = (int(1/ratio * target_shape_before_pad[2]), int(1/ratio *target_shape_before_pad[3]))
    t_pad0, t_pad1 = cal_pad_para(t_new_shape, target_rescale.shape)

    target_foreground_dir = os.path.join(mask_base, slide_id, 'contours_geojson', target_name+'.geojson')
    target_foreground_load = get_foreground_biopsy(target_foreground_dir, (t_new_shape[1], t_new_shape[0]), target_slide.level_dimensions[0])
    
    if args.t_f_mode == 'trident':
        target_foreground = target_foreground_load
    elif args.t_f_mode == 'L_225':
        mask_base_L = '/nas/username/code/patch_feature/seg_biopsy_L/225_255_15_c16_at_16_r_32'
        target_foreground = None
        target_foreground_L_dir = os.path.join(mask_base_L, slide_id, 'mask_new', target_name + '.png')
        target_foreground_L = get_foreground_b1b2(t_new_shape, args, mask_dir = target_foreground_L_dir) # Open then Resize
        target_foreground = target_foreground_L
    else:
        raise NotImplementedError

    target_foreground = np.pad(target_foreground, ((t_pad0[0], t_pad0[1]), (t_pad1[0], t_pad1[1])), mode='constant', constant_values=pad_value)
    t_f_save = target_rescale * np.repeat(1-0.5*target_foreground[:, :, None], 3, axis=2)
    Image.fromarray(t_f_save.astype(np.uint8)).save(os.path.join(dir_name, 'target_foreground.jpg'))
    # --------------------------------------------------------------------------------------------------
    s_new_shape = (int(1/ratio * source_shape_before_pad[2]), int(1/ratio *source_shape_before_pad[3]))

    s_pad0, s_pad1 = cal_pad_para(s_new_shape, source_rescale.shape)
    s_fullsize_pad0, s_fullsize_pad1 = cal_pad_para((source_shape_before_pad[2], source_shape_before_pad[3]), (pad_source.shape[2], pad_source.shape[3]))
    print('s_pad', s_pad0, s_pad1, 't_pad', t_pad0, t_pad1, 's_fullsize_pad', s_fullsize_pad0, s_fullsize_pad1, target_rescale.shape) # (179, 179) (714, 714)
    
    source_foreground_dir = os.path.join(mask_base, slide_id, 'contours_geojson', source_name+'.geojson') # slide_name+'.geojson'
    source_foreground_load = get_foreground_biopsy(source_foreground_dir, (s_new_shape[1], s_new_shape[0]), source_slide.level_dimensions[0]) 
    source_foreground_load_fullsize = get_foreground_biopsy(source_foreground_dir, (source_shape_before_pad[3], source_shape_before_pad[2]), source_slide.level_dimensions[0]) 
    
    source_foreground_ = np.pad(source_foreground_load, ((s_pad0[0], s_pad0[1]), (s_pad1[0], s_pad1[1])), mode='constant', constant_values=pad_value)
    
    s_f_save = source_rescale * np.repeat(1-0.5*source_foreground_[:, :, None], 3, axis=2)
    Image.fromarray(s_f_save.astype(np.uint8)).save(os.path.join(dir_name, 'source_foreground.jpg'))

    if args.s_nomask:
        source_foreground = np.ones((s_new_shape[0], s_new_shape[1])).astype(np.uint8)
        source_foreground = np.pad(source_foreground, ((s_pad0[0], s_pad0[1]), (s_pad1[0], s_pad1[1])), mode='constant', constant_values=pad_value)
        print('Using the whole source image as the source_forground', source_foreground.shape)
    else:
        source_foreground = source_foreground_
    # endregion
    
    clean=True
    if clean:
        no_pad_shape = (s_new_shape, t_new_shape)
        s_k, t_k = clean_pair_points(s_k, t_k, source_foreground, target_foreground, no_pad_shape, threshold=10, clean_control=True) 
        # index 499 is out of bounds for axis 1 with size 499
        print('after cleaning', len(s_k))

    print(len(s_k), len(t_k))
    img_s_p = display_one_image(s_k, source_rescale, rescale_=1, indictor=np.array([255,0,0]), num=len(s_k), idx=[t for t in range(len(s_k))], binary_map=None, size=2)
    img_s_p.save(os.path.join(dir_name, source_save_name.replace('.jpg', '_%s.jpg' % (len(s_k)))))
    img_t_p = display_one_image(t_k, target_rescale, rescale_=1, indictor=np.array([255,0,0]), num=len(t_k), idx=[t for t in range(len(t_k))], binary_map=None, size=2)
    img_t_p.save(os.path.join(dir_name, '%s_p_t.png' % slide_id))
    
    # --------------------------------------------------------------------------
    anno_dir = os.path.join(anno_base_dir, slide_id, source_name+'.json')
    anno_mask_dir = os.path.join(mask_base, slide_id, 'contours', source_name+'.jpg') 
    
    binary_masks = load_source_mask(anno_dir, anno_mask_dir, (s_new_shape[1], s_new_shape[0]), source_foreground_load) # source_foreground_load w.o. padding
    binary_masks_fullsize = load_source_mask(anno_dir, anno_mask_dir, (source_shape_before_pad[3], source_shape_before_pad[2]), source_foreground_load_fullsize)
    
    target_save_name = get_target_name(args=args, stain=target_stain)
    source_save_name = get_source_name(args=args, stain=source_stain)

    resize_foreground_ratio = 1
    source_foreground_area = np.sum(source_foreground)
    target_foreground_area = np.sum(target_foreground)

    # if source_foreground_area > 160000: # CUDA OOM
    #     resize_foreground_ratio = 0.8
    ori_dsize = (source_foreground.shape[1], source_foreground.shape[0])
    dsize = (round(source_foreground.shape[1] * resize_foreground_ratio), round(source_foreground.shape[0] * resize_foreground_ratio))

    final_idxs, final_maps, final_binary_masks = {}, {}, {}
    return_patches, return_target_masks, return_points = {}, {}, {}
    
    for anno_id in binary_masks.keys():
        binary_mask = binary_masks[anno_id] 
        binary_mask = np.pad(binary_mask, ((s_pad0[0], s_pad0[1]), (s_pad1[0], s_pad1[1])), mode='constant', constant_values=pad_value)

        binary_mask_fullsize = binary_masks_fullsize[anno_id] # pad to pad_souce
        binary_mask_fullsize = np.pad(binary_mask_fullsize, ((s_fullsize_pad0[0], s_fullsize_pad0[1]), (s_fullsize_pad1[0], s_fullsize_pad1[1])), mode='constant', constant_values=pad_value) # s_fullsize_pad0, s_fullsize_pad1
        final_binary_masks[anno_id] = binary_mask
        
        # -----------------------------------Patch Mask----------------------------------------
        # region Patch_and_resize
        # Patch mask first
        patch_mask_dir = os.path.join(dir_name, 'first', 'patch_mask_%s_%d_%d_%d' % (anno_id, args.patch_size, args.stride, args.num_p))
        os.makedirs(patch_mask_dir, exist_ok=True)
        
        patch_masks = extract_mask_patches_fullsize(mask=binary_mask, mask_fullsize=binary_mask_fullsize, patch_size=args.patch_size, stride=args.stride, 
                            patch_mask_dir=patch_mask_dir, ignore_area=100)   
        target_maps = []
        final_idx = set()

        #======================================================================================================================================
        if resize_foreground_ratio != 1: # default=1
            source_foreground_resize = cv2.resize(source_foreground, dsize, interpolation=cv2.INTER_NEAREST) 
            target_foreground_resize = cv2.resize(target_foreground, dsize, interpolation=cv2.INTER_NEAREST)
            target_rescale_resize = cv2.resize(target_rescale, dsize)
            
            patch_masks_resample = []
            for item in patch_masks: 
                p_mask_ori = item[0]
                p_mask_resample = cv2.resize(p_mask_ori, dsize, interpolation=cv2.INTER_NEAREST)
                patch_masks_resample.append(p_mask_resample)
        else:
            source_foreground_resize = source_foreground
            target_foreground_resize = target_foreground
            target_rescale_resize = target_rescale
    
        # endregion
        #======================================================================================================================================
        return_patches[anno_id] = []
        return_target_masks[anno_id] = []
        return_points[anno_id] = []
        for k in range(len(patch_masks)):
            p_mask, left_top, area = patch_masks[k][0], patch_masks[k][1], patch_masks[k][2]
            num, n_component = patch_masks[k][3], patch_masks[k][4]
            # ---------------------------get ref_idxs-----------------------------------
            num_p_ = args.num_p

            max_len = 0
            ignore_len=30
            while max_len < 3: 
                idx, _ = get_k_nearest_points_to_mask_new(s_k, p_mask, num_p_, rescale_=1, ignore_len=ignore_len)
                # ------------------------------------------------------------------------
                if args.split_mode == "none":
                    idxs = [idx]
                else:
                    idxs = split_pair_points(s_k, t_k, idx, distance_diff_threshold=args.dist_thre, mode=args.split_mode)
                    # idxs = split_pair_points(s_k, t_k, idx, distance_diff_threshold=args.dist_thre, mode=args.split_mode, print_projection=True) 

                max_len = max(len(idx_) for idx_ in idxs)
                ignore_len *=2
                if ignore_len > 200:
                    break
            # print(num, '---', idxs, idx)
            final_idx = final_idx.union(set(idx))
            # endregion
            
            return_patches_split_idx = []
            return_target_masks_split_idx= []
            return_points_split_idx = []
            for idx_num in range(len(idxs)):
                idx = idxs[idx_num]
                if len(idx) < 3:
                    continue
                
                # ---------------------------save source------------------------------------
                img_s_p = display_one_image(s_k, source_rescale, rescale_=1, indictor=np.array([255,0,0]), num=len(idx), idx=idx, binary_map=p_mask, size=2)
                img_s_p.save(os.path.join(patch_mask_dir, 'mask_%d_%d_%d_' % (num, n_component, idx_num) + source_save_name))
                
                if resize_foreground_ratio != 1: 
                    p_mask_ = patch_masks_resample[k]
                else:
                    p_mask_ = p_mask
        
                source_ref, source_calc = get_ref_calc_mask(mask=source_foreground_resize, ps=resize_foreground_ratio*s_k, idx=idx, device=device, args=args) 
                target_ref, target_calc = get_ref_calc_mask(mask=target_foreground_resize, ps=resize_foreground_ratio*t_k, idx=idx, device=device, args=args) 

                source_ref_, source_calc_, target_ref_, target_calc_ = get_input(source_ref, source_calc, target_ref, target_calc, 
                        source_foreground_resize, target_foreground_resize, device=device, patch_ratio=args.patch_ratio, set_ori=args.set_ori)

                with torch.no_grad(): 
                    ori_feat_ref = encoder(source_ref_, source_calc_) 
                    target_feat_ref = encoder(target_ref_, target_calc_) # [H*W, dim]
                    # torch.cuda.empty_cache()
                # ---------------------------------------------------       
                # Get similarity map
                ori_w, ori_h = target_rescale_resize.shape[1], target_rescale_resize.shape[0]
                W, H = int(1*ori_w), int(1*ori_h) 

                target_calc = target_calc.detach().cpu() 
                source_calc = source_calc.detach().cpu()
             
                target_sim_map = compute_target_sim_map_new(ori_feat_ref, source_calc, target_feat_ref, target_calc, p_mask_, H, W, b_s=1000, top_k=args.top_k, low_thre=args.low_thre) 

                if resize_foreground_ratio != 1: 
                    target_sim_map = cv2.resize(target_sim_map, ori_dsize , interpolation=cv2.INTER_NEAREST) 
                
                target_maps.append(target_sim_map)
                
                img_target_mask = display_one_image(t_k, target_rescale, rescale_=1, indictor=np.array([255,0,0]), num=len(idx), idx=idx, binary_map=target_sim_map, size=2)
                img_target_mask.save(os.path.join(patch_mask_dir, 'mask_%d_%d_%d_' % (num, n_component, idx_num) + target_save_name))  

                return_patches_split_idx.append(patch_masks[k])
                return_target_masks_split_idx.append(target_sim_map)
                return_points_split_idx.append(idx)

            return_patches[anno_id].append(patch_masks[k])
            return_target_masks[anno_id].append(return_target_masks_split_idx)
            return_points[anno_id].append(return_points_split_idx)
            
            if k % 10 == 0:
                torch.cuda.empty_cache()
        
        final_map = np.zeros_like(target_foreground, dtype=np.uint8)
        for target_map in target_maps:
            final_map = np.logical_or(final_map, target_map).astype(np.uint8) # final_map | target_map
        
        final_idxs[anno_id] = list(final_idx)
        final_maps[anno_id] = final_map
    
    return_result = (return_patches, return_target_masks, return_points)
    return_tmp = (final_idxs, final_maps, final_binary_masks, landmark_paras, (s_pad0, t_pad0), (t_pad0, t_pad1), (source_rescale, target_rescale),
                   (source_foreground, target_foreground), (s_k, t_k), (pad_source, pad_target), (source_shape_before_pad, target_shape_before_pad))
    return return_result, return_tmp



def get_bounding_box(coords: np.ndarray):
    h_min, w_min = np.min(coords, axis=0)
    h_max, w_max = np.max(coords, axis=0)

    top_left = np.array([h_min, w_min])
    bottom_right = np.array([h_max, w_max])

    return top_left, bottom_right


def get_box_from_two_points(coords1, coords2, mode='tl', cut_thre=0):
    
    h1, w1 = coords1 
    h2, w2 = coords2 
    if mode == 'tl':
        if cut_thre > 0:
            top_left = np.array([max(min(h1, h2), h2-cut_thre), max(min(w1, w2), w2-cut_thre)])
        else:
            top_left = np.array([min(h1, h2), min(w1, w2)])
        return top_left, None
    elif mode == 'br':
        if cut_thre > 0:
            bottom_right = np.array([min(max(h1, h2), h2+cut_thre), min(max(w1, w2), w2+cut_thre)])
        else:
            bottom_right = np.array([max(h1, h2), max(w1, w2)])
        return None, bottom_right
    else:
        return None, None


def first_round_post(pad_source, pad_target, first_round_result, base_params, second_params):
    (patches, masks, points_id, first_pair_points, first_round_shape, source_rescale_first, target_rescale_first, source_shape_before_pad, 
     target_shape_before_pad, first_patch_size, first_stride, first_reg_paras) = first_round_result
    (first_s_k_ori, first_t_k_ori) = first_pair_points
    args, slide_id, target_stain, device, source_path, target_path, anno_id = base_params
    second_size, first_size, use_first_points = second_params['size'], first_reg_paras['reg_size'], second_params['use_first_points']
    merge_points_second = args.merge_points
    second_first_ratio = second_size/first_size
    fullsize_first_ratio = pad_source.shape[2] / source_rescale_first.shape[0] # [1, 3, h, w] 2 -> h, [h, w, 3] 0 -> h
    fullsize_second_ratio = fullsize_first_ratio / second_first_ratio

    sec_patch_ratio = 2 * args.patch_ratio
    sec_patch_ratio = args.patch_ratio
    # print('patch_ratio_first_post', sec_patch_ratio)
    low_thre = args.low_thre
    low_thre = 0.8

    first_s_k, first_t_k = second_first_ratio * first_s_k_ori, second_first_ratio * first_t_k_ori

    resampled_source, resampled_target = u.initial_resampling_fix_reg(pad_source, pad_target, second_size)
    source_rescale = resampled_source.squeeze().permute(1, 2, 0).numpy() # [1, 3, h, w] ->[h, w, 3]
    target_rescale = resampled_target.squeeze().permute(1, 2, 0).numpy() # [1, 3, h, w] ->[h, w, 3]
    dir_name = get_dir_name(args=args, slide_id=slide_id, target_stain=target_stain)
    second_dir = os.path.join(dir_name, 'first_post_size_%d' % second_size)
    if use_first_points:
        second_dir = second_dir + '_use_p1'
    os.makedirs(os.path.join(second_dir, 'anno_%s' % anno_id), exist_ok=True)

    # --------------------------------------------------------------------------
    # region Foreground
    pad_value = 0
    source_slide = opensdpc.OpenSdpc(str(source_path))
    target_slide = opensdpc.OpenSdpc(str(target_path))
    ratio = pad_source.shape[2] / resampled_source.shape[2]
    mask_base = '/nas/username/code/biopsy_thumb/seg_biopsy_trident' # union_new
    
    t_new_shape = (int(1/ratio * target_shape_before_pad[2]), int(1/ratio *target_shape_before_pad[3]))
    t_pad0, t_pad1 = cal_pad_para(t_new_shape, target_rescale.shape) # before_pad, after_pad

    target_foreground_dir = os.path.join(mask_base, slide_id, 'contours_geojson', target_name+'.geojson')
    target_foreground_load = get_foreground_biopsy(target_foreground_dir, (t_new_shape[1], t_new_shape[0]), target_slide.level_dimensions[0])
    
    if args.t_f_mode == 'trident':
        target_foreground = target_foreground_load
    elif args.t_f_mode == 'L_225':
        mask_base_L = '/nas/username/code/patch_feature/seg_biopsy_L/225_255_15_c16_at_16_r_32'
        target_foreground = None
        target_foreground_L_dir = os.path.join(mask_base_L, slide_id, 'mask_new', target_name + '.png')
        target_foreground_L = get_foreground_b1b2(t_new_shape, args, mask_dir = target_foreground_L_dir) # Open then Resize    
        target_foreground = target_foreground_L
    else:
        raise NotImplementedError

    target_foreground = np.pad(target_foreground, ((t_pad0[0], t_pad0[1]), (t_pad1[0], t_pad1[1])), mode='constant', constant_values=pad_value)
    t_f_save = target_rescale * np.repeat(1-0.5*target_foreground[:, :, None], 3, axis=2)
    Image.fromarray(t_f_save.astype(np.uint8)).save(os.path.join(second_dir, 'target_foreground.jpg'))
    # --------------------------------------------------------------------------------------------------
    s_new_shape = (int(1/ratio * source_shape_before_pad[2]), int(1/ratio *source_shape_before_pad[3]))

    s_pad0, s_pad1 = cal_pad_para(s_new_shape, source_rescale.shape)
    
    source_foreground_dir = os.path.join(mask_base, slide_id, 'contours_geojson', source_name+'.geojson') # slide_name+'.geojson'
    source_foreground_load = get_foreground_biopsy(source_foreground_dir, (s_new_shape[1], s_new_shape[0]), source_slide.level_dimensions[0]) 
    source_foreground_ = np.pad(source_foreground_load, ((s_pad0[0], s_pad0[1]), (s_pad1[0], s_pad1[1])), mode='constant', constant_values=pad_value)
    
    s_f_save = source_rescale * np.repeat(1-0.5*source_foreground_[:, :, None], 3, axis=2)
    Image.fromarray(s_f_save.astype(np.uint8)).save(os.path.join(second_dir, 'source_foreground.jpg'))

    if args.s_nomask:
        source_foreground = np.ones((s_new_shape[0], s_new_shape[1])).astype(np.uint8)
        source_foreground = np.pad(source_foreground, ((s_pad0[0], s_pad0[1]), (s_pad1[0], s_pad1[1])), mode='constant', constant_values=pad_value)
        print('Using the whole source image as the source_forground', source_foreground.shape)
    else:
        source_foreground = source_foreground_
    # endregion
    # ========================================================================================================

    landmark_paras = {
            'is_warp': False,
            'device': device,
            'resample_ratio': None,
            'initial_resample': False,
            'initial_resample_size':620, 
            'reg_size': -1, # no resize
            'angle_step': 15,
            'angle_range': 180,
            'merge': merge_points_second,
            'merge_thre': 5, #
            'multi_anchor': False
        } 
    
    sec_thres = [0.7, 0.75]
    target_maps = [[] for n_ in range(len(sec_thres))]
    final_ids = set()
    for i in range(len(patches)):
        patch_ = patches[i]
        mask_split_idx = masks[i] # [idx1_map, idx2_map, ...]
        if len(mask_split_idx) == 0:
            continue
        points_split_id = points_id[i]
        s_mask, left_top, area, num, n_component, s_mask_fullsize = patch_[0], patch_[1], patch_[2], patch_[3], patch_[4], patch_[5]
        # print('---num: %s, n_component: %s-------' % (num, n_component))

        t_mask = np.zeros_like(mask_split_idx[0], dtype=np.uint8)
        points_all_id = set()
        for idx_num in range(len(mask_split_idx)):
            mask = mask_split_idx[idx_num]
            p_id = points_split_id[idx_num]
            t_mask = np.logical_or(t_mask, mask).astype(np.uint8) 
            points_all_id = points_all_id.union(set(p_id))
            
        final_ids.union(points_all_id)
        points_all_id = list(points_all_id)
        first_s_k_, first_t_k_ = first_s_k[points_all_id], first_t_k[points_all_id] # second_first_ratio at the beginning

        pad_width = 25
       
        s_tl, s_br = get_mask_bound(s_mask, return_mode='coords', pad_width=pad_width) # tl_h, br_h, tl_w, br_w
        t_tl, t_br = get_mask_bound(t_mask, return_mode='coords', pad_width=pad_width) # tl_h, br_h, tl_w, br_w

        s_k_tl, s_k_br = get_bounding_box(first_s_k_ori[points_all_id]) 
        t_k_tl, t_k_br = get_bounding_box(first_t_k_ori[points_all_id]) # first_s_k_ori, first_t_k_ori
        
        cut_thre=150 
        s_tl, _ = get_box_from_two_points(s_tl, s_k_tl, mode='tl', cut_thre=cut_thre)
        _, s_br = get_box_from_two_points(s_br, s_k_br, mode='br', cut_thre=cut_thre)
        t_tl, _ = get_box_from_two_points(t_tl, t_k_tl, mode='tl', cut_thre=cut_thre)
        _, t_br = get_box_from_two_points(t_br, t_k_br, mode='br', cut_thre=cut_thre)

        expand = True
        if expand:
            pad_s_tl_h, pad_s_tl_w = int(min(pad_width, s_tl[0])), int(min(pad_width, s_tl[1])) 
            pad_t_tl_h, pad_t_tl_w = int(min(pad_width, t_tl[0])), int(min(pad_width, t_tl[1]))
            pad_s_br_h, pad_s_br_w = int(min(pad_width, s_mask.shape[0]-s_br[0])), int(min(pad_width, s_mask.shape[1]-s_br[1])) 
            pad_t_br_h, pad_t_br_w = int(min(pad_width, t_mask.shape[0]-t_br[0])), int(min(pad_width, t_mask.shape[1]-t_br[1])) 
            
            s_tl_, s_br_, t_tl_, t_br_ = s_tl - np.array([pad_s_tl_h, pad_s_tl_w]), s_br + np.array([pad_s_br_h, pad_s_br_w]), t_tl - np.array([pad_t_tl_h, pad_t_tl_w]), t_br + np.array([pad_t_br_h, pad_t_br_w])
            # s_tl_, s_br_, t_tl_, t_br_ = s_tl - pad_width, s_br + pad_width, t_tl - pad_width, t_br + pad_width
            # s_mask = np.pad(s_mask, pad_width=pad_width, mode='constant', constant_values=pad_value)
            # t_mask = np.pad(t_mask, pad_width=pad_width, mode='constant', constant_values=pad_value)

            s_mask = np.pad(s_mask, ((pad_s_tl_h, pad_s_br_h), (pad_s_tl_w, pad_s_br_w)), mode='constant', constant_values=pad_value)
            t_mask = np.pad(t_mask, ((pad_t_tl_h, pad_t_br_h), (pad_t_tl_w, pad_t_br_w)), mode='constant', constant_values=pad_value)
            # np.pad(source_foreground_load, ((s_pad0[0], s_pad0[1]), (s_pad1[0], s_pad1[1])), mode='constant', constant_values=pad_value)

        else:
            pad_width = 0
            s_tl_, s_br_, t_tl_, t_br_ = s_tl, s_br, t_tl, t_br
        
        s_tl_full, s_br_full, t_tl_full, t_br_full = fullsize_first_ratio * s_tl_, fullsize_first_ratio * s_br_, fullsize_first_ratio * t_tl_, fullsize_first_ratio * t_br_
        s_tl_sec, s_br_sec, t_tl_sec, t_br_sec = second_first_ratio * s_tl_, second_first_ratio * s_br_, second_first_ratio * t_tl_, second_first_ratio * t_br_
        
        first_s_k_sec, first_t_k_sec = first_s_k_ - s_tl_sec, first_t_k_ - t_tl_sec # set new coord-system

        s_patch  = source_rescale[int(s_tl_sec[0]):int(s_br_sec[0]), int(s_tl_sec[1]):int(s_br_sec[1])]
        t_patch  = target_rescale[int(t_tl_sec[0]):int(t_br_sec[0]), int(t_tl_sec[1]):int(t_br_sec[1])]

        s_patch_fore  = source_foreground[int(s_tl_sec[0]):int(s_br_sec[0]), int(s_tl_sec[1]):int(s_br_sec[1])]
        t_patch_fore  = target_foreground[int(t_tl_sec[0]):int(t_br_sec[0]), int(t_tl_sec[1]):int(t_br_sec[1])]
        # print(s_patch_fore.shape, t_patch_fore.shape)

        best, sift_ransac, superpoint_superglue = load_anchors_np(s_patch, t_patch, landmark_paras=landmark_paras)
        transform_new, s_k_base, t_k_base, new_add_s, new_add_t, reg_size, angle = best

        if merge_points_second:
            s_k = np.concatenate([s_k_base, new_add_s], axis=0)
            t_k = np.concatenate([t_k_base, new_add_t], axis=0)
        else:
            s_k = s_k_base
            t_k = t_k_base
        s_k = np.column_stack((s_k[:, 1], s_k[:, 0])) 
        t_k = np.column_stack((t_k[:, 1], t_k[:, 0]))
        
        if use_first_points:
            s_k = np.concatenate([s_k, first_s_k_sec], axis=0)
            t_k = np.concatenate([t_k, first_t_k_sec], axis=0)
             
        clean=True
        if clean:
            s_k, t_k = clean_pair_points(s_k, t_k, s_patch_fore, t_patch_fore, None, threshold=5, clean_control=False) 
            # print('after cleaning', len(s_k))
        
        s_patch_mask  = s_mask_fullsize[int(s_tl_full[0]):int(s_br_full[0]), int(s_tl_full[1]):int(s_br_full[1])]
        s_patch_mask = skimage.transform.resize(s_patch_mask, s_patch_fore.shape,order=0, anti_aliasing=False, preserve_range=True) 
        # Image.fromarray(255*s_patch_mask).save(os.path.join(second_dir, 'patch', 's_%d_%d_patch_mask.jpg' % (num, n_component)))
        img_s_p = display_one_image(s_k, s_patch, rescale_=1, indictor=np.array([255,0,0]), num=len(s_k), idx=[t for t in range(len(s_k))], binary_map=s_patch_mask, size=2)
        img_s_p.save(os.path.join(second_dir, 'anno_%s' % anno_id, '%d_%d_s_points.jpg' % (num, n_component)))
        img_t_p = display_one_image(t_k, t_patch, rescale_=1, indictor=np.array([255,0,0]), num=len(t_k), idx=[t for t in range(len(t_k))], binary_map=t_patch_fore, size=2)
        img_t_p.save(os.path.join(second_dir, 'anno_%s' % anno_id, '%d_%d_t_points.jpg' % (num, n_component)))
        
        # ===============================================================================================================
        second_patch_mask_dir = os.path.join(second_dir, 'anno_%s' % anno_id, '%d_%d' % (num, n_component))
        os.makedirs(second_patch_mask_dir, exist_ok=True)
        second_patch_masks = extract_mask_patches_second(s_patch_mask, patch_size=100, stride=100, patch_mask_dir=second_patch_mask_dir)
        
        sec_patch_target_maps = [[] for n_ in range(len(sec_thres))]
        sec_patch_final_ids = set() # [set() for n_ in range(len(sec_thres))]
        for k_ in range(len(second_patch_masks)):
            sec_p_mask, left_top, p_id = second_patch_masks[k_][0], second_patch_masks[k_][1], second_patch_masks[k_][2]
            max_len = 0
            ignore_len=30
            while max_len < 3:
                idx, _ = get_k_nearest_points_to_mask_new(s_k, sec_p_mask, 40, rescale_=1, ignore_len=ignore_len)
                # ------------------------------------------------------------------------
                idxs = split_pair_points(s_k, t_k, idx, distance_diff_threshold=10, mode=args.split_mode, multi=False)
                # print('------------------------------------')
                max_len = max(len(idx_) for idx_ in idxs)
                ignore_len *= 2
                if ignore_len > 200:
                    break

            a = [len(idxs[idx_num]) for idx_num in range(len(idxs))]
            print(num, n_component, k_, 'idx_num', a, len(idx))

            # sec_patch_final_ids.union(set(idx))
            for idx_num in range(len(idxs)): 
                idx = idxs[idx_num]
                if idx_num == 0: 
                    if len(idx) < 3:
                        continue
                else:
                    if len(idx) < 7:
                        continue
                sec_patch_final_ids = sec_patch_final_ids.union(set(idx))
      
                # ---------------------------save source------------------------------------
                img_s_p = display_one_image(s_k, s_patch, rescale_=1, indictor=np.array([255,0,0]), num=len(idx), idx=idx, binary_map=sec_p_mask, size=2)
                img_s_p.save(os.path.join(second_patch_mask_dir, 'mask_%d_%d_source.png' % (p_id, idx_num)))
                # print('s_patch_fore', s_patch_fore.shape)
                source_ref, source_calc = get_ref_calc_mask(mask=s_patch_fore, ps=s_k, idx=idx, device=device, args=args, source_expand=args.s_expand, delta=5) 

                target_ref, target_calc = get_ref_calc_mask(mask=t_patch_fore, ps=t_k, idx=idx, device=device, args=args, source_expand=False)
                # patch_ratio = max(1, args.patch_ratio / second_first_ratio)
                
                source_ref_, source_calc_, target_ref_, target_calc_ = get_input(source_ref, source_calc, target_ref, target_calc, 
                        s_patch_fore, t_patch_fore, device=device, patch_ratio=sec_patch_ratio, set_ori=args.set_ori)

                with torch.no_grad(): 
                    ori_feat_ref = encoder(source_ref_, source_calc_) 
                    target_feat_ref = encoder(target_ref_, target_calc_) # [H*W, dim]
                    # torch.cuda.empty_cache()
                # ---------------------------------------------------       
                # Get similarity map
                ori_w, ori_h = t_patch.shape[1], t_patch.shape[0] # [h, w, 3]
                W, H = int(1*ori_w), int(1*ori_h) 

                target_calc = target_calc.detach().cpu()
                source_calc = source_calc.detach().cpu()
                
                for n_ in range(len(sec_thres)):
                    low_thre = sec_thres[n_]
                    target_sim_map_thre = compute_target_sim_map_new(ori_feat_ref, source_calc, target_feat_ref, target_calc, sec_p_mask, 
                         H, W, b_s=1000, top_k=args.top_k, low_thre=low_thre, clip=args.s_expand)

                    img_target_mask = display_one_image(t_k, t_patch, rescale_=1, indictor=np.array([255,0,0]), num=len(t_k), idx=idx, binary_map=target_sim_map_thre, size=2)
                    img_target_mask.save(os.path.join(second_patch_mask_dir, 'mask_%d_%d_%s.png' % (p_id, idx_num, low_thre))) 

                    sec_patch_target_maps[n_].append(target_sim_map_thre)
                
        for n_ in range(len(sec_thres)):
            low_thre = sec_thres[n_]
            sec_patch_final_map = np.zeros_like(t_patch_fore, dtype=np.uint8)
            for target_map in sec_patch_target_maps[n_]:
                sec_patch_final_map = np.logical_or(sec_patch_final_map, target_map).astype(np.uint8)
            sec_patch_final_ids = list(sec_patch_final_ids)

            # print('sec_patch_final_ids', sec_patch_final_ids)
            sec_patch_result = display_one_image(t_k, t_patch, rescale_=1, indictor=np.array([255,0,0]), num=len(final_ids), idx=sec_patch_final_ids, binary_map=sec_patch_final_map, size=2)
            sec_patch_result.save(os.path.join(second_patch_mask_dir, 'mask_target_%s.png' % low_thre)) 
            sec_patch_result.save(os.path.join(second_dir, 'anno_%s' % anno_id, '%d_%d_%s.png' % (num, n_component, low_thre))) 

            target_sim_map_orisize = np.zeros_like(target_foreground, dtype=np.uint8)
            target_sim_map_orisize[int(t_tl_sec[0]):int(t_br_sec[0]), int(t_tl_sec[1]):int(t_br_sec[1])] = sec_patch_final_map
            
            target_maps[n_].append(target_sim_map_orisize)
            final_ids.union(set(sec_patch_final_ids))

    all_final_map = {}
    for n_ in range(len(sec_thres)):
        low_thre = sec_thres[n_]   
        final_map = np.zeros_like(target_foreground, dtype=np.uint8) 
        for target_map in target_maps[n_]:
            final_map = np.logical_or(final_map, target_map).astype(np.uint8) # final_map | target_map
        
        all_final_map[low_thre] = final_map
        
        final_ids = list(final_ids)
        img_target_mask = display_one_image(first_t_k, target_rescale, rescale_=1, indictor=np.array([255,0,0]), num=len(final_ids), idx=final_ids, binary_map=final_map, size=2)
        img_target_mask.save(os.path.join(second_dir, 'target_mask_%s_%s.png' % (anno_id, low_thre))) 

        
    return all_final_map, second_dir


# ---------------------------------------------------
parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument('--loss_mode', type=str, default='infonce')
parser.add_argument('--batch_size', type=int, default=5120)
parser.add_argument('--train_num', type=int, default=25600000, help='number of total nums for training (step/batch_size)')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--save_dir', type=str, default="./checkpoints")
parser.add_argument('--model_name', type=str, default="num_51200000_bs_5120_lr_3e-4")
parser.add_argument('--step', type=int, default=9900)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--num_p', type=int, default=5)
parser.add_argument('--rescale', type=float, default=1)
parser.add_argument('--s_ratio', type=float, default=2.0)
# --------------------------------------------------------------
parser.add_argument('--temperature', type=float, default=1, help='softmax temperature')
parser.add_argument('--not_in_wsi', action='store_true', help='if True, only find the mask in the tissue of the slide')
parser.add_argument('--top_k', type=int, default=1, help='max if top_k=1')
parser.add_argument('--low_thre', type=float, default=0.7, help='for radicious matching')
parser.add_argument('--s_as_t', action='store_true', help='use the source image as the target one')

parser.add_argument('--select_point', type=str, default='half', help='half / close')
parser.add_argument('--search_mode', type=str, default='topk', choices=['thre', 'topk'])

parser.add_argument('--out_dim', type=int, default=256)
parser.add_argument('--attn_fc', action='store_true')
parser.add_argument('--set_ori', action='store_true', help='if True, reset the original')
parser.add_argument('--load_mask', action='store_true', help='Load the pre-extracted foreground mask')

parser.add_argument('--patch_size', type=int, default=100)
parser.add_argument('--stride', type=int, default=100)

parser.add_argument('--machine', type=str, default=19)
parser.add_argument('--s_nomask', action='store_true', help='No mask for source slides')
parser.add_argument('--merge_points', action='store_true', help='Merge the source_target points from different angle')
parser.add_argument('--t_f_mode', type=str, default='trident', help='where to load ther target foreground')
parser.add_argument('--split_mode', type=str, default='leiden', help='split_pair_points, ransac_home,ransac_affine, connect, xxx')
parser.add_argument('--dist_thre', type=int, default=10, help='split_pair_points')
parser.add_argument('--patch_ratio', type=float, default=1, help='patch_size 100 is too small for the model trained for 400')

parser.add_argument('--r_rot', type=float, default=1)
parser.add_argument('--r_site', type=float, default=0)

parser.add_argument('--angle_step', type=int, default=15)
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--end', type=int, default=200)
parser.add_argument('--multi_anchor', action='store_true', help='Use multi size for anchor selection')
parser.add_argument('--run_second', action='store_true')
parser.add_argument('--s_expand', action='store_true', help='In second round, expand source mask for cal')
args = parser.parse_args()

if torch.cuda.is_available():
    device = torch.device(args.device)
else:
    device = torch.device("cpu")

if 'weight' in args.model_name or 'kneg' in args.save_dir:
    if 'big' in args.save_dir:
        if 'split' in args.model_name:
            encoder = GeometryFeatureNet_weight_big_split(input_dim=4, hidden_dim=128, output_dim=args.out_dim, using_weight=True, temp=args.temperature, attn_fc=args.attn_fc)
        else:
            encoder = GeometryFeatureNet_weight_big(input_dim=4, hidden_dim=128, output_dim=args.out_dim, using_weight=True, temp=args.temperature, attn_fc=args.attn_fc)
    else:
        raise NotImplementedError

if 'center' in args.save_dir:
    model_dir = os.path.join(args.save_dir, args.model_name, 'rot_%s_site_%s' % (args.r_rot, args.r_site), f"{args.loss_mode}_step_{args.step}_b_{args.batch_size}.pth")
else:
    model_dir = os.path.join(args.save_dir, args.model_name, f"{args.loss_mode}_step_{args.step}_b_{args.batch_size}.pth")
encoder.load_state_dict(torch.load(model_dir, map_location=device, weights_only=True))
encoder.to(device)
encoder.eval()


def get_stain(slide_name):
    splits = slide_name.split('_')
    splits_ = slide_name.split('-')
    if len(splits) > 2:
        stain = splits[2]
    elif len(splits_) > 1:
        stain = splits_[1]
    else:
        stain = slide_name
    return stain

### Load Biopsy cases ###
biopsy_json = '/nas/username/code/biopsy_thumb/biopsy_all.json' 
if args.machine == '19':
    biopsy_data_dir = '/data/username/xijing/biopsy'
else:
    biopsy_data_dir = '/17data/username/private/xijing/biopsy'
with open(biopsy_json, "r", encoding="utf-8") as f:
    data = json.load(f)

slide_ids = list(data.keys())

anno_base_dir = '/nas/username/code/biopsy_thumb/anno_mask'

for i in tqdm(range(len(slide_ids))):
    if i < args.start or i > args.end:
        continue

    slide_id = slide_ids[i] 
    pair = data[slide_id]
    if ('source' not in pair.keys()) or ('target' not in pair.keys()):
        continue
    source_path = os.path.join(biopsy_data_dir, slide_id, pair['source'])
    source_stain, source_name = 'HE', pair['source'].replace('.sdpc','')
    num_target = len(pair['target'])
    for j in range(num_target):
        target_slide_name = pair['target'][j]
        target_name = target_slide_name.replace('.sdpc', '')
        seed_torch(args.seed, device=device)
        target_path = os.path.join(biopsy_data_dir, slide_id, target_slide_name)      
        target_stain = get_stain(target_name) 

        print('--------------------%s--%s--------------------'% (slide_id, target_stain))
        
        dir_name = get_dir_name(args=args, slide_id=slide_id, target_stain=target_stain)
        target_save_name = get_target_name(args=args, stain=target_stain)
        source_save_name = get_source_name(args=args, stain=source_stain)

        names = (source_stain, source_name, source_save_name, target_stain, target_name, target_save_name, dir_name)
        return_result, return_tmp = first_round(source_path, target_path, slide_id, device, names)
        return_patches, return_target_masks, return_points = return_result
        final_idxs, final_maps, final_binary_masks, landmark_paras, (s_pad0, t_pad0), (t_pad0, t_pad1), (source_rescale, target_rescale), (
            source_foreground, target_foreground), (s_k, t_k), (pad_source, pad_target), (source_shape_before_pad, target_shape_before_pad) =  return_tmp
        
        # --------------------------------Display First round------------------------------------------
        anno_ids = return_patches.keys()
        total_idx, total_map = set(), np.zeros_like(target_foreground, dtype=np.uint8) # target_foreground
        for anno_id in anno_ids:
            patch_mask_dir = os.path.join(dir_name, 'first', 'patch_mask_%s_%d_%d_%d' % (anno_id, args.patch_size, args.stride, args.num_p))
            
            final_idx = final_idxs[anno_id]
            final_map = final_maps[anno_id]

            img_target_mask = display_one_image(t_k, target_rescale, rescale_=1, indictor=np.array([255,0,0]), num=len(final_idx), idx=final_idx, binary_map=final_map, size=2)
            img_target_mask.save(os.path.join(patch_mask_dir, 'mask_fianl_' + target_save_name))
            
            final_map_unpad = unpad(final_map, pad_para=(t_pad0, t_pad1))

            Image.fromarray((255*final_map_unpad).astype(np.uint8)).save(os.path.join(patch_mask_dir, 'mask_top_%d.png' % (args.top_k)))
            Image.fromarray((255*final_map_unpad).astype(np.uint8)).save(os.path.join(dir_name, 'first', 'mask_%s_top_%d.png' % (anno_id, args.top_k)))

            target_rescale_unpad = unpad(target_rescale, pad_para=(t_pad0, t_pad1)) # (500, 854, 3) -> (479, 854, 3)
            t_k_unpad = unpad_coordinates(t_k,  pad_para=(t_pad0, t_pad1))
            
            img_target_mask = display_one_image(t_k_unpad, target_rescale_unpad, rescale_=1, indictor=np.array([255,0,0]), num=len(final_idx), idx=final_idx, binary_map=final_map_unpad, size=2)
            img_target_mask.save(os.path.join(dir_name, 'first', 'mask_%s_target_top_%d.png' % (anno_id, args.top_k))) 
            img_s_p = display_one_image(s_k, source_rescale, rescale_=1, indictor=np.array([255,0,0]), num=len(final_idx), idx=final_idx, binary_map=final_binary_masks[anno_id], size=2)
            img_s_p.save(os.path.join(dir_name, 'mask_%s_' % (anno_id) + source_save_name)) 

            idx = final_idx
            map = final_map
            total_idx = total_idx.union(set(idx))
            total_map = np.logical_or(total_map, map).astype(np.uint8)
        
        img_target_mask = display_one_image(t_k, target_rescale, rescale_=1, indictor=np.array([255,0,0]), num=None, idx=list(total_idx), binary_map=total_map, size=2)
        img_target_mask.save(os.path.join(dir_name, 'first', 'mask_total_' + target_save_name))
        Image.fromarray(total_map.astype(np.uint8)).save(os.path.join(dir_name, 'first', 'mask.png'))

        if args.run_second:
            # ---------------------------------------------------------------------------------------
            for anno_id in anno_ids:
                patches, masks, points_id = return_patches[anno_id], return_target_masks[anno_id], return_points[anno_id]
                first_round_shape = target_rescale.shape
                first_pair_points = (s_k, t_k)
                first_round_result = (patches, masks, points_id, first_pair_points, first_round_shape, source_rescale, target_rescale, source_shape_before_pad, target_shape_before_pad, args.patch_size, args.stride, landmark_paras)
                base_params = (args, slide_id, target_stain, device, source_path, target_path, anno_id)
                
                for second_size in [500]: # [500,1000]:
                    second_params = {
                        'size': second_size,
                        'use_first_points': True
                    }
                    all_final_map, second_dir = first_round_post(pad_source, pad_target, first_round_result, base_params, second_params)
                    for low_thre in all_final_map.keys():
                        final_map = all_final_map[low_thre]
                        final_map_unpad = unpad(final_map, pad_para=(t_pad0, t_pad1))
                        Image.fromarray((255*final_map_unpad).astype(np.uint8)).save(os.path.join(second_dir, 'mask_%s_%s.png' % (anno_id, low_thre)))