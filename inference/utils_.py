import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import pathlib
from typing import Union
import json
import cv2
from scipy.ndimage import distance_transform_edt
import numpy as np
from scipy.ndimage import label

import math
import community as community_louvain  # pip install python-louvain
import igraph as ig
import leidenalg
import networkx as nx
#from skimage.measure import label, regionprops
Image.MAX_IMAGE_PIXELS = 2000000000


def extract_mask_patches(mask: np.ndarray, patch_size=100, stride=100, patch_mask_dir=None):
    H, W = mask.shape
    patches = []
    
    ys, xs = np.where(mask == 1)
    if len(ys) > 0 and len(xs) > 0:
        top_left_y = np.min(ys)
        bottom_right_y = np.max(ys)
        top_left_x = np.min(xs)
        bottom_right_x = np.max(xs)
    num = 0

    for y in range(top_left_y, bottom_right_y, stride):
        for x in range(top_left_x, bottom_right_x, stride):
            patch = mask[y:y+patch_size, x:x+patch_size]
            num_one = np.sum(patch)
            
            if np.any(patch == 1):
                patch_on_mask = np.zeros_like(mask, dtype=np.uint8)
                patch_on_mask[y:y+patch_size, x:x+patch_size] = patch
                patches.append((patch_on_mask, num_one, (y, x)))
                # patch_mask_dir = os.path.join(dir_name, 'patch_mask_%d' % (stride))
                # os.makedirs(patch_mask_dir, exist_ok=True)
                Image.fromarray(255*patch_on_mask).save(os.path.join(patch_mask_dir, 'try_s_%d.jpg' % num))
                num += 1

    return patches

# -------------------------------------------------------------------------------------------------------------------
def get_anchor_bound(pts, return_mode='coords'):

    y_min, x_min = np.min(pts[:, 0]), np.min(pts[:, 1])
    y_max, x_max = np.max(pts[:, 0]), np.max(pts[:, 1])

    top_left = np.array([y_min, x_min])
    bottom_right = np.array([y_max, x_max])

    
    if return_mode == 'coords':
        return top_left, bottom_right
    elif return_mode == '4xy':
        return y_min, y_max, x_min, x_max
    else:
        raise NotImplementedError

def get_mask_bound(mask, return_mode='coords', pad_width=25):
    ys, xs = np.where(mask == 1) 
    if len(ys) > 0 and len(xs) > 0:
        top_left_y = int(np.min(ys))
        bottom_right_y = int(np.max(ys))
        top_left_x = int(np.min(xs))
        bottom_right_x = int(np.max(xs))
    else: #
        top_left_y = 0 + pad_width # 防止后面pad超限
        bottom_right_y = mask.shape[0] - pad_width
        top_left_x = 0 + pad_width
        bottom_right_x = mask.shape[1] - pad_width
    
    if return_mode == 'coords':
        tl, br = np.array([top_left_y, top_left_x]), np.array([bottom_right_y, bottom_right_x])
        return tl, br
    elif return_mode == '4xy':
        return top_left_y, bottom_right_y, top_left_x, bottom_right_x
    else:
        raise NotImplementedError

# -------------------------------------------------------------------------------------------------------------------
def extract_mask_patches_second(mask, patch_size=100, stride=100, patch_mask_dir=None):  
    ignore_area = min(20, (0.1*patch_size)**2)
    H, W = mask.shape
    patches = []
    delta = patch_size - stride
    
    top_left_y, bottom_right_y, top_left_x, bottom_right_x = get_mask_bound(mask, return_mode='4xy')

    record = {} # 0: skip, 1: save
    y_max_idx = (bottom_right_y - top_left_y - 1) // stride # form top_left_y (y0), to next value > bottom_right_y 
    x_max_idx = (bottom_right_x - top_left_x - 1) // stride
    num = 0
    for y in range(top_left_y, bottom_right_y, stride): 
        y_idx = (y-top_left_y) // stride
        for x in range(top_left_x, bottom_right_x, stride):
            x_idx = (x-top_left_x) // stride
            # print(y_idx, x_idx)
            patch = mask[y:y+patch_size, x:x+patch_size]

            num_one = np.sum(patch) 
            if num_one > ignore_area:
                tl_y, br_y, tl_x, br_x = get_mask_bound(patch, return_mode='4xy')
                if ((patch_size - tl_y) < delta) or (br_y < delta) or ((patch_size - tl_x) < delta) or (br_x < delta):
                    if ((patch_size - tl_y) < delta) and (y_idx < y_max_idx): # mask本身在下方, 需要下方还有patch才能skip
                        
                        record[(y_idx, x_idx)] = '0' # skip (num_one > ignore_area)
                        continue

                    elif (br_y < delta) and (y_idx > 0): # mask本身在上方, 需要上方还有patch才能skip
                        # 如果上方还有patch，且上方patch所含mask区域和现在一样，那该区域在上方patch中一定位于下方且先前被记录
                        # patch upside [y_idx-1, x_idx]
                        # print(record.keys(), y_idx, x_idx, record[(y_idx-1, x_idx)] )
                        # dict_keys([(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 0), (1, 1), (1, 2), (1, 3), (2, 0), (2, 1), (2, 2)]) 2 4
                        if record[(y_idx-1, x_idx)] == '1': # 上方没有skip
                            record[(y_idx, x_idx)] = '0'
                            continue
                        else:
                            record[(y_idx, x_idx)] = '1'

                    elif ((patch_size - tl_x) < delta) and (x_idx < x_max_idx): # mask本身在右侧, 需要右侧还有patch才能skip
                        # skip后如果右侧patch所含mask区域和现在一样，就不能skip了，记录位置
                        record[(y_idx, x_idx)] = '0' # 直接skip，右侧存在的patch直接无法skip
                        continue

                    elif (br_x < delta) and (x_idx > 0): # mask本身在左侧, 需要左侧还有patch才能skip
                        # 如果左侧还有patch，且左侧patch所含mask区域和现在一样，那该区域在左侧patch中一定位于右侧且先前被记录
                        # 左侧patch [y_idx, x_idx-1]
                        if record[(y_idx, x_idx-1)] == '1': # 左侧没有skip
                            record[(y_idx, x_idx)] = '0'
                            continue 
                        else:
                            record[(y_idx, x_idx)] = '1'
                    else:
                        record[(y_idx, x_idx)] = '1' # 没有skip
                else:
                    record[(y_idx, x_idx)] = '1' # 没有skip
                
                patch_on_mask = np.zeros_like(mask, dtype=np.uint8)
                patch_on_mask[y:y+patch_size, x:x+patch_size] = patch
                Image.fromarray(255*patch_on_mask).save(os.path.join(patch_mask_dir, 'try_s_%d.jpg' % (num)))
                patches.append((patch_on_mask, (y, x), num))
                num += 1
            else:
                record[(y_idx, x_idx)] = '0'

    return patches


def extract_mask_patches_new(mask: np.ndarray, patch_size=100, stride=100, patch_mask_dir=None, ignore_area=100):
    ignore_area = min(ignore_area, (0.11*patch_size)**2)
    H, W = mask.shape
    patches = []
    delta = patch_size - stride
    
    top_left_y, bottom_right_y, top_left_x, bottom_right_x = get_mask_bound(mask, return_mode='4xy')
    print(top_left_y, bottom_right_y, top_left_x, bottom_right_x)
    """
    ys, xs = np.where(mask == 1) 
    if len(ys) > 0 and len(xs) > 0:
        top_left_y = np.min(ys)
        bottom_right_y = np.max(ys)
        top_left_x = np.min(xs)
        bottom_right_x = np.max(xs)
    """
    num = 0
    record = {}
    y_max_idx = (bottom_right_y - top_left_y - 1) // stride 
    x_max_idx = (bottom_right_x - top_left_x - 1) // stride
    for y in range(top_left_y, bottom_right_y, stride): 
        y_idx = (y-top_left_y) // stride
        for x in range(top_left_x, bottom_right_x, stride):
            x_idx = (x-top_left_x) // stride

            n_component = 0
            patch = mask[y:y+patch_size, x:x+patch_size]
            Image.fromarray(255*patch).save(os.path.join(patch_mask_dir, 'ori_try_s_%d_%d.jpg' % (y, x)))
            num_one = np.sum(patch) # 还没有区分连通分量的

        
            if num_one > ignore_area:
                tl_y, br_y, tl_x, br_x = get_mask_bound(patch, return_mode='4xy')
                if ((patch_size - tl_y) < delta) or (br_y < delta) or ((patch_size - tl_x) < delta) or (br_x < delta):
                    if ((patch_size - tl_y) < delta) and (y_idx < y_max_idx): # mask本身在下方, 需要下方还有patch才能skip
                        # skip后如果下侧patch所含mask区域和现在一样，就不能skip了，记录位置
                        record[(y_idx, x_idx)] = '0' # 直接skip，下方存在的patch直接无法skip (下方也包含这些，肯定num_one > ignore_area)
                        continue

                    elif (br_y < delta) and (y_idx > 0): # mask本身在上方, 需要上方还有patch才能skip
                        # 如果上方还有patch，且上方patch所含mask区域和现在一样，那该区域在上方patch中一定位于下方且先前被记录
                        # 上方patch [y_idx-1, x_idx]
                        if record[(y_idx-1, x_idx)] == '1': # 上方没有skip
                            record[(y_idx, x_idx)] = '0'
                            continue
                        else:
                            record[(y_idx, x_idx)] = '1'

                    elif ((patch_size - tl_x) < delta) and (x_idx < x_max_idx): # mask本身在右侧, 需要右侧还有patch才能skip
                        # skip后如果右侧patch所含mask区域和现在一样，就不能skip了，记录位置
                        record[(y_idx, x_idx)] = '0' # 直接skip，右侧存在的patch直接无法skip
                        continue

                    elif (br_x < delta) and (x_idx > 0): # mask本身在左侧, 需要左侧还有patch才能skip
                        # 如果左侧还有patch，且左侧patch所含mask区域和现在一样，那该区域在左侧patch中一定位于右侧且先前被记录
                        # 左侧patch [y_idx, x_idx-1]
                        if record[(y_idx, x_idx-1)] == '1': # Left have not skipped
                            record[(y_idx, x_idx)] = '0'
                            continue 
                        else:
                            record[(y_idx, x_idx)] = '1'
                    else:
                        record[(y_idx, x_idx)] = '1' # No skip
                else:
                    record[(y_idx, x_idx)] = '1' # No skip

                    
                component_masks = extract_components_custom(patch, n=1)

                for component_mask in component_masks:
                    num_one_component = np.sum(component_mask)
                    if num_one_component > ignore_area:
                        patch_on_mask = np.zeros_like(mask, dtype=np.uint8)
                        patch_on_mask[y:y+patch_size, x:x+patch_size] = component_mask
                        patches.append((patch_on_mask, (y, x), num_one_component, num, n_component))
                        Image.fromarray(255*patch_on_mask).save(os.path.join(patch_mask_dir, 'try_s_%d_%d.jpg' % (num, n_component)))
                        # print(num, n_component,'---------', tl_y, br_y, tl_x, br_x)
                        n_component +=1
                num += 1

    return patches


def extract_mask_patches_fullsize(mask: np.ndarray, mask_fullsize: np.ndarray, patch_size=100, stride=100, patch_mask_dir=None, ignore_area=100):
    from skimage.transform import resize
    ratio =  mask_fullsize.shape[0] / mask.shape[0]
    ignore_area = min(ignore_area, (0.11*patch_size)**2) * ratio * ratio
    patch_size_, stride_ = int(ratio * patch_size), int(ratio * stride)
    patches = []

    delta = patch_size_ - stride_
    # print(patch_size_, stride_, delta)
    
    ys, xs = np.where(mask == 1)
    if len(ys) > 0 and len(xs) > 0:
        top_left_y = int(np.min(ys) * ratio)
        bottom_right_y = int(np.max(ys) * ratio)
        top_left_x = int(np.min(xs) * ratio)
        bottom_right_x = int(np.max(xs) * ratio)
    
    num = 0
    record = {} 
    y_max_idx = (bottom_right_y - top_left_y - 1) // stride_ 
    x_max_idx = (bottom_right_x - top_left_x - 1) // stride_
    for y in range(top_left_y, bottom_right_y, stride_):
        y_idx = (y-top_left_y) // stride_
        for x in range(top_left_x, bottom_right_x, stride_):
            x_idx = (x-top_left_x) // stride_

            n_component = 0
            patch = mask_fullsize[y:y+patch_size_, x:x+patch_size_]
            num_one = np.sum(patch) 

            if num_one > ignore_area: # num_one < ignore_area的也要加入 record[(y_idx, x_idx)] = '1'
                tl_y, br_y, tl_x, br_x = get_mask_bound(patch, return_mode='4xy')
                # 下面取出边界上重复且覆盖太小的patch
                if ((patch_size_ - tl_y) < delta) or (br_y < delta) or ((patch_size_ - tl_x) < delta) or (br_x < delta):
                    if ((patch_size_ - tl_y) < delta) and (y_idx < y_max_idx): # mask本身在下方, 需要下方还有patch才能skip
                        # skip后如果下侧patch所含mask区域和现在一样，就不能skip了，记录位置
                        record[(y_idx, x_idx)] = '0' # 直接skip，下方存在的patch直接无法skip (下方也包含这些，肯定num_one > ignore_area)
                        continue

                    elif (br_y < delta) and (y_idx > 0): # mask本身在上方, 需要上方还有patch才能skip
                        # 如果上方还有patch，且上方patch所含mask区域和现在一样，那该区域在上方patch中一定位于下方且先前被记录
                        # 上方patch [y_idx-1, x_idx]
                        if record[(y_idx-1, x_idx)] == '1': # 上方没有skip
                            record[(y_idx, x_idx)] = '0'
                            continue
                        else:
                            record[(y_idx, x_idx)] = '1'

                    elif ((patch_size_ - tl_x) < delta) and (x_idx < x_max_idx): # mask本身在右侧, 需要右侧还有patch才能skip
                        # skip后如果右侧patch所含mask区域和现在一样，就不能skip了，记录位置
                        record[(y_idx, x_idx)] = '0' # 直接skip，右侧存在的patch直接无法skip
                        continue

                    elif (br_x < delta) and (x_idx > 0): # mask本身在左侧, 需要左侧还有patch才能skip
                        # 如果左侧还有patch，且左侧patch所含mask区域和现在一样，那该区域在左侧patch中一定位于右侧且先前被记录
                        # 左侧patch [y_idx, x_idx-1]
                        if record[(y_idx, x_idx-1)] == '1': # 左侧没有skip
                            record[(y_idx, x_idx)] = '0'
                            continue 
                        else:
                            record[(y_idx, x_idx)] = '1'
                    else:
                        record[(y_idx, x_idx)] = '1' # 没有skip
                else:
                    record[(y_idx, x_idx)] = '1' # 没有skip

                # print(y,x,'--------', num, n_component,'---------', tl_y, br_y, tl_x, br_x, 'yes')
                component_masks = extract_components_custom(patch, n=1)

                for component_mask in component_masks:
                    num_one_component = np.sum(component_mask)
                    if num_one_component > ignore_area:
                        patch_on_mask_fullsize = np.zeros_like(mask_fullsize, dtype=np.uint8)
                        patch_on_mask_fullsize[y:y+patch_size_, x:x+patch_size_] = component_mask

                        patch_on_mask = resize(
                            patch_on_mask_fullsize,
                            mask.shape,
                            order=0,            
                            anti_aliasing=False, 
                            preserve_range=True  
                        )
                       
                        patch_on_mask = (patch_on_mask > 0.5).astype(np.uint8)

                        patches.append((patch_on_mask, (y, x), num_one_component, num, n_component, patch_on_mask_fullsize))
                        Image.fromarray(255*patch_on_mask).save(os.path.join(patch_mask_dir, 'try_s_%d_%d.jpg' % (num, n_component)))
                        n_component +=1
                num += 1
            else:
                record[(y_idx, x_idx)] = '1'

    return patches


def extract_components(mask, connectivity=1):
    labeled, num = label(mask, structure=np.ones((3,3)) if connectivity==2 else None)
    return labeled, num

def extract_components_custom(mask, n=1):
    mask = mask.astype(np.uint8)
    structure = np.ones((2*n+1, 2*n+1), dtype=np.uint8)

    labeled, num = label(mask, structure=structure)
    masks = []
    for i in range(1, num + 1):
        comp_mask = (labeled == i).astype(np.uint8)
        masks.append(comp_mask)

    return masks

# -------------------------------------------------------------------------------------------------------------------

def cal_pad_para(shape_before_pad, shape_after_pad):
    pad_size0 = shape_after_pad[0] - shape_before_pad[0] # source_rescale.shape[0] - source_foreground.shape[0]
    pad_size1 = shape_after_pad[1] - shape_before_pad[1] # source_rescale.shape[1] - source_foreground.shape[1]
    pad1 = (math.floor(pad_size1 / 2), math.ceil(pad_size1 / 2))
    pad0 = (math.floor(pad_size0 / 2), math.ceil(pad_size0 / 2))
    return pad0, pad1

def unpad(pad_image, pad_para):
    # pad_para: (t_pad0, t_pad1)
    pad0, pad1 = pad_para
    end_index_0 = -pad0[1] if pad0[1] > 0 else None
    end_index_1 = -pad1[1] if pad1[1] > 0 else None

    original = pad_image[pad0[0] : end_index_0, pad1[0] : end_index_1]
    return original

# (a, b) -> (a - t_pad0[0], b - t_pad1[0])
def unpad_coordinates(coords,  pad_para):
    pad0, pad1 = pad_para
    y_offset = pad0[0]
    x_offset = pad1[0]
    offset_vector = np.array([y_offset, x_offset])
    coords_unpad = coords - offset_vector
    return coords_unpad

# -----------------------------------------------------------------------

def warp_img(transform_file, img, return_mode='Image', is_mask=False, already_tensor=False):
    # [1, 3, 6451, 13306] [1, 3, H, W]
    if not already_tensor:
        if isinstance(img, Image.Image):
            img = np.array(img)
            img = torch.tensor(img)
        elif isinstance(img, np.ndarray):
            img = torch.tensor(img)
        elif not isinstance(img, torch.tensor):
            raise TypeError("img must be a numpy array or PIL.Image.Image or torch.tensor")
        
        if img.ndim == 2:
            img = img.unsqueeze(-1)
        # input img: [H,W,C]
        
        img = img.permute(2,0,1).unsqueeze(0) # [H,W,3] -> [C,H,W] -> [1,C,H,W]

    if is_mask:
        img = img.float()
    else:
        img = img.float() / 255.0
    
    grid = F.affine_grid(transform_file, img.shape, align_corners=False)

    interpolation_mode = 'nearest' if is_mask else 'bilinear'
    # interpolation_mode = 'bilinear'
    warped = F.grid_sample(img, grid, align_corners=False, mode=interpolation_mode) # [1,C,H,W]
    warped_img = warped.squeeze(0).permute(1, 2, 0)  # [1,C,H,W] -> [C,H,W] -> [H,W,C]

    if is_mask:
        print(torch.max(warped_img))
        warped_img = (warped_img > 0).byte()
        print(torch.sum(warped_img))
        print(warped_img.shape)
        print('-----------')
    else:
        warped_img = (warped_img * 255).clamp(0, 255).byte()


    # warped_img = (warped_img * 255).clamp(0, 255).byte()
    warped_img = warped_img.cpu().numpy()
    if return_mode == 'Image':
        warped_img = Image.fromarray(warped_img)
    elif return_mode == 'torch':
        warped_img = torch.tensor(warped_img)
    elif return_mode == 'numpy':
        warped_img = warped_img
    return warped_img


def warp_coords(forward_transform, coords, img_shape):
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


def get_forward_transform(back_transform):
    M_inv = back_transform.squeeze(0)
    
    rot_scale_inv = M_inv[:, :2]
    translation_inv = M_inv[:, 2:]
    
    rot_scale_forward = torch.inverse(rot_scale_inv)
    
    translation_forward = -torch.matmul(rot_scale_forward, translation_inv)

    M_forward = torch.cat((rot_scale_forward, translation_forward), dim=1)
    return M_forward


def get_forward_transform_simple(back_transform):
    theta = back_transform[0]        # [2,3]
    theta_h = torch.eye(3)           # 变成 3x3
    theta_h[:2,:] = theta

    theta_inv = torch.inverse(theta_h)[0:2,:] 
    return theta_inv


def points_from_csv(csv_path):
    LANDMARK_COORDS = ['X', 'Y']
    df = pd.read_csv(csv_path)
    points = df[LANDMARK_COORDS].values
    points_np = np.column_stack((points[:, 1], points[:, 0]))
    return points_np

def in_image_bound(coords, h, w):
    y_coords = coords[:, 0]
    x_coords = coords[:, 1]

    
    y_mask = (y_coords >= 0) & (y_coords < h) # y in [0, w-1]
    x_mask = (x_coords >= 0) & (x_coords < w) # x in [0, h-1]

    in_bounds_mask = x_mask & y_mask

    return in_bounds_mask

def clean_pair_points(source_points, target_points, source_fore, target_fore, no_pad_shape=None, threshold=10, clean_control=False):
    # remove points faraway from the foreground
    import skimage
    assert source_points.shape == target_points.shape
   
    if clean_control: # control tissue in the left
        ori_source_shape, ori_target_shape = no_pad_shape # (321, 607) (500, 1321), np.array, (h,w)
        ratio_h = ori_target_shape[0] / ori_source_shape[0]
        ratio_w = ori_target_shape[1] / ori_source_shape[1]

        
        if (ratio_w > 1.6):
            H , W = target_fore.shape
            label_image = skimage.measure.label(target_fore)
            regions = skimage.measure.regionprops(label_image)
            
            # small noisy area
            min_area_threshold = W * H * 0.001
            valid_regions = [r for r in regions if r.area > min_area_threshold]
                
            leftmost_region = min(valid_regions, key=lambda r: r.bbox[1]) # min_h, min_w, max_h, max_w
            # w_thre = min(leftmost_region.bbox[3], 0.3*W)
            if leftmost_region.bbox[3] > 0.3*W:
                w_thre = 0.3*W
            else:
                w_thre = 0.2*W
            print('存在IHC左对照', 'ratio_w: %.2f' %ratio_w, ' w_thre/H: %s/%s' % (w_thre, W))
            
            target_w = target_points[:, 1]
            out_control = target_w > w_thre
            out_control_source = source_points[out_control]
            out_control_target = target_points[out_control]
            print(f"取出IHC左对照, 共 {len(source_points)} 对点，保留 {len(out_control_source)} 对")
            source_points, target_points = out_control_source, out_control_target

    src_mask = (source_fore > 0).astype(np.uint8)
    tgt_mask = (target_fore > 0).astype(np.uint8)
    
    dist_src = cv2.distanceTransform(1 - src_mask, cv2.DIST_L2, 5)
    dist_tgt = cv2.distanceTransform(1 - tgt_mask, cv2.DIST_L2, 5)
    
    source_points_0_clip = np.clip(source_points[:, 0].astype(int), None, dist_src.shape[0]-1)
    source_points_1_clip = np.clip(source_points[:, 1].astype(int), None, dist_src.shape[1]-1)
    target_points_0_clip = np.clip(target_points[:, 0].astype(int), None, dist_tgt.shape[0]-1)
    target_points_1_clip = np.clip(target_points[:, 1].astype(int), None, dist_tgt.shape[1]-1)
    
    src_dists = dist_src[source_points_0_clip, source_points_1_clip] # IndexError: index 499 is out of bounds for axis 1 with size 499
    tgt_dists = dist_tgt[target_points_0_clip, target_points_1_clip]

    keep_mask = (src_dists <= threshold) & (tgt_dists <= threshold)

    clean_source = source_points[keep_mask]
    clean_target = target_points[keep_mask]

    return clean_source, clean_target



def cluster_consistent_pairs_louvain(G):
    # Louvain methods
    partition = community_louvain.best_partition(G, random_state=42)
    clusters = {}
    for node, cid in partition.items():
        clusters.setdefault(cid, []).append(node)
    return list(clusters.values())
        
def cluster_consistent_pairs_leiden(G, partition_type='modularity'):
    """
    Leiden for NetworkX G
    
    Args:
        G (networkx.Graph): input graph
        partition_type (str): Leiden  'modularity' or 'CPM'
    
    Returns:
        List[List]
    """
    # NetworkX -> iGraph
    nx_nodes = list(G.nodes)
    G_ig = ig.Graph.from_networkx(G)

    if partition_type == 'modularity':
        partition = leidenalg.find_partition(G_ig, leidenalg.ModularityVertexPartition, seed=42)
    elif partition_type == 'CPM':
        partition = leidenalg.find_partition(G_ig, leidenalg.CPMVertexPartition, seed=42)
    else:
        raise ValueError("partition_type must be 'modularity' or 'CPM'")

    # communities = [[nx_nodes[v.index] for v in cluster] for cluster in partition]
    communities = [[nx_nodes[v] for v in cluster] for cluster in partition]
    
    return communities

def split_pair_points(s_k, t_k, idx, distance_diff_threshold=10, mode='leiden', return_matrix=False, print_projection=False, multi=True):
    source_points = s_k[idx]
    target_points = t_k[idx]

    if mode == 'ransac_homo':
        M, mask = cv2.findHomography(source_points, target_points, method=cv2.RANSAC, ransacReprojThreshold=distance_diff_threshold) # maxIters default 2000
        ransac_idx = np.where(mask.ravel() == 1)[0]
        left_ransac_idx = np.where(mask.ravel() == 0)[0]
        idxs = [idx[ransac_idx]]
        left_idxs = idx[left_ransac_idx]

        left_source_points = s_k[left_idxs[0]]
        left_target_points = t_k[left_idxs[0]]


        if len(left_idxs) > 4:
            M, mask = cv2.findHomography(left_source_points, left_target_points, cv2.RANSAC, distance_diff_threshold) 
            if not (mask is None or mask.size == 0):
                ransac_idx = np.where(mask.ravel() == 1)[0]
                idxs.append(left_idxs[ransac_idx])
                print('Add second ransac: %d points' % len(ransac_idx))


        if print_projection:
            if source_points.ndim == 2: # [N,1,2]
                source_points_h = source_points[:, np.newaxis, :]
            else:
                source_points_h = source_points

            projected_points = cv2.perspectiveTransform(source_points_h, M)
            projected_points = projected_points.squeeze()
            target_points_flat = target_points.squeeze()

            print('all_source', source_points[ransac_idx])
            print('projected_points', projected_points[ransac_idx])
            diff = projected_points - target_points_flat
            squared_diff = diff ** 2

            reprojection_errors = np.sqrt(np.sum(squared_diff, axis=1))
            print('reprojection_errors')
            print(reprojection_errors)
        
    elif mode == 'ransac_affine':
        # cv2.estimateAffinePartial2D
        # cv2.USAC_DEFAULT -> cv2.RANSAC
        M, mask = cv2.estimateAffine2D(source_points, target_points, method=cv2.RANSAC, ransacReprojThreshold=distance_diff_threshold) # maxIters default 2000
        # idxs = (mask.ravel()==1)
        ransac_idx = np.where(mask.ravel() == 1)[0]
        left_ransac_idx = np.where(mask.ravel() == 0)[0]
        idxs = [idx[ransac_idx]]
        left_idxs = idx[left_ransac_idx]

        left_source_points = s_k[left_idxs]
        left_target_points = t_k[left_idxs]


        if len(left_idxs) > 4:
            M, mask = cv2.estimateAffine2D(left_source_points, left_target_points, method=cv2.RANSAC, ransacReprojThreshold=distance_diff_threshold)
            if not (mask is None or mask.size == 0):
                ransac_idx = np.where(mask.ravel() == 1)[0]
                idxs.append(left_idxs[ransac_idx])
                print('Add second ransac: %d points' % len(ransac_idx))

    elif mode == 'ransac_partial':
        M, mask = cv2.estimateAffinePartial2D(source_points, target_points, method=cv2.RANSAC, ransacReprojThreshold=distance_diff_threshold) # maxIters default 2000
        # idxs = (mask.ravel()==1)
        ransac_idx = np.where(mask.ravel() == 1)[0]
        left_ransac_idx = np.where(mask.ravel() == 0)[0]
        idxs = [idx[ransac_idx]]
        left_idxs = idx[left_ransac_idx]


        left_source_points = s_k[left_idxs]
        left_target_points = t_k[left_idxs]


        if len(left_idxs) > 4:
            M, mask = cv2.cv2.estimateAffinePartial2D(left_source_points, left_target_points, method=cv2.RANSAC, ransacReprojThreshold=distance_diff_threshold)
            if not (mask is None or mask.size == 0):
                ransac_idx = np.where(mask.ravel() == 1)[0]
                idxs.append(left_idxs[ransac_idx])
                print('Add second ransac: %d points' % len(ransac_idx))
        
        # print_projection = True
        if print_projection:
            if source_points.ndim == 2: # [N,1,2]
                source_points_h = source_points[:, np.newaxis, :]
            else:
                source_points_h = source_points
            print('source_points_h', source_points_h)

            projected_points = cv2.transform(source_points_h, M)
            # print(' projected_points',  projected_points)
            projected_points = projected_points.squeeze()
            target_points_flat = target_points.squeeze()

            print('all_source', source_points[ransac_idx])
            print('projected_points', projected_points[ransac_idx])
            print('target_points_flat', target_points_flat)
            diff = projected_points - target_points_flat
            squared_diff = diff ** 2

            reprojection_errors = np.sqrt(np.sum(squared_diff, axis=1))
            print('reprojection_errors')
            print(reprojection_errors)
    elif mode == 'ransac_rigid':
        from skimage.transform import estimate_transform
        from skimage.measure import ransac
        from skimage.transform import EuclideanTransform

        
        try:
            model, inliers = ransac((source_points, target_points), EuclideanTransform, min_samples=3, residual_threshold=distance_diff_threshold)
        except:
            inliers = None
        if inliers is None:
            idxs = [[]]
            return idxs
        
        
        ransac_idx = np.where(inliers)[0]
        left_ransac_idx = np.where(inliers == False)[0]


        idxs = [idx[ransac_idx]]
        left_idxs = idx[left_ransac_idx]

        left_source_points = s_k[left_idxs]
        left_target_points = t_k[left_idxs]

        # print('left_source_points', len(left_source_points))
        # print('left_idxs', left_idxs) # left_idxs[0] [93 77 80 74 75]
        if multi and (len(left_idxs) > 3): 
            # M, mask = cv2.cv2.estimateAffinePartial2D(left_source_points, left_target_points, method=cv2.RANSAC, ransacReprojThreshold=distance_diff_threshold)
            try:
                model, inliers_ = ransac((left_source_points, left_target_points), EuclideanTransform, min_samples=3, residual_threshold=distance_diff_threshold)
            except:
                inliers = None
            # print('inliers', inliers)
            if inliers is not None:
                try:
                    if np.sum(inliers_) >= 3:
                        ransac_idx_ = np.where(inliers_)[0]
                        new_idx = left_idxs[ransac_idx_]
                        avg_dist_sk, avg_dist_tk = check_average_distance_within_threshold(s_k[idxs[0]], t_k[idxs[0]], s_k[new_idx], t_k[new_idx])
                        # print(abs(avg_dist_tk - avg_dist_sk))
                        is_within = abs(avg_dist_tk - avg_dist_sk) <= 25
                        if is_within:
                            idxs.append(left_idxs[ransac_idx_])
                            print('Add second ransac: %d points' % len(ransac_idx_))
                except:
                    second_ransac = False
            #print('Fail to do second ransac')
        
        print_projection = True      
    elif mode in ['connect', 'complete', 'louvain', 'leiden']:
        # print('s/t_points', source_points.shape, target_points.shape)
        source_dist_matrix = np.linalg.norm(source_points[:, np.newaxis] - source_points[np.newaxis, :], axis=2)
        target_dist_matrix = np.linalg.norm(target_points[:, np.newaxis] - target_points[np.newaxis, :], axis=2)

        dist_diff_matrix = np.abs(source_dist_matrix - target_dist_matrix)
        # print('dist_diff_matrix', np.max(dist_diff_matrix))
    
        if np.max(dist_diff_matrix) > distance_diff_threshold:
            num_points = source_points.shape[0]
            
            G = nx.Graph()
            for i in range(num_points):
                G.add_node(i)

            for i in range(num_points):
                for j in range(i + 1, num_points):
                    if dist_diff_matrix[i, j] < distance_diff_threshold:
                        G.add_edge(i, j)
        
            if mode == 'connect': # 'connected_components'
                communities = list(nx.connected_components(G))
            elif mode == 'complete': # 'complete_subgraph' 
                communities = list(nx.find_cliques(G)) # cliques
            elif mode == 'louvain':
                communities = cluster_consistent_pairs_louvain(G)
                # print('louvain', communities, len(communities))
            elif mode == 'leiden':
                communities = cluster_consistent_pairs_leiden(G, partition_type='modularity')
                # print('leiden', communities, len(communities))
            else:
                raise NotImplementedError
        
            if len(communities) > 1:
                idxs = [idx[list(item)].tolist() for item in communities] # [idx[np.array(list(item))] for item in communities]
            else:
                idxs = [idx]
        else:
            # print('single', idxs)
            idxs = [idx]
    
    if return_matrix:
        return idxs, dist_diff_matrix
    else:
        return idxs

def check_average_distance_within_threshold(sk1, tk1, sk2, tk2):
    def pairwise_dist(a, b):
        return np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2))

    dist_sk = pairwise_dist(sk2, sk1)
    dist_tk = pairwise_dist(tk2, tk1)
    
    # avg_dist_sk = np.mean(np.min(dist_sk, axis=1))
    # avg_dist_tk = np.mean(np.min(dist_tk, axis=1))

    avg_dist_sk1 = np.mean(dist_sk)
    avg_dist_tk1 = np.mean(dist_tk)
    # print(avg_dist_sk, avg_dist_tk)
    # print(avg_dist_sk1, avg_dist_tk1)
    # is_within = abs(avg_dist_tk - avg_dist_sk) <= threshold
    
    return avg_dist_sk1, avg_dist_tk1

# get new_ori according to ref_points
def get_new_slide_ori_refp(ref_points, set_ori=False, delta=15):
    # ref_points = torch.from_numpy(ref_points.astype(np.float32)).unsqueeze(0).repeat(B, 1, 1)
    if set_ori:
        # min_x = min(p[0] for p in ref_points)
        # min_y = min(p[1] for p in ref_points)
        # origin = (min_x, min_y)
        min_h, min_w = torch.min(ref_points[0], axis=0).values
        # print(ref_points[0].shape, min_h, min_w)
        min_h = max(min_h-delta, 0)
        min_w = max(min_w-delta, 0)

        origin = torch.tensor([min_h, min_w]) 
    else:
        origin = torch.tensor([0,0]) 

    return origin

def get_new_slide_ori_foreground(foreground_mask, set_ori=False, delta=15):

    if set_ori:
        xs, ys = np.where(foreground_mask == 1)

        if len(xs) > 0 and len(ys) > 0:
            top_left_x = np.min(xs)
            top_left_y = np.min(ys)

        origin = torch.tensor([top_left_x, top_left_y]) 
    else:
        origin = torch.tensor([0,0]) 

    return origin

def get_input(source_ref, source_calc, target_ref, target_calc, source_foreground, target_foreground, device=None, patch_ratio=1, set_ori=False, return_ori=False):
    # source_new_ori = get_new_slide_ori_foreground(source_foreground, set_ori=set_ori) # 如果source_foreground 换成binary_mask，效果很差，说明对平移还没学好
    # target_new_ori = get_new_slide_ori_foreground(target_foreground, set_ori=set_ori)
    
    source_new_ori = get_new_slide_ori_refp(source_ref, set_ori=set_ori) # 如果source_foreground 换成binary_mask，效果很差，说明对平移还没学好
    target_new_ori = get_new_slide_ori_refp(target_ref, set_ori=set_ori)
    # source_new_ori = torch.tensor([ 0, 300])
    # target_new_ori = torch.tensor([ 0, 500])
    # print('new_ori', source_new_ori, target_new_ori) # new_ori tensor([ 43, 223]) tensor([19, 21])
    
    source_ref_ = source_ref - source_new_ori # torch.from_numpy
    source_calc_ = source_calc - source_new_ori
    target_ref_ = target_ref - target_new_ori
    target_calc_ = target_calc - target_new_ori
    
    
    if patch_ratio != 1:
        source_ref_ = patch_ratio * source_ref_
        source_calc_ = patch_ratio * source_calc_
        target_ref_ = patch_ratio * target_ref_
        target_calc_ = patch_ratio * target_calc_


    source_ref_ = source_ref_.to(device)
    source_calc_ = source_calc_.to(device)
    target_ref_ = target_ref_.to(device)
    target_calc_ = target_calc_.to(device)
    if return_ori:
        return source_ref_, source_calc_, target_ref_, target_calc_, (source_new_ori, target_new_ori)
    return source_ref_, source_calc_, target_ref_, target_calc_


def get_foreground_b1b2(img_shape, args=None, mask_dir=None):
    img_foreground = Image.open(mask_dir).resize((img_shape[1], img_shape[0]))
    img_foreground = (np.array(img_foreground) > 100).astype(np.uint8)
    
    return img_foreground

def get_foreground_biopsy(geo_file_dir, result_size, slide_size):
    mask = np.zeros((result_size[1], result_size[0]), dtype=np.uint8)
    with open(geo_file_dir, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
        polygons = json_data['features']
    
    # print(slide_size, result_size)
    mask_ratio, _ = get_ratio(slide_size, result_size)
    for polygon in polygons:
        coords = polygon["geometry"]["coordinates"][0]
        vertices = np.array(coords) 
        vertices = vertices / mask_ratio

        vertices = vertices.astype(np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [vertices], color=1)

    
    return mask

def get_ratio(ori_size, new_size):
    ratio_w = ori_size[0] / new_size[0]
    ratio_h = ori_size[1] / new_size[1]
    assert abs(ratio_w - ratio_h) < 1
    return ratio_w, ratio_h


def load_source_mask(anno_dir, anno_mask_dir, result_size, source_foreground):   
    mask_image = np.array(Image.open(anno_mask_dir))
    anno_size = (mask_image.shape[1], mask_image.shape[0])

    anno_masks = {}
    # anno_dir = os.path.join(anno_patient_dir, slide_name+'.json')
    with open(anno_dir, 'r', encoding='utf-8') as f_a:
        json_anno_data = json.load(f_a)
    
    anno_ratio, _ = get_ratio(anno_size, result_size)
    for anno_id in json_anno_data.keys():
        mask_anno = np.zeros((result_size[1], result_size[0]), dtype=np.uint8)
        anno_coords = json_anno_data[anno_id]
        anno_vertices = np.array(anno_coords) / anno_ratio
        anno_vertices = anno_vertices.astype(np.int32).reshape((-1, 1, 2))
        # print(anno_vertices.shape)
        cv2.fillPoly(mask_anno, [anno_vertices], color=1)
        mask_anno = np.logical_and(mask_anno, source_foreground) # 这里是没有pad之前的
        # mask_anno = (mask_anno.astype(np.uint8)) * 255
        mask_anno = mask_anno.astype(np.uint8)
        anno_masks[anno_id] = mask_anno
    
    """
    mask_bool = binary_mask.astype(bool)
    mask_ori = binary_mask_ori.astype(bool)
    return mask_bool, mask_ori
    """
    return anno_masks


def load_draw_mask(anno_dir, anno_mask_dir, result_size, source_foreground):   
    mask_image = np.array(Image.open(anno_mask_dir))
    anno_size = (mask_image.shape[1], mask_image.shape[0])

    anno_masks = {}
    # anno_dir = os.path.join(anno_patient_dir, slide_name+'.json')
    with open(anno_dir, 'r', encoding='utf-8') as f_a:
        json_anno_data = json.load(f_a)
    
    anno_ratio, _ = get_ratio(anno_size, result_size)
    for anno_id in json_anno_data.keys():
        mask_anno = np.zeros((result_size[1], result_size[0]), dtype=np.uint8)
        anno_coords = json_anno_data[anno_id]
        anno_vertices = np.array(anno_coords) / anno_ratio
        anno_vertices = anno_vertices.astype(np.int32).reshape((-1, 1, 2))
        # print(anno_vertices.shape)
        cv2.fillPoly(mask_anno, [anno_vertices], color=1)
        mask_anno = np.logical_and(mask_anno, source_foreground) # 这里是没有pad之前的
        # mask_anno = (mask_anno.astype(np.uint8)) * 255
        mask_anno = mask_anno.astype(np.uint8)
        anno_masks[anno_id] = mask_anno
    
    return anno_masks


def get_k_nearest_points_to_mask(points, mask, k, rescale_=1):
    distance_map = distance_transform_edt(mask == 0) 

    h, w = mask.shape
    valid_points = []
    distances = []

    for pt in points:
        x, y = int(pt[0] * rescale_), int(pt[1] * rescale_)
        if 0 <= x < h and 0 <= y < w:
            dist = distance_map[x, y]
        else:
            dist = np.inf 
            print('Error', x, h,'--', y,w)
        valid_points.append(pt)
        distances.append(dist)

    # get topk
    distances = np.array(distances)
    valid_points = np.array(valid_points)

    topk_indices = np.argsort(distances)[:k]
    nearest_k_points = valid_points[topk_indices]

    return topk_indices, nearest_k_points


def get_k_nearest_points_to_mask_new(points, mask, k, rescale_=1, ignore_len=30, min_num_p=5):
    area = np.sum(mask)
    distance_map = distance_transform_edt(mask == 0) 
   
    h, w = mask.shape
    valid_points = []
    distances = []
    
    for pt in points:
        x, y = int(pt[0] * rescale_), int(pt[1] * rescale_)
        if 0 <= x < h and 0 <= y < w:
            dist = distance_map[x, y] 
        else:
            dist = np.inf 
        
        valid_points.append(pt)
        distances.append(dist)
    # print('points', points)
    # print('distances', distances)

    # get topk
    distances = np.array(distances)
    valid_points = np.array(valid_points)
    

    sorted_indices = np.argsort(distances)
    sorted_distances = distances[sorted_indices]


    zero_count = np.argmax(sorted_distances != 0)  
    if zero_count == 0 and sorted_distances[0] == 0: 
        zero_count = len(sorted_distances)
    
    zero_count += 1 
    if zero_count >= k:
        indices = sorted_indices[:zero_count]
    else:
        indices = sorted_indices[:k] # 头k个
    
    # indices = np.argsort(distances)[:k]
    top_k_distances = distances[indices]

    ignore_len = max(ignore_len, 0.3*100)
    close_idx = np.where(top_k_distances < ignore_len)[0]
    if len(close_idx) > min_num_p:
        indices = indices[close_idx]
        top_k_distances = distances[indices]
    else:
        indices = indices[:min_num_p]
        top_k_distances = distances[indices]
    # ---------------------------------------------------------------------------------------------
    nearest_k_points = valid_points[indices]

    return indices, nearest_k_points


from scipy.ndimage import binary_dilation

def expand_mask(mask, delta=5):
    structure = np.ones((delta, 2*delta+1))  
    expanded_mask = binary_dilation(mask, structure=structure).astype(np.uint8)
    return expanded_mask


def get_ref_calc_mask(mask, ps, idx, device, args=None, source_expand=False, delta=5):
    coords = np.argwhere(mask == 1) # assert mask[coords[i][0], coords[i][1]] == 1
    
    ref_points = [ps[idx[i]] * args.rescale for i in range(len(idx))]
    ref_points = np.stack(ref_points) # ref_points[0]: [247.3 262.3]

    if source_expand:
        # print('expand source')
        h, w = mask.shape
        x_range = np.arange(-delta, h + delta)
        y_range = np.arange(-delta, w + delta)
        
        x_coords, y_coords = np.meshgrid(x_range, y_range)
        x_coords = x_coords.ravel()
        y_coords = y_coords.ravel()

        valid_mask_coords = ((x_coords < 0) | (x_coords >= h) | (y_coords < 0) | (y_coords >= w))
        expand_coords1 = np.column_stack((x_coords[valid_mask_coords], y_coords[valid_mask_coords]))

        mask_expand = expand_mask(mask, delta=10) # 外扩了10 pixel
        expand_coords2 = np.argwhere(mask_expand == 0)


        expand_calc_points = np.vstack([coords, expand_coords1, expand_coords2])
        calc_points = torch.from_numpy(expand_calc_points.astype(np.float32)).unsqueeze(1)
        
    else:
        calc_points = torch.from_numpy(coords.astype(np.float32)).unsqueeze(1)
    B = len(calc_points)
    ref_points = torch.from_numpy(ref_points.astype(np.float32)).unsqueeze(0).repeat(B, 1, 1)
    
    return ref_points, calc_points


def get_thumbnail(img, rescale=0.1):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    elif not isinstance(img, Image.Image):
        raise TypeError("img must be a numpy array or PIL.Image.Image")
    
    ori_w, ori_h = img.size  # w, h
    new_image = img.resize([int(rescale*ori_w), int(rescale*ori_h)])
    return new_image
  
def display_one_image(ps, img_, rescale_=0.1, indictor=np.array([255,0,0]), num=None, idx=None, binary_map=None, size=100):
    """
    indictor: color to draw the landmark
    rescale_: rescale size to get the thumbnail
    num: the max number of points to display
    """
    size = size 
    img = img_.copy()
    num = len(idx)
    for i in range(num):
        p = ps[idx[i]]
        p_x, p_y = int(p[0]), int(p[1])
        img[p_x-size:p_x+size, p_y-size:p_y+size, :] = indictor
    
    img_rescale = get_thumbnail(img, rescale=rescale_) # get_thumb后失去标红landmarks

    if binary_map is not None:
        img_rescale = np.array(img_rescale)
        mask_rgb = np.repeat(1-0.5*binary_map[:, :, None], 3, axis=2)
        img_rescale = img_rescale * mask_rgb
        img_rescale = np.clip(img_rescale, 0, 255).astype(np.uint8)
        img_rescale = Image.fromarray(img_rescale)

    return img_rescale

def seed_torch(seed=7, device=None):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True






    