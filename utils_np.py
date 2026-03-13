import json
import numpy as np
import torch
import random
import torch.nn.functional as F
# ------------------------------------------

def add_border_numpy(img, border_width=2, color=(255, 0, 0)):
    img_with_border = img.copy()
    h, w, _ = img.shape

    img_with_border[:border_width, :, :] = color
    img_with_border[-border_width:, :, :] = color
    img_with_border[:, :border_width, :] = color
    img_with_border[:, -border_width:, :] = color

    return img_with_border

def get_random_point(w, h, num=3):
    ref_points = []
    x_coords = np.random.randint(0, w, size=num).tolist()
    y_coords = np.random.randint(0, h, size=num).tolist()
    # ref_points = list(zip(x_coords, y_coords))
    ref_points = np.stack((x_coords, y_coords), axis=1)
    return ref_points

def get_random_point_range(x_range, y_range, num=3):
    x_start, x_end = x_range
    y_start, y_end = y_range
    ref_points = []
    x_coords = np.random.randint(x_start, x_end, size=num).tolist()
    y_coords = np.random.randint(y_start, y_end, size=num).tolist()
    # ref_points = list(zip(x_coords, y_coords))
    ref_points = np.stack((x_coords, y_coords), axis=1)
    return ref_points

def sample_points_mask(w, h, p, r, num, around_p_ratio=0):
    # p: Existing cal_point
    # r: Minimum dx/dy of the remaining neg_cal samples from p

    mask = np.ones((w, h), dtype=bool)
    p = p.squeeze()
    # print(p, p[0], p.shape)
   
    x_start = max(int(p[0])-r, 0)
    x_end = min(int(p[0])+r, w)
    y_start = max(int(p[1])-r, 0)
    y_end = min(int(p[1])+r, h)
    # print(x_start, x_end, y_start, y_end)
    mask[x_start:x_end, y_start:y_end] = False

    xs, ys = np.where(mask) 
    left_num = num - int(around_p_ratio * num)
    idx = np.random.choice(len(xs), size=left_num, replace=False)

    sampled_points = np.stack([xs[idx], ys[idx]], axis=1)
    if around_p_ratio > 0:
        points_around_p = sample_points_around_p(length=80, p=p, r=1.5*r, num=int(around_p_ratio*num))
        sampled_points = np.vstack([sampled_points, points_around_p])
    return sampled_points


def sample_points_around_p(length, p, r, num):
    # p: Existing cal_point
    # r: Minimum dx/dy of the remaining neg_cal samples from p

    mask = np.ones((2 *length, 2 * length), dtype=bool)
    p = p.squeeze()
    # print(p, p[0], p.shape)
    center = np.array([length, length])
   
    x_start = max(int(center[0]-r), 0)
    x_end = min(int(center[0]+r), 2 * length)
    y_start = max(int(center[1]-r), 0)
    y_end = min(int(center[1]+r), 2 * length)
    
    # print(x_start, x_end, y_start, y_end)
    mask[x_start:x_end, y_start:y_end] = False

    xs, ys = np.where(mask) 

    idx = np.random.choice(len(xs), size=int(num), replace=False)

    sampled_points = np.stack([xs[idx], ys[idx]], axis=1)
    sampled_points = sampled_points - center + p
    return sampled_points


def sample_site_points_mask(w, h, p, ref_ps, r, delta, num):
    # p: Existing cal_point
    # r: Minimum dx/dy of the remaining neg_cal samples from p

    mask = np.zeros((int(w+r), int(h+r)), dtype=bool)
    p = p.squeeze()
    # print(p, p[0], p.shape)
    for ref_p in ref_ps: 
        ref_x_start = max(int(ref_p[0]-r-3), 0)
        ref_x_end = min(int(ref_p[0]+r+3), int(w+r))
        ref_y_start = max(int(ref_p[1]-r-3), 0)
        ref_y_end = min(int(ref_p[1]+r+3), int(h+r))
        
        mask[ref_x_start:ref_x_end, ref_y_start:ref_y_end] = 1
        # print(np.sum(mask))

        """
        ref_x_start = max(int(ref_p[0]-r//2), 0)
        ref_x_end = min(int(ref_p[0]+r//2), w)
        ref_y_start = max(int(ref_p[1]-r//2), 0)
        ref_y_end = min(int(ref_p[1]+r//2), h)
        mask[ref_x_start:ref_x_end, ref_y_start:ref_y_end] = 0
        """
    # print('r', r)
    # print('-----------------')
   
    x_start = max(int(p[0])-delta, 0)
    x_end = min(int(p[0])+delta, int(w+r))
    y_start = max(int(p[1])-delta, 0)
    y_end = min(int(p[1])+delta, int(h+r))
    # print(x_start, x_end, y_start, y_end)
    mask[x_start:x_end, y_start:y_end] = 0

    xs, ys = np.where(mask)

    idx = np.random.choice(len(xs), size=num, replace=False)

    sampled_points = np.stack([xs[idx], ys[idx]], axis=1)
    return sampled_points


# ------------------------------------------------------
def random_translation(points, max_translation=5):
    delta_x = random.uniform(-max_translation, max_translation)
    delta_y = random.uniform(-max_translation, max_translation)
    translated_points = points + np.array([delta_x, delta_y])
    return translated_points

def random_rotation(points, max_angle=30):
    angle = random.uniform(-max_angle, max_angle)
    angle_rad = np.deg2rad(angle)
    rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)], 
                                [np.sin(angle_rad), np.cos(angle_rad)]])
    
    rotated_points = np.dot(points, rotation_matrix)
    return rotated_points



def random_rotation_strong(points, max_angle=60, min_angle=10, return_matrix=False):
    assert 0 <= min_angle <= max_angle
    
    r_random = random.random()
    if r_random < 0.45:
        angle = random.uniform(-max_angle, -min_angle)
    elif r_random < 0.9:
        angle = random.uniform(min_angle, max_angle)
    else:
        angle = random.uniform(165, 195) 
    # print('angle', angle)
    
    angle_rad = np.deg2rad(angle)
    rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)], 
                                [np.sin(angle_rad), np.cos(angle_rad)]])
    
    rotated_points = np.dot(points, rotation_matrix)
    if return_matrix:
        return rotated_points, rotation_matrix
    return rotated_points

def apply_ratation(points, rotation_matrix):
    rotated_points = np.dot(points, rotation_matrix)
    return rotated_points


def random_scaling(points, min_scale=0.8, max_scale=1.2):
    scale = random.uniform(min_scale, max_scale)
    scaled_points = points * scale
    return scaled_points

def apply_random_rigid_transform(points):
    points = random_translation(points)
    points = random_rotation(points)
    # points = random_scaling(points)
    return points

# ------------------------------------------------------
def fix_translation(points, delta):
    delta_x, delta_y = delta
    translated_points = points + np.array([delta_x, delta_y])
    return translated_points

def fix_rotation(points, angle=5):
    angle_rad = np.deg2rad(angle)
    rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)], 
                                [np.sin(angle_rad), np.cos(angle_rad)]])
    rotated_points = np.dot(points, rotation_matrix)
    return rotated_points

def fix_scaling(points, scale=0.99):
    assert scale < 1.01 and scale > 0.99
    scaled_points = points * scale
    return scaled_points

def apply_fix_rigid_transform(points):
    points = fix_translation(points, delta=(10, 10))
    print('translation', points)
    points = fix_rotation(points, angle=5)
    print('rota', points)
    points = fix_scaling(points, scale=1)
    return points
# ------------------------------------------------------



def get_ori(points, delta=5):
    """
    min_x = min(p[0] for p in points)
    min_y = min(p[1] for p in points)
    origin = (min_x, min_y)
    """
    origin = points.min(axis=0)      # shape: (2,), i.e., [min_x, min_y]
    new_points = points - origin + delta
    return origin, new_points

def get_ori_tensor(points, delta=5):
    # torch: points = torch.tensor([[x1, y1], [x2, y2], ..., [xn, yn]])  # shape: (n, 2)
    # origin = points.min(dim=0).values    # shape: (2,)
    # new_points = points - origin 

    origin = points.min(dim=0)      # shape: (2,), i.e., [min_x, min_y]
    new_points = points - origin.values + delta
    return origin, new_points

def cal_sim(coding1, coding2):
    coding1 = coding1.squeeze(1)
    coding2 = coding2.squeeze(1)

    # L2 normalize
    coding1 = F.normalize(coding1, dim=-1)
    coding2 = F.normalize(coding2, dim=-1)

    similarity = torch.sum(coding1 * coding2, dim=-1, keepdim=True)  # shape: [N, 1]
    result = torch.cat([similarity, 1 - similarity], dim=1)  # shape: [N, 2]

def cal_dist(coding1, coding2):
    coding1 = coding1.squeeze(1)  # [N, 256]
    coding2 = coding2.squeeze(1)  # [N, 256]
    
    l2_dist = torch.norm(coding1 - coding2, dim=1)  # shape: [N, 1]

    return l2_dist
