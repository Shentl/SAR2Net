import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
import random

from utils_np import *
from LGN.generate_near_site_both_rot import get_near_points_both_rot, rotate_around_center, get_near_points_both_rot_multi_opp

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image
from tqdm import tqdm
import  os


def is_in_range(r, range):
    if (r > range[0]) and (r < range[1]):
        return True
    else:
        return False

# 1 pos, k neg
class Infinite_Num_Kneg_new1(IterableDataset):
    def __init__(self, 
        mode='triplet', 
        img_size=(500, 500), 
        batch_size=100, 
        r_rot=0.7, 
        r_tra=0.1, 
        r_site=0.1, 
        r_close=0.1, 
        ref_num=[4, 10],
        k_neg=128,
        neg_delta=20,
        max_rotation=30,
        min_rotation=10,
        sample_ref_site=False,
        pertu = 0,
        around_p_ratio = 0
        ):
        """
        r_rot/tra/site/close: sample cal_points
        ref_num: ref_points
        k_neg: 1 anchor-pos, k neg
        neg_delta: when sampling cal_neg, the min xy offset with cal_point
        """

        self.batch_size = batch_size
        self.img_size = img_size
        self.w = img_size[0]
        self.h = img_size[1]
        self.delta = np.array([5, 5])
        self.ref_min = ref_num[0]
        self.ref_max = ref_num[1]
        self.k_neg = k_neg
        self.neg_delta = neg_delta
        self.max_rotation = max_rotation
        self.min_rotation = min_rotation
        self.sample_ref_site = sample_ref_site
        self.pertu = pertu 
        self.around_p_ratio = around_p_ratio
        
        # Set ratio
        ratio_sum = 0
        ratios = [r_rot, r_tra, r_site, r_close]
        self.ratios = [[0, 0], [0, 0], [0, 0], [0, 0]]
        self.ratios_name = ['rot', 'tra', 'site', 'close']
        self.r_rot, self.r_tra, self.r_site, self.r_close = [0, 0], [0, 0], [0, 0], [0, 0]
        for i in range(len(ratios)):
            self.ratios[i][0] = ratio_sum
            ratio_sum += ratios[i]
            self.ratios[i][1] = ratio_sum
        # print(ratio_sum)
        assert (ratio_sum < 1.000001 and ratio_sum > 0.999999) 
    
    
    def add_random_perturbation_with_mask(self, points_array, no_pertu_idx=None):
        max_perturbation = self.pertu
       
        perturbed_points = points_array.copy()
        perturbed_points = perturbed_points.astype(np.float64)
        
        # num points
        n = points_array.shape[0]
        
        perturb_mask = np.ones(n, dtype=bool)
        
        if no_pertu_idx is not None:
            no_pertu_idx = np.array(no_pertu_idx)
            if no_pertu_idx.size > 0:
                perturb_mask[no_pertu_idx] = False

        num_to_perturb = np.sum(perturb_mask)
        
        if num_to_perturb == 0:
            return perturbed_points

        perturbation = np.random.uniform(
            low=-max_perturbation, 
            high=max_perturbation, 
            size=(num_to_perturb, 2)
        )
        
        perturbed_points[perturb_mask] += perturbation
        
        return perturbed_points
    
    def get_points_rot_bound_patch(self, num_ref_points):
        random_num = random.random()
        if random_num < 0.1:
            bound_size = 150
        elif random_num < 0.4:
            bound_size = 200
        elif random_num < 0.7:
            bound_size = 300
        else:
            bound_size = self.w

        # bound_size = self.w
        x_start = (self.w - bound_size) // 2
        x_end = x_start + bound_size
        y_start = (self.h - bound_size) // 2
        y_end = y_start + bound_size
        x_range = (x_start, x_end)
        y_range = (y_start, y_end)

        ref_points = get_random_point_range(x_range, y_range, num_ref_points) # [n,2]
        cal_point = get_random_point(self.w, self.h, 1) # [1,2]

        
        pos = np.concatenate([ref_points, cal_point], axis=0) # [n+1, 2]

        neg_cal_points = sample_points_mask(self.w, self.h, p=cal_point.squeeze(), r=self.neg_delta, num=self.k_neg, around_p_ratio=self.around_p_ratio) # (w, h, p, r, num)
        neg_ref_expanded = np.repeat(ref_points[np.newaxis, :, :], self.k_neg, axis=0)  # [k, n, 2]
        neg_cal_expanded = neg_cal_points[:, np.newaxis, :]  # [k, 1, 2]
        neg = np.concatenate([neg_ref_expanded, neg_cal_expanded], axis=1)


        anchor = random_rotation_strong(pos, max_angle=self.max_rotation, min_angle=self.min_rotation) # [n+1, 2]
        
        anchor_ref = anchor[:-1]
        if self.pertu > 0:
            anchor_ref_pertu = self.add_random_perturbation_with_mask(anchor_ref, no_pertu_idx=None)
        else:
            anchor_ref_pertu = anchor_ref
        anchor = np.vstack([anchor_ref_pertu, anchor[-1]]) 
  
        pos_neg_ref_origin, _ = get_ori(ref_points)
        anchor_ref_origin, _ = get_ori(anchor_ref_pertu) # return origin, new_points (origin w.o delta, new_points w. delta)
        
        # print(pos.shape, pos_neg_ref_origin.shape)
        
    
        delta = random.randint(0, 20)
        pos = pos - pos_neg_ref_origin + delta
        neg = neg - pos_neg_ref_origin + delta
        anchor = anchor - anchor_ref_origin + delta

        return anchor, pos, neg, 'rot' 

    def get_points_site(self, num_ref_points):
    
        min_dist, max_dist = 10, 24 # sample range
        rot_ang = 15 # rot in convex hull
        return_points, ref_p = None, None

        # ref_points = get_random_point(self.w, self.h, num_ref_points) # self.num_points-1 为cal_point空出一个位置

        random_num = random.random()
        if random_num < 0.1:
            bound_size = 150
        elif random_num < 0.4:
            bound_size = 200
        elif random_num < 0.7:
            bound_size = 300
        else:
            bound_size = self.w
        # bound_size = self.w
        x_start = (self.w - bound_size) // 2
        x_end = x_start + bound_size
        y_start = (self.h - bound_size) // 2
        y_end = y_start + bound_size
        x_range = (x_start, x_end)
        y_range = (y_start, y_end)

        ref_points = get_random_point_range(x_range, y_range, num_ref_points) # [n,2]
        # cal_points = get_random_point(self.w, self.h, 1) # [1,2]


        for ref_idx in range(ref_points.shape[0]):
            # return_points = get_near_points(ref_points, ref_idx, min_dist=min_dist, max_dist=max_dist, rot_ang=rot_ang, ref_must_in=ref_must_in) 
            ref_p = ref_points[ref_idx]
            try:
                return_points = get_near_points_both_rot_multi_opp(ref_points, ref_p, min_dist=min_dist, max_dist=max_dist, rot_ang=rot_ang, ref_must_in=False) 
            except:
                return_points = None
            
            if return_points:
                cal_point, opp_points, rotated_cw, rotated_ccw, inside_cal, inside_opp, ref_dist = return_points
                break
        
        if not return_points: # back to rot_bound
            cal_point = get_random_point(self.w, self.h, 1) # [1,2]
            num_opp = 0
        else:
            num_opp = opp_points.shape[0]
            max_samples = int(self.k_neg * 0.1) # 20 -> 50
    
            if num_opp > max_samples:
                sample_indices = np.random.choice(num_opp, size=max_samples, replace=False)
                opp_points = opp_points[sample_indices]
                num_opp = max_samples
        
            
            if self.sample_ref_site:
                try:
                    neg_cal_sample_sie = sample_site_points_mask(self.w, self.h, p=cal_point, ref_ps=ref_points, r=ref_dist, delta=30, num=max_samples)
                    opp_points = np.vstack((opp_points, neg_cal_sample_sie))
                    num_opp = 2 * max_samples 
                except:
                    print('ref_dist', ref_dist, 'ref_points', ref_points)
        
        pos = np.vstack([ref_points, cal_point]) # [n+1, 2]

        # pos = np.vstack([ref_points, cal_point])
        anchor, rotation_matrix = random_rotation_strong(pos, max_angle=self.max_rotation, min_angle=self.min_rotation, return_matrix=True)
        anchor_ref = anchor[:-1]

        if self.pertu > 0:
            anchor_ref_pertu = self.add_random_perturbation_with_mask(anchor_ref, no_pertu_idx=ref_idx)
        else:
            anchor_ref_pertu = anchor_ref
        anchor = np.vstack([anchor_ref_pertu, anchor[-1]])


        
        neg_ref_points = pos[:-1]
        if num_opp > 0:
            # opp_points_rot = apply_ratation(opp_points, rotation_matrix)
            # opp_points_rot now in neg, without rot
            
            neg_ref_expanded = np.repeat(neg_ref_points[np.newaxis, :, :], num_opp, axis=0)  # [num_opp, n, 2]
            neg_opp_expanded = opp_points[:, np.newaxis, :]  # [num_opp, 1, 2]
            neg_site = np.concatenate([neg_ref_expanded, neg_opp_expanded], axis=1)

        # -------------------------------------------
        sample_num = self.k_neg - num_opp
        neg_cal_points = sample_points_mask(self.w, self.h, p=pos[-1], r=self.neg_delta, num=sample_num, around_p_ratio=self.around_p_ratio) # (w, h, p, r, num)
        neg_ref_expanded = np.repeat(neg_ref_points[np.newaxis, :, :], sample_num, axis=0)  # [k, n, 2]
        neg_cal_expanded = neg_cal_points[:, np.newaxis, :]  # [k, 1, 2]
        neg_rot = np.concatenate([neg_ref_expanded, neg_cal_expanded], axis=1)  # [k, n+1, 2]

        if num_opp > 0:
            neg = np.concatenate([neg_site, neg_rot], axis=0)
        else:
            neg = neg_rot
        
        pos_neg_ref_origin, _ = get_ori(ref_points) 
        anchor_ref_origin, _ = get_ori(anchor_ref_pertu) # return origin, new_points (origin w.o. delta, new_points w. delta)

        # in get_new_slide_ori_refp, delta=20
        delta = random.randint(0, 20)
        pos = pos - pos_neg_ref_origin + delta
        neg = neg - pos_neg_ref_origin + delta
        anchor = anchor - anchor_ref_origin + delta

        return anchor, pos, neg, 'site'

    

    def generate_batch(self):
        num_ref_points = np.random.randint(self.ref_min, self.ref_max+1) 
        anchor_, pos_, neg_, mode_ = [], [], [], []
        
        for i in range(self.batch_size):
            r = np.random.random()
            if is_in_range(r, self.ratios[0]):
                anchor, pos, neg, mode = self.get_points_rot_bound_patch(num_ref_points)
            elif is_in_range(r, self.ratios[1]):
                anchor, pos, neg, mode = self.get_points_rot_bound_patch(num_ref_points) # self.get_points_tra()
            elif is_in_range(r, self.ratios[2]):
                # print(r, self.ratios[2])
                # print('------------Site---------------------')
                anchor, pos, neg, mode = self.get_points_site(num_ref_points)
                # if anchor is None: 
                #     anchor, pos, neg, mode = self.get_points_rot(num_ref_points)
                # print('back to rot')
            elif is_in_range(r, self.ratios[3]):
                anchor, pos, neg, mode = self.get_points_rot_bound_patch(num_ref_points) # self.get_points_close()
            else:
                raise NotImplementedError("Ratio not implemented")

            anchor = torch.from_numpy(anchor.astype(np.float32))  # [bs, num_points, 2]
            pos = torch.from_numpy(pos.astype(np.float32)) # [bs, num_points, 2]
            neg = torch.from_numpy(neg.astype(np.float32)) # [bs, num_neg, num_points, 2]
        
            anchor_.append(anchor)
            pos_.append(pos)
            neg_.append(neg)
            mode_.append(mode)
        
        return {
                'anchor': torch.stack(anchor_),
                'pos': torch.stack(pos_),
                'neg': torch.stack(neg_),
                'mode': mode_
            }
        # stack expects each tensor to be equal size, but got [9, 2] at entry 0 and [128, 9, 2] at entry 1
        # yield anchor, pos, neg, new_img_size, mode

    def __iter__(self):
        while True:
            yield self.generate_batch()
        

def custom_collate_fn(batch):
    return batch[0]


def get_image_size(anchor, pos, ori_size, delta=5):
    all_points = torch.concatenate([anchor, pos], axis=0) # (15360, 8, 2) [3B, N, 2]
    flatten_coords = all_points.view(-1, 2) # [3BN, 2]

    max_x = flatten_coords[:, 0].max()  
    max_y = flatten_coords[:, 1].max() 

    new_img_size = np.maximum(np.array(ori_size), np.array([int(max_x), int(max_y)]))
    new_img_size = np.ceil(new_img_size + delta).astype(int)
    return new_img_size



def draw_points_on_image1(points, img_size, use_border=False):
    img = np.zeros((img_size[0], img_size[1], 3)).astype(np.uint8)
    for i in range(len(points)-1):
        x, y = int(points[i][0]), int(points[i][1])
        img[x-3:x+3, y-3:y+3, :] = [255, 0, 0]
    x, y = int(points[-1][0]), int(points[-1][1])
    img[x-3:x+3, y-3:y+3, :] = [0, 255, 0]

    if use_border:
        img = add_border_numpy(img, border_width=1, color=[255, 255, 255])
    return img

def draw_neg_points(points, img):
    # blue: neg_points
    for i in range(len(points)):
        x, y = int(points[i][0]), int(points[i][1])
        img[x-3:x+3, y-3:y+3, :] = [0, 0, 255]
    return img


if __name__ == '__main__':
    save_base = 'user_dir'
    import time
    pertu = 3
    dataset = Infinite_Num_Kneg_new1(batch_size=10, ref_num=[4, 10], img_size=(400, 400), k_neg=500,
                                neg_delta=20,
                                r_rot=0.5, 
                                r_tra=0, 
                                r_site=0.5, 
                                r_close=0,
                                sample_ref_site=True,
                                pertu = pertu)
    dataloader = DataLoader(
        dataset, 
        batch_size=1,        
        num_workers=1,       
        pin_memory=True,      
        collate_fn = custom_collate_fn
    )
    start = time.time()
    epoch = 100
    # 迭代获取批次数据
    img_size = (900, 900)
    r = dataset.neg_delta
    neg_N = dataset.k_neg
    save_dir_ = os.path.join(save_base, 'points_center_newratio', 'sample_ref_site', 'site_r_%d_n%d_pertu_%s' % (r, neg_N, pertu))
    save_dir = save_dir_
    os.makedirs(save_dir, exist_ok=True)
    for i, batch in enumerate(dataloader):
        anchor, pos, neg, mode = batch['anchor'], batch['pos'], batch['neg'], batch['mode']
        """
        neg_ref_expanded = np.repeat(neg_ref_points[np.newaxis, :, :], self.k_neg, axis=0)  # [k, n, 2]
        neg_cal_expanded = neg_cal_points[:, np.newaxis, :]  # [k, 1, 2]
        neg = np.concatenate([neg_ref_expanded, neg_cal_expanded], axis=1)  # [k, n+1, 2]
        """
        delta = torch.tensor([150, 150])
        for b in range(anchor.shape[0]):
            m = mode[b]
            anchor_ = anchor[b] + delta # [n,2]
            pos_ = pos[b] + delta # [n,2]
            neg_ = neg[b][:, -1, :].squeeze() + delta # [500, n, 2] -> [500,2]
            # print(anchor_.shape,neg_.shape)

            # print('pos_', pos_[-1, :].unsqueeze(0), 'neg_', neg_.shape)
            # print('pos_', pos_.shape, 'neg_', neg_.shape) # neg_ torch.Size([128, 11, 2])
            img1 = draw_points_on_image1(anchor_, img_size, use_border=True) # points_ (5, 2)   anchor
            img2 = draw_points_on_image1(pos_, img_size, use_border=True) # trans_points_ (5, 2)   pos
            draw_neg_points(neg_, img2) # neg_points (1000, 2)   neg
            concat_img = np.concatenate((img1, img2), axis=1) 
            Image.fromarray(concat_img).save(os.path.join(save_dir, '%d_%d_%s.png' % (b,i,m)))
        
