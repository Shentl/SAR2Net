import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import argparse
from PIL import Image
from tqdm import tqdm

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append('..')
from train_utils import *
from loss_utils import cal_loss
from models_weight import *
from datasets import Infinite_Num_Kneg_new1, custom_collate_fn, get_image_size

parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument('--loss_mode', type=str, default='npair')
parser.add_argument('--batch_size', type=int, default=256)

parser.add_argument('-j', '--workers', type=int, default=8)
parser.add_argument('--train_num', type=int, default=256000, help='number of total nums for training (step/batch_size)')
parser.add_argument('--print_every', type=int, default=100)
parser.add_argument('--save_every', type=int, default=1000)


parser.add_argument('--lr', type=float, default=0.0003)
parser.add_argument('--device', type=str, default='cuda:1')
parser.add_argument('--seed', type=int, default=42)

parser.add_argument('--save_dir', type=str, default="./checkpoints")
parser.add_argument('--wd', '--weight-decay', type=float, default=1e-4)
parser.add_argument('--out_dim', type=int, default=256)

parser.add_argument('--model_temp', type=float, default=1, help='Encoder softmax temperature (default: 1)')
parser.add_argument('--loss_temp', type=float, default=0.07, help='temperature for InfoNCE Loss (default: 0.07)')
parser.add_argument('--margin', type=float, default=0.2, help='margin for Triplet Loss (cos_sim, default: 0.2)')

parser.add_argument('--no_weight', action='store_true', help='do not use the weight for different ref_points')
parser.add_argument('--r_rot', type=float, default=1)
parser.add_argument('--r_tra', type=float, default=0)
parser.add_argument('--r_site', type=float, default=0)
parser.add_argument('--r_close', type=float, default=0)
parser.add_argument('--k_neg', type=int, default=128)

# Rotation to generate the neg_pairs
parser.add_argument('--max_rot', type=int, default=30)
parser.add_argument('--min_rot', type=int, default=0)

parser.add_argument('--img_size', type=int, default=500)
parser.add_argument('--neg_delta', type=int, default=10, help='the minimun delta for the positive and negative ref_points')

parser.add_argument('--pertu', type=int, default=0)
parser.add_argument('--attn_fc', action='store_true')
parser.add_argument('--split_fc', action='store_true')
parser.add_argument('--sample_ref_site', action='store_true')

parser.add_argument('--max_ref_num', type=int, default=10)
parser.add_argument('--around_p_ratio', type=float, default=0)
args = parser.parse_args()

if args.loss_mode == 'infonce':
    model_name = 'kneg_%d_delta_%d_num_%d_bs_%d_lr_%.0e_out_%d_ltemp_%s_img_%d_rot_%d_%d' % (args.k_neg, args.neg_delta, args.train_num, args.batch_size, args.lr, args.out_dim, args.loss_temp, args.img_size, args.min_rot, args.max_rot)
elif args.loss_mode == 'triplet_sim':
    model_name = 'kneg_%d_delta_%d_num_%d_bs_%d_lr_%.0e_out_%d_margin_%s_img_%d_rot_%d_%d' % (args.k_neg, args.neg_delta, args.train_num, args.batch_size, args.lr, args.out_dim, args.margin, args.img_size, args.min_rot, args.max_rot)
else:
    model_name = 'kneg_%d_delta_%d_num_%d_bs_%d_lr_%.0e_out_%d_img_%d_rot_%d_%d' % (args.k_neg, args.neg_delta, args.train_num, args.batch_size, args.lr, args.out_dim, args.img_size, args.min_rot, args.max_rot)

model_name = model_name + '_%d' % args.max_ref_num
if args.around_p_ratio > 0:
    model_name = model_name + '_%s' % args.around_p_ratio

if args.attn_fc:
    model_name = 'afc_' + model_name
if args.pertu > 0:
    model_name = 'pertu%d_' % (args.pertu) + model_name
if args.split_fc:
    model_name = 'split_' + model_name
if args.sample_ref_site:
    model_name = 'refsite_' + model_name
print('model_name', model_name)
save_dir = os.path.join(args.save_dir, args.loss_mode, model_name, 'rot_%s_site_%s' % (args.r_rot, args.r_site))

if torch.cuda.is_available():
    device = torch.device(args.device)
else:
    device = torch.device("cpu")

os.makedirs(save_dir, exist_ok=True)

def seed_torch(seed=7):
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


def worker_init_fn(worker_id):
    worker_seed = args.seed + worker_id # let each worker has a unique seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


seed_torch(args.seed)



# =======================
# train
# =======================

def train_model_dynamic(args=None):
    assert args.loss_mode in ['npair', 'triplet_sim', 'infonce', 'supcon', 'contrastive']
    img_size = (args.img_size, args.img_size)
    dataset = Infinite_Num_Kneg_new1(
            batch_size=args.batch_size, 
            ref_num=[4, args.max_ref_num], 
            k_neg=args.k_neg,
            neg_delta=args.neg_delta,
            img_size=img_size, 
            r_rot=args.r_rot, 
            r_tra=args.r_tra, 
            r_site=args.r_site, 
            r_close=args.r_close,
            max_rotation=args.max_rot,
            min_rotation=args.min_rot,
            sample_ref_site = args.sample_ref_site,
            pertu = args.pertu,
            around_p_ratio = args.around_p_ratio
        )
    """
        r_rot/tra/site/close: sample cal_points
        ref_num: num of ref_points
        k_neg: num of negs for 1 anchor-pos 
        neg_delta: min xy offset with cal_point (cal_neg)
    """ 

    dataloader = DataLoader(
            dataset, 
            batch_size=1, 
            pin_memory=True, 
            num_workers=args.workers, 
            worker_init_fn=worker_init_fn,
            collate_fn=custom_collate_fn
        )

    using_weight = not args.no_weight
    # model = GeometryFeatureNet_weight_big(input_dim=4, hidden_dim=128, output_dim=args.out_dim, using_weight=using_weight, temp=args.model_temp, attn_fc=args.attn_fc)
    if args.split_fc:
        model = GeometryFeatureNet_weight_big_split(input_dim=4, hidden_dim=128, output_dim=args.out_dim, using_weight=using_weight, temp=args.model_temp, attn_fc=args.attn_fc)
    else:
        model = GeometryFeatureNet_weight_big(input_dim=4, hidden_dim=128, output_dim=args.out_dim, using_weight=using_weight, temp=args.model_temp, attn_fc=args.attn_fc)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) # lr=1e-3

    step = 0
    total_loss, tmp_loss = 0.0, 0.0
    
    max_steps = args.train_num // args.batch_size

    for data in dataloader:
        if step >= max_steps:
            break

        optimizer.zero_grad()

        # -------------------------------------------------------------------------------
        anchor, pos, neg, mode = data['anchor'], data['pos'], data['neg'], data['mode']  # [bs, num_points, 2] [bs, num_neg, num_points, 2]
        # new_img_size = get_image_size(anchor, pos, ori_size=img_size, delta=5)

        num_p = anchor.shape[1]
        neg = neg.view(args.batch_size * args.k_neg, num_p, 2)  # Flatten, [bs, k_neg, num_p, 2] -> [bs*k_neg, num_p, 2] 

        anchor = anchor.float().to(device) # [B, N+1, 2]
        pos = pos.float().to(device)
        neg = neg.float().to(device)
        #print(torch.min(neg))

        f_a = model(anchor[:, :-1, :], anchor[:, -1, :].unsqueeze(1)) # ref_points, cal_points
        f_p = model(pos[:, :-1, :], pos[:, -1, :].unsqueeze(1))
        f_n = model(neg[:, :-1, :], neg[:, -1, :].unsqueeze(1))
        f_n = f_n.view(args.batch_size, args.k_neg, args.out_dim) # f_a.shape[-1]
        # k_neg=100: torch.Size([bs, 128]) torch.Size([bs, 128]) torch.Size([bs, 100, 128]
        # default output with L2-norm

        # loss = triplet_loss(f_a, f_p, f_n, margin=args.margin)
        loss = cal_loss(f_a, f_p, f_n, args=args, device=device)
       
        # ------------------------------------------------------------------------------- 

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        tmp_loss += loss.item()

        file = open(os.path.join(save_dir, 'result.txt'),'a')
        file.write('\n Step: %d, Loss: %s, Num_points: %d ' % (step, loss.item(), num_p))
        # file.write('-----------------------------------------------------------------\n')
        file.close()

        if step % args.print_every == 0:
            avg_loss = total_loss / (step + 1)
            avg_loss_print_every = tmp_loss / args.print_every
            print(f"Step {step}, Loss: {avg_loss:.4f}, Current Loss: {avg_loss_print_every:.4f}, Num Points: {num_p}")
            # torch.save(model.state_dict(), os.path.join(save_dir, f"{args.loss_mode}_step_{step}_b_{args.batch_size}.pth"))
            tmp_loss = 0 
        
        if step % args.save_every == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f"{args.loss_mode}_step_{step}_b_{args.batch_size}.pth"))
        
        step += 1

if __name__ == '__main__':
    command = " ".join(sys.argv)
    file = open(os.path.join(save_dir, 'result.txt'),'a')
    file.write(" python %s\n"% command)
    file.close()
    seed_torch(args.seed)
    train_model_dynamic(args)

