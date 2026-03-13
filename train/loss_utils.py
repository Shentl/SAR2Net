import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------------
# --------------- 1_anchor - 1_pos - k_negs ---------------
def contrastive_loss(f1, f2, label, margin=1.0):
    d = F.pairwise_distance(f1, f2)
    return (label * d**2 + (1 - label) * F.relu(margin - d)**2).mean()


def triplet_loss_distance(anchor, positive, negatives, margin=1.0, reduction='mean'):
    """
    anchor:     [B, D]
    positive:   [B, D]
    negatives:  [B, N, D]
    margin:     float, margin value
    reduction:  'mean' or 'none'
    
    Returns:
        scalar loss or [B] loss if reduction='none'
    """

    #  anchor-to-positive [B, 1]
    d_pos = F.pairwise_distance(anchor, positive).unsqueeze(1)  # [B, 1]

    # anchor-to-negative [B, N]
    # (a - n)^2.sum(dim=-1)
    d_neg = torch.norm(anchor.unsqueeze(1) - negatives, p=2, dim=2)  # [B, N]

    # triplet loss: max(0, d_pos - d_neg + margin)
    losses = F.relu(d_pos - d_neg + margin)  # [B, N]

    loss_per_sample = losses.mean(dim=1)  # [B]
    if reduction == 'mean':
        return loss_per_sample.mean()
    else:
        return loss_per_sample  # [B]


def triplet_loss_sim(anchor, positive, negatives, margin=0.2, reduction='mean'):
    """
    anchor:     [B, D]
    positive:   [B, D]
    negatives:  [B, N, D]
    margin:     float, margin value
    reduction:  'mean' or 'none'
    
    Returns:
        scalar loss (mean) or [B] loss if reduction='none'
    """
    
    #  dot product，if normalized, use cosine sim
    sim_pos = (anchor * positive).sum(dim=-1, keepdim=True)       # [B, 1]
    sim_negs = torch.bmm(negatives, anchor.unsqueeze(2)).squeeze(2)  # [B, N]

    # Triplet Loss: max(0, sim_neg - sim_pos + margin)
    triplet_losses = F.relu(sim_negs - sim_pos + margin)          # [B, N]
    # sim_pos -sim_neg > margin 时 loss = 0

    # mean over negatives / hard negative mining
    loss_per_sample = triplet_losses.mean(dim=1)  # average over N negatives
    # hardest negative：
    # loss_per_sample = triplet_losses.max(dim=1).values

    if reduction == 'mean':
        return loss_per_sample.mean()
    else:
        return loss_per_sample  # [B]



# InfoNCE with temperature=1
def npair_loss(anchor, positive, negatives):
    # already normalized
    """
    anchor:     [B, D]
    positive:   [B, D]
    negatives:  [B, N, D]
    return:     scalar loss
    """
    # Compute sim(a, p): [B]
    sim_pos = (anchor * positive).sum(dim=-1, keepdim=True)  # [B, 1]

    # Compute sim(a, n): [B, N]     [B, N, D] * [B, D, 1] -> [B, N, 1] -> [B, N]
    sim_negs = torch.bmm(negatives, anchor.unsqueeze(2)).squeeze(2)  # [B, N]
    
    # Loss: log(1 + sum(exp(sim_neg - sim_pos)))
    loss = torch.log1p(torch.exp(sim_negs - sim_pos).sum(dim=1)).mean()
    return loss


def info_nce_loss(anchor, positive, negatives, temperature=0.07, device=None):
    # Inout no need fornormalize
    """
    anchor:    [B, D]
    positive:  [B, D]
    negatives: [B, N, D]  # 每个 anchor 有 N 个负样本
    """
    B, D = anchor.shape
    N = negatives.shape[1]

    # Cosine similarity
    sim_pos = F.cosine_similarity(anchor, positive, dim=-1)  # [B]
    sim_negs = F.cosine_similarity(anchor.unsqueeze(1), negatives, dim=-1)  # [B, N]
    # [B, 1, D] - [B, N, D]
    """
    anchor_expand = anchor_f.unsqueeze(1).expand(-1, N, -1)     # [B, N, D]
    sim_neg = F.cosine_similarity(anchor_expand, neg_f, dim=-1) # [B, N]
    """

    # logits: [B, 1+N]
    logits = torch.cat([sim_pos.unsqueeze(1), sim_negs], dim=1)  # [B, 1+N]
    logits = logits / temperature

    labels = torch.zeros(B, dtype=torch.long, device=device)

    # CrossEntropy over manually constructed logits
    loss = F.cross_entropy(logits, labels)

    return loss



def simcle_info_nce_loss(z, temperature=0.07, device=None):
    z = F.normalize(z, dim=1)
    sim = z @ z.T  # [2N, 2N]
    sim /= temperature

    # mask out self-comparisons
    mask = torch.eye(sim.size(0), dtype=torch.bool, device=device)
    sim.masked_fill_(mask, -1e9)

    # construct label: for i in [0, N), positive is i+N; for i in [N, 2N), positive is i-N
    N = z.size(0) // 2
    labels = torch.arange(N, device=z.device)
    labels = torch.cat([labels + N, labels], dim=0)

    loss = F.cross_entropy(sim, labels)
    return loss


def cal_loss(f_a, f_p, f_n, device=None, args=None):
    if args.loss_mode == 'infonce':
        loss = info_nce_loss(f_a, f_p, f_n, temperature=args.loss_temp, device=device)
    elif args.loss_mode == 'npair':
        loss = npair_loss(f_a, f_p, f_n)
    elif args.loss_mode == 'triplet_sim':  # default margin=0.2
        loss = triplet_loss_sim(f_a, f_p, f_n, margin=args.margin, reduction='mean')
    elif args.loss_mode == 'contrastive':
        loss_pos = contrastive_loss(f_a, f_p, label=1, margin=args.margin)
        loss_neg = contrastive_loss(f_a, f_n, label=0, margin=args.margin)
        loss = loss_pos + loss_neg
    else: 
        raise NotImplementedError
    
    return loss


if __name__ == '__main__':
    device = 'cpu'
    B, N, D = 32, 10, 128 
    anchor = torch.randn(B, D)
    pos = torch.randn(B, D)
    neg = torch.randn(B, N, D)

    anchor = F.normalize(anchor, dim=1)
    pos = F.normalize(pos, dim=1)
    neg = F.normalize(neg, dim=2)
    # x = F.normalize(x, dim=1)

    infonce = info_nce_loss(anchor, pos, neg, temperature=1, device=device)
    npair = npair_loss(anchor, pos, neg)
    triplet_002 = triplet_loss_sim(anchor, pos, neg, margin=0.2, reduction='mean')
    triplet_005 = triplet_loss_sim(anchor, pos, neg, margin=0.5, reduction='mean')
    triplet_100 = triplet_loss_sim(anchor, pos, neg, margin=1, reduction='mean')
    triplet_200 = triplet_loss_sim(anchor, pos, neg, margin=1, reduction='mean')
    print('infonce: %s, npair: %s' % (infonce, npair))
    print(triplet_002, triplet_005, triplet_100, triplet_200)
