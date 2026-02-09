import numpy as np
import random
import torch

from datetime import datetime
import copy
import csv
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import math

from util_quant_ import *

now = datetime.now()
torch.cuda.empty_cache()

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from torchvision.datasets import CocoDetection

import albumentations as AA
from albumentations.pytorch import ToTensorV2

class CocoDetectionDataset(CocoDetection):
    def __init__(self, root, annFile, alb_transforms=None):
        # base transforms=None to avoid base applying Albumentations
        super().__init__(root, annFile, transforms=None, transform=None, target_transform=None)
        self.alb_transforms = alb_transforms
        self.valid_indices = [i for i, img_id in enumerate(self.ids)
                              if len(self.coco.imgToAnns[img_id]) > 0]

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        # super returns (PIL.Image, annotations_list)
        img, annots = super().__getitem__(actual_idx)

        bboxes = []
        labels = []
        for ann in annots:
            x, y, w, h = ann['bbox']
            
            x1, y1, x2, y2 = x, y, x + w, y + h
            if (x2 > x1) and (y2 > y1):
                bboxes.append([x1, y1, x2, y2])
                labels.append(ann['category_id'])

        if len(bboxes) == 0:
            return None
        
        sample = self.alb_transforms(image=np.array(img), bboxes=bboxes, labels=labels)
        image = sample['image']               # Tensor[3,H,W]
        boxes = torch.tensor(sample['bboxes'], dtype=torch.float32)
        labels = torch.tensor(sample['labels'], dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([self.ids[actual_idx]]),
        }
        return image, target

def collate_fn(batch):
    images, targets = [], []
    for img, tgt in batch:
        boxes = tgt['boxes']
        if boxes.ndim == 1:
            boxes = boxes.view(-1, 4)  # ⇒ (0,4)

        images.append(img)
        targets.append({**tgt, 'boxes': boxes})
    return images, targets


def gradient_GD_loader(X, model, y):
    gradient_sum = [torch.zeros_like(param).to(device) for param in model.parameters()]

    model.train()
    loss_dict = model(X, y)
    loss = sum(loss for loss in loss_dict.values())

    loss.backward()

    # Compute gradients and update gradient_sum
    count_temp = 0
    for param in model.parameters():
        if param.grad is not None:
            gradient_sum[count_temp] += clip_grad(param.grad.clone(), C)
        count_temp += 1

    return gradient_sum

def gradient_GC_loader(X, model, y, stragglers, data_idx, encoding_mat, decoding_vec, bit_alloc = None, is_1bitquant = False):
    gradient_sum = [torch.zeros_like(param).to(device) for param in model.parameters()]
    idx_nonstragg = np.where(stragglers == False)[0]
    for worker_id in idx_nonstragg:
        decoding_weight = decoding_vec[worker_id]
        encoding_weight = encoding_mat[worker_id,data_idx]
        model.zero_grad()
        
        model.train()
        loss_dict = model(X, y)
        loss = sum(loss for loss in loss_dict.values())

        loss.backward()

        params = list(model.parameters())
        
        indices = []
        sizes   = []
        total_elems = 0
        for idx, param in enumerate(params):
            if param.grad is not None:
                n = param.grad.numel()
                indices.append(idx)
                sizes.append(n)
                total_elems += n

        if total_elems == 0:
            pass
        else:
            bits = None
            try:
                if bit_alloc is not None:
                    bits = bit_alloc[worker_id]
            except NameError:
                bits = None

            if (bits is None) or (bits <= 0):
                for idx in indices:
                    g = params[idx].grad.detach() * encoding_weight * decoding_weight
                    gradient_sum[idx] += g
            else:
                flat_msg = params[indices[0]].grad.detach().new_empty(total_elems)
                offset = 0
                for idx, n in zip(indices, sizes):
                    g = params[idx].grad.detach() * encoding_weight     # encode
                    flat_msg[offset:offset+n].copy_(g.view(-1))
                    offset += n

                flat_msg_q  = quantize_tensor(flat_msg, bits, True, is_1bitquant)
                flat_dec = flat_msg_q * decoding_weight
                offset = 0
                for idx, n in zip(indices, sizes):
                    part = flat_dec[offset:offset+n].view_as(params[idx].grad)
                    gradient_sum[idx] += part
                    offset += n

    for i in range(len(gradient_sum)):
        gradient_sum[i] = clip_grad(gradient_sum[i], C)
        
    return gradient_sum

def adam_proposed(X, model, y, stragglers, data_idx, encoding_mat, decoding_vec, m, v, t, b1=0.9, b2=0.999, scale = 1e-0, eps = 1e-8):
    grad = [torch.zeros_like(param).to(device) for param in model.parameters()]
    idx_nonstragg = np.where(stragglers == False)[0]
    for worker_id in idx_nonstragg:
        decoding_weight = decoding_vec[worker_id]
        encoding_weight = encoding_mat[worker_id,data_idx]
        model.zero_grad()
        
        model.train()
        loss_dict = model(X, y)
        loss = sum(loss for loss in loss_dict.values())

        loss.backward()

        count_temp = 0
        for param in model.parameters():
            if param.grad is not None:
                gradtmp = clip_grad(param.grad.clone(), C)
                m[count_temp] = b1 * m[count_temp]  + (1-b1) * gradtmp 
                v[count_temp] = b2 * v[count_temp] + (1-b2) * ((gradtmp) ** 2 / scale)
                gradtmp2 = (m[count_temp] / (1-b1 ** (t+1))) / (torch.sqrt(v[count_temp] / (1-b2 ** (t+1))) + eps)
                grad[count_temp] += gradtmp2 * encoding_weight * decoding_weight
            count_temp += 1
            
    return grad, m, v

def gradient_GC_loader2(X, model, y, stragglers, data_idx, encoding_mat, decoding_vec, decoding_vec2, bit_alloc = None, is_1bitquant = False):
    gradient_sum = [torch.zeros_like(param).to(device) for param in model.parameters()]
    gradient_sum2 = [torch.zeros_like(param).to(device) for param in model.parameters()]
    idx_nonstragg = np.where(stragglers == False)[0]
    for worker_id in idx_nonstragg:
        decoding_weight = decoding_vec[worker_id]
        decoding_weight2 = decoding_vec2[worker_id]
        encoding_weight = encoding_mat[worker_id,data_idx]
        model.zero_grad()
        
        model.train()
        loss_dict = model(X, y)
        loss = sum(loss for loss in loss_dict.values())

        loss.backward()
        
        params = list(model.parameters())
        indices = []
        sizes   = []
        total_elems = 0
        for idx, param in enumerate(params):
            if param.grad is not None:
                n = param.grad.numel()
                indices.append(idx)
                sizes.append(n)
                total_elems += n

        if total_elems == 0:
            pass
        else:
            bits = None
            try:
                if bit_alloc is not None:
                    bits = bit_alloc[worker_id]
            except NameError:
                bits = None

            if (bits is None) or (bits <= 0):
                for idx in indices:
                    g = params[idx].grad.detach() * encoding_weight * decoding_weight
                    gradient_sum[idx] += g
                    
                    g2 = params[idx].grad.detach() * encoding_weight * decoding_weight2
                    gradient_sum2[idx] += g2
            else:
                flat_msg = params[indices[0]].grad.detach().new_empty(total_elems)
                offset = 0
                for idx, n in zip(indices, sizes):
                    g = params[idx].grad.detach() * encoding_weight     # encode
                    flat_msg[offset:offset+n].copy_(g.view(-1))
                    offset += n

                flat_msg_q  = quantize_tensor(flat_msg, bits, True, is_1bitquant)
                flat_dec = flat_msg_q * decoding_weight
                flat_dec2 = flat_msg_q * decoding_weight2
                offset = 0
                for idx, n in zip(indices, sizes):
                    part = flat_dec[offset:offset+n].view_as(params[idx].grad)
                    gradient_sum[idx] += part
                    
                    part2 = flat_dec2[offset:offset+n].view_as(params[idx].grad)
                    gradient_sum2[idx] += part2
                    offset += n

    for i in range(len(gradient_sum)):
        gradient_sum[i] = clip_grad(gradient_sum[i], C)
        gradient_sum2[i] = clip_grad(gradient_sum2[i], C)
        
    return gradient_sum, gradient_sum2



def genran_vec(n, d_avg):
    d_i = [random.randint(mn, mx) for _ in range(n)]
    while (sum(d_i) / n) >= d_avg + 0.05 or (sum(d_i) / n) <= d_avg - d_gap:
        d_i = [random.randint(mn, mx) for _ in range(n)]
    return d_i

def clip_grad(grad, C):
    return torch.clip(grad, -C, C)

def calculate_optimal_decoding(assignment_matrix, stragglers):
    n, m = assignment_matrix.shape
    A = assignment_matrix.copy()

    if isinstance(stragglers, np.ndarray) and stragglers.dtype == bool:
        if len(stragglers) != m:
            raise ValueError("Stragglers array length must match the number of machines.")
        A[:, stragglers] = 0
    else:
        for s in stragglers:
            A[:, s] = 0  

    try:
        pseudo_inverse = np.linalg.pinv(A)
    except np.linalg.LinAlgError as e:
        print("Error in pseudoinverse calculation:", e)
        return None
    
    ones_vector = np.ones(n)
    optimal_decoding = pseudo_inverse @ ones_vector

    return optimal_decoding

transform = AA.Compose(
    [
        AA.RandomSizedBBoxSafeCrop(width=640, height=640, erosion_rate=0.0, p=1.0), 
        AA.HorizontalFlip(p=0.5),
        AA.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        AA.Normalize(mean=(0.485, 0.456, 0.406),
                     std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ],
    bbox_params=AA.BboxParams(
        format="pascal_voc",
        label_fields=["labels"],
        min_visibility=0.05,          
    ),
)

is_COCO = True

is_comp_norm = False

Btot = 250

ema_nu = 0.2
update_period = 10 

n = 50
Z_th = 1.1 
mu_min = 0.01
mu_max = 2 
mu = np.random.uniform(mu_min, mu_max, n)
p = np.exp(- mu * (Z_th - 1)) 
p_avg = np.mean(p)
sigma = 0.2
alpha_param = 2 
period_validate = 10 
period_disp = 10
period_straggler = 1
img_id   = 724 
n_split = n 
d = 2 # 
d2 = 2 
d_gap = 0.1

adamparam = 0.9 
adamparam2 = 0.999 
scale_proposed = 1e0

if is_COCO:
    m = 100000 
    test_batch_size = 64 
    batch_size = 32
    T = 101
    gamma = 0.05 
    
    
    lambda_sq = lambda t: 1 
    gamma_t = lambda t: gamma 
    C = 10
    C2 = 1

def adam_apply(m, v, model, grad, norm, b1, b2, t, scale = 1e-0, eps = 1e-8):
    with torch.no_grad():
        count_tmp = 0
        for param in model.parameters():
            m[count_tmp] = b1 * m[count_tmp]  + (1-b1) * grad[count_tmp]  
            v[count_tmp] = b2 * v[count_tmp] + (1-b2) * ((grad[count_tmp] / 1) ** 2 / scale)
            param -= gamma_t(t) * (m[count_tmp] / (1-b1 ** (t+1))) / (torch.sqrt(v[count_tmp] / (1-b2 ** (t+1))) + eps) / norm
            count_tmp += 1
    
    return m, v
def adam_apply2(m, v, model, grad, grad2, norm, b1, b2, t, scale = 1e-0, eps = 1e-8):
    with torch.no_grad():
        count_tmp = 0
        for param in model.parameters():
            m[count_tmp] = b1 * m[count_tmp]  + (1-b1) * grad[count_tmp] 
            v[count_tmp] = b2 * v[count_tmp] + (1-b2) * ((grad2[count_tmp] / 1) ** 2 / scale)
            param -= gamma_t(t) * (m[count_tmp] / (1-b1 ** (t+1))) / (torch.sqrt(v[count_tmp] / (1-b2 ** (t+1))) + eps) / norm
            count_tmp += 1
    return m, v


partitions = np.array_split(range(m), n_split)

b = np.zeros(n, dtype=int)
b[-1] = 1
b[:-1] = 2
b[0] = b[0] + n + n_split - 2 - sum(b[:-1])

delta_inv = (1-p) / p
sum_delta_inv = sum(delta_inv)
Y1 = delta_inv * n / sum_delta_inv
alpha = np.zeros((n,n_split))


if is_COCO:
    base_norm = 1

avg_num = base_norm
avg_num2 = base_norm
avg_num3 = base_norm

if is_comp_norm:
    avg_num2 *= ((n + n_split - 1) / n) 
    avg_num3 *= (d) 


Y = delta_inv * n_split / sum_delta_inv
prev_idx = 0
for i in range(n): 
    if i == 0:
        alpha[i, prev_idx] = 1
    else:
        alpha[i, prev_idx] = 1 - alpha[i - 1, prev_idx]
    alpha[i, prev_idx + 1: prev_idx + b[i] - 1] = np.ones_like(alpha[i, prev_idx + 1: prev_idx + b[i] - 1])
    alpha[i, prev_idx + b[i] - 1] = Y[i] - sum(alpha[i, :prev_idx + b[i] - 1])

    prev_idx = prev_idx + b[i] - 1

alpha0 = np.zeros((n,n_split))
mn = 0
mx = np.ceil(2 * d)
prev_idx = 0
for i in range(n): 
    alpha0[i, 0] = Y[i] - (b[i] - 1)
    alpha0[i, prev_idx + 1: prev_idx + b[i]] = np.ones_like(alpha0[i, prev_idx + 1: prev_idx + b[i]])

    prev_idx = prev_idx + b[i] - 1
    
w_tilde0 = np.random.randn(n) 

w0 = w_tilde0 / (1-p)
A0 = np.zeros_like(alpha0)
for i in range(n):
    for j in range(n_split):
        A0[i,j] = alpha0[i,j] / w_tilde0[i]


prev_idx = 0
for i in range(n): 
    if i == 0:
        alpha[i, prev_idx] = 1
    else:
        alpha[i, prev_idx] = 1 - alpha[i - 1, prev_idx]
    alpha[i, prev_idx + 1: prev_idx + b[i] - 1] = np.ones_like(alpha[i, prev_idx + 1: prev_idx + b[i] - 1])
    alpha[i, prev_idx + b[i] - 1] = Y[i] - sum(alpha[i, :prev_idx + b[i] - 1])

    prev_idx = prev_idx + b[i] - 1
w_tilde = np.random.randn(n) 
w = w_tilde / (1-p)
A = np.zeros_like(alpha)
for i in range(n):
    for j in range(n_split):
        A[i,j] = alpha[i,j] / w_tilde[i]

workers_proposed = [[] for _ in range(n)]
for i in range(n):
    assigned_workers = np.where(alpha[i,:] != 0)[0]
    for data in assigned_workers:
        workers_proposed[i].append(data)

d_i = np.zeros(n_split)
d_i = genran_vec(n_split, d)


A2 = np.zeros_like(alpha)
workers_sgc = [[] for _ in range(n)]
for i in range(n_split): 
    assigned_workers = np.random.choice(n, size = d_i[i], replace=False)
    for worker in assigned_workers:
        workers_sgc[worker].append(i) 
        A2[worker, i] = 1 / ((1-p[worker])*d_i[i])
            
def bernoulli(p):
    return 1 if np.random.rand() < p else 0
A3 = np.zeros_like(alpha)
for i in range(n):
    for j in range(n_split):
        A3[i,j] = bernoulli(d2/n)

workers_bgc = [[] for _ in range(n)]
for i in range(n):
    assigned_workers = np.where(A3[i,:] != 0)[0]
    for data in assigned_workers:
        workers_bgc[i].append(data)
        
A4 = np.zeros_like(alpha)
sp = 0
prev = 0 
for i in range(n):
    if np.floor(i/d2) != prev:
        sp += int(d2 * n_split / n)
        prev = np.floor(i/d2)
    A4[i,sp:int(sp+d2 * n_split / n)] = 1


workers_erasurehead = [[] for _ in range(n)]
for i in range(n):
    assigned_workers = np.where(A4[i,:] != 0)[0]
    for data in assigned_workers:
        workers_erasurehead[i].append(data)

A5 = np.zeros_like(alpha)
sp = 0
prev = 0 
for i in range(n):
    A5[i,sp:int(sp+n_split / n)] = 1
    sp += int(n_split / n)
    prev = np.floor(i/d2)


workers_issgd = [[] for _ in range(n)]
for i in range(n):
    assigned_workers = np.where(A5[i,:] != 0)[0]
    for data in assigned_workers:
        workers_issgd[i].append(data)

G = nx.random_regular_graph(d2, n_split, seed=42)
A6 = nx.adjacency_matrix(G).toarray()

workers_od = [[] for _ in range(n)]
for i in range(n):
    assigned_workers = np.where(A6[i,:] != 0)[0]
    for data in assigned_workers:
        workers_od[i].append(data)


norm_gd = avg_num 
norm_proposed = avg_num2 
norm_sgc = avg_num3 
norm_bgc = avg_num3 
norm_issgd = avg_num3 
norm_ehd = avg_num3 
norm_od = avg_num3 


###############################################################################################
""" Quanization """
###############################################################################################

def quantize_tensor(x: torch.Tensor, bits: int, stochastic: bool = True, is_1bitquant: bool = False, eps: float = 1e-8):
    if bits is None or bits <= 0:
        return x

    scale_factor =  1.0 / (x.shape[0] ** 0.5)
    
    x_shape = x.shape
    x_flat = x.detach().view(-1)

    with torch.no_grad():
        s = torch.norm(x_flat, p=2).item()
    if s < eps:
        return torch.zeros_like(x)
    
    sign = torch.sign(x_flat)
    sign[sign == 0] = 1.0

    u = (x_flat.abs() / s).clamp(0.0, 1.0)

    M = 2 ** (bits - 1) - 1
    if M <= 0:
        prob = (x_flat / s + 1.0) / 2.0
        prob = torch.clamp(prob, 0.0, 1.0)
        noise = torch.rand_like(prob)
        h_flat = torch.where(noise < prob,
                             torch.ones_like(prob),
                             -torch.ones_like(prob))
        x_q_flat = s * h_flat 
        return x_q_flat.view(x_shape)

    y = u * M
    j = torch.floor(y)
    j_clamped = torch.clamp(j, 0, M).to(torch.long)

    if stochastic:
        r = (y - j).clamp(0.0, 1.0)
        noise = torch.rand_like(r)
        can_increase = (j_clamped < M).float()
        add_one = (noise < r).float() * can_increase
        j_hat = j_clamped + add_one.to(torch.long)
    else:
        j_hat = torch.round(y).to(torch.long).clamp(0, M)

    r_hat = j_hat.float() / float(M)

    x_q_flat = s * sign * r_hat 
    return x_q_flat.view(x_shape)

def recalculate_allocations(current_D, Btot, p):
    if current_D < 1e-6: 
        current_D = 1e-6

    # 1. DP 
    F_dp, b_dp = solve_dp(Btot, p, current_D)

    # 2. Greedy
    b0 = np.zeros(n)
    F_zero = objective_bit_allocation(b0, p, current_D)
    F_greed, b_greedy, _ = solve_greedy(Btot, p, current_D, F_zero, F_dp)

    # 3. Top-K
    # F_topk, b_tk = solve_topk(Btot, p, current_D)
    
    F_equal, b_equal = solve_equal(Btot, p, current_D)

    # 4. Top-K-LAG (Proposed)
    F_prop, b_prop = solve_topk_waterfilling(Btot, p, current_D)

    # 5. Continuous Relaxation (KKT)
    _, b_cont = solve_continuous(Btot, p, current_D)
    b_int = np.floor(b_cont).astype(int)
    rem = Btot - np.sum(b_int)
    if rem > 0:
        diffs = b_cont - b_int
        sorted_indices = np.argsort(diffs)[::-1]
        for k_idx in range(int(rem)):
            b_int[sorted_indices[k_idx]] += 1
    b_kkt = b_int
    
    return (b_prop + 1), (b_dp + 1), (b_greedy + 1), (b_equal + 1), (b_kkt + 1)
