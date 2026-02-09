import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import random
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split, Subset
from util_param_ import *
from util_ import *

import torchvision
import torchvision.transforms as transfroms

from torchvision.models.detection import (
    retinanet_resnet50_fpn,
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
    SSDLite320_MobileNet_V3_Large_Weights
)
 
train_set = CocoDetectionDataset(
    root='./data/COCO/train2017',
    annFile='./data/COCO/annotations/instances_train2017.json',
    alb_transforms=transform   
)
test_set = CocoDetectionDataset(
    root='./data/COCO/val2017',
    annFile='./data/COCO/annotations/instances_val2017.json',
    alb_transforms=transform   
)

train_set = Subset(train_set, list(range(m)))

indices = list(range(len(train_set)))
random.shuffle(indices)
partitions = np.array_split(indices, n)
train_subsets = [Subset(train_set, p.tolist()) for p in partitions]

test_set = Subset(test_set, list(range(test_batch_size)))

indices = list(range(len(test_set)))
random.shuffle(indices)
partitions = np.array_split(indices, n)
test_subsets = [Subset(test_set, p.tolist()) for p in partitions]


partitions = np.array_split(range(m), n)

beta_gd = models.detection.ssdlite320_mobilenet_v3_large(weights=None, num_classes=91).to(device)
beta_proposed0 = models.detection.ssdlite320_mobilenet_v3_large(weights=None, num_classes=91).to(device)
beta_osgc = models.detection.ssdlite320_mobilenet_v3_large(weights=None, num_classes=91).to(device)
beta_sgc = models.detection.ssdlite320_mobilenet_v3_large(weights=None, num_classes=91).to(device)
beta_bgc = models.detection.ssdlite320_mobilenet_v3_large(weights=None, num_classes=91).to(device)
beta_ehd = models.detection.ssdlite320_mobilenet_v3_large(weights=None, num_classes=91).to(device)
beta_od = models.detection.ssdlite320_mobilenet_v3_large(weights=None, num_classes=91).to(device)
beta_issgd = models.detection.ssdlite320_mobilenet_v3_large(weights=None, num_classes=91).to(device)


from copy import deepcopy
initial_state_dict = {k: v.detach().clone() for k, v in beta_gd.state_dict().items()}
beta_proposed0.load_state_dict(initial_state_dict)
beta_osgc.load_state_dict(initial_state_dict)
beta_sgc.load_state_dict(initial_state_dict)
beta_bgc.load_state_dict(initial_state_dict)
beta_ehd.load_state_dict(initial_state_dict)
beta_od.load_state_dict(initial_state_dict)
beta_issgd.load_state_dict(initial_state_dict)



#####################################################################################################################
dim_l = sum(p.numel() for p in beta_gd.parameters())
D = dim_l / 4 * C2 
eta = p.mean() / (dim_l / (4 * ((2 ** ((Btot+2*n) / n - 1) - 1) ** 2)) * C2)
current_scale = 0.005 
quant_denom_avg = (2 ** ((Btot+2*n) / n - 1) - 1) ** 2
current_D = p.mean() * quant_denom_avg * current_scale
if Btot > 0:
    bit_alloc_proposed, bit_alloc_dp, bit_alloc_greedy, bit_alloc_equal, bit_alloc_kkt = \
        recalculate_allocations(current_D, Btot, p)

    bit_alloc_proposed += 1
    bit_alloc_dp += 1
    bit_alloc_greedy += 1
    bit_alloc_equal += 1
    bit_alloc_kkt += 1
else:# 
    bit_alloc_proposed = np.ones(n)
    bit_alloc_dp = np.ones(n)
    bit_alloc_greedy = np.ones(n)
    bit_alloc_equal = np.ones(n)
    bit_alloc_kkt = np.ones(n)


omega0 = current_D / ((2 ** (bit_alloc_proposed - 1) - 1).astype(np.float64) ** 2) 
c_inv = (1-p) / (p + omega0)

mean_straggler = p.mean()
mean_quant_noise = omega0.mean()

Y2 = c_inv * n_split / sum(c_inv)
prev_idx = 0
alpha2 = np.zeros((n,n_split))
for i in range(n): 
    if i == 0:
        alpha2[i, prev_idx] = 1
    else:
        alpha2[i, prev_idx] = 1 - alpha2[i - 1, prev_idx]
    alpha2[i, prev_idx + 1: prev_idx + b[i] - 1] = np.ones_like(alpha2[i, prev_idx + 1: prev_idx + b[i] - 1])
    alpha2[i, prev_idx + b[i] - 1] = Y2[i] - sum(alpha2[i, :prev_idx + b[i] - 1])

    prev_idx = prev_idx + b[i] - 1

alpha02 = np.zeros((n,n_split))
mn = 0
mx = np.ceil(2 * d)
prev_idx = 0
for i in range(n): 
    alpha02[i, 0] = Y2[i] - (b[i] - 1)
    alpha02[i, prev_idx + 1: prev_idx + b[i]] = np.ones_like(alpha02[i, prev_idx + 1: prev_idx + b[i]])

    prev_idx = prev_idx + b[i] - 1
    
w_tilde02 = np.random.randn(n) 

w02 = w_tilde02 / (1-p)
A02 = np.zeros_like(alpha02)
for i in range(n):
    for j in range(n_split):
        A02[i,j] = alpha02[i,j] / w_tilde02[i]

        
A01 = np.zeros_like(alpha02)
for i in range(n):
    for j in range(n_split):
        A01[i,j] = alpha2[i,j] / w_tilde02[i]


iteration = 0
losses_gd = []
losses_proposed0 = []
losses_osgc = []
losses_proposed4 = []
losses_proposed5 = []
losses_sgc = []
losses_bgc = []
losses_ehd = []
losses_od = []
losses_issgd = []
itq = 0
for t in range(T):
    w0_sq = lambda_sq(t) * n_split / (p * sum(A0.T) * (1 + lambda_sq(t) * sum(delta_inv))) 
    
    current_worker_norms_sq = np.zeros(n)
    valid_worker_mask = np.zeros(n, dtype=bool) #

    stragglers = np.random.rand(n) < p  # Determine stragglers for this iteration
    count = 0

    grad_gd = []
    grad_proposed0 = []
    grad_proposed02 = []
    grad_proposed1 = []
    grad_proposed2 = []
    grad_osgc = []
    grad_proposed4 = []
    grad_proposed5 = []
    grad_sgc = []
    grad_bgc = []
    grad_ehd = []
    grad_od = []
    grad_issgd = []

    gradient_gd_sum = [torch.zeros_like(param).to(device) for param in beta_gd.parameters()]
    gradient_proposed0_sum = [torch.zeros_like(param).to(device) for param in beta_proposed0.parameters()]
    gradient_osgc = [torch.zeros_like(param).to(device) for param in beta_proposed0.parameters()]
    gradient_sgc_sum = [torch.zeros_like(param).to(device) for param in beta_sgc.parameters()]
    gradient_bgc_sum = [torch.zeros_like(param).to(device) for param in beta_bgc.parameters()]
    gradient_ehd_sum = [torch.zeros_like(param).to(device) for param in beta_ehd.parameters()]
    gradient_od_sum = [torch.zeros_like(param).to(device) for param in beta_od.parameters()]
    gradient_issgd_sum = [torch.zeros_like(param).to(device) for param in beta_issgd.parameters()]
    it = 0
    it2 = 0
    
    accumulated_grad_norm_sq = 0.0
    valid_worker_count = 0
    for cid, subset in enumerate(train_subsets):
        loader = DataLoader(
            subset,
            batch_size=batch_size, 
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            collate_fn=collate_fn
        )
        stragglers_tmp_proposed = np.ones_like(stragglers, dtype=bool)
        stragglers_tmp_proposed0 = np.ones_like(stragglers, dtype=bool)
        stragglers_tmp_sgc = np.ones_like(stragglers, dtype=bool)
        stragglers_tmp_bgc = np.ones_like(stragglers, dtype=bool)
        stragglers_tmp_od = np.ones_like(stragglers, dtype=bool)
        stragglers_tmp_ehd = np.ones_like(stragglers, dtype=bool)
        stragglers_tmp_issgd = np.ones_like(stragglers, dtype=bool)

        stragglers_tmp_proposed[np.where(A0[:,it] != 0)[0]] = stragglers[np.where(A0[:,it] != 0)[0]]
        stragglers_tmp_proposed0[np.where(A0[:,it] != 0)[0]] = stragglers[np.where(A0[:,it] != 0)[0]]
        stragglers_tmp_sgc[np.where(A2[:,it] != 0)[0]] = stragglers[np.where(A2[:,it] != 0)[0]]
        stragglers_tmp_bgc[np.where(A3[:,it] != 0)[0]] = stragglers[np.where(A3[:,it] != 0)[0]]
        stragglers_tmp_od[np.where(A6[:,it] != 0)[0]] = stragglers[np.where(A6[:,it] != 0)[0]]
        stragglers_tmp_ehd[np.where(A4[:,it] != 0)[0]] = stragglers[np.where(A4[:,it] != 0)[0]]
        stragglers_tmp_issgd[np.where(A5[:,it] != 0)[0]] = stragglers[np.where(A5[:,it] != 0)[0]]

        for images, targets in loader:
            X_batch = [img.to(device) for img in images]
            y_batch = [{k: v.to(device) for k, v in t.items()} for t in targets]

            grad_gd_tmp = gradient_GD_loader(X_batch, beta_gd, y_batch)
            beta_gd.zero_grad()
            
            
            grad_proposed0_tmp = gradient_GC_loader(X_batch, beta_proposed0, y_batch, stragglers_tmp_proposed, it, A02, w02, bit_alloc_proposed) 
            beta_proposed0.zero_grad()
            

            grad_osgc_tmp = gradient_GC_loader(X_batch, beta_osgc, y_batch, stragglers_tmp_proposed, it, A0, w0, bit_alloc_equal) 
            beta_osgc.zero_grad()


            grad_sgc_tmp = gradient_GC_loader(X_batch, beta_sgc, y_batch, stragglers_tmp_sgc, it, A2, np.ones(n), bit_alloc_equal) 
            beta_sgc.zero_grad()
            
            grad_bgc_tmp = gradient_GC_loader(X_batch, beta_bgc, y_batch, stragglers_tmp_bgc, it, A3, np.ones(n), bit_alloc_equal)
            beta_bgc.zero_grad()
            
            grad_ehd_tmp = gradient_GC_loader(X_batch, beta_ehd, y_batch, stragglers_tmp_ehd, it, A4, np.ones(n), bit_alloc_equal)
            beta_ehd.zero_grad()
            
            w_od = calculate_optimal_decoding(A6, stragglers)
            grad_od_tmp = gradient_GC_loader(X_batch, beta_od, y_batch, stragglers_tmp_od, it, A6, w_od, bit_alloc_equal) 
            beta_od.zero_grad()
            
            grad_issgd_tmp = gradient_GC_loader(X_batch, beta_issgd, y_batch, stragglers_tmp_issgd, it, A5, np.ones(n), bit_alloc_equal) 
            beta_issgd.zero_grad()
            
            
            
            for i in range(len(grad_gd_tmp)):
                gradient_gd_sum[i] += grad_gd_tmp[i]
                gradient_proposed0_sum[i] += grad_proposed0_tmp[i] 
                gradient_osgc[i] += grad_osgc_tmp[i] 

                gradient_sgc_sum[i] += grad_sgc_tmp[i]
                gradient_bgc_sum[i] += grad_bgc_tmp[i]
                gradient_ehd_sum[i] += grad_ehd_tmp[i]
                gradient_od_sum[i] += grad_od_tmp[i]
                gradient_issgd_sum[i] += grad_issgd_tmp[i]
            it2 += 1
            
        
        it += 1
        
    
    for i in range(len(gradient_gd_sum)):
        grad_gd.append(gradient_gd_sum[i] / it)
        grad_proposed0.append(gradient_proposed0_sum[i] / it)
        grad_osgc.append(gradient_osgc[i] / it)
        
        grad_sgc.append(gradient_sgc_sum[i] / it)
        grad_bgc.append(gradient_bgc_sum[i] / it)
        grad_ehd.append(gradient_ehd_sum[i] / it)
        grad_od.append(gradient_od_sum[i] / it)
        grad_issgd.append(gradient_issgd_sum[i] / it)
    
    

    if t % period_validate == 0:
        with torch.no_grad():
            i = 0
            
            loss_gd = 0
            loss_proposed0 = 0
            loss_osgc = 0
            loss_sgc = 0
            loss_bgc = 0
            loss_ehd = 0
            loss_od = 0
            loss_issgd = 0

            loader = DataLoader(
                test_set,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
                collate_fn=collate_fn
            )

            for images, targets in loader:
                X_test = [img.to(device) for img in images]
                y_test = [{k: v.to(device) for k, v in t.items()} for t in targets]
                
                i += len(X_test)
                
                beta_gd.train()
                beta_proposed0.train()
                beta_osgc.train()
                beta_sgc.train()
                beta_bgc.train()
                beta_ehd.train()
                beta_od.train()
                beta_issgd.train()

                loss_gd += sum(loss for loss in beta_gd(X_test, y_test).values()).item() 
                loss_proposed0 += sum(loss for loss in beta_proposed0(X_test, y_test).values()).item() 
                loss_osgc += sum(loss for loss in beta_osgc(X_test, y_test).values()).item() 
                loss_sgc += sum(loss for loss in beta_sgc(X_test, y_test).values()).item()  
                loss_bgc += sum(loss for loss in beta_bgc(X_test, y_test).values()).item() 
                loss_ehd += sum(loss for loss in beta_ehd(X_test, y_test).values()).item() 
                loss_od += sum(loss for loss in beta_od(X_test, y_test).values()).item()  
                loss_issgd += sum(loss for loss in beta_issgd(X_test, y_test).values()).item()  

            print(f"iteration {t},  GD: {loss_gd / i:.4f}, Proposed: {loss_proposed0 / i:.4f}, OSGC: {loss_osgc / i:.4f}")
            print(f"iteration {t},  SGC: {loss_sgc / i:.4f}, BGC: {loss_bgc / i:.4f}, EHD: {loss_ehd / i:.4f}, OD: {loss_od / i:.4f}, ISSGD: {loss_issgd / i:.4f}")
            
            losses_gd.append(loss_gd / i)
            losses_proposed0.append(loss_proposed0 / i)
            losses_osgc.append(loss_osgc / i)
            losses_sgc.append(loss_sgc / i)
            losses_bgc.append(loss_bgc / i)
            losses_ehd.append(loss_ehd / i)
            losses_od.append(loss_od / i)
            losses_issgd.append(loss_issgd / i)
        
    ###############################################################################################################
    '''
    Update model parameters
    '''
    ###############################################################################################################
    with torch.no_grad():
        count_tmp = 0
        for param in beta_gd.parameters():
            param -= gamma_t(t) * grad_gd[count_tmp] / norm_gd
            count_tmp += 1
        count_tmp = 0
        for param in beta_proposed0.parameters():
            param -= gamma_t(t) * grad_proposed0[count_tmp] / norm_proposed
            count_tmp += 1
        count_tmp = 0
        for param in beta_osgc.parameters():
            param -= gamma_t(t) * grad_osgc[count_tmp] / norm_proposed
            count_tmp += 1
        count_tmp = 0
        for param in beta_sgc.parameters():
            param -= gamma_t(t) * grad_sgc[count_tmp] / norm_sgc
            count_tmp += 1
        count_tmp = 0
        for param in beta_bgc.parameters():
            param -= gamma_t(t) * grad_bgc[count_tmp] / norm_bgc
            count_tmp += 1
        count_tmp = 0
        for param in beta_ehd.parameters():
            param -= gamma_t(t) * grad_ehd[count_tmp] / norm_ehd
            count_tmp += 1
        count_tmp = 0
        for param in beta_od.parameters():
            param -= gamma_t(t) * grad_od[count_tmp] / norm_od
            count_tmp += 1
        count_tmp = 0
        for param in beta_issgd.parameters():
            param -= gamma_t(t) * grad_issgd[count_tmp] / norm_issgd
            count_tmp += 1

    # m_gd, v_gd = adam_apply(m_gd, v_gd, beta_gd, grad_gd, norm_gd, adamparam, adamparam2, t)
    # # m_proposed, v_proposed = adam_apply(m_proposed, v_proposed, beta_proposed, grad_proposed, norm_proposed,  adamparam, adamparam2, t, scale_proposed) #2
    # # m_proposed0, v_proposed0 = adam_apply(m_proposed0, v_proposed0, beta_proposed0, grad_proposed0, norm_proposed,  adamparam, adamparam2, t, scale_proposed) #2
    # m_proposed0, v_proposed0 = adam_apply2(m_proposed0, v_proposed0, beta_proposed0, grad_proposed0, grad_proposed02, norm_proposed,  adamparam, adamparam2, t, scale_proposed) #2
    # m_sgc,       v_sgc       = adam_apply(m_sgc,       v_sgc,       beta_sgc,       grad_sgc,       norm_sgc,       adamparam, adamparam2, t)
    # # m_bgc,       v_bgc       = adam_apply(m_bgc,       v_bgc,       beta_bgc,       grad_bgc,       norm_proposed,       adamparam, adamparam2, t)
    # m_bgc,       v_bgc       = adam_apply(m_bgc,       v_bgc,       beta_bgc,       grad_bgc,       norm_bgc,       adamparam, adamparam2, t)
    # m_ehd,       v_ehd       = adam_apply(m_ehd,       v_ehd,       beta_ehd,       grad_ehd,       norm_ehd,       adamparam, adamparam2, t)
    # m_od,        v_od        = adam_apply(m_od,        v_od,        beta_od,        grad_od,        norm_od,        adamparam, adamparam2, t)
    # # m_issgd,     v_issgd     = adam_apply(m_issgd,     v_issgd,     beta_issgd,     grad_issgd,     norm_proposed,     adamparam, adamparam2, t)
    # m_issgd,     v_issgd     = adam_apply(m_issgd,     v_issgd,     beta_issgd,     grad_issgd,     norm_issgd,     adamparam, adamparam2, t)
    