import torch
from tqdm import tqdm
import torchvision.utils as tvu
import os

def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a

def efficient_generalized_steps(x, seq, model, b, y_0, sigma_0, sigma_0_map, etaB, etaA, etaC, cls_fn=None, classes=None):
    with torch.no_grad():
        largest_alphas = compute_alpha(b, (torch.ones(x.size(0)) * seq[-1]).to(x.device).long())
        largest_sigmas = (1 - largest_alphas).sqrt() / largest_alphas.sqrt()

        if largest_sigmas[0, 0, 0, 0] > sigma_0:  
            # 观测的权重
            init_y = (sigma_0 / largest_sigmas[0, 0, 0, 0]) * y_0 + torch.sqrt(1 - (sigma_0 / largest_sigmas[0, 0, 0, 0])**2) * x
        else:
            init_y = y_0 + largest_sigmas[0, 0, 0, 0] * x
            
        x = init_y / largest_alphas.sqrt()
        
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]

        for i, j in tqdm(zip(reversed(seq), reversed(seq_next))):
            t = (torch.ones(n) * i).to(x.device)
            current_step = t / 50 # TODO step [0,19]
            eta_est = 0.8 + 0.1 * (current_step / 19)        # 越来越小，从 0.8 → 0.8  etaA_bias是alpha
            eta_b_est = 1 - 0.1 * (current_step / 19)      # 越来越大，从 0.7 → 0.8 TODO 设置个可调参数 etaB_bias garma
            # print("-----current_step=", current_step, "----eta_est=", eta_est,"-----eta_b_est=",eta_b_est)
            etaA = eta_est
            etaC = eta_est
            etaB = eta_b_est

            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to('cuda')
            
            if cls_fn == None:
                et = model(xt, t)
            else:
                et = model(xt, t, classes)
                et = et[:, :3]
                et = et - (1 - at).sqrt()[0,0,0,0] * cls_fn(x,t,classes)
            
            if et.size(1) == 6:
                et = et[:, :3]
            
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

            sigma_next = (1 - at_next).sqrt()[0, 0, 0, 0] / at_next.sqrt()[0, 0, 0, 0]
            
            std_nextC = sigma_next * etaC
            sigma_tilde_nextC = torch.sqrt((sigma_next ** 2 - std_nextC ** 2).clamp(min=0))
            
            std_nextA = sigma_next * etaA
            sigma_tilde_nextA = torch.sqrt((sigma_next ** 2 - std_nextA ** 2).clamp(min=0))
            
            xt_next = x0_t + sigma_tilde_nextC * et + std_nextC * torch.randn_like(x0_t)
            
            sigma_0_map_device = sigma_0_map.to(x.device)
            
            less_noisy_mask = sigma_next < sigma_0_map_device  
            
            noisier_mask = sigma_next > sigma_0_map_device 
            
            less_noisy_result = x0_t + sigma_tilde_nextA * (y_0 - x0_t) / sigma_0_map_device + std_nextA * torch.randn_like(x0_t)
            
            xt_next = torch.where(less_noisy_mask, less_noisy_result, xt_next)
            
            diff_sigma_t_nextB = torch.sqrt((sigma_next ** 2 - sigma_0_map_device ** 2 * (etaB ** 2)).clamp(min=0))
            noisier_result = etaB * y_0 + (1 - etaB) * x0_t + diff_sigma_t_nextB * torch.randn_like(x0_t)
            
            xt_next = torch.where(noisier_mask, noisier_result, xt_next)
            
            xt_next = at_next.sqrt() * xt_next

            x0_preds.append(x0_t.to('cpu'))
            xs.append(xt_next.to('cpu'))

    return xs, x0_preds

































































