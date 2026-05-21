import torch.nn.functional as F
import torch

def estimate_sigma(y_tensor, kernel_size=5): 
    y_blur = F.avg_pool2d(y_tensor, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    noise = y_tensor - y_blur
    sigma_eq = torch.std(noise)
    return sigma_eq.item()

def estimate_sigma_local(y_tensor, kernel_size=5): 
    y_blur = F.avg_pool2d(y_tensor, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    noise = y_tensor - y_blur
    noise_var_map = torch.abs(noise)
    return noise_var_map





























# TODO ----------------------2025-12-04 SPADAP使用的BNN 输出的局部标准差图--------------------------
def get_texture_noise_map(bnn_output, win_size=7, thr_low=1.0, thr_high=5.0):
    # ========= 支持 batch 维度 =========
    if bnn_output.dim() == 4:                    # (1,3,256,256) → (3,256,256)
        bnn_output = bnn_output.squeeze(0)
    
    # ========= RGB → 灰度 =========
    if bnn_output.dim() == 3 and bnn_output.shape[0] == 3:
        gray = bnn_output.mean(0, keepdim=True)   # (1, H, W)
    else:
        gray = bnn_output.unsqueeze(0) if bnn_output.dim() == 2 else bnn_output
        gray = gray.mean(0, keepdim=True)         # 强制转灰度

    if gray.max() > 1.1:
        gray = gray / 255.0

    # ========= 关键修复：确保是 (1,1,H,W) =========
    x = gray.unsqueeze(0)                        # (1,1,H,W)

    # ========= 计算局部标准差 =========
    unfold = torch.nn.Unfold(kernel_size=(win_size, win_size), padding=win_size//2)
    patches = unfold(x)                          # (1, win*win, H*W)
    patches = patches.view(1, win_size*win_size, -1)   # (1, 49, 65536)
    std = patches.std(dim=1)                     # (1, 65536) ← 改这里！dim=1
    std = std.view(x.shape[2], x.shape[3])       # (256, 256) ← 直接 view

    # ========= 论文公式 (3) =========
    sigma = std
    alpha = torch.zeros_like(sigma)
    mask1 = sigma <= thr_low
    mask2 = (sigma > thr_low) & (sigma <= thr_high)
    mask3 = sigma > thr_high

    alpha[mask1] = torch.sigmoid(sigma[mask1] - 1.0)
    alpha[mask2] = 0.5
    alpha[mask3] = torch.sigmoid(sigma[mask3] - 5.0)

    texture_map = alpha
    flat_map = 1.0 - alpha

    return texture_map.cpu().numpy(), flat_map.cpu().numpy()
