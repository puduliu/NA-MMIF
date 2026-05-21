import os
import time

import cv2
import torch

from model_vit_weight import build_model

from torchvision import transforms
from torch.autograd import Variable

from PIL import Image

from utils.myTransforms import denorm

from utils.myDatasets import ImagePair
from thop import profile
from thop import clever_format

def fuse(): 
    model = build_model(type = 0) # TODO edit  0 max, 1 sum ,2 mean
    model.load_state_dict(torch.load(f'model/MFIF.pth'))
    model.eval()
    model = model.cuda()
    # # 3. 计算 FLOPs 和参数量
    dummy_input = torch.randn(1, 3, 256, 256).cuda()
    flops, params = profile(model, inputs=(dummy_input,dummy_input))
    flops, params = clever_format([flops, params], "%.3f")

    print(f"模型 FLOPs: {flops}")
    print(f"模型参数量: {params}")
    

    def get_image_paths(path1, path2):
        files1 = sorted([f for f in os.listdir(path1) if f.endswith(('.jpg', '.png'))])
        files2 = sorted([f for f in os.listdir(path2) if f.endswith(('.jpg', '.png'))])
        img_paths1 = [os.path.join(path1, f) for f in files1]
        img_paths2 = [os.path.join(path2, f) for f in files2]

        return img_paths1, img_paths2

    datasets = ['Lytro-adaptive-pixel']

    dataset = 'Lytro-adaptive-pixel'
    method = 'NA-MMIF'
    base_out = f'output/{dataset}'
    base_in = f'output/{dataset}'

    path_pairs = [
    
    (f"{base_out}/{method}_gau_0.1/",
     f"{base_in}/source_1_gau_0.1_deno/",
     f"{base_in}/source_2_gau_0.1_deno/"),
    
    (f"{base_out}/{method}_gau_0.2/",
     f"{base_in}/source_1_gau_0.2_deno/",
     f"{base_in}/source_2_gau_0.2_deno/"),
    
    (f"{base_out}/{method}_gau_0.05/",
     f"{base_in}/source_1_gau_0.05_deno/",
     f"{base_in}/source_2_gau_0.05_deno/"),
    
    (f"{base_out}/{method}_impulse_0.02/",
     f"{base_in}/source_1_impulse_0.02_deno/",
     f"{base_in}/source_2_impulse_0.02_deno/"),
    
    (f"{base_out}/{method}_impulse_0.04/",
     f"{base_in}/source_1_impulse_0.04_deno/",
     f"{base_in}/source_2_impulse_0.04_deno/"),

    (f"{base_out}/{method}_poisson/",
     f"{base_in}/source_1_poisson_deno/",
     f"{base_in}/source_2_poisson_deno/"),
    
    (f"{base_out}/{method}_ray_0.2/",
     f"{base_in}/source_1_ray_0.2_deno/",
     f"{base_in}/source_2_ray_0.2_deno/"),
    
    (f"{base_out}/{method}_uni/",
     f"{base_in}/source_1_uni_deno/",
     f"{base_in}/source_2_uni_deno/"),
    
    (f"{base_out}/{method}_gau_0.1_poisson/",
     f"{base_in}/source_1_gau_0.1_poisson_deno/",
     f"{base_in}/source_2_gau_0.1_poisson_deno/"),
    
    (f"{base_out}/{method}_gau_0.1_uni/",
     f"{base_in}/source_1_gau_0.1_uni_deno/",
     f"{base_in}/source_2_gau_0.1_uni_deno/"),
    

    (f"{base_out}/{method}_impulse_0.04_ray_0.2/",
     f"{base_in}/source_1_impulse_0.04_ray_0.2_deno/",
     f"{base_in}/source_2_impulse_0.04_ray_0.2_deno/"),

    (f"{base_out}/{method}_ray_0.2_uni/",
     f"{base_in}/source_1_ray_0.2_uni_deno/",
     f"{base_in}/source_2_ray_0.2_uni_deno/"),
    ]

    from pathlib import Path

    dataset = datasets[0] 
    mean = [0.485, 0.456, 0.406] 
    std = [0.229, 0.224, 0.225]

    for output_path, path1, path2 in path_pairs:
        os.makedirs(output_path, exist_ok=True)
        img_paths1, img_paths2 = get_image_paths(path1, path2)
        for index, (img_path1, img_path2) in enumerate(zip(img_paths1, img_paths2)):
            file_name = Path(img_path1).name
            save_name = os.path.join(output_path, file_name)

            pair_loader = ImagePair(impath1=img_path1, impath2=img_path2,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=mean, std=std)
                                    ]))
            img1, img2 = pair_loader.get_pair() # tensor

            img1.unsqueeze_(0)
            img2.unsqueeze_(0)

            with torch.no_grad():
                res = model(Variable(img1.cuda()), Variable(img2.cuda()))
                res = denorm(mean, std, res[0]).clamp(0, 1) * 255
                res_img = res.cpu().data.numpy().astype('uint8')
                img = res_img.transpose([1, 2, 0])

            img = Image.fromarray(img)
            print("----------------------save_name = ", save_name)
            img.save(save_name)

if __name__ == "__main__":
    fuse()