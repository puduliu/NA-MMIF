import os
import time

import cv2
import torch

# from global_model_CBAM_vit_dynamic import build_model
from model_vit_weight import build_model

from torchvision import transforms
from torch.autograd import Variable

from PIL import Image

from utils.myTransforms import denorm

from utils.myDatasets import ImagePair

from utils.fuseY2RGB import fuseY2RGB


def fuse(): 
    model = build_model(type = 2) # TODO edit  0 max, 1 sum ,2 mean
    model.load_state_dict(torch.load(f'model/VIF.pth'))
    model.eval()
    model = model.cuda()
    

    def get_image_paths(path1, path2):
        files1 = sorted([f for f in os.listdir(path1) if f.endswith(('.jpg', '.png'))])
        files2 = sorted([f for f in os.listdir(path2) if f.endswith(('.jpg', '.png'))])

        img_paths1 = [os.path.join(path1, f) for f in files1]
        img_paths2 = [os.path.join(path2, f) for f in files2]

        return img_paths1, img_paths2
    datasets = ['TNO_256-adaptive-pixel']

    dataset = 'TNO_256-adaptive-pixel'
    base_out = f'output/{dataset}'
    base_in = f'output/{dataset}'


    path_pairs = [
        (f"{base_out}/result/", 
         f"{base_in}/vi_deno/",
         f"{base_in}/ir_deno/"),
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
            img1, img2 = pair_loader.get_pair() 
            img1.unsqueeze_(0)
            img2.unsqueeze_(0)
            with torch.no_grad():
                res = model(Variable(img1.cuda()), Variable(img2.cuda()))
                res = denorm(mean, std, res[0]).clamp(0, 1) * 255
                res_img = res.cpu().data.numpy().astype('uint8')
                img = res_img.transpose([1, 2, 0])
            
            img = Image.fromarray(img).convert('L')  
            print("----------------------save_name = ", save_name)
            img.save(save_name)

if __name__ == "__main__":
    fuse()