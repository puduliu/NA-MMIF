import os
import random

import numpy as np
import torch
import torch.utils.data as data
import torchvision.utils as tvu
import tqdm

from datasets import get_dataset, data_transform, inverse_data_transform
from functions.ckpt_util import get_ckpt_path, download
from functions.denoising_pixel_adaptive import efficient_generalized_steps
from guided_diffusion.script_util import create_model, create_classifier, classifier_defaults, args_to_dict
from datasets.estimate_noise import estimate_sigma, estimate_sigma_local


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
                np.linspace(
                    beta_start ** 0.5,
                    beta_end ** 0.5,
                    num_diffusion_timesteps,
                    dtype=np.float64,
                )
                ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas

dataset = 'PET-MRI'
base_out = f'output/{dataset}-adaptive-pixel'
base_in = f'datasets/{dataset}'

path_pairs = [
    (f"{base_in}/MRI_gau_0.1",
     f"{base_in}/PET_gau_0.1",
     f"{base_out}/MRI_gau_0.1_deno",
     f"{base_out}/PET_gau_0.1_deno"),
    
    (f"{base_in}/MRI_gau_0.2",
     f"{base_in}/PET_gau_0.2",
     f"{base_out}/MRI_gau_0.2_deno",
     f"{base_out}/PET_gau_0.2_deno"),
    
    (f"{base_in}/MRI_gau_0.05",
     f"{base_in}/PET_gau_0.05",
     f"{base_out}/MRI_gau_0.05_deno",
     f"{base_out}/PET_gau_0.05_deno"),

    (f"{base_in}/MRI_impulse_0.02",
     f"{base_in}/PET_impulse_0.02",
     f"{base_out}/MRI_impulse_0.02_deno",
     f"{base_out}/PET_impulse_0.02_deno"), 
    
    (f"{base_in}/MRI_impulse_0.04",
     f"{base_in}/PET_impulse_0.04",
     f"{base_out}/MRI_impulse_0.04_deno",
     f"{base_out}/PET_impulse_0.04_deno"),

    (f"{base_in}/MRI_poisson",
     f"{base_in}/PET_poisson",
     f"{base_out}/MRI_poisson_deno",
     f"{base_out}/PET_poisson_deno"), 


    (f"{base_in}/MRI_ray_0.2",
     f"{base_in}/PET_ray_0.2",
     f"{base_out}/MRI_ray_0.2_deno",
     f"{base_out}/PET_ray_0.2_deno"),

    (f"{base_in}/MRI_uni",
     f"{base_in}/PET_uni",
     f"{base_out}/MRI_uni_deno",
     f"{base_out}/PET_uni_deno"), 


    # todo 组合噪声
    (f"{base_in}/MRI_gau_0.1_poisson",
     f"{base_in}/PET_gau_0.1_poisson",
     f"{base_out}/MRI_gau_0.1_poisson_deno",
     f"{base_out}/PET_gau_0.1_poisson_deno"),
    
    (f"{base_in}/MRI_gau_0.1_uni",
     f"{base_in}/PET_gau_0.1_uni",
     f"{base_out}/MRI_gau_0.1_uni_deno",
     f"{base_out}/PET_gau_0.1_uni_deno"),

    (f"{base_in}/MRI_impulse_0.04_ray_0.2",
     f"{base_in}/PET_impulse_0.04_ray_0.2",
     f"{base_out}/MRI_impulse_0.04_ray_0.2_deno",
     f"{base_out}/PET_impulse_0.04_ray_0.2_deno",),

    (f"{base_in}/MRI_ray_0.2_uni",
     f"{base_in}/PET_ray_0.2_uni",
     f"{base_out}/MRI_ray_0.2_uni_deno",
     f"{base_out}/PET_ray_0.2_uni_deno"),

]

class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        self.alphas_cumprod_prev = alphas_cumprod_prev
        posterior_variance = (
                betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
            # torch.cat(
            # [posterior_variance[1:2], betas[1:]], dim=0).log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

    def sample(self):
        cls_fn = None
        if self.config.model.type == 'openai':
            config_dict = vars(self.config.model)
            model = create_model(**config_dict)
            if self.config.model.use_fp16:
                model.convert_to_fp16()
            if self.config.model.class_cond:
                ckpt = os.path.join(self.args.exp, 'logs/imagenet/%dx%d_diffusion.pt' % (
                    self.config.data.image_size, self.config.data.image_size))
                if not os.path.exists(ckpt):
                    download(
                        'https://openaipublic.blob.core.windows.net/diffusion/jul-2021/%dx%d_diffusion_uncond.pt' % (
                            self.config.data.image_size, self.config.data.image_size), ckpt)
            else:
                ckpt = os.path.join(self.args.exp, "logs/imagenet/256x256_diffusion_uncond.pt") # TODO 
                print("----------------ckpt = ", ckpt)
                if not os.path.exists(ckpt):
                    print("----------------please download pretrained model")

            model.load_state_dict(torch.load(ckpt, map_location=self.device))
            model.to(self.device)
            model.eval()

            if self.config.model.class_cond:
                ckpt = os.path.join(self.args.exp, 'logs/imagenet/%dx%d_classifier.pt' % (
                    self.config.data.image_size, self.config.data.image_size))
                if not os.path.exists(ckpt):
                    image_size = self.config.data.image_size
                    download(
                        'https://openaipublic.blob.core.windows.net/diffusion/jul-2021/%dx%d_classifier.pt' % image_size,
                        ckpt)
                classifier = create_classifier(**args_to_dict(self.config.classifier, classifier_defaults().keys()))
                classifier.load_state_dict(torch.load(ckpt, map_location=self.device))
                classifier.to(self.device)
                if self.config.classifier.classifier_use_fp16:
                    classifier.convert_to_fp16()
                classifier.eval()
                # classifier = torch.nn.DataParallel(classifier)

                import torch.nn.functional as F
                def cond_fn(x, t, y):
                    with torch.enable_grad():
                        x_in = x.detach().requires_grad_(True)
                        logits = classifier(x_in, t)
                        log_probs = F.log_softmax(logits, dim=-1)
                        selected = log_probs[range(len(logits)), y.view(-1)]
                        return torch.autograd.grad(selected.sum(), x_in)[0] * self.config.classifier.classifier_scale

                cls_fn = cond_fn

        self.sample_sequence(model, cls_fn)

    def sample_sequence(self, model, cls_fn=None):
        args, config = self.args, self.config

        for source_path1, source_path2, output_path1, output_path2 in path_pairs:
            print("---------------source_path1 = ", source_path1, "----source_path2 = ", source_path2)
            os.makedirs(output_path1, exist_ok=True)
            os.makedirs(output_path2, exist_ok=True)
            # get original images and corrupted y_0
            config.data.source_A = source_path1
            config.data.source_B = source_path2
            dataset, test_dataset = get_dataset(args, config)

            if args.subset_start >= 0 and args.subset_end > 0:
                assert args.subset_end > args.subset_start
                test_dataset = torch.utils.data.Subset(test_dataset, range(args.subset_start, args.subset_end))
            else:
                args.subset_start = 0
                args.subset_end = len(test_dataset)

            print(f'Dataset has size {len(test_dataset)}')

            def seed_worker(worker_id):
                worker_seed = args.seed % 2 ** 32
                np.random.seed(worker_seed)
                random.seed(worker_seed)

            g = torch.Generator()
            g.manual_seed(args.seed)
            val_loader = data.DataLoader(
                test_dataset,
                batch_size=config.sampling.batch_size,
                shuffle=False, # todo True -> false
                num_workers=config.data.num_workers,
                worker_init_fn=seed_worker,
                generator=g,
            )


            print(f'Start from {args.subset_start}')
            idx_so_far = args.subset_start
            pbar = tqdm.tqdm(val_loader)
            for x_orig_A, x_orig_B, classes, save_name in pbar: 
                save_name = save_name[0]
                
                x_orig_A = x_orig_A.to(self.device)
                x_orig_B = x_orig_B.to(self.device)
                
                sigma_0_map_A = estimate_sigma_local(x_orig_A)
                sigma_0_map_A[sigma_0_map_A <= 0.4] = 0
                sigma_0_map_A[sigma_0_map_A > 0.4] = 0.5

                sigma_0_map_B = estimate_sigma_local(x_orig_B)
                sigma_0_map_B[sigma_0_map_B <= 0.4] = 0
                sigma_0_map_B[sigma_0_map_B > 0.4] = 0.5

                sigma_es_A = estimate_sigma(x_orig_A) 
                sigma_es_B = estimate_sigma(x_orig_B)
                                                
                sigma_0_map_A = 2 * (sigma_0_map_A + sigma_es_A)
                sigma_0_map_B = 2 * (sigma_0_map_B + sigma_es_B)
                
                x_orig_A = data_transform(self.config, x_orig_A) 
                print(f"x_orig_A2222: min={x_orig_A.min().item():.4f}, max={x_orig_A.max().item():.4f}")
                x_orig_B = data_transform(self.config, x_orig_B)


                # todo start A
                x_A = torch.randn(
                    x_orig_A.shape[0],
                    config.data.channels,
                    config.data.image_size,
                    config.data.image_size,
                    device=self.device,
                )
                with torch.no_grad():
                    x_A, _ = self.sample_image(x_A, model, x_orig_A, sigma_es_A, sigma_0_map_A, last=False, cls_fn=cls_fn, classes=classes)

                x_A = [inverse_data_transform(config, y) for y in x_A]

                for i in [-1]:  # range(len(x)): [-1]:
                    for j in range(x_A[i].size(0)):
                        tvu.save_image(
                            x_A[i][j],
                            os.path.join(output_path1, save_name)
                        )  # todo 得到步长图片



                # todo start B
                x_B = torch.randn(
                    x_orig_B.shape[0],
                    config.data.channels,
                    config.data.image_size,
                    config.data.image_size,
                    device=self.device,
                )

                with torch.no_grad():
                    x_B, _ = self.sample_image(x_B, model, x_orig_B, sigma_es_B, sigma_0_map_B, last=False, cls_fn=cls_fn,
                                               classes=classes)

                x_B = [inverse_data_transform(config, y) for y in x_B]


                for i in [-1]:  
                    for j in range(x_B[i].size(0)):
                        tvu.save_image(
                            x_B[i][j],
                            os.path.join(output_path2, save_name)
                        )  

                idx_so_far += x_orig_A.shape[0]


    def sample_image(self, x, model, y_0, sigma_0, sigma_0_map, last=True, cls_fn=None, classes=None):
        skip = self.num_timesteps // self.args.timesteps
        seq = range(0, self.num_timesteps, skip)

        x = efficient_generalized_steps(x, seq, model, self.betas, y_0, sigma_0, sigma_0_map, \
                                        etaB=self.args.etaB, etaA=self.args.eta, etaC=self.args.eta, cls_fn=cls_fn,
                                        classes=classes)
        if last:
            x = x[0][-1]
        return x