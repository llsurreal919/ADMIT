# Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
# Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import time
from models import create_model
from options.test_options import TestOptions
import numpy as np
import torch
import os
# import torchvision
import torchvision.transforms as transforms
# from collections import defaultdict
from typing import List
from PIL import Image
import torch.nn.functional as F
# from models import ImageFolder
# from torch.utils.data import DataLoader
import time
from torchvision.utils import save_image
from pytorch_msssim import ms_ssim

# from torchvision.datasets.folder
IMG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
)
def read_image(filepath: str) -> torch.Tensor:
    assert os.path.isfile(filepath)
    img = Image.open(filepath).convert("RGB")
    img = transforms.ToTensor()(img)
    img = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(img)
    return img

def test(model, filepaths, opt):
    
    PSNR_list = []
    Inf_time = []
    for f in filepaths:
        x = read_image(f).to('cuda')

        x = x.unsqueeze(0)

        h, w = x.size(2), x.size(3)
        p = 16  # maximum 2 strides of 2, and window size 4 for the smallest latent fmap: 4*2^2=16
        new_h = (h + p - 1) // p * p
        new_w = (w + p - 1) // p * p
        padding_left = (new_w - w) // 2
        padding_right = new_w - w - padding_left
        padding_top = (new_h - h) // 2
        padding_bottom = new_h - h - padding_top
        x_padded = F.pad(
            x,
            (padding_left, padding_right, padding_top, padding_bottom),
            mode="constant",
            value=0,
        )

        model.set_input(x_padded)
        start_time = time.time()
        model.forward()
        end_time = time.time() - start_time

        fake = model.fake
        
        
        fake= F.pad(
        fake, (-padding_left, -padding_right, -padding_top, -padding_bottom)
        )
        
        # mse = torch.nn.MSELoss(reduction='none')(fake.detach().cpu()* 255., x.detach().cpu() * 255.)
        # PSNR = 10 * (torch.log(255. * 255. / mse.mean()) / np.log(10))
        # PSNR_list.append(PSNR)

        img_gen_numpy = fake.detach().cpu().float().numpy()
        img_gen_numpy = (np.transpose(img_gen_numpy, (0, 2, 3, 1)) + 1) / 2.0 * 255.0
        img_gen_int8 = img_gen_numpy.astype(np.uint8)
        
        
        origin_numpy = x.detach().cpu().float().numpy()
        origin_numpy = (np.transpose(origin_numpy, (0, 2, 3, 1)) + 1) / 2.0 * 255.0
        origin_int8 = origin_numpy.astype(np.uint8)

        diff = np.mean((np.float64(img_gen_int8) - np.float64(origin_int8))**2, (1, 2, 3))

        PSNR = 10 * np.log10((255**2) / diff)
        MS_SSIM = ms_ssim(x, fake, data_range=1.0).item(),
        _, filename = os.path.split(f)
        os.makedirs('Ours-L' + '/' + str(opt.SNR), exist_ok=True)
        
        new_filename = filename[0:7]
        # fake.save('kodak_rec'+ '/' + str(opt.SNR) + '/' +  new_filename)
        # new_filename_org = re.sub(r'(\d+)', r'\1_org', filename)
        save_image((fake[0] + 1) / 2.0, 'Ours-L'+ '/' + str(opt.SNR) + '/' + new_filename + '.png' )
        # save_image((x[0] + 1) / 2.0, 'kodak_rec'+ '/' + str(opt.SNR) + '/' +  new_filename_org)
        
        PSNR_list.append(np.mean(PSNR))
        Inf_time.append(end_time)

    return np.mean(PSNR_list), np.mean(Inf_time)

def collect_images(rootpath: str) -> List[str]:
    return [
        os.path.join(rootpath, f)
        for f in os.listdir(rootpath)
        if os.path.splitext(f)[-1].lower() in IMG_EXTENSIONS
    ]

if __name__ == "__main__":
    # Extract the options
    opt = TestOptions().parse()

    filepaths = collect_images('kodak/')
    # filepaths = collect_images('kodim01')

    opt.name = str(opt.band)
    # path = os.path.join("Checkpoints", opt.name)
    model = create_model(opt)      # create a model given opt.model and other options

    model.setup(opt)               # regular setup: load and print networks; create schedulers
    # checkpoint_path = opt.checkpoint
    # checkpoint = torch.load(checkpoint_path, map_location='cuda')
    # model.load_state_dict(checkpoint["state_dict"])
    model_list = ['SE', 'SD', 'CA','BCM_' + str(opt.band)]
    model.load(model_list, opt)
    # print('Loaded the checkpoint in: %s', checkpoint_path)
    model.eval()
    snr_list = [0,5,7,10,15]
    cpp = opt.band / 512
    print('MEAN_CPP is %f' % (cpp))
    for snr in snr_list:
    # for snr in range(0,21):
        opt.SNR = snr
        opt.gpu_ids=0 ,
        mean_psnr, inf_time = test(model, filepaths, opt)
        print('MEAN_PSNR  is %f' % ( mean_psnr))
        logfile = os.path.join(opt.test_dir, opt.name)
        f = open(logfile + 'test.json', 'a')
        print("{:.4f} {:.4f} {:.4f}".format(mean_psnr, opt.SNR, inf_time), file=f)

    # mean_psnr = test(model, filepaths, opt)
    # cpp = opt.band / 128
    # print('MEAN_PSNR  is %f' % ( mean_psnr))
    # print('MEAN_CPP is %f' % (cpp))