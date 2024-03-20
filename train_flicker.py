# Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
# Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import time
from models import create_model
from options.train_options import TrainOptions
from options.test_options import TestOptions
import random
import os
import torch
import shutil
import torchvision
import torchvision.transforms as transforms
import numpy as np
from models import ImageFolder
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(20)

def train_one_epoch(model, dataset, epoch, epoch_iter, total_iters, opt):
    model.train()
    for i, data in enumerate(dataset):  # inner loop within one epoch
        iter_start_time = time.time()  # timer for computation per iteration

        total_iters += opt.batch_size
        epoch_iter += opt.batch_size
        
        model.set_input(data)         # unpack data from dataset and apply preprocessing
        model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

        if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
            losses = model.get_current_losses()
            t_comp = (time.time() - iter_start_time) / opt.batch_size
            message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, epoch_iter, t_comp)
            for k, v in losses.items():
                message += '%s: %.5f ' % (k, v)
            print(message)  # print the message
            log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
            with open(log_name, "a") as log_file:
                log_file.write('%s\n' % message)  # save the message
        
def val_one_epoch(model, test_dataset, epoch, total_epoch, opt):
    model.eval()
    with torch.no_grad():
        PSNR_list = []
        loss_list = []
        for i, data in enumerate(test_dataset):
            input = data
            model.set_input(input)
            model.forward()
            fake = model.fake
            
            img_gen_numpy = fake.detach().cpu().float().numpy()
            img_gen_numpy = (np.transpose(img_gen_numpy, (0, 2, 3, 1)) + 1) / 2.0 * 255.0
            img_gen_int8 = img_gen_numpy.astype(np.uint8)

            origin_numpy = input.detach().cpu().float().numpy()
            origin_numpy = (np.transpose(origin_numpy, (0, 2, 3, 1)) + 1) / 2.0 * 255.0
            origin_int8 = origin_numpy.astype(np.uint8)

            diff = np.mean((np.float64(img_gen_int8) - np.float64(origin_int8))**2, (1, 2, 3))

            PSNR = 10 * np.log10((255**2) / diff)
            PSNR_list.append(np.mean(PSNR))
            loss = model.get_current_losses()
            loss_list.append(loss['G_L2'])

        mean_loss = np.mean(loss_list)
        mean_psnr = np.mean(PSNR_list)

        return mean_loss, mean_psnr
            
def save_checkpoint(state, path, name):
    save_filename = '%s_epoch.pth' % (name)
    save_path = os.path.join(path, save_filename)
    torch.save(state, save_path)
    
def save_best_checkpoint(state, is_best, path, name, epoch):
    save_filename = '%s_epoch.pth' % (name)
    save_path = os.path.join(path, save_filename)
    torch.save(state, save_path)
    
    save_best = 'epoch_best_loss.pth'
    best_path = os.path.join(path, save_best)
    if is_best:
        shutil.copyfile(save_path, best_path)

def main(train_opt):
    # Prepare the train_dataset   
    train_transforms = transforms.Compose(
        [transforms.RandomHorizontalFlip(),
         transforms.RandomVerticalFlip(),
        transforms.RandomCrop(256),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    test_transforms = transforms.Compose(
        [transforms.CenterCrop(256), transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    train_dataset = ImageFolder("/home/dannyluo/Dataset/flicker_2W_images/", split="train", transform=train_transforms)
    test_dataset = ImageFolder("/home/dannyluo/Dataset/flicker_2W_images/", split="test", transform=test_transforms)
    
    train_dataset = DataLoader(
        train_dataset,
        batch_size = train_opt.batch_size,
        num_workers=8,
        shuffle=True,
        pin_memory=("cuda"),
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=8,
        shuffle=False,
        pin_memory=("cuda"),
    )
    
    dataset_size = len(train_dataset)
    print('#training images = %d' % dataset_size)
    test_dataset_size = len(test_dataset)
    print('#training images = %d' % test_dataset_size)

    # Create the checkpoint folder  
    # train_opt.name = 'cpp' + str(train_opt.band / 32) + '_Pre_0.375'
    train_opt.name = 'cpp' + str(train_opt.band / 512) + '_Pre_0.375' #16*16*2
    path = os.path.join(train_opt.checkpoints_dir, train_opt.name)
    if not os.path.exists(path):
        os.makedirs(path) 

    writer = SummaryWriter(log_dir=path + '/log/')

    # Initialize the model
    model = create_model(train_opt)      # create a model given opt.model and other options
    model.setup(train_opt)               # print networks;  load checkpoint放在在后面
    print(model.optimizer_G.param_groups[0]['lr'])

    if train_opt.pretrain:  # load from previous checkpoint
        # checkpoint_path = 'Checkpoints_natten_adaptive/cpp0.375/latest_epoch.pth'
        # checkpoint = torch.load(checkpoint_path, map_location='cuda')
        # model.load_state_dict(checkpoint["state_dict"], strict=True)
        # print('Loaded the checkpoint')
        model_list = ['SE', 'CA']
        model.load(model_list, train_opt)
        # 冻结SE
        for param in model.netSE.parameters():
            param.requires_grad = False
        for param in model.netCA.parameters():
            param.requires_grad = False    
        print('SE and CA is fixed')
        
    total_iters = 0                # the total number of training iterations
    total_epoch = train_opt.n_epochs_joint + train_opt.n_epochs_decay + train_opt.n_epochs_fine
    best_loss = float("inf")
    for epoch in range(train_opt.epoch_count, total_epoch + 1):# outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_iter = 0                 # the number of training iterations in current epoch, reset to 0 every epoch
        epoch_start_time = time.time()  # timer for entire epoch

        if epoch == train_opt.n_epochs_joint + 1:
            model.optimizer_G.param_groups[0]['lr'] = train_opt.lr_decay
            print(f'Learning rate changed to {train_opt.lr_decay}')
            
        if epoch == train_opt.n_epochs_joint + train_opt.n_epochs_decay + 1:
            # print('Fine-tuning stage begins!')
            model.optimizer_G.param_groups[0]['lr'] = train_opt.lr_fine
            print(f'Learning rate changed to {train_opt.lr_fine}')

        #train
        model.opt.isTrain = True # 训练时是否adaptive
        train_one_epoch(model, train_dataset, epoch, epoch_iter, total_iters, train_opt)
        if total_iters % train_opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
            print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
            save_suffix = 'iter_%d' % total_iters if train_opt.save_by_iter else 'latest'
            model.save_networks(save_suffix)
        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, total_epoch, time.time() - epoch_start_time))
        #val
        # model.update_temp()
        # print(f'Update temperature to {model.temp}')

        model.opt.isTrain = False   #是否adaptive
        start_time = time.time()
        mean_loss, mean_psnr = val_one_epoch(model, test_dataloader, epoch, total_epoch, train_opt)

        f = open(path + '/' + 'val.json', 'a')
        print(mean_psnr,file=f)
        print('End of val %d / %d \t Time Taken: %d sec' % (epoch, total_epoch, time.time() - start_time))
        writer.add_scalar(tag='ValLoss', scalar_value = mean_loss, global_step=epoch)
        writer.add_scalar(tag='ValPSNR', scalar_value = mean_psnr, global_step=epoch)
        writer.add_scalar(tag='lr', scalar_value = model.optimizer_G.param_groups[0]['lr'], global_step=epoch)
        f.close()
        
        is_best = mean_psnr < best_loss
        best_loss = min(mean_loss, best_loss)

        save_best_checkpoint(
            {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": model.optimizer_G.state_dict(),
            },
            is_best,
            path,
            'latest',
            epoch
            )

        #save model
        if epoch % train_opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))

            save_checkpoint(
            {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": model.optimizer_G.state_dict(),
            }, 
            path,
            epoch
            )

             

if __name__ == "__main__":
    # Extract the options
    train_opt = TrainOptions().parse()
    main(train_opt)