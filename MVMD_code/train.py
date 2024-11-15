import datetime
import os
import torch
from torch import nn, optim
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import joint_transforms
from config import training_root, validation_root
from dataset.Dataset import DataLoad
from misc import AvgMeter, check_mkdir
from networks.MVMD_network import MVMD_Network
from torch.optim.lr_scheduler import StepLR
import math
import time
from losses import lovasz_hinge, mse_loss_per_image
import numpy as np
import matplotlib.pyplot as plt
import argparse
import importlib
from utils import backup_code


def setup_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def initialize():
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        num_gpus = torch.cuda.device_count()
        print("Number of available GPUs:", num_gpus)
        for i in range(num_gpus):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("No CUDA GPUs are available.")
    cudnn.deterministic = True
    cudnn.benchmark = False


def create_dataloader():
    joint_transform = joint_transforms.Compose([
        joint_transforms.RandomHorizontallyFlip(),
        joint_transforms.Resize((args['scale'], args['scale']))
    ])
    val_joint_transform = joint_transforms.Compose([
        joint_transforms.Resize((args['scale'], args['scale']))
    ])
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    target_transform = transforms.ToTensor()

    print('=====>Dataset loading<======')
    train_set = DataLoad([training_root], joint_transform, img_transform, target_transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, drop_last=True, num_workers=cmd_args.num_workers,
                              shuffle=True)

    val_set = DataLoad([validation_root], val_joint_transform, img_transform, target_transform)
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=cmd_args.num_workers, shuffle=False)

    return train_loader, val_loader


def main():
    print("NEWEST MODEL: NO DEPTH, USE MIUNS NET IN REFINE  *use only coarse mask*")
    global train_loader, val_loader, args, batch_size, cmd_args, exp_name, model_name, gpu_ids, ckpt_path, log_path, val_log_path, ce_loss

    ckpt_path = './models'

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='MVMD_network', help='exp name')
    parser.add_argument('--model', type=str, default='MVMD_network', help='model name')
    parser.add_argument('--gpu', type=str, default='0', help='used gpu id')
    parser.add_argument('--batchsize', type=int, default=10, help='train batch')
    parser.add_argument('--bestonly', action="store_true", help='only best model')
    parser.add_argument('--num_workers', type=int, default=2, help='number of workers')

    cmd_args = parser.parse_args()
    exp_name = cmd_args.exp
    model_name = cmd_args.model
    gpu_ids = cmd_args.gpu
    train_batch_size = cmd_args.batchsize

    MVMD_file = importlib.import_module('networks.' + model_name)
    MVMD_Network = MVMD_file.MVMD_Network

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids

    args = {
        'max_epoch': 25,
        'last_iter': 0,
        'finetune_lr': 5e-5, # 1e-4
        'scratch_lr': 5e-4, # 1e-3
        'weight_decay': 5e-4, # 5e-4
        'momentum': 0.9,
        'snapshot': '',
        'scale': 416,
        'multi-scale': None,
        'fp16': False,
        'warm_up_epochs': 3,
        'seed': 2020
    }

    setup_seed(args['seed'])

    if len(gpu_ids.split(',')) > 1:
        batch_size = train_batch_size * len(gpu_ids.split(','))
    else:
        torch.cuda.set_device(0)
        batch_size = train_batch_size

    train_loader, val_loader = create_dataloader()

    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = os.path.join(ckpt_path, exp_name, current_time + '.txt')
    val_log_path = os.path.join(ckpt_path, exp_name, 'val_log_' + current_time + '.txt')

    ce_loss = nn.CrossEntropyLoss()

    print('=====>Prepare Network {}<======'.format(exp_name))
    if len(gpu_ids.split(',')) > 1:
        net = torch.nn.DataParallel(MVMD_Network()).cuda().train()
        for name, param in net.named_parameters():
            if 'backbone' in name:
                print(name)
        params = [
            {"params": [param for name, param in net.named_parameters() if 'backbone' in name],
             "lr": args['finetune_lr']},
            {"params": [param for name, param in net.named_parameters() if 'backbone' not in name],
             "lr": args['scratch_lr']}
        ]
    else:
        net = MVMD_Network().cuda().train()
        params = [
            {"params": [param for name, param in net.named_parameters() if 'backbone' in name],
             "lr": args['finetune_lr']},
            {"params": [param for name, param in net.named_parameters() if 'backbone' not in name],
             "lr": args['scratch_lr']}
        ]

    optimizer = optim.Adam(params, betas=(0.9, 0.99), eps=6e-8, weight_decay=args['weight_decay'])
    warm_up_with_cosine_lr = lambda epoch: epoch / args['warm_up_epochs'] if epoch <= args['warm_up_epochs'] else 0.5 * \
                            (math.cos((epoch -args['warm_up_epochs']) / (args['max_epoch'] -args['warm_up_epochs']) * math.pi) + 1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_cosine_lr)

    check_mkdir(ckpt_path)
    check_mkdir(os.path.join(ckpt_path, exp_name))

    train(net, optimizer, scheduler)

def train_save_masks(pred_maskc, pred_maskf, gt_mask, epoch, iteration, output_dir="./train_output"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    predc_path = os.path.join(output_dir, f"pre_mask_{epoch}_{iteration}_c.png")
    predf_path = os.path.join(output_dir, f"pre_mask_{epoch}_{iteration}_f.png")
    gt_path = os.path.join(output_dir, f"gt_mask_{epoch}_{iteration}.png")

    predc_np = pred_maskc.cpu().detach().numpy()
    predf_np = pred_maskf.cpu().detach().numpy()
    gt_np = gt_mask.cpu().detach().numpy()

    plt.imsave(predc_path, predc_np, cmap='gray')
    plt.imsave(predf_path, predf_np, cmap='gray')
    plt.imsave(gt_path, gt_np, cmap='gray')

def train(net, optimizer, scheduler):
    curr_epoch = 1
    curr_iter = 1
    start = 0
    best_mae = 100.0

    print('=====>Start training<======')
    while True:
        loss_record1, loss_record2, loss_record3, loss_record4 = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
        loss_record5, loss_record6, loss_record7, loss_record8 = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
        train_iterator = tqdm(train_loader, total=len(train_loader))
        # train_iterator = tqdm(train_loader, desc=f'Epoch: {curr_epoch}', ncols=100, ascii=' =', bar_format='{l_bar}{bar}|')
        # tqdm(train_loader, total=len(train_loader))
        for i, sample in enumerate(train_iterator):
            image1, image1_mask = sample['image1'].cuda(), sample['image1_mask'].cuda()
            image2 = sample['image2'].cuda()
            image3 = sample['image3'].cuda()

            optimizer.zero_grad()

            mask_c, mask_f = net(image1, image2, image3)


            # MSE loss
            loss_mse1 = mse_loss_per_image(mask_c.squeeze(), image1_mask.squeeze())
            loss_mse2 = mse_loss_per_image(mask_f.squeeze(), image1_mask.squeeze())

            loss = loss_mse1 + 2 * loss_mse2

            loss.backward()

            torch.nn.utils.clip_grad_norm_(net.parameters(), 12)  # gradient clip
            optimizer.step()  # change gradient
            
            loss_record1.update(loss.item(), image1.size(0))
            loss_record2.update(loss_mse1.item(), batch_size)
            loss_record3.update(loss_mse2.item(), batch_size)

            train_iterator.set_description(f'Epoch: {curr_epoch} | Loss: {loss_record1.avg:.4f}')

            curr_iter += 1

            log = "epochs:%d, iter: %d, mask_coarse: %f5, mask_fine: %f5, lr: %f8"%\
                  (curr_epoch, curr_iter, loss_record2.avg, loss_record3.avg, scheduler.get_last_lr()[0])

            if (curr_iter-1) % 20 == 0:
                elapsed = (time.perf_counter() - start)
                start = time.perf_counter()
                log_time = log + ' [time {}]'.format(elapsed)
                print(log_time)
            open(log_path, 'a').write(log + '\n')

        if curr_epoch % 1 == 0 and not cmd_args.bestonly:
            if len(gpu_ids.split(',')) > 1:
                checkpoint = {
                        'model': net.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                }
                torch.save(checkpoint, os.path.join(ckpt_path, exp_name, f'{curr_epoch}.pth'))
            else:
                checkpoint = {
                        'model': net.state_dict(),
                        'optimizer': optimizer.state_dict(),
                }
                torch.save(checkpoint, os.path.join(ckpt_path, exp_name, f'{curr_epoch}.pth'))


        current_mae = val(net, curr_epoch)

        net.train() # val -> train
        if current_mae < best_mae:
            best_mae = current_mae
            if len(gpu_ids.split(',')) > 1:
                checkpoint = {
                        'model': net.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                }
            else:
                checkpoint = {
                        'model': net.state_dict(),
                        'optimizer': optimizer.state_dict(),
                }
            torch.save(checkpoint, os.path.join(ckpt_path, exp_name, 'best_mae.pth'))



        if curr_epoch > args['max_epoch']:
            return
        curr_epoch += 1
        scheduler.step()  # change learning rate after epoch


def val_save_masks(pred_mask_c, pred_mask_f, gt_mask, epoch, iteration, output_dir="./val_output"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    pred_path_c = os.path.join(output_dir, f"pre_mask_{epoch}_{iteration}_c.png")
    pred_path_f = os.path.join(output_dir, f"pre_mask_{epoch}_{iteration}_f.png")
    gt_path = os.path.join(output_dir, f"gt_mask_{epoch}_{iteration}.png")

    pred_np_c = pred_mask_c.cpu().detach().numpy()
    pred_np_f = pred_mask_f.cpu().detach().numpy()
    gt_np = gt_mask.cpu().detach().numpy()

    plt.imsave(pred_path_c, pred_np_c, cmap='gray')
    plt.imsave(pred_path_f, pred_np_f, cmap='gray')
    plt.imsave(gt_path, gt_np, cmap='gray')


def val(net, epoch):
    mae_record = AvgMeter()
    net.eval()
    with torch.no_grad():
        val_iterator = tqdm(val_loader)
        for i, sample in enumerate(val_iterator):
            image1, image1_mask = sample['image1'].cuda(), sample['image1_mask'].cuda()
            image2 = sample['image2'].cuda()
            image3 = sample['image3'].cuda()

            mask_c, mask_f = net(image1, image2, image3)

            res_c = (mask_c.data > 0).to(torch.float32).squeeze(0)
            res = (mask_f.data > 0).to(torch.float32).squeeze(0)
            mae = torch.mean(torch.abs(res - image1_mask.squeeze(0)))

            batch_size = mask_f.size(0)
            mae_record.update(mae.item(), batch_size)

            pred_mask_c = res_c.squeeze(1)
            pred_mask_f = res.squeeze(1)
            gt_mask = image1_mask.squeeze(1)

            if i % 1 == 0 and epoch != 1:
                val_save_masks(pred_mask_c[0, :, :], pred_mask_f[0, :, :],gt_mask[0, :, :], epoch, i)

        log = "val: iter: %d, mae: %f5" % (epoch, mae_record.avg)
        print(log)
        open(val_log_path, 'a').write(log + '\n')
        return mae_record.avg


if __name__ == '__main__':
    initialize()
    main()
