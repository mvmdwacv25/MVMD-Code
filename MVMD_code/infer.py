import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import joint_transforms
from dataset.Dataset import DataLoad
import argparse
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import matplotlib.pyplot as plt
from networks.MVMD_network import MVMD_Network
from config import validation_root
import importlib
import sys

def create_dataloader():
    val_joint_transform = joint_transforms.Compose([
        joint_transforms.Resize((args['scale'], args['scale']))
    ])
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    target_transform = transforms.ToTensor()

    val_set = DataLoad([validation_root], val_joint_transform, img_transform, target_transform)
    val_loader = DataLoader(val_set, batch_size=1, num_workers=cmd_args.num_workers, shuffle=False)

    return val_loader

def infer(net, dataloader, output_dir="infer_output"):
    print("=================== start inference ================")
    net.eval()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with torch.no_grad():
        for i, sample in enumerate(dataloader):
            image1, image1_mask, image1_deptha, image1_depthb = sample['image1'].cuda(), sample['image1_mask'].cuda(), sample['depth_img1a'].cuda(), sample['depth_img1b'].cuda()
            image2 = sample['image2'].cuda()
            image3 = sample['image3'].cuda()

            mask_c, mask_f = net(image1, image2, image3)

            res = (mask_f.data > 0).to(torch.float32).squeeze(0)

            pred_mask = res.squeeze(1)
            gt_mask = image1_mask.squeeze(1)

            # Save or display the mask
            pred_path = os.path.join(output_dir, f"pred_mask_{i}.png")
            gt_path = os.path.join(output_dir, f"gt_mask_{i}.png")

            pred_np = pred_mask.cpu().detach().squeeze().numpy()
            gt_np = gt_mask.cpu().detach().squeeze().numpy()

            plt.imsave(pred_path, pred_np, cmap='gray')
            plt.imsave(gt_path, gt_np, cmap='gray')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='MVMD_network', help='exp name')
    parser.add_argument('--model', type=str, default='MVMD_network', help='model name')
    parser.add_argument('--gpu', type=str, default='0', help='used gpu id')
    parser.add_argument('--checkpoint', type=str, default='models/MVMD_network/best_mae.pth', help='path to the checkpoint')
    parser.add_argument('--num_workers', type=int, default=1, help='number of workers')

    cmd_args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = cmd_args.gpu

    MVMD_file = importlib.import_module('networks.' + cmd_args.model)
    MVMD_Network = MVMD_file.MVMD_Network

    net = MVMD_Network().cuda()
    checkpoint = torch.load(cmd_args.checkpoint)
    net.load_state_dict(checkpoint['model'])

    val_loader = create_dataloader()

    infer(net, val_loader)
