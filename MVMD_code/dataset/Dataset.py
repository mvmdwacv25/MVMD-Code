import os
import os.path
import re
import cv2
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import torch
import numpy as np
from networks.opencv_depth import depth_map

def linear_map(img):
    valid_mask = (img != -16)
    valid_values = img[valid_mask]

    valid_values_sorted = np.sort(valid_values)
    n = len(valid_values_sorted)

    min_index = int(n * 0.1)
    max_index = int(n * 0.9)

    min_val = valid_values_sorted[min_index]
    max_val = valid_values_sorted[max_index]

    img_mapped = np.zeros_like(img, dtype=np.float32)
    if max_val - min_val != 0:
        img_mapped[valid_mask] = (valid_values - min_val) / (max_val - min_val)

    img_mapped[img == -16] = -16
    return img_mapped

def align_images(img1, img2):
    img1 = np.array(img1)
    img2 = np.array(img2)
    # ??ORB?????????
    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

    # ??BFMatcher??????
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)

    # ??????
    matches = sorted(matches, key=lambda x: x.distance)

    # ?????
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 2)

    # ???????
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # ??????????????
    height, width = img1.shape[:2]
    img1_aligned = cv2.warpPerspective(img1, H, (width, height))

    return img1_aligned, img2, H

def listdirs_only(folder):
    return [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]


class DataLoad(data.Dataset):
    def __init__(self, root, joint_transform=None, img_transform=None, target_transform=None, interval=1):  # interval set here
        self.img_root, self.scene_root = self.split_root(root)
        self.joint_transform = joint_transform
        self.img_transform = img_transform
        self.target_transform = target_transform
        self.input_folder = 'JPEGImages'
        self.label_folder = 'SegmentationClassPNG'
        self.img_exts = ['.jpg', '.png']
        self.interval = interval
        # get all frames from scene datasets
        self.sceneImg_dict = self.generateImgFromScene(self.scene_root)


    def __getitem__(self, index):

        image_list = self.sceneImg_dict
        # Ensure wrap-around within the same folder
        triple_index = index % len(image_list)

        image1_path, image2_path, image3_path, image1_mask_path = image_list[triple_index]

        image1 = Image.open(image1_path).convert('RGB').resize((640, 480), Image.LANCZOS)
        image2 = Image.open(image2_path).convert('RGB').resize((640, 480), Image.LANCZOS)
        image3 = Image.open(image3_path).convert('RGB').resize((640, 480), Image.LANCZOS)
        image1_mask = Image.open(image1_mask_path).convert('L').resize((640, 480), Image.LANCZOS)

        image1 = transforms.ToTensor()(image1)
        image2 = transforms.ToTensor()(image2)
        image3 = transforms.ToTensor()(image3)
        image1_mask = transforms.ToTensor()(image1_mask)



        sample = {
            'image1': image1.squeeze(),
            'image1_mask': image1_mask.squeeze(),
            'image2': image2.squeeze(),
            'image3': image3.squeeze(),
        }
        return sample

    def generateImgFromScene(self, root):
        imgs = []
        root = root[0]
        scene_list = listdirs_only(root[0])
        for scene in scene_list:
            images_path = os.path.join(root[0], scene, self.input_folder)
            masks_path = os.path.join(root[0], scene, self.label_folder)
            image_files = sorted([f for f in os.listdir(images_path) if f.endswith(('.jpg', '.png'))])
    
            # Set interval based on the first character of the folder name
            interval = self.interval if scene[0].isdigit() else 1
    
            scene_imgs = []
            for i in range(0, len(image_files) - 2 * interval, interval):
                img1 = os.path.join(images_path, image_files[i])
                img2 = os.path.join(images_path, image_files[i + interval])
                img3 = os.path.join(images_path, image_files[i + 2 * interval])
    
                # Extract numeric part from img1 filename
                img_number = re.findall(r'\d+', os.path.splitext(image_files[i])[0])[0]
    
                # Find corresponding mask file
                mask1 = None
                for mask_file in os.listdir(masks_path):
                    mask_number = re.findall(r'\d+', os.path.splitext(mask_file)[0])
                    if mask_number and mask_number[0] == img_number:
                        mask1 = os.path.join(masks_path, mask_file)
                        break
    
                if not mask1:
                    raise FileNotFoundError(f"Mask file not found for {masks_path} {image_files[i]}")
    
                # Only add the trio if all three images exist
                if os.path.exists(img1) and os.path.exists(img2) and os.path.exists(img3):
                    scene_imgs.append((img1, img2, img3, mask1))
    
            imgs.append(scene_imgs)
    
        # Flatten the list of lists
        flat_imgs = [item for sublist in imgs for item in sublist]
    
        return flat_imgs

    def is_valid_image(self, filename):
        return any(filename.endswith(ext) for ext in self.img_exts)

    def get_ext(self, filename):
        for ext in self.img_exts:
            if filename.endswith(ext):
                return ext
        return '.png'

    def sortImg(self, img_list):
        img_int_list = [int(os.path.splitext(f)[0]) for f in img_list]
        sort_index = [i for i, v in sorted(enumerate(img_int_list), key=lambda x: x[1])]  # sort img to 001,002,003...
        return [img_list[i] for i in sort_index]

    def split_root(self, root):
        if not isinstance(root, list):
            raise TypeError('root should be a list')
        img_root_list = []
        scene_root_list = []
        for tmp in root:
            if tmp[1] == 'image':
                img_root_list.append(tmp)
            elif tmp[1] == 'scene':
                scene_root_list.append(tmp)
            else:
                raise TypeError('you should input scene or image')
        return img_root_list, scene_root_list

    def __len__(self):
        return len(self.sceneImg_dict)

