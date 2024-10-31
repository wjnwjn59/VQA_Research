from torchvision import transforms
from PIL import Image
import os

import matplotlib.pyplot as plt

def augment_image_merge(aug_imgs_path):
    return [Image.open(os.path.join(aug_imgs_path))]


# Function return all augmented images in a folder as a list of PIL images
def augment_image_multi_unique(ori_img, aug_imgs_folderpath, max_aug_imgs=3):
    aug_imgs = [] 

    ori_img = ori_img.convert('RGB')
    
    transform = transforms.Compose([
        transforms.RandomCrop((128, 128)),
        transforms.Resize(ori_img.size),
    ])

    for img_file in os.listdir(aug_imgs_folderpath):
        img_file_path = os.path.join(aug_imgs_folderpath, img_file)
        img_pil = Image.open(img_file_path).convert('RGB')  # Convert to RGB to ensure 3 channels
        img_pil = img_pil.resize(ori_img.size)
        aug_imgs.append(img_pil)
                
        max_aug_imgs -= 1
        if max_aug_imgs <= 0: break
    
    while max_aug_imgs > 0:
        augmented_img = transform(ori_img)
        aug_imgs.append(augmented_img)
        max_aug_imgs -= 1

    return aug_imgs
