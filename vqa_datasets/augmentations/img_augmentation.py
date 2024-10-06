from torchvision import transforms
from PIL import Image
import os

import matplotlib.pyplot as plt

def augment_image(img_pil, n_img_augmentations=1):
    augmented_imgs = [] 
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=1),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=45),
        # transforms.RandomPosterize(bits=3),
        transforms.RandomPerspective(distortion_scale=0.1, p=0.2)
    ])
    for _ in range(n_img_augmentations):
        augmented_img = transform(img_pil)
        augmented_imgs.append(augmented_img)

    return augmented_imgs

# os.makedirs('augmented_images', exist_ok=True)
# img_path = '/home/VLAI/datasets/OpenViVQA/dev-images/000000003757.jpg'
# img_pils = Image.open(img_path).convert('RGB')
# augmented_imgs_pil = augment_image(img_pils, 5)
# idx = 0
# for img_pil in augmented_imgs_pil:
#     img_pil.save(f'augmented_images/augmented_image_{idx}.png')
#     idx += 1