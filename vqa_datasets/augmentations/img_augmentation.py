from torchvision import transforms
from PIL import Image
import os

import matplotlib.pyplot as plt

def augment_image(img_pil, n_img_augmentations=1):
    augmented_imgs = [] 
    # transform = transforms.Compose([
    #     # transforms.RandomHorizontalFlip(p=0.5), # Flips the image horizontally with a fixed probability.
    #     #transforms.RandomVerticalFlip(p=1), # Flips the image vertically with a fixed probability.
    #     # transforms.RandomPerspective(p=0.3, p=0.5),
    #     # transforms.RandomRotation(degrees=10) # Rotates the image by a given degree.
    # ])
    # for _ in range(n_img_augmentations):
    #     # Apply the transformation to the input image and store the result.
    #     augmented_img = transform(img_pil)
    #     augmented_imgs.append(augmented_img)
    cropper = transforms.RandomCrop(size=(64, 64))
    augmented_imgs = [cropper(img_pil) for _ in range(4)]

    return augmented_imgs


# os.makedirs('augmented_images', exist_ok=True)
# img_path = '/home/VLAI/datasets/OpenViVQA/dev-images/000000003757.jpg'
# img_pils = Image.open(img_path).convert('RGB')
# augmented_imgs_pil = augment_image(img_pils, 5)
# idx = 0
# for img_pil in augmented_imgs_pil:
#     img_pil.save(f'augmented_images/augmented_image_{idx}.png')
#     idx += 1

