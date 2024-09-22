from torchvision import transforms

def augment_image(img_pil, n_img_augmentations=1):
    augmented_imgs = [] 
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=1),
        transforms.RandomVerticalFlip(p=1),
        transforms.RandomRotation(degrees=90)
    ])
    for _ in range(n_img_augmentations):
        augmented_img = transform(img_pil)
        augmented_imgs.append(augmented_img)

    return augmented_imgs