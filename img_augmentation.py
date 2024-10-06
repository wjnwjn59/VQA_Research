from torchvision import transforms

def augment_image(img_pil, n_img_augmentations=2):
    augmented_imgs = [] 
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=1), # Flips the image horizontally with a fixed probability.
        transforms.RandomVerticalFlip(p=1), # Flips the image vertically with a fixed probability.
        transforms.RandomRotation(degrees=90) # Rotates the image by a given degree.
    ])
    for _ in range(n_img_augmentations):
        # Apply the transformation to the input image and store the result.
        augmented_img = transform(img_pil)
        augmented_imgs.append(augmented_img)

    return augmented_imgs