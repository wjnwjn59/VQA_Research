from torchvision import transforms

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


    