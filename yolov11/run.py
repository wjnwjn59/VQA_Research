from ultralytics import YOLO
import pandas as pd
import argparse
import time
import torch
import random
import os
from PIL import Image 
import numpy as np
from tqdm import tqdm

# Set environment variables for CUDA devices and world size for distributed training
os.environ["CUDA_VISIBLE_DEVICES"] = '3'  # Specify which GPU to use
os.environ["WORLD_SIZE"] = '1'  # Set the world size for distributed training


# Select the device: CUDA if available, otherwise CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


# Set the random seed for reproducibility in all libraries (random, NumPy, PyTorch, CUDA).
def set_seed(random_seed):
    random.seed(random_seed)  # Set the random seed for the 'random' module
    np.random.seed(random_seed)  # Set the random seed for NumPy
    torch.manual_seed(random_seed)  # Set the seed for PyTorch CPU operations
    torch.cuda.manual_seed(random_seed)  # Set the seed for PyTorch CUDA operations (single GPU)
    torch.cuda.manual_seed_all(random_seed)  # Set the seed for PyTorch CUDA operations (multi-GPU)
    
    # Ensure deterministic behavior in cuDNN (for consistent training results)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    
    # Set environment variables for Python and CUDA to further enforce determinism
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ['CUDNN_DETERMINISTIC'] = '1'
    
    
# Function to predict and save bounding boxes 
def generate_aug_imgs(model, df, COCO_path, conf, root_save_path):
    # Create root directory
    if not os.path.exists(root_save_path):
        os.makedirs(root_save_path)
    
    # Iterate over images in the DataFrame
    for img_id in tqdm(df, desc="Processing images"):
        img_path = f'{img_id:012}.jpg'
        img = Image.open(os.path.join(COCO_path, img_path))
        results = model.predict(img, conf=conf)

        # Create directory to save bounding boxes if it does not exist
        folder_path = os.path.join(root_save_path, str(img_id))
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
                
        count = 0
        # Iterate over results to process bounding boxes
        for result in results:
            for box in result.boxes:
                # Extract coordinates for the bounding box
                x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
                
                # Crop the bounding box from the original image
                cropped_img = img.crop((x_min, y_min, x_max, y_max))
                
                save_path = os.path.join(folder_path, f'{img_id}_{count}.jpg')
                cropped_img.save(save_path)
                
                count += 1
        
    return True


# Function to predict and merge bbox into unique canvas, then save it
def generate_aug_imgs_unique(model, df, COCO_path, conf, root_save_path):
    # Create root directory
    if not os.path.exists(root_save_path):
        os.makedirs(root_save_path)
    
    # Iterate over images in the DataFrame
    for img_id in tqdm(df, desc="Processing images"):
        img_path = f'{img_id:012}.jpg'
        img = Image.open(os.path.join(COCO_path, img_path))
        print(img.mode)
        exit()
        results = model.predict(img, conf=conf)
                
        width, height = img.size
        canvas = Image.new('RGB', (width, height), (0, 0, 0))
                        
        count = 0
        # Iterate over results to process bounding boxes
        for result in results:
            for box in result.boxes:
                # Extract coordinates for the bounding box
                x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
                
                # Crop the bounding box from the original image
                cropped_img = img.crop((x_min, y_min, x_max, y_max))
                
                # Paste the cropped image onto the canvas
                canvas.paste(cropped_img, (x_min, y_min, x_max, y_max))
                
        save_path = os.path.join(root_save_path, f'{img_id}.jpg')
        canvas.save(save_path)
        
    return True


# Parse command line arguments for the script
def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate a new dataset")
    parser.add_argument("--train_filepath", type=str, required=True)
    parser.add_argument("--root_savepath", type=str, required=True)

    return parser.parse_args()


# Main function to drive the paraphrase generation process
def main():
    args = parse_arguments()
    train_filepath = args.train_filepath
    root_save_path = args.root_savepath

    print('Start processing...')
    start_time = time.time()
    set_seed(59)
    
    model = YOLO('yolo11x.pt')  # Load a pre-trained YOLOv11 nano model
    model = model.to(device)  # Move the model to the specified device (GPU or CPU)
    print('Model loaded!')
    
    df = pd.read_csv(train_filepath)  # Load the original dataset from a CSV file
    img_df = df['img_id']  # Initialize an empty DataFrame to store the paraphrased data
    
    result = generate_aug_imgs_unique(model=model, 
                                    df=img_df, 
                                    COCO_path="/home/VLAI/datasets/COCO_Images/merge", 
                                    conf=0.15, 
                                    root_save_path=root_save_path)
            
    if result:
        print('Generating augmentation images successfully!')
        print(f'Processing time: {time.time() - start_time}')
    else:
        print('Failed to generate augmentation images.')
    

if __name__ == "__main__":
    main()