import os
import sys

# Add path to parent folder
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

# Set environment variables for CUDA devices and world size for distributed training
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
os.environ["WORLD_SIZE"] = '1'

# Load environment variables from a .env file
from dotenv import load_dotenv
load_dotenv()

# Retrieve the WANDB API key from environment variables
WANDB_API_KEY = os.getenv('WANDB_API_KEY')

import time
import random
import numpy as np
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm  
from matplotlib import pyplot as plt

from kd_train.teacher_config import pipeline_config
from vqa_datasets import get_dataset
from text_encoder import load_text_encoder
from img_encoder import load_img_encoder
from kd_train.kd_scheduler_vqa_model import ViVQAModel
from utils import get_label_encoder

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def set_seed(random_seed):    
    random.seed(random_seed)                                # Set seed for random module
    np.random.seed(random_seed)                             # Set seed for random module
    torch.manual_seed(random_seed)                          # Set seed for PyTorch
    torch.cuda.manual_seed(random_seed)                     # Set seed for current CUDA device
    torch.cuda.manual_seed_all(random_seed)                 # Set seed for all CUDA devices
    torch.backends.cudnn.deterministic = True               # Enable deterministic mode
    torch.use_deterministic_algorithms(True)                # Enforce deterministic behavior
    torch.backends.cudnn.benchmark = False                  # Disable benchmark for reproducibility
    os.environ['PYTHONHASHSEED']= str(random_seed)          # Set hash seed
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"       # Set CUBLAS workspace config
    os.environ['CUDNN_DETERMINISTIC'] = '1'                 # Ensure cuDNN is deterministic


def set_threads(n_threads, is_print_available_threads=False):
    """
    Set the number of threads for PyTorch and optionally print the current number of threads.
    
    Parameters:
    - n_threads: int, number of threads to set
    - is_print_available_threads: bool, whether to print current number of threads
    """
    
    if is_print_available_threads:
        num_threads = torch.get_num_threads()  # Get the current number of threads
        print(f"Current number of threads: {num_threads}")
        
    torch.set_num_threads(n_threads)  # Set the number of threads
    torch.set_num_interop_threads(n_threads)  # Set interop threads for parallelism
    
    
def save_failcases(model1, model2, text_encoder_dict, test_loader, idx2label, bothfail_path='./bothfailcases', model1fail_path='./model1failcases', model2fail_path='./model2failcases'):
    model1.eval()
    model2.eval()
    
    os.makedirs(bothfail_path, exist_ok=True)
    os.makedirs(model1fail_path, exist_ok=True)
    os.makedirs(model2fail_path, exist_ok=True)
    
    batch_count = 0
    test = 10
    for i, batch in enumerate(tqdm(test_loader, desc=f'Checking {batch_count}/{len(test_loader)}', unit='batch')):
        batch_count = i
        with torch.no_grad():
            text_inputs_lst = batch.pop('text_inputs_lst')
            img_inputs_lst = batch.pop('img_inputs')
            labels = batch.pop('labels')
            img_ids = batch.pop('img_ids')
            
            text_inputs_lst = [
                {k: v.squeeze().to(device, non_blocking=True) for k, v in input_ids.items()} \
                    for input_ids in text_inputs_lst]
            img_inputs_lst = [inputs.to(device, non_blocking=True) for inputs in img_inputs_lst]
            labels = labels.to(device, non_blocking=True)
            
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=True):
                logits1 = model1(text_inputs_lst, img_inputs_lst)
                logits2 = model2(text_inputs_lst, img_inputs_lst)
                
            _, preds1 = torch.max(logits1, 1)
            _, preds2 = torch.max(logits2, 1)
            questions_decode = text_encoder_dict['decode'](text_inputs_lst[0]['input_ids'])
                
            for i, samples in enumerate(zip(preds1, preds2, questions_decode, img_ids)):
                pred1 = samples[0].item()
                pred2 = samples[1].item()
                ques = samples[2]
                img_id = samples[3]
                label = labels[i].item()
                
                res = [pred == label for pred in [pred1, pred2]]
                
                if res == [True, True]:
                    continue
                elif res == [False, False]:
                    print("case1: both fail")
                    save_path = os.path.join(bothfail_path, f'{batch_count}_{i}.jpg')
                elif res == [False, True]:
                    print("case2: model1 fail")
                    save_path = os.path.join(model1fail_path, f'{batch_count}_{i}.jpg')
                else:
                    print("case3: model2 fail")
                    save_path = os.path.join(model2fail_path, f'{batch_count}_{i}.jpg')

                img_name = f"{int(img_id.split('.')[0]):012}.jpg"
                img = plt.imread(f'/home/VLAI/datasets/COCO_Images/merge/{img_name}')
                
                plt.imshow(img)
                plt.axis('off')
                plt.title(f"Question: {ques}\nGround Truth: {idx2label[label]}\nPrediction\nModel1: {idx2label[pred1]}\nModel2: {idx2label[pred2]}")
                plt.savefig(f'{save_path}', dpi=300, bbox_inches='tight')
                plt.close()
                
                if test == 0:
                    exit()
                test -= 1
                                                
                    
def parse_args():
    parser = argparse.ArgumentParser(description='ViVQA Training Script')
    parser.add_argument('--seed', type=int, default=pipeline_config.seed, help='Random seed')
    parser.add_argument('--gpus', type=str, default='0', help='Indices of GPUs used count from 0')
    parser.add_argument('--project_name', type=str, default='vivqa_paraphrase_augmentation_kd', help='Project name for wandb')
    parser.add_argument('--exp_name', type=str, help='Experiment name for wandb')
    parser.add_argument('--dataset_name', default=pipeline_config.dataset_name, type=str, help='Name of the dataset')
    parser.add_argument('--data_dir', default=pipeline_config.data_dir, type=str, help='Dataset directory')
    parser.add_argument('--learning_rate', type=float, default=pipeline_config.learning_rate, help='Learning rate')
    parser.add_argument('--test_batch_size', type=int, default=pipeline_config.test_batch_size, help='Validation batch size')
    parser.add_argument('--hidden_dim', type=int, default=pipeline_config.hidden_dim, help='Hidden dimension')
    parser.add_argument('--projection_dim', type=int, default=pipeline_config.projection_dim, help='Projection dimension')
    parser.add_argument('--text_encoder_id', type=str, default=pipeline_config.text_encoder_id, help='Text encoder ID')
    parser.add_argument('--img_encoder_id', type=str, default=pipeline_config.img_encoder_id, help='Image encoder ID')
    parser.add_argument('--use_amp', type=lambda x: (str(x).lower() == 'true'), default=True, help='Use mixed precision training')
    parser.add_argument('--model_path1', type=str, default=None, help='Path to the model checkpoint') 
    parser.add_argument('--model_path2', type=str, default=None, help='Path to the model checkpoint') 
    
    return parser.parse_args()


def main(): 
    args = parse_args()  # Parse command-line arguments
    
    set_seed(args.seed)  # Set random seed for reproducibility
    set_threads(4)  # Set the number of CPU threads for training

    # Load the text encoder based on the provided ID
    if args.text_encoder_id:
        text_encoder_dict = load_text_encoder(args.text_encoder_id)
    else:
        raise Exception('No text encoder specified!')

    # Load the image encoder based on the provided ID
    if args.img_encoder_id:
        img_encoder_dict = load_img_encoder(args.img_encoder_id)
    else:
        raise Exception('No img encoder specified!')
    
    # Get label encoders for converting labels to indices
    label2idx, idx2label, answer_space_len = get_label_encoder(data_dir=args.data_dir, 
                                                               dataset_name=args.dataset_name)
    
    # Load validation dataset
    test_dataset = get_dataset(text_encoder_dict=text_encoder_dict,
                               img_encoder_dict=img_encoder_dict,
                               label_encoder=label2idx,
                               is_train=False, 
                               **vars(args))

    # Create DataLoader for validation set
    test_loader = DataLoader(test_dataset,
                             batch_size=args.test_batch_size,
                             pin_memory=True,
                             shuffle=False)


    # Initialize the best model for evaluation
    model1 = ViVQAModel(projection_dim=args.projection_dim,
                        hidden_dim=args.hidden_dim,
                        answer_space_len=answer_space_len,
                        text_encoder_dict=text_encoder_dict, 
                        img_encoder_dict=img_encoder_dict,
                        is_text_augment=False, 
                        is_img_augment=False).to(device)
    
    model2 = ViVQAModel(projection_dim=args.projection_dim,
                        hidden_dim=args.hidden_dim,
                        answer_space_len=answer_space_len,
                        text_encoder_dict=text_encoder_dict, 
                        img_encoder_dict=img_encoder_dict,
                        is_text_augment=False, 
                        is_img_augment=False).to(device)
    
    
    dev = torch.cuda.current_device()  # Get current CUDA device
    checkpoint = torch.load(args.model_path1,
                            weights_only=True,
                            map_location = lambda storage, loc: storage.cuda(dev))
    
    # Remove prefix _orig_mod.
    state_dict = checkpoint['model']
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("_orig_mod.", "") 
        new_state_dict[new_key] = value
    
    model1.load_state_dict(new_state_dict) 
    
    
    checkpoint = torch.load(args.model_path2,
                            weights_only=True,
                            map_location = lambda storage, loc: storage.cuda(dev))
    
    state_dict = checkpoint['model']
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("_orig_mod.", "") 
        new_state_dict[new_key] = value
    
    model2.load_state_dict(new_state_dict) 
    
    
    save_failcases(
        model1=model1,
        model2=model2,
        text_encoder_dict=text_encoder_dict,
        test_loader=test_loader,
        idx2label=idx2label,
        bothfail_path='./bothfailcases',
        model1fail_path='./model1failcases',
        model2fail_path='./model2failcases'
    )
 
if __name__ == '__main__':
    main()