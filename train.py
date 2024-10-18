import os

# Set environment variables for CUDA devices and world size for distributed training
os.environ["CUDA_VISIBLE_DEVICES"] = '3,0'
os.environ["WORLD_SIZE"] = '2'

# Load environment variables from a .env file
from dotenv import load_dotenv
load_dotenv()

# Retrieve the WANDB API key from environment variables
WANDB_API_KEY = os.getenv('WANDB_API_KEY')

import time
import random
import wandb  # Weights and Biases for experiment tracking
import numpy as np
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader 
from tqdm import tqdm  # Progress bar for loops

# Import custom modules for configuration and model components
from config import pipeline_config
from vqa_datasets import get_dataset
from text_encoder import load_text_encoder
from img_encoder import load_img_encoder
from scheduler_vqa_model import ViVQAModel
from utils import get_label_encoder
from eval import evaluate, compute_accuracy, compute_cider

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def set_seed(random_seed):
    """
    Set the random seed for reproducibility across different libraries.
    
    Parameters:
    - random_seed: int, seed value to ensure reproducibility
    """
    
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


def save_model(save_path, model, optimizer, scaler):
    """
    Save the model, optimizer state, and scaler to the specified path.
    
    Parameters:
    - save_path: str, path to save the model checkpoint
    - model: nn.Module, the model to save
    - optimizer: optimizer, the optimizer state to save
    - scaler: scaler, the gradient scaler for mixed precision training
    """
    
    try:
        model_state_dict = model.module.state_dict()
    except AttributeError:
        model_state_dict = model.state_dict()

    # Create a checkpoint dictionary
    checkpoint = {
        'model': model_state_dict,
        'optimizer': optimizer.state_dict(),
        'scaler': scaler.state_dict()
    }
    
    # Save the checkpoint
    torch.save(checkpoint, save_path)
    

def free_vram(model, optimizer, scaler):
    """
    Free up VRAM by deleting model, optimizer, and scaler objects, and clearing cache.
    
    Parameters:
    - model: nn.Module, the model to delete
    - optimizer: optimizer, the optimizer to delete
    - scaler: scaler, the scaler to delete
    """
    
    del model
    del optimizer
    del scaler
    torch.cuda.empty_cache()


# def train(model, 
#           train_loader, 
#           val_loader, 
#           epochs, 
#           criterion, 
#           optimizer, 
#           scaler,
#           dataset_name,  
#           idx2label, 
#           patience=5,
#           save_best_path='./weights/best.pt',
#           is_log_result=True,
#           is_multi_gpus=True,
#           use_amp=True):
    
#     """
#     Train the model with the specified parameters and evaluate on validation set.
    
#     Parameters:
#     - model: nn.Module, the model to train
#     - train_loader: DataLoader, DataLoader for training dataset
#     - val_loader: DataLoader, DataLoader for validation dataset
#     - epochs: int, number of epochs to train
#     - criterion: loss function, to calculate loss
#     - optimizer: optimizer, to update model weights
#     - scaler: scaler, for mixed precision training
#     - dataset_name: str, dataset name to determine evaluation metric (accuracy or CIDEr)
#     - idx2label: dict, mapping of index to label for text-based evaluation (for CIDEr)
#     - patience: int, number of epochs to wait for improvement before early stopping
#     - save_best_path: str, path to save the best model
#     - is_log_result: bool, whether to log results to Weights and Biases
#     - is_multi_gpus: bool, whether to use multiple GPUs
#     - use_amp: bool, whether to use automatic mixed precision

#     Returns:
#     - train_loss_lst: list, training losses for each epoch
#     - train_acc_lst: list, training accuracies (None for `openvivqa`)
#     - train_cider_lst: list, training CIDEr scores (None for non-`openvivqa`)
#     - val_loss_lst: list, validation losses for each epoch
#     - val_acc_lst: list, validation accuracies (None for `openvivqa`)
#     - val_cider_lst: list, validation CIDEr scores (None for non-`openvivqa`)
#     """
    
#     best_val_metric = -np.inf  # Initialize the best validation metric (CIDEr or accuracy)
#     epochs_no_improve = 0  # Counter for epochs without improvement
    
#     train_loss_lst = []
#     train_acc_lst = []  
#     train_cider_lst = []  
#     val_loss_lst = []
#     val_acc_lst = []  
#     val_cider_lst = []  

#     for epoch in range(epochs):
#         if is_multi_gpus:
#             model.module.update_epoch(epoch)  # Update epoch for DataParallel model
#         else:
#             model.update_epoch(epoch)  # Update epoch for single GPU model
            
#         total_correct = 0
#         total_loss = 0
#         total_samples = 0
#         all_predictions = [] 
#         all_references = []   

#         # Progress bar for training batches
#         epoch_iterator = tqdm(train_loader, 
#                               desc=f'Epoch {epoch + 1}/{epochs}', 
#                               unit='batch')
        
#         model.train()  # Set model to training mode
#         for batch in epoch_iterator:
#             # Extract data from batch
#             text_inputs_lst = batch.pop('text_inputs_lst')
#             img_inputs_lst = batch.pop('img_inputs')
#             labels = batch.pop('labels')
            
#             # Move inputs to device
#             text_inputs_lst = [
#                 {k: v.squeeze().to(device, non_blocking=True) for k, v in input_ids.items()} \
#                     for input_ids in text_inputs_lst]
#             img_inputs_lst = [inputs.to(device, non_blocking=True) for inputs in img_inputs_lst]
#             labels = labels.to(device, non_blocking=True)

#             # Forward pass with mixed precision
#             with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
#                 logits = model(text_inputs_lst, img_inputs_lst)
#                 loss = criterion(logits, labels)

#             _, preds = torch.max(logits, 1)

#             total_batch_samples = labels.size(0)
#             batch_loss_sum = loss.item() * total_batch_samples

#             total_samples += total_batch_samples
#             total_loss += batch_loss_sum

#             if dataset_name == 'openvivqa':
#                 pred_texts = [idx2label[pred.item()] for pred in preds]
#                 label_texts = [idx2label[label.item()] for label in labels]

#                 all_predictions += pred_texts
#                 all_references += label_texts
#             else: 
#                 correct = (preds == labels).sum().item()
#                 total_correct += correct

#             # Backward pass and optimization
#             scaler.scale(loss).backward()
#             scaler.step(optimizer)
#             scaler.update()
#             optimizer.zero_grad() 

#             epoch_iterator.set_postfix({'Batch Loss': loss.item()})  # Update progress bar

#         val_loss, val_acc, val_cider = evaluate(model=model, 
#                                                 val_loader=val_loader, 
#                                                 criterion=criterion, 
#                                                 idx2label=idx2label, 
#                                                 dataset_name=dataset_name)

#         # Compute average training loss, accuracy, and CIDEr
#         train_loss = total_loss / total_samples

#         # Print the progress for the current epoch
#         print(f'EPOCH {epoch + 1}: Train loss: {train_loss:.4f}\tVal loss: {val_loss:.4f}')

#         if dataset_name == 'openvivqa':
#             train_acc = -1 
#             train_cider = compute_cider(all_predictions, all_references)
#             val_metric = val_cider
#             print(f'Train CIDEr: {train_cider:.4f}\tVal CIDEr: {val_cider:.4f}')
#         else:
#             train_acc = total_correct / total_samples
#             train_cider = -1
#             val_metric = val_acc
#             print(f'Train acc: {train_acc:.4f}\tVal acc: {val_acc:.4f}')

#         print(f'val_metric: {val_metric}')

#         # Append to lists
#         train_loss_lst.append(train_loss)
#         train_acc_lst.append(train_acc)
#         train_cider_lst.append(train_cider)
#         val_loss_lst.append(val_loss)
#         val_acc_lst.append(val_acc)
#         val_cider_lst.append(val_cider)

#         # Log results to Weights and Biases
#         if is_log_result:
#             log_data = {
#                 'epoch': epoch + 1,
#                 'train_loss': train_loss,
#                 'train_acc': train_acc,
#                 'train_cider': train_cider,
#                 'val_loss': val_loss,
#                 'val_acc': val_acc,
#                 'val_cider': val_cider
#             }

#             wandb.log(log_data)

#         if val_metric > best_val_metric:
#             best_val_metric = val_metric
#             epochs_no_improve = 0  # Reset no improvement counter
#             save_model(save_best_path, model, optimizer, scaler)  # Save the best model
#         else:
#             epochs_no_improve += 1
#             if epochs_no_improve >= patience:  # Early stopping condition
#                 print(f'Early stopping triggered after {epochs_no_improve} epochs without improvement.')
#                 break  # Exit training loop

#     return train_loss_lst, train_acc_lst, train_cider_lst, val_loss_lst, val_acc_lst, val_cider_lst


def train(model, 
          train_loader, 
          val_loader, 
          epochs, 
          criterion, 
          optimizer, 
          scaler,
          dataset_name,  
          idx2label, 
          patience=5,
          save_best_path='./weights/best.pt',
          is_log_result=True,
          is_multi_gpus=True,
          use_amp=True):
    
    """
    Train the model with the specified parameters and evaluate on validation set.
    
    Parameters:
    - model: nn.Module, the model to train
    - train_loader: DataLoader, DataLoader for training dataset
    - val_loader: DataLoader, DataLoader for validation dataset
    - epochs: int, number of epochs to train
    - criterion: loss function, to calculate loss
    - optimizer: optimizer, to update model weights
    - scaler: scaler, for mixed precision training
    - dataset_name: str, dataset name to determine evaluation metric (accuracy or CIDEr)
    - idx2label: dict, mapping of index to label for text-based evaluation (for CIDEr)
    - patience: int, number of epochs to wait for improvement before early stopping
    - save_best_path: str, path to save the best model
    - is_log_result: bool, whether to log results to Weights and Biases
    - is_multi_gpus: bool, whether to use multiple GPUs
    - use_amp: bool, whether to use automatic mixed precision

    Returns:
    - train_loss_lst: list, training losses for each epoch
    - train_acc_lst: list, training accuracies (None for `openvivqa`)
    - train_cider_lst: list, training CIDEr scores (None for non-`openvivqa`)
    - val_loss_lst: list, validation losses for each epoch
    - val_acc_lst: list, validation accuracies (None for `openvivqa`)
    - val_cider_lst: list, validation CIDEr scores (None for non-`openvivqa`)
    """
    
    best_val_loss = np.inf  # Initialize the best validation loss
    epochs_no_improve = 0  # Counter for epochs without improvement
    
    train_loss_lst = []
    train_acc_lst = []  
    train_cider_lst = []  
    val_loss_lst = []
    val_acc_lst = []  
    val_cider_lst = []  

    for epoch in range(epochs):
        if is_multi_gpus:
            model.module.update_epoch(epoch)  # Update epoch for DataParallel model
        else:
            model.update_epoch(epoch)  # Update epoch for single GPU model
            
        total_correct = 0
        total_loss = 0
        total_samples = 0
        all_predictions = [] 
        all_references = []   

        # Progress bar for training batches
        epoch_iterator = tqdm(train_loader, 
                              desc=f'Epoch {epoch + 1}/{epochs}', 
                              unit='batch')
        
        model.train()  # Set model to training mode
        for batch in epoch_iterator:
            # Extract data from batch
            text_inputs_lst = batch.pop('text_inputs_lst')
            img_inputs_lst = batch.pop('img_inputs')
            labels = batch.pop('labels')
            
            # Move inputs to device
            text_inputs_lst = [
                {k: v.squeeze().to(device, non_blocking=True) for k, v in input_ids.items()} \
                    for input_ids in text_inputs_lst]
            img_inputs_lst = [inputs.to(device, non_blocking=True) for inputs in img_inputs_lst]
            labels = labels.to(device, non_blocking=True)

            # Forward pass with mixed precision
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                logits = model(text_inputs_lst, img_inputs_lst)
                loss = criterion(logits, labels)

            _, preds = torch.max(logits, 1)

            total_batch_samples = labels.size(0)
            batch_loss_sum = loss.item() * total_batch_samples

            total_samples += total_batch_samples
            total_loss += batch_loss_sum

            if dataset_name == 'openvivqa':
                pred_texts = [idx2label[pred.item()] for pred in preds]
                label_texts = [idx2label[label.item()] for label in labels]

                all_predictions += pred_texts
                all_references += label_texts
            else: 
                correct = (preds == labels).sum().item()
                total_correct += correct

            # Backward pass and optimization
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad() 

            epoch_iterator.set_postfix({'Batch Loss': loss.item()})  # Update progress bar

        val_loss, val_acc, val_cider = evaluate(model=model, 
                                                val_loader=val_loader, 
                                                criterion=criterion, 
                                                idx2label=idx2label, 
                                                dataset_name=dataset_name)

        # Compute average training loss, accuracy, and CIDEr
        train_loss = total_loss / total_samples

        # Print the progress for the current epoch
        print(f'EPOCH {epoch + 1}: Train loss: {train_loss:.4f}\tVal loss: {val_loss:.4f}')

        if dataset_name == 'openvivqa':
            train_acc = -1 
            train_cider = compute_cider(all_predictions, all_references)
            print(f'Train CIDEr: {train_cider:.4f}\tVal CIDEr: {val_cider:.4f}')
        else:
            train_acc = total_correct / total_samples
            train_cider = -1
            print(f'Train acc: {train_acc:.4f}\tVal acc: {val_acc:.4f}')

        # Append to lists
        train_loss_lst.append(train_loss)
        train_acc_lst.append(train_acc)
        train_cider_lst.append(train_cider)
        val_loss_lst.append(val_loss)
        val_acc_lst.append(val_acc)
        val_cider_lst.append(val_cider)

        # Log results to Weights and Biases
        if is_log_result:
            log_data = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'train_cider': train_cider,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_cider': val_cider
            }

            wandb.log(log_data)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0  # Reset no improvement counter
            save_model(save_best_path, model, optimizer, scaler)  # Save the best model
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:  # Early stopping condition
                print(f'Early stopping triggered after {epochs_no_improve} epochs without improvement.')
                break  # Exit training loop

    return train_loss_lst, train_acc_lst, train_cider_lst, val_loss_lst, val_acc_lst, val_cider_lst


# Function to parse command-line arguments for the training script
def parse_args():
    parser = argparse.ArgumentParser(description='ViVQA Training Script')
    parser.add_argument('--seed', type=int, default=pipeline_config.seed, help='Random seed')
    parser.add_argument('--gpus', type=str, default='0,1', help='Indices of GPUs used count from 0')
    parser.add_argument('--project_name', type=str, default='vivqa_paraphrase_augmentation', help='Project name for wandb')
    parser.add_argument('--exp_name', type=str, help='Experiment name for wandb')
    parser.add_argument('--dataset_name', default=pipeline_config.dataset_name, type=str, help='Name of the dataset')
    parser.add_argument('--data_dir', default=pipeline_config.data_dir, type=str, help='Dataset directory')
    parser.add_argument('--learning_rate', type=float, default=pipeline_config.learning_rate, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=pipeline_config.epochs, help='Number of epochs')
    parser.add_argument('--train_batch_size', type=int, default=pipeline_config.train_batch_size, help='Training batch size')
    parser.add_argument('--test_batch_size', type=int, default=pipeline_config.test_batch_size, help='Validation batch size')
    parser.add_argument('--hidden_dim', type=int, default=pipeline_config.hidden_dim, help='Hidden dimension')
    parser.add_argument('--projection_dim', type=int, default=pipeline_config.projection_dim, help='Projection dimension')
    parser.add_argument('--weight_decay', type=float, default=pipeline_config.weight_decay, help='Weight decay')
    parser.add_argument('--patience', type=int, default=pipeline_config.patience, help='Patience for early stopping')
    parser.add_argument('--text_encoder_id', type=str, default=pipeline_config.text_encoder_id, help='Text encoder ID')
    parser.add_argument('--img_encoder_id', type=str, default=pipeline_config.img_encoder_id, help='Image encoder ID')
    parser.add_argument('--is_text_augment', type=lambda x: (str(x).lower() == 'true'), default=pipeline_config.is_text_augment, help='Augment with text paraphrases')
    parser.add_argument('--n_text_paras', type=int, default=pipeline_config.n_text_paras, help='Number of paraphrases')
    parser.add_argument('--text_para_thresh', type=float, default=pipeline_config.text_para_thresh, help='Paraphrase threshold')
    parser.add_argument('--n_text_para_pool', type=int, default=pipeline_config.n_text_para_pool, help='The number of paraphrase in the paraphrase pool')
    parser.add_argument('--is_img_augment', type=lambda x: (str(x).lower() == 'true'), default=pipeline_config.is_img_augment, help='Augment with img geometric shift')
    parser.add_argument('--n_img_augments', type=int, default=pipeline_config.n_img_augments, help='Number of image augments')
    parser.add_argument('--img_augment_thresh', type=float, default=pipeline_config.img_augment_thresh, help='Image augmentation threshold')
    parser.add_argument('--use_dynamic_thresh', type=lambda x: (str(x).lower() == 'true'), default=pipeline_config.use_dynamic_thresh, help='Use dynamic threshold scaled by epochs')
    parser.add_argument('--save_ckpt_dir', type=str, default='runs/train', help='Directory to save checkpoints')
    parser.add_argument('--is_log_result', type=lambda x: (str(x).lower() == 'true'), default=True, help='Log training and eval results to wandb')
    parser.add_argument('--use_amp', type=lambda x: (str(x).lower() == 'true'), default=True, help='Use mixed precision training')
    
    return parser.parse_args()


# Main function for executing the training script
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

    # Initialize the ViVQA model with various parameters
    model = ViVQAModel(projection_dim=args.projection_dim,
                       hidden_dim=args.hidden_dim,
                       answer_space_len=answer_space_len,
                       text_encoder_dict=text_encoder_dict, 
                       img_encoder_dict=img_encoder_dict,
                       is_text_augment=args.is_text_augment, 
                       is_img_augment=args.is_img_augment,
                       total_epochs=args.epochs,
                       use_dynamic_thresh=args.use_dynamic_thresh,
                       text_para_thresh=args.text_para_thresh,
                       img_augment_thresh=args.img_augment_thresh)

    # Check if multiple GPUs are available and set up DataParallel if necessary
    is_multi_gpus = len(args.gpus.split(',')) > 1
    if is_multi_gpus:
        model = nn.DataParallel(model,
                                device_ids=list(map(int, args.gpus.split(','))))  # Wrap model in DataParallel

    # Compile the model for optimization
    model = torch.compile(model, mode='default') 
    model = model.to(device)
        
    # Set up the optimizer   
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.learning_rate,
                                  weight_decay=args.weight_decay)

    # Set up the loss function
    criterion = nn.CrossEntropyLoss()
    
    # Initialize the GradScaler for mixed precision training
    scaler = torch.amp.GradScaler(enabled=args.use_amp)

    # Load training dataset
    train_dataset = get_dataset(text_encoder_dict=text_encoder_dict,
                                img_encoder_dict=img_encoder_dict,
                                label_encoder=label2idx,
                                is_train=True, 
                                **vars(args))
    
    # Load validation dataset
    test_dataset = get_dataset(text_encoder_dict=text_encoder_dict,
                               img_encoder_dict=img_encoder_dict,
                               label_encoder=label2idx,
                               is_train=False, 
                               **vars(args))

    # Create DataLoader for training set
    train_loader = DataLoader(train_dataset,
                              batch_size=args.train_batch_size,
                              pin_memory=True,
                            #   num_workers=1,
                            #   multiprocessing_context='spawn',
                              shuffle=True)

    # Create DataLoader for validation set
    test_loader = DataLoader(test_dataset,
                             batch_size=args.test_batch_size,
                             pin_memory=True,
                             shuffle=False)

    # Generate experiment name for logging
    if not args.exp_name:
        text_augment_info = f'istextaug{args.is_text_augment}_ntextpara{args.n_text_paras}_random{args.text_para_thresh}_nparapool{args.n_text_para_pool}'
        img_augment_info = f'isimgaug{args.is_img_augment}_nimgaug{args.n_img_augments}_random{args.img_augment_thresh}'
        # text_augment_info = f'istextaug{args.is_text_augment}'
        # img_augment_info = f'isimgaug{args.is_img_augment}'
        #exp_name = f'phase6_seed{args.seed}_{args.dataset_name}_curr{args.use_dynamic_thresh}&0.6_{text_augment_info}_{img_augment_info}'
        exp_name = f'phase5_seed{args.seed}_{args.dataset_name}_{text_augment_info}_{img_augment_info}'
    else:
        exp_name = args.exp_name

    # Initialize wandb for logging results if enabled
    if args.is_log_result:
        wandb.init(
            project=args.project_name,
            name=exp_name,
            config=pipeline_config.__dict__)
    
    # Create a directory to save model weights
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    weights_dirname= f'{args.save_ckpt_dir}/weights_{timestamp}'

    os.makedirs(weights_dirname, exist_ok=True)  # Create the directory if it doesn't exist
    save_best_path = f'./{weights_dirname}/{exp_name}_best.pt'  # Path for saving the best model

    # Training
    train_loss_lst, train_acc_lst, train_cider_lst, val_loss_lst, val_acc_lst, val_cider_lst = train(model, 
                                                                                                     train_loader, 
                                                                                                     test_loader, 
                                                                                                     epochs=args.epochs, 
                                                                                                     criterion=criterion, 
                                                                                                     optimizer=optimizer, 
                                                                                                     scaler=scaler,
                                                                                                     dataset_name=args.dataset_name, 
                                                                                                     idx2label=idx2label,
                                                                                                     patience=args.patience,
                                                                                                     save_best_path=save_best_path,
                                                                                                     is_log_result=args.is_log_result,
                                                                                                     is_multi_gpus=is_multi_gpus,
                                                                                                     use_amp=args.use_amp)
    
    free_vram(model, optimizer, scaler)  # Free VRAM to avoid memory overflow

    # Initialize the best model for evaluation
    best_model = ViVQAModel(projection_dim=args.projection_dim,
                            hidden_dim=args.hidden_dim,
                            answer_space_len=answer_space_len,
                            text_encoder_dict=text_encoder_dict, 
                            img_encoder_dict=img_encoder_dict,
                            is_text_augment=False, 
                            is_img_augment=False).to(device)

    # Load the best model's state from the saved checkpoint
    dev = torch.cuda.current_device()  # Get current CUDA device
    checkpoint = torch.load(save_best_path,
                            weights_only=True,
                            map_location = lambda storage, loc: storage.cuda(dev))

    best_model.load_state_dict(checkpoint['model'])  # Load the model state

    # Evaluate the model on the test set
    test_loss, test_acc, test_cider = evaluate(model=best_model,
                                               val_loader=test_loader,
                                               criterion=criterion,
                                               idx2label=idx2label,
                                               dataset_name=args.dataset_name)
    
    test_loss = round(test_loss, 4)
    test_acc = round(test_acc, 4) if test_acc > 0 else test_acc 
    test_cider = round(test_cider, 4) if test_cider > 0 else test_cider

    args_dict = vars(args)  # Convert arguments to a dictionary for logging

    # Log the test results to wandb if enabled
    if args.is_log_result:
        exp_table = wandb.Table(
            columns=list(args_dict.keys()) + ['test_loss', 'test_acc', 'test_cider'], 
            data=[list(args_dict.values()) + [test_loss, test_acc, test_cider]]
        )
        wandb.log({"Exp_table": exp_table})  # Log the experiment results

    # Print the final test results
    print(f'Test loss: {test_loss}\tTest acc: {test_acc}')
    

if __name__ == '__main__':
    main()