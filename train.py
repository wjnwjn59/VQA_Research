from torch.utils.data import DataLoader
from eval import evaluate, compute_accuracy, compute_cider
from utils import get_label_encoder
from lightweight_vqa_model import ViVQAModel
from img_encoder import load_img_encoder
from text_encoder import load_text_encoder
from vqa_datasets import get_dataset
from config import pipeline_config
from tqdm import tqdm
import torch.nn as nn
import torch
import argparse
import numpy as np
import wandb
import random
import time
from dotenv import load_dotenv
import os

load_dotenv()
WANDB_API_KEY = os.getenv('WANDB_API_KEY')

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def set_seed(random_seed):
    # Set seed for random module
    random.seed(random_seed)
    # Set seed for random module
    np.random.seed(random_seed)
    # Set seed for PyTorch
    torch.manual_seed(random_seed)
    # Set seed for current CUDA device
    torch.cuda.manual_seed(random_seed)
    # Set seed for all CUDA devices
    torch.cuda.manual_seed_all(random_seed)
    # Enable deterministic mode
    torch.backends.cudnn.deterministic = True
    # Enforce deterministic behavior
    torch.use_deterministic_algorithms(True)
    # Disable benchmark for reproducibility
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    # Set CUBLAS workspace config
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    # Ensure cuDNN is deterministic
    os.environ['CUDNN_DETERMINISTIC'] = '1'


def set_threads(n_threads, is_print_available_threads=False):
    if is_print_available_threads:
        num_threads = torch.get_num_threads()
        print(f"Current number of threads: {num_threads}")

    torch.set_num_threads(n_threads)
    torch.set_num_interop_threads(n_threads)


def save_model(save_path, model):
    try:
        model_state_dict = model.module.state_dict()
    except AttributeError:
        model_state_dict = model.state_dict()

    # Create a checkpoint dictionary
    checkpoint = {
        'model': model_state_dict,
    }

    torch.save(checkpoint, save_path)


def free_vram(model, optimizer, scaler):
    del model
    del optimizer
    del scaler
    torch.cuda.empty_cache()


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
          use_amp=True):

    best_val_loss = np.inf
    epochs_no_improve = 0

    train_loss_lst = []
    train_acc_lst = []
    train_cider_lst = []
    val_loss_lst = []
    val_acc_lst = []
    val_cider_lst = []

    for epoch in range(epochs):
        model.update_epoch(epoch)

        total_correct = 0
        total_loss = 0
        total_samples = 0
        all_predictions = []
        all_references = []

        epoch_iterator = tqdm(train_loader,
                              desc=f'Epoch {epoch + 1}/{epochs}',
                              unit='batch')

        model.train()
        for batch in epoch_iterator:
            text_inputs_lst = batch.pop('text_inputs_lst')
            img_inputs_lst = batch.pop('img_inputs')
            labels = batch.pop('labels')

            text_inputs_lst = [
                {k: v.squeeze().to(device, non_blocking=True)
                 for k, v in input_ids.items()}
                for input_ids in text_inputs_lst]
            img_inputs_lst = [inputs.to(device, non_blocking=True)
                              for inputs in img_inputs_lst]
            labels = labels.to(device, non_blocking=True)

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

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            epoch_iterator.set_postfix(
                {'Batch Loss': loss.item()})

        val_loss, val_acc, val_cider = evaluate(model=model,
                                                val_loader=val_loader,
                                                criterion=criterion,
                                                idx2label=idx2label,
                                                dataset_name=dataset_name)

        train_loss = total_loss / total_samples

        print(
            f'EPOCH {epoch + 1}: Train loss: {train_loss:.4f}\tVal loss: {val_loss:.4f}')

        if dataset_name == 'openvivqa':
            train_acc = -1
            train_cider = compute_cider(all_predictions, all_references)
            print(
                f'Train CIDEr: {train_cider:.4f}\tVal CIDEr: {val_cider:.4f}')
        else:
            train_acc = total_correct / total_samples
            train_cider = -1
            print(f'Train acc: {train_acc:.4f}\tVal acc: {val_acc:.4f}')

        train_loss_lst.append(train_loss)
        train_acc_lst.append(train_acc)
        train_cider_lst.append(train_cider)
        val_loss_lst.append(val_loss)
        val_acc_lst.append(val_acc)
        val_cider_lst.append(val_cider)

        model.update_loss(val_loss)

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
            epochs_no_improve = 0
            save_model(save_best_path, model, optimizer,
                       scaler)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(
                    f'Early stopping triggered after {epochs_no_improve} epochs without improvement.')
                break

    return train_loss_lst, train_acc_lst, train_cider_lst, val_loss_lst, val_acc_lst, val_cider_lst


def parse_args():
    parser = argparse.ArgumentParser(description='ViVQA Training Script')
    parser.add_argument('--seed', type=int,
                        default=pipeline_config.seed, help='Random seed')
    parser.add_argument('--gpus', type=str, default='0',
                        help='Indices of GPUs used count from 0')
    parser.add_argument('--project_name', type=str,
                        default='vivqa_paraphrase_augmentation_new', help='Project name for wandb')
    parser.add_argument('--exp_name', type=str,
                        help='Experiment name for wandb')
    parser.add_argument('--dataset_name', default=pipeline_config.dataset_name,
                        type=str, help='Name of the dataset')
    parser.add_argument('--data_dir', default=pipeline_config.data_dir,
                        type=str, help='Dataset directory')
    parser.add_argument('--learning_rate', type=float,
                        default=pipeline_config.learning_rate, help='Learning rate')
    parser.add_argument('--epochs', type=int,
                        default=pipeline_config.epochs, help='Number of epochs')
    parser.add_argument('--train_batch_size', type=int,
                        default=pipeline_config.train_batch_size, help='Training batch size')
    parser.add_argument('--test_batch_size', type=int,
                        default=pipeline_config.test_batch_size, help='Validation batch size')
    parser.add_argument('--hidden_dim', type=int,
                        default=pipeline_config.hidden_dim, help='Hidden dimension')
    parser.add_argument('--projection_dim', type=int,
                        default=pipeline_config.projection_dim, help='Projection dimension')
    parser.add_argument('--weight_decay', type=float,
                        default=pipeline_config.weight_decay, help='Weight decay')
    parser.add_argument('--patience', type=int,
                        default=pipeline_config.patience, help='Patience for early stopping')
    parser.add_argument('--text_encoder_id', type=str,
                        default=pipeline_config.text_encoder_id, help='Text encoder ID')
    parser.add_argument('--img_encoder_id', type=str,
                        default=pipeline_config.img_encoder_id, help='Image encoder ID')
    parser.add_argument('--is_text_augment', type=lambda x: (str(x).lower() == 'true'),
                        default=pipeline_config.is_text_augment, help='Augment with text paraphrases')
    parser.add_argument('--n_text_paras', type=int,
                        default=pipeline_config.n_text_paras, help='Number of paraphrases')
    parser.add_argument('--text_para_thresh', type=float,
                        default=pipeline_config.text_para_thresh, help='Paraphrase threshold')
    parser.add_argument('--n_text_para_pool', type=int, default=pipeline_config.n_text_para_pool,
                        help='The number of paraphrase in the paraphrase pool')
    parser.add_argument('--use_dynamic_thresh', type=lambda x: (str(x).lower() == 'true'),
                        default=pipeline_config.use_dynamic_thresh, help='Use dynamic threshold scaled by epochs')
    parser.add_argument('--start_threshold', type=float,
                        default=pipeline_config.start_threshold, help='Start dynamic threshold value')
    parser.add_argument('--update_threshold_method', type=str, default='linear',
                        help='Update dynamic threshold with specified method')
    parser.add_argument('--save_ckpt_dir', type=str,
                        default='runs/train', help='Directory to save checkpoints')
    parser.add_argument('--is_log_result', type=lambda x: (str(x).lower() ==
                        'true'), default=True, help='Log training and eval results to wandb')
    parser.add_argument('--use_amp', type=lambda x: (str(x).lower()
                        == 'true'), default=True, help='Use mixed precision training')

    return parser.parse_args()


def main():
    args = parse_args()

    set_seed(args.seed)
    set_threads(4)

    if args.text_encoder_id:
        text_encoder_dict = load_text_encoder(args.text_encoder_id)
    else:
        raise Exception('No text encoder specified!')

    if args.img_encoder_id:
        img_encoder_dict = load_img_encoder(args.img_encoder_id)
    else:
        raise Exception('No img encoder specified!')

    label2idx, idx2label, answer_space_len = get_label_encoder(data_dir=args.data_dir,
                                                               dataset_name=args.dataset_name)

    model = ViVQAModel(projection_dim=args.projection_dim,
                       hidden_dim=args.hidden_dim,
                       answer_space_len=answer_space_len,
                       text_encoder_dict=text_encoder_dict,
                       img_encoder_dict=img_encoder_dict,
                       is_text_augment=args.is_text_augment,
                       total_epochs=args.epochs,
                       use_dynamic_thresh=args.use_dynamic_thresh,
                       start_threshold=args.start_threshold,
                       update_threshold_method=args.update_threshold_method,
                       text_para_thresh=args.text_para_thresh
                       )

    model = torch.compile(model, mode='default')
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.learning_rate,
                                  weight_decay=args.weight_decay)

    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(enabled=args.use_amp)

    train_dataset = get_dataset(text_encoder_dict=text_encoder_dict,
                                img_encoder_dict=img_encoder_dict,
                                label_encoder=label2idx,
                                is_train=True,
                                **vars(args))

    test_dataset = get_dataset(text_encoder_dict=text_encoder_dict,
                               img_encoder_dict=img_encoder_dict,
                               label_encoder=label2idx,
                               is_train=False,
                               **vars(args))

    train_loader = DataLoader(train_dataset,
                              batch_size=args.train_batch_size,
                              pin_memory=True,
                              #   num_workers=1,
                              #   multiprocessing_context='spawn',
                              shuffle=True)

    test_loader = DataLoader(test_dataset,
                             batch_size=args.test_batch_size,
                             pin_memory=True,
                             shuffle=False)

    if not args.exp_name:
        text_augment_info = f'istextaug{args.is_text_augment}_ntextpara{args.n_text_paras}_random{args.text_para_thresh}_nparapool{args.n_text_para_pool}'
        # text_augment_info = f'istextaug{args.is_text_augment}'
        # img_augment_info = f'isimgaug{args.is_img_augment}'
        # exp_name = f'phase6_seed{args.seed}_{args.dataset_name}_curr{args.use_dynamic_thresh}&0.6_{text_augment_info}_{img_augment_info}'
        exp_name = f'CrossAttention_ConfigSetting'
        # exp_name = f'norm_seed{args.seed}_{args.dataset_name}_{text_augment_info}'
    else:
        exp_name = args.exp_name

    if args.is_log_result:
        wandb.init(
            project=args.project_name,
            name=exp_name,
            config=pipeline_config.__dict__)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    weights_dirname = f'{args.save_ckpt_dir}/weights_{timestamp}'
    os.makedirs(weights_dirname, exist_ok=True)
    save_best_path = f'./{weights_dirname}/{exp_name}_best.pt'

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
                                                                                                     use_amp=args.use_amp)

    free_vram(model, optimizer, scaler)

    best_model = ViVQAModel(projection_dim=args.projection_dim,
                            hidden_dim=args.hidden_dim,
                            answer_space_len=answer_space_len,
                            text_encoder_dict=text_encoder_dict,
                            img_encoder_dict=img_encoder_dict,
                            is_text_augment=args.is_text_augment).to(device)

    dev = torch.cuda.current_device()
    checkpoint = torch.load(save_best_path,
                            weights_only=True,
                            map_location=lambda storage, loc: storage.cuda(dev))

    state_dict = checkpoint['model']
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("_orig_mod.", "")
        new_state_dict[new_key] = value

    best_model.load_state_dict(new_state_dict)

    test_loss, test_acc, test_cider = evaluate(model=best_model,
                                               val_loader=test_loader,
                                               criterion=criterion,
                                               idx2label=idx2label,
                                               dataset_name=args.dataset_name)

    test_loss = round(test_loss, 4)
    test_acc = round(test_acc, 4) if test_acc > 0 else test_acc
    test_cider = round(test_cider, 4) if test_cider > 0 else test_cider

    args_dict = vars(args)
    if args.is_log_result:
        exp_table = wandb.Table(
            columns=list(args_dict.keys()) +
            ['test_loss', 'test_acc', 'test_cider'],
            data=[list(args_dict.values()) + [test_loss, test_acc, test_cider]]
        )
        wandb.log({"Exp_table": exp_table})

    print(f'Test acc: {test_acc}\tTest loss: {test_loss}')


if __name__ == '__main__':
    main()
