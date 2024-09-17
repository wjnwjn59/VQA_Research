import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2,3'
os.environ["WORLD_SIZE"] = '2'

from dotenv import load_dotenv

load_dotenv()

WANDB_API_KEY = os.getenv('WANDB_API_KEY')

import time
import random
import wandb
import numpy as np
import argparse
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from tqdm import tqdm

from config import pipeline_config
from torch_dataset import ViVQADataset
from text_encoder import load_text_encoder
from img_encoder import load_img_encoder
from vqa_model import ViVQAModel
from utils import get_label_encoder

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def set_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED']= str(random_seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ['CUDNN_DETERMINISTIC'] = '1'

def save_model(save_path, model):
    try:
        state_dict = model.module.state_dict()
    except AttributeError:
        state_dict = model.state_dict()

    torch.save(state_dict, save_path)

def free_vram(model, optimizer):
    del model
    del optimizer
    torch.cuda.empty_cache()

def compute_accuracy(logits, labels):
    _, preds = torch.max(logits, 1)
    correct = (preds == labels).sum().item()
    accuracy = correct / logits.size(0)

    return accuracy

def evaluate(model, val_loader, criterion):
    model.eval()
    eval_losses = []
    eval_accs = []
    with torch.no_grad():
        for idx, batch in enumerate(val_loader):
            text_inputs_lst = batch.pop('text_inputs_lst')
            img_inputs = batch.pop('img_inputs')
            labels = batch.pop('labels')

            logits = model(text_inputs_lst, img_inputs)

            loss = criterion(logits, labels)
            acc = compute_accuracy(logits, labels)

            eval_losses.append(loss.item())
            eval_accs.append(acc)

    eval_loss = sum(eval_losses) / len(eval_losses)
    eval_acc = sum(eval_accs) / len(eval_accs)

    return eval_loss, eval_acc


def train(model, 
          train_loader, 
          val_loader, 
          epochs, 
          criterion, 
          optimizer, 
          #scheduler,
          patience=5,
          save_best_path='./weights/best.pt'):
    
    best_val_loss = np.inf
    epochs_no_improve = 0
    
    train_loss_lst = []
    train_acc_lst = []
    val_loss_lst = []
    val_acc_lst = []
    for epoch in range(epochs):
        train_batch_loss_lst = []
        train_batch_acc_lst = []

        epoch_iterator = tqdm(train_loader, 
                              desc=f'Epoch {epoch + 1}/{epochs}', 
                              unit='batch')
        model.train()
        for batch in epoch_iterator:
            text_inputs_lst = batch.pop('text_inputs_lst')
            img_inputs = batch.pop('img_inputs')
            labels = batch.pop('labels')

            logits = model(text_inputs_lst, img_inputs)

            loss = criterion(logits, labels)
            acc = compute_accuracy(logits, labels)

            train_batch_loss_lst.append(loss.item())
            train_batch_acc_lst.append(acc)

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            epoch_iterator.set_postfix({'Batch Loss': loss.item()})

        # scheduler.step()

        val_loss, val_acc = evaluate(model,
                                     val_loader,
                                     criterion)

        train_loss = sum(train_batch_loss_lst) / len(train_batch_loss_lst)
        train_acc = sum(train_batch_acc_lst) / len(train_batch_acc_lst)

        train_loss_lst.append(train_loss)
        train_acc_lst.append(train_acc)
        val_loss_lst.append(val_loss)
        val_acc_lst.append(val_acc)

        wandb.log({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        })

        print(f'EPOCH {epoch + 1}: Train loss: {train_loss:.4f}\tTrain acc: {train_acc:.4f}\tVal loss: {val_loss:.4f}\tVal acc: {val_acc:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            save_model(save_best_path, model)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f'Early stopping triggered after {epochs_no_improve} epochs without improvement.')
                break

    return train_loss_lst, train_acc_lst, val_loss_lst, val_acc_lst

def parse_args():
    parser = argparse.ArgumentParser(description='ViVQA Training Script')
    parser.add_argument('--seed', type=int, default=pipeline_config.seed, help='Random seed')
    parser.add_argument('--gpus', type=str, default='0,1', help='GPU ID. Default is GPU ID 0,1.')
    parser.add_argument('--project_name', type=str, default='vivqa_paraphrase_augmentation', help='Project name for wandb')
    parser.add_argument('--exp_name', type=str, help='Experiment name for wandb')
    parser.add_argument('--data_dir', default=pipeline_config.data_dir, type=str, help='Dataset directory')
    parser.add_argument('--learning_rate', type=float, default=pipeline_config.learning_rate, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=pipeline_config.epochs, help='Number of epochs')
    parser.add_argument('--train_batch_size', type=int, default=pipeline_config.train_batch_size, help='Training batch size')
    parser.add_argument('--test_batch_size', type=int, default=pipeline_config.test_batch_size, help='Validation batch size')
    parser.add_argument('--hidden_dim', type=int, default=pipeline_config.hidden_dim, help='Hidden dimension')
    parser.add_argument('--projection_dim', type=int, default=pipeline_config.projection_dim, help='Projection dimension')
    parser.add_argument('--weight_decay', type=float, default=pipeline_config.weight_decay, help='Weight decay')
    parser.add_argument('--patience', type=int, default=pipeline_config.patience, help='Patience for early stopping')
    parser.add_argument('--text_max_len', type=int, default=pipeline_config.text_max_len, help='Maximum text length')
    parser.add_argument('--fusion_strategy', type=str, default=pipeline_config.fusion_strategy, help='Fusion strategy')
    parser.add_argument('--text_encoder_id', type=str, default=pipeline_config.text_encoder_id, help='Text encoder ID')
    parser.add_argument('--img_encoder_id', type=str, default=pipeline_config.img_encoder_id, help='Image encoder ID')
    parser.add_argument('--paraphraser_id', type=str, default=pipeline_config.paraphraser_id, help='Paraphraser ID')
    parser.add_argument('--is_text_augment', type=lambda x: (str(x).lower() == 'true'), default=pipeline_config.is_text_augment, help='Augment with text paraphrases')
    parser.add_argument('--n_text_paras', type=int, default=pipeline_config.n_text_paras, help='Number of paraphrases')
    parser.add_argument('--text_para_thresh', type=float, default=pipeline_config.text_para_thresh, help='Paraphrase threshold')
    parser.add_argument('--n_text_para_pool', type=int, default=pipeline_config.n_text_para_pool, help='The number of paraphrase in the paraphrase pool')
    parser.add_argument('--is_img_augment', type=lambda x: (str(x).lower() == 'true'), default=pipeline_config.is_img_augment, help='Augment with img geometric shift')
    parser.add_argument('--n_img_augments', type=int, default=pipeline_config.n_text_paras, help='Number of image augments')
    parser.add_argument('--img_augment_thresh', type=float, default=pipeline_config.img_augment_thresh, help='Image augmentation threshold')
    parser.add_argument('--save_ckpt_dir', type=str, default='runs/train', help='Directory to save checkpoints')
    parser.add_argument('--is_evaluate', type=lambda x: (str(x).lower() == 'true'), default=True, help='Evaluate the performance of the model on test set')
    
    return parser.parse_args()

def main(): 
    args = parse_args()

    set_seed(args.seed)

    # if args.gpus:
    #     os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    #     os.environ["WORLD_SIZE"] = str(len(args.gpus.split(',')))

    if not args.exp_name:
        text_augment_info = f'istextaug{args.is_text_augment}_ntextpara{args.n_text_paras}_random{args.text_para_thresh}_nparapool{args.n_text_para_pool}'
        img_augment_info = f'isimgaug{args.is_img_augment}_nimgaug{args.n_img_augments}_random{args.img_augment_thresh}'
        exp_name = f'phase4_vivqa_{text_augment_info}_{img_augment_info}'
    else:
        exp_name = args.exp_name

    if args.text_encoder_id:
        text_encoder_dict = load_text_encoder(args.text_encoder_id)
    else:
        raise Exception('No text encoder specified!')

    if args.img_encoder_id:
        img_encoder_dict = load_img_encoder(args.img_encoder_id)
    else:
        raise Exception('No img encoder specified!')
    

    wandb.init(
        project=args.project_name,
        name=exp_name,
        config=pipeline_config.__dict__)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    weights_dirname= f'{args.save_ckpt_dir}/weights_{timestamp}'

    os.makedirs(weights_dirname, exist_ok=True)
    save_best_path = f'./{weights_dirname}/{exp_name}_best.pt'

    label2idx, idx2label, answer_space_len = get_label_encoder(args.data_dir)
        
    # Enable augmentation in training set
    train_dataset = ViVQADataset(data_dir=args.data_dir,
                                 data_mode='train',
                                 text_encoder_dict=text_encoder_dict,
                                 img_encoder_dict=img_encoder_dict, 
                                 label_encoder=label2idx,
                                 is_text_augment=args.is_text_augment,
                                 is_img_augment=args.is_img_augment,
                                 n_text_paras=args.n_text_paras,
                                 text_para_thresh=args.text_para_thresh,
                                 n_para_pool=args.n_text_para_pool,
                                 n_img_augments=args.n_img_augments,
                                 img_augment_thresh=args.img_augment_thresh)

    # Disable augmentation for all modalities in test set
    test_dataset = ViVQADataset(data_dir=args.data_dir,
                                data_mode='val',
                                text_encoder_dict=text_encoder_dict,
                                img_encoder_dict=img_encoder_dict, 
                                label_encoder=label2idx,
                                is_text_augment=False,
                                is_img_augment=False)

    train_loader = DataLoader(train_dataset,
                              batch_size=args.train_batch_size,
                              shuffle=True)

    test_loader = DataLoader(test_dataset,
                             batch_size=args.test_batch_size,
                             shuffle=False)

    model = ViVQAModel(projection_dim=args.projection_dim,
                       hidden_dim=args.hidden_dim,
                       answer_space_len=answer_space_len,
                       text_encoder_dict=text_encoder_dict, 
                       img_encoder_dict=img_encoder_dict,
                       is_text_augment=args.is_text_augment, 
                       is_img_augment=args.is_img_augment)
    
    if len(args.gpus.split(',')) > 1:
        model = nn.DataParallel(model,
                                device_ids=list(map(int, args.gpus.split(','))))
    model = model.to(device)
        
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    # step_size = EPOCHS * 0.4
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
    #                                             step_size=step_size, 
    #                                             gamma=0.1)
    criterion = nn.CrossEntropyLoss()


    train_loss_lst, train_acc_lst, val_loss_lst, val_acc_lst = train(model, 
                                                                     train_loader, 
                                                                     test_loader, 
                                                                     epochs=args.epochs, 
                                                                     criterion=criterion, 
                                                                     optimizer=optimizer, 
                                                                     #scheduler=scheduler,
                                                                     patience=args.patience,
                                                                     save_best_path=save_best_path)
    
    free_vram(model, optimizer)

    best_model = ViVQAModel(projection_dim=args.projection_dim,
                            hidden_dim=args.hidden_dim,
                            answer_space_len=answer_space_len,
                            text_encoder_dict=text_encoder_dict, 
                            img_encoder_dict=img_encoder_dict,
                            is_text_augment=False, 
                            is_img_augment=False).to(device)

    best_model.load_state_dict(torch.load(save_best_path))

    test_loss, test_acc = evaluate(model=best_model,
                                   val_loader=test_loader,
                                   criterion=criterion)
    test_loss, test_acc = round(test_loss, 4), round(test_acc, 4)

    args_dict = vars(args)

    exp_table = wandb.Table(
        columns=list(args_dict.keys()) + ['test_loss', 'test_acc'], 
        data=[list(args_dict.values()) + [test_loss, test_acc]]
    )
    wandb.log({"Exp_table": exp_table})

    print(f'Test loss: {test_loss}\tTest acc: {test_acc}')
    

if __name__ == '__main__':
    main()