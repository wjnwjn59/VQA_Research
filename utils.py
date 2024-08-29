import random
import shutil
import pandas as pd
import os

def generate_answer_space_file(data_dir):
    train_csv_path = os.path.join(data_dir, 'ViVQA', 'train.csv')
    test_csv_path = os.path.join(data_dir, 'ViVQA', 'test.csv')

    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)
    
    train_answers = train_df['answer'].unique().tolist()
    test_answers = test_df['answer'].unique().tolist()

    print(train_answers)
    print(test_answers)

    answer_space = set(list(train_answers + test_answers))

    save_path = os.path.join(data_dir, 'ViVQA', 'answer_space.txt')
    with open(save_path, 'w+') as f:
        f.write('\n'.join(answer_space))

def get_label_encoder(data_dir):
    save_path = os.path.join(data_dir, 'ViVQA', 'answer_space.txt')
    with open(save_path, 'r') as f:
        lines = f.read().splitlines()

    label2idx = {label: idx for idx, label in enumerate(lines)}
    idx2label = {idx: label for idx, label in enumerate(lines)}
    answer_space_len = len(lines)

    return label2idx, idx2label, answer_space_len

def copy_and_rename_images(source_folder, destination_folder):
    os.makedirs(destination_folder, exist_ok=True)

    for filename in os.listdir(source_folder):
        source_path = os.path.join(source_folder, filename)
        
        if not os.path.isfile(source_path):
            continue

        new_filename = filename.split('_')[-1]
        
        destination_path = os.path.join(destination_folder, new_filename)

        shutil.copy2(source_path, destination_path)
        print(f"Copied and renamed: {filename} -> {new_filename}")


if __name__ == '__main__':
    #generate_answer_space_file('/home/VLAI/datasets')
    source_folder = '/home/VLAI/datasets/COCO_Images/val2014'
    destination_folder = '/home/VLAI/datasets/COCO_Images/merge'
    copy_and_rename_images(source_folder, destination_folder)