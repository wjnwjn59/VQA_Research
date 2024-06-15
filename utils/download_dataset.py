from datasets import load_dataset

import json
import os
from pathlib import Path

DATASET_DIR = Path('../../datasets/InfographicsVQA')
INFOGRAPHIC_IMG_DIR = DATASET_DIR / 'inforgraphicsvqa_images'
TRAIN_JSON_PATH = DATASET_DIR / 'infographicsVQA_train_v1.0.json'
VAL_JSON_PATH = DATASET_DIR / 'infographicsVQA_val_v1.0_withQT.json'
TEST_JSON_PATH = DATASET_DIR / 'infographicsVQA_test_v1.0.json'

def read_dataset(json_path):
    with open(json_path, 'r') as f:
        data_dict = json.load(f)
        dataset_name = data_dict['dataset_name']
        dataset_version = data_dict['dataset_version']
        dataset_split_name = data_dict['dataset_split']
        dataset_data = data_dict['data']
        print(f'Dataset name: {dataset_name} - {dataset_split_name}')
        for idx in range(len(dataset_data)):
            sample_idx_dict = dataset_data[idx]
            sample_idx_question_id = sample_idx_dict['questionId']
            sample_idx_question = sample_idx_dict['question']
            sample_idx_img_local_name = sample_idx_dict['image_local_name']
            sample_idx_ocr_output_file = sample_idx_dict['ocr_output_file']
            if dataset_split_name in ['train', 'val']:
                sample_idx_answer_lst = sample_idx_dict['answers']
            print(sample_idx_dict)
            sample_idx_img_pil = Image.open(INFOGRAPHIC_IMG_DIR / sample_idx_img_local_name)
            display(sample_idx_img_pil)
            break

def main():
    pass

if __name__ == '__main__':
    main()