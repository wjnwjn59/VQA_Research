import os
import ast
import random
import torch
import pandas as pd

from PIL import Image
from torch.utils.data import Dataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class ViVQADataset(Dataset):
    def __init__(self, data_dir, data_mode, text_encoder_dict, img_encoder_dict,
                 label_encoder=None, is_text_augment=True,
                 n_text_paras=2, text_para_thresh=0.5, n_para_pool=30
                 ):
        self.data_dir = data_dir
        self.data_mode = data_mode

        self.is_text_augment = is_text_augment
        self.n_text_paras = n_text_paras
        self.text_para_thresh = text_para_thresh

        if self.data_mode == 'train':
            train_filename = f'{n_para_pool}_paraphrases_train.csv'

            data_path = os.path.join(data_dir, 'ViVQA', train_filename)
            if not os.path.exists(data_path):
                print(
                    'Data training file with number of paraphrases pool not found! Select default (20) file.')
                data_path = os.path.join(
                    data_dir, 'ViVQA', '30_paraphrases_train.csv')
            self.data_path = data_path
        else:
            self.data_path = os.path.join(data_dir, 'ViVQA', 'test.csv')

        self.img_dirpath = os.path.join(data_dir, 'COCO_Images', 'merge')
        self.text_encoder_dict = text_encoder_dict
        self.img_encoder_dict = img_encoder_dict
        self.device = device

        self.questions, self.para_questions, self.img_paths, self.answers = self.get_data()
        self.label_encoder = label_encoder

    def get_data(self):
        df = pd.read_csv(self.data_path, index_col=0)
        questions = []
        para_questions = []
        answers = []
        img_paths = []

        for _, row in df.iterrows():
            question = row['question']
            answer = row['answer']
            img_id = row['img_id']
            # question_type = row['type'] # 0: object, 1: color, 2: how many, 3: where

            if self.data_mode == 'train' and self.is_text_augment:
                question_paraphrases = row['question_paraphrase']
                para_questions.append(question_paraphrases)

            img_name = f'{img_id:012}.jpg'
            img_path = os.path.join(self.img_dirpath, img_name)

            questions.append(question)
            answers.append(answer)
            img_paths.append(img_path)

        return questions, para_questions, img_paths, answers

    def __getitem__(self, idx):
        questions = self.questions[idx]
        answers = self.answers[idx]
        img_paths = self.img_paths[idx]

        img_pils = Image.open(img_paths).convert('RGB')
        label = self.label_encoder[answers]

        img_inputs_lst = [self.img_encoder_dict['img_processor'](img_pils)]
        text_inputs_lst = [self.text_encoder_dict['text_processor'](questions)]

        if self.data_mode == 'train' and self.is_text_augment:
            para_questions = self.para_questions[idx]
            para_questions = ast.literal_eval(para_questions)
            selected_para_questions = random.sample(
                para_questions, self.n_text_paras)
            paraphrase_inputs_lst = [self.text_encoder_dict['text_processor'](
                text) for text in selected_para_questions]

            text_inputs_lst += paraphrase_inputs_lst

        labels = torch.tensor(label, dtype=torch.long)

        data_outputs = {
            'text_inputs_lst': text_inputs_lst,
            'img_inputs': img_inputs_lst,
            'labels': labels
        }

        return data_outputs

    def __len__(self):
        return len(self.questions)
