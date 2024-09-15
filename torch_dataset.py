import os
import ast
import random
import torch
import pandas as pd

from PIL import Image
from torch.utils.data import Dataset
from text_encoder import text_model
from config import pipeline_config
    
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# class ViVQADataset(Dataset):
#     def __init__(self, questions, img_pils, answers, 
#                  data_mode, text_processor, img_processor, 
#                  paraphraser, label_encoder=None):
#         self.questions = questions
#         self.img_pils = img_pils
#         self.answers = answers
#         self.data_mode = data_mode
#         self.text_processor = text_processor
#         self.img_processor = img_processor
#         self.paraphraser = paraphraser
#         self.label_encoder = label_encoder
#         self.device = device

#     def __getitem__(self, idx):
#         questions = self.questions[idx]
#         answers = self.answers[idx]
#         img_pils = self.img_pils[idx]

#         question_inputs = self.text_processor(questions)
#         img_inputs = self.img_processor(img_pils).to(device)
#         label = self.label_encoder[answers]
        
#         labels = torch.tensor(label, dtype=torch.long).to(device)

#         text_inputs_lst = [question_inputs]

#         r = random.random()
#         if self.data_mode == 'train' and config['num_paraphrase'] > 0:
#             paraphrase_questions = self.paraphraser(questions, config['num_paraphrase'])
#             paraphrase_inputs_lst = [self.text_processor(text) for text in paraphrase_questions]

#             if r < config['paraphrase_thresh']:
#                 is_fuse_para_t = torch.ones(text_model.config.hidden_size).to(device)
#             else: 
#                 is_fuse_para_t = torch.zeros(text_model.config.hidden_size).to(device)

#             text_inputs_lst += paraphrase_inputs_lst + [is_fuse_para_t]
        
#         data_outputs = {
#             'text_inputs_lst': text_inputs_lst,
#             'img_inputs': img_inputs,
#             'labels': labels
#         }
        
#         return data_outputs

#     def __len__(self):
#         return len(self.questions)
    
class ViVQADataset(Dataset):
    def __init__(self, data_dir, data_mode, text_processor, 
                img_processor, label_encoder=None, is_text_augment=True, 
                n_text_paras=3, text_para_thresh=0.9, n_para_pool=20):
        self.data_dir = data_dir
        self.data_mode = data_mode
        self.is_text_augment = is_text_augment
        self.n_text_paras = n_text_paras
        self.text_para_thresh = text_para_thresh

        if self.data_mode == 'train':
            train_filename = f'{n_para_pool}_paraphrases_train.csv'
            data_path = os.path.join(data_dir, 'ViVQA', train_filename)
            if not os.path.exists(data_path):
                print('Data training file with number of paraphrases pool not found! Select default (20) file.')
                data_path = os.path.join(data_dir, 'ViVQA', '20_paraphrases_train.csv')
            self.data_path = data_path
        else:
            self.data_path = os.path.join(data_dir, 'ViVQA', 'test.csv')
        
        self.img_dirpath = os.path.join(data_dir, 'COCO_Images', 'merge')
        self.text_processor = text_processor
        self.img_processor = img_processor
        self.device = device

        self.questions, self.para_questions, self.img_paths, self.answers = self.get_data()
        self.label_encoder = label_encoder

    def get_data(self):
        df = pd.read_csv(self.data_path, index_col=0)
        questions = [] 
        para_questions = []
        answers = []
        img_paths = []

        for idx, row in df.iterrows():
            question = row['question']
            answer = row['answer']
            img_id = row['img_id']
            #question_type = row['type'] # 0: object, 1: color, 2: how many, 3: where

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

        question_inputs = self.text_processor(questions)
        img_pils = Image.open(img_paths).convert('RGB')
        img_inputs = self.img_processor(img_pils).to(device)
        label = self.label_encoder[answers]
        
        labels = torch.tensor(label, dtype=torch.long).to(device)

        text_inputs_lst = [question_inputs]
        
        r = random.random()
        if self.data_mode == 'train' and self.is_text_augment:
            para_questions = self.para_questions[idx]
            para_questions = ast.literal_eval(para_questions)
            selected_para_questions = random.sample(para_questions, self.n_text_paras)
            paraphrase_inputs_lst = [self.text_processor(text) for text in selected_para_questions]

            if r < self.text_para_thresh:
                is_fuse_para_t = torch.ones(text_model.config.hidden_size).to(device)
            else: 
                is_fuse_para_t = torch.zeros(text_model.config.hidden_size).to(device)

            text_inputs_lst += paraphrase_inputs_lst + [is_fuse_para_t]
        
        data_outputs = {
            'text_inputs_lst': text_inputs_lst,
            'img_inputs': img_inputs,
            'labels': labels
        }
        
        return data_outputs

    def __len__(self):
        return len(self.questions)