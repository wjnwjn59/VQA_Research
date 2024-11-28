import os
import json
import random
import torch

from PIL import Image
from torch.utils.data import Dataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class OpenViVQADataset(Dataset):
    def __init__(self, data_dir, data_mode, text_encoder_dict, img_encoder_dict,
                 label_encoder=None, is_text_augment=True,
                 n_text_paras=2, text_para_thresh=0.5, n_para_pool=20
                 ):
        self.data_dir = data_dir
        self.data_mode = data_mode

        self.is_text_augment = is_text_augment
        self.n_text_paras = n_text_paras
        self.text_para_thresh = text_para_thresh

        if self.data_mode == 'train':
            train_filename = f'vlsp2023_train_data_{n_para_pool}_paraphrases.json'
            data_path = os.path.join(data_dir, 'OpenViVQA', train_filename)
                
            if not os.path.exists(data_path):
                print(
                    'Data training file with number of paraphrases pool not found! Select original file.')
                data_path = os.path.join(
                    data_dir, 'OpenViVQA', 'vlsp2023_train_data_20_paraphrases.json')
            self.data_path = data_path
            self.img_dirpath = os.path.join(
                data_dir, 'OpenViVQA', 'training-images')
        elif self.data_mode == 'dev':
            self.data_path = os.path.join(
                data_dir, 'OpenViVQA', 'vlsp2023_dev_data.json')
            self.img_dirpath = os.path.join(
                data_dir, 'OpenViVQA', 'dev-images')
        else:
            self.data_path = os.path.join(
                data_dir, 'OpenViVQA', 'vlsp2023_test_data.json')
            self.img_dirpath = os.path.join(
                data_dir, 'OpenViVQA', 'test-images')

        self.text_encoder_dict = text_encoder_dict
        self.img_encoder_dict = img_encoder_dict
        self.device = device

        self.questions, self.para_questions, self.img_paths, self.answers = self.get_data()
        self.label_encoder = label_encoder

    def load_data(self):
        with open(self.data_path, 'r') as file:
            data = json.load(file)
        images = data.get('images', {})
        annotations = data.get('annotations', {})

        return images, annotations

    def get_data(self):
        images, annotations = self.load_data()

        questions = []
        para_questions = []
        img_paths = []
        answers = []

        for annotation_id, annotation in annotations.items():
            img_id = annotation['image_id']
            question = annotation['question']

            if self.data_mode == 'train' and self.is_text_augment:
                para_question = annotation['paraphrases']
                para_questions.append(para_question)

            answer = annotation['answer']

            img_filename = images.get(str(img_id), "Unknown Image")
            img_path = f'{self.img_dirpath}/{img_filename}'

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
