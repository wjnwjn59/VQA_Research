import os
import json
from typing import Literal
import pandas as pd
import torch
from torch.utils.data import Dataset    
from PIL import Image

class ViVQADataset(Dataset):
    def __init__(self, 
                 data_dir: str, 
                 split: Literal["train", "test"],
                 img_trasnsform = None,
                 image_processor = None,
                 text_processor = None,
                 max_length = 256
                ) -> None:
        self.data_dir = data_dir    
        self.split = split
        assert self.split in ["train", "test"], "split must be either 'train' or 'test'"
        self.img_transform = img_trasnsform
        self.image_processor = image_processor
        self.text_processor = text_processor
        self.max_length = max_length
        self.get_data()
        self.load_encoder()

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, index):
        question = self.questions[index]
        answer = self.answers[index]
        img_path = self.img_paths[index]

        # Image proccessing
        img = Image.open(img_path).convert('RGB')
        if self.img_transform:
            img = self.img_transform(img)
        
        # >>> dict_keys(['pixel_values'])
        img_inputs = self.image_processor(
            images = img,
            return_tensors="pt"
        )
        # >>> dict_keys(['input_ids', 'token_type_ids', 'attention_mask'])
        text_inputs = self.text_processor(
            question,
            max_length = self.max_length, 
            truncation=True, 
            padding="max_length",
            return_tensors="pt"
        )

        inputs = {**img_inputs, **text_inputs}
        inputs.pop("token_type_ids", None)
        # >>> dict_keys(['pixel_values', 'input_ids', 'attention_mask'])

        label = [0 if idx != self.label2idx[answer] else 1 for idx in range(len(self.label2idx.keys()))]
        
        items = {k: v.squeeze() for k, v in inputs.items()}
        items['labels'] = torch.tensor(label)
        
        return items

    
    def get_data(self):
        data_path = os.path.join(self.data_dir, f'{self.split}.csv')
        df = pd.read_csv(data_path, index_col=0)

        questions, answers, img_paths = [], [], []

        for _, row in df.iterrows():
            question = row["question"]
            answer = row["answer"]
            img_id = row["img_id"]
            img_path = self.data_dir / 'MSCOCO2014_selected' / self.split / f'{img_id:012}.jpg'

            questions.append(question)
            answers.append(answer)
            img_paths.append(img_path)
        
        self.questions = questions
        self.answers = answers
        self.img_paths = img_paths

    def load_encoder(self):
        with open(self.data_dir / 'label2idx.json', 'r', encoding='utf-8') as f:
            self.label2idx = json.load(f)

        with open(self.data_dir / 'idx2label.json', 'r', encoding='utf-8') as f:
            self.idx2label = json.load(f)