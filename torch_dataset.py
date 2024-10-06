import os
import ast
import random
import torch
import pandas as pd

from PIL import Image
from torch.utils.data import Dataset
from img_augmentation import augment_image
    
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class ViVQADataset(Dataset):
    def __init__(self, data_dir, data_mode, text_encoder_dict, img_encoder_dict, 
                 label_encoder=None, is_text_augment=True, is_img_augment=True,
                 n_text_paras=3, text_para_thresh=0.9, n_para_pool=20,
                 n_img_augments=1, img_augment_thresh=0.9):
        
        """
        Initialize the dataset class.

        Parameters:
        - data_dir: Directory containing the dataset files.
        - data_mode: Mode of operation ('train' or 'test').
        - text_encoder_dict: Dictionary containing text encoding functions.
        - img_encoder_dict: Dictionary containing image encoding functions.
        - label_encoder: Optional label encoding function.
        - is_text_augment: Flag to enable text augmentation.
        - is_img_augment: Flag to enable image augmentation.
        - n_text_paras: Number of paraphrases to sample.
        - text_para_thresh: Threshold for selecting paraphrases.
        - n_para_pool: Number of paraphrase options available.
        - n_img_augments: Number of image augmentations to apply.
        - img_augment_thresh: Threshold for applying image augmentations.
        """
        
        self.data_dir = data_dir
        self.data_mode = data_mode

        # Set parameters for text augmentation
        self.is_text_augment = is_text_augment
        self.n_text_paras = n_text_paras
        self.text_para_thresh = text_para_thresh

        # Set parameters for image augmentation
        self.is_img_augment = is_img_augment
        self.n_img_augments = n_img_augments 
        self.img_augment_thresh = img_augment_thresh

        # Determine the data file based on the mode (train/test)
        if self.data_mode == 'train':
            train_filename = f'{n_para_pool}_paraphrases_train.csv'
            data_path = os.path.join(data_dir, 'ViVQA', train_filename)
            
            # If the specified file does not exist, fall back to the default
            if not os.path.exists(data_path):
                print('Data training file with number of paraphrases pool not found! Select default (20) file.')
                data_path = os.path.join(data_dir, 'ViVQA', '20_paraphrases_train.csv')
            self.data_path = data_path
        else:
            self.data_path = os.path.join(data_dir, 'ViVQA', 'test.csv')
        
        # Set image directory path
        self.img_dirpath = os.path.join(data_dir, 'COCO_Images', 'merge')
        self.text_encoder_dict = text_encoder_dict
        self.img_encoder_dict = img_encoder_dict
        self.device = device

        # Load data from the specified CSV file
        self.questions, self.para_questions, self.img_paths, self.answers = self.get_data()
        self.label_encoder = label_encoder

    def get_data(self):
        """
        Load data from the CSV file and parse questions, images, and answers.

        Returns:
        - questions: List of questions.
        - para_questions: List of paraphrased questions.
        - img_paths: List of image file paths.
        - answers: List of answers.
        """
        
        df = pd.read_csv(self.data_path, index_col=0)
        questions = [] 
        para_questions = []
        answers = []
        img_paths = []

        # Iterate through the rows in the DataFrame to extract data
        for idx, row in df.iterrows():
            question = row['question']
            answer = row['answer']
            img_id = row['img_id']
            #question_type = row['type'] # 0: object, 1: color, 2: how many, 3: where

            # If in training mode and text augmentation is enabled, collect paraphrases
            if self.data_mode == 'train' and self.is_text_augment:
                question_paraphrases = row['question_paraphrase']
                para_questions.append(question_paraphrases)

            # Construct the image file name and path
            img_name = f'{img_id:012}.jpg'
            img_path = os.path.join(self.img_dirpath, img_name)

            # Append the collected data to respective lists
            questions.append(question)
            answers.append(answer)
            img_paths.append(img_path)

        return questions, para_questions, img_paths, answers 

    def __getitem__(self, idx):
        """
        Get a single data item (question, answer, and image) by index.

        Parameters:
        - idx: Index of the item to retrieve.

        Returns:
        - data_outputs: Dictionary containing text inputs, image inputs, and labels.
        """
        
        questions = self.questions[idx]
        answers = self.answers[idx]
        img_paths = self.img_paths[idx]

        # Open and preprocess the image
        img_pils = Image.open(img_paths).convert('RGB')
        label = self.label_encoder[answers]

        # Process the image and add to input list
        img_inputs_lst = [self.img_encoder_dict['img_processor'](img_pils).to(device)]
        
        
        r = random.random()
        # Apply image augmentation if in training mode
        if self.data_mode == 'train' and self.is_img_augment:
            augmented_imgs_pil = augment_image(img_pils, self.n_img_augments)
            # Process augmented images
            augmented_imgs = [self.img_encoder_dict['img_processor'](img).to(self.device) for img in augmented_imgs_pil]

            # Determine if paraphrase features should be fused based on random threshold
            if r < self.img_augment_thresh:
                is_fuse_para_t = torch.ones(self.img_encoder_dict['features_dim']).to(self.device)
            else:
                is_fuse_para_t = torch.zeros(self.img_encoder_dict['features_dim']).to(self.device)

            img_inputs_lst += augmented_imgs + [is_fuse_para_t]

        # Process the text input
        text_inputs_lst = [self.text_encoder_dict['text_processor'](questions)]
        
        
        r = random.random()
        # Apply text augmentation if in training mode
        if self.data_mode == 'train' and self.is_text_augment:
            para_questions = self.para_questions[idx]
            para_questions = ast.literal_eval(para_questions) # Convert string representation of list to list
            selected_para_questions = random.sample(para_questions, self.n_text_paras) # Randomly select n_text_paras from the list
            paraphrase_inputs_lst = [self.text_encoder_dict['text_processor'](text) for text in selected_para_questions] # Process each paraphrase

            # Determine if features should be fused based on random threshold
            if r < self.text_para_thresh: 
                is_fuse_para_t = torch.ones(self.text_encoder_dict['features_dim']).to(device)
            else: 
                is_fuse_para_t = torch.zeros(self.text_encoder_dict['features_dim']).to(device)

            text_inputs_lst += paraphrase_inputs_lst + [is_fuse_para_t]
        
        labels = torch.tensor(label, dtype=torch.long).to(device)

        # Create output dictionary
        data_outputs = {
            'text_inputs_lst': text_inputs_lst, 
            'img_inputs': img_inputs_lst,
            'labels': labels
        }
        
        return data_outputs

    def __len__(self):
        return len(self.questions)