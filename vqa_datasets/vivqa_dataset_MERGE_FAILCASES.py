import os
import ast
import random
import torch
import pandas as pd

from PIL import Image
from torch.utils.data import Dataset
from .augmentations.img_augmentation import augment_image_merge
from tqdm import tqdm

# SBert
from sentence_transformers import SentenceTransformer, util
# KNN
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

device = 'cuda' if torch.cuda.is_available() else 'cpu'



class ViVQADataset(Dataset):
    def __init__(self, data_dir, data_mode, text_encoder_dict, img_encoder_dict, 
                 label_encoder=None, is_text_augment=True, is_img_augment=True,
                 n_text_paras=3, text_para_thresh=0.9, n_para_pool=20,
                 n_img_augments=1, img_augment_thresh=0.9):

        self.data_dir = data_dir
        self.data_mode = data_mode

        self.is_text_augment = is_text_augment
        self.n_text_paras = n_text_paras
        self.text_para_thresh = text_para_thresh

        self.is_img_augment = is_img_augment
        self.n_img_augments = n_img_augments 
        self.img_augment_thresh = img_augment_thresh

        if self.data_mode == 'train':
            train_filename = f'{n_para_pool}_paraphrases_train.csv'
            data_path = os.path.join(data_dir, 'ViVQA', train_filename)
            
            if not os.path.exists(data_path):
                print('Data training file with number of paraphrases pool not found! Select default (20) file.')
                data_path = os.path.join(data_dir, 'ViVQA', '20_paraphrases_train.csv')
            self.data_path = data_path
            
            self.aug_imgs_rootpath = os.path.join(data_dir, 'ViVQA', 'aug_imgs_merge')
            # Sentence transformer model for calculation of text similarity scores.
            self.sen_model = SentenceTransformer('paraphrase-MiniLM-L6-v2').to(device)
        else:
            self.data_path = os.path.join(data_dir, 'ViVQA', 'test.csv')
        
        self.img_dirpath = os.path.join(data_dir, 'COCO_Images', 'merge')
        self.text_encoder_dict = text_encoder_dict
        self.img_encoder_dict = img_encoder_dict
        self.device = device

        self.filter = 'no'
        self.questions, self.para_questions, self.img_paths, self.img_ids, self.answers = self.get_data()
        self.label_encoder = label_encoder
        
        

    def get_data(self):
        df = pd.read_csv(self.data_path, index_col=0)
        questions = [] 
        para_questions = []
        answers = []
        img_paths = []
        img_ids = []

        
        for idx, row in tqdm(df.iterrows(), desc='Loading data', total=len(df)):
            question = row['question']
            answer = row['answer']
            img_id = row['img_id']
            #question_type = row['type'] # 0: object, 1: color, 2: how many, 3: where

            if self.data_mode == 'train' and self.is_text_augment:
                if self.filter == 'sbert':
                    question_paraphrases = row['question_paraphrase'] 
                    question_paraphrases = ast.literal_eval(question_paraphrases)   
                    
                    # Encoder câu hỏi gốc và các câu paraphrase
                    ori_sen = self.sen_model.encode(question)
                    paraphrase_sens = self.sen_model.encode(question_paraphrases)

                    # Tính điểm tương đồng cosine giữa câu gốc và các câu paraphrase
                    scores = util.pytorch_cos_sim(ori_sen, paraphrase_sens).flatten()
                    
                    # Tạo danh sách gồm các câu paraphrase và điểm số tương ứng
                    paraphrase_with_scores = list(zip(question_paraphrases, scores))
                    
                    # Đặt trước giới hạn
                    # max_score = 0.95
                    # min_score = 0.9
                    
                    # Lọc các câu paraphrase trong vùng giới hạn từ min_score đến max_score
                    # filtered_paraphrases = [(para, score) for para, score in paraphrase_with_scores if min_score <= score <= max_score]
                    # filtered_paraphrases = [para for para, score in paraphrase_with_scores if min_score <= score <= max_score]
                    # if len(filtered_paraphrases) < self.min_para_size:
                    #     self.min_para_size = len(filtered_paraphrases)
                    #     print("New minimum para size: ", self.min_para_size)
                    
                    # Sắp xếp danh sách theo thứ tự điểm số từ cao đến thấp 
                    paraphrase_with_scores_sorted = sorted(paraphrase_with_scores, key=lambda x: x[1], reverse=True)
                    
                    # Đặt trước giới hạn
                    from_index = 1
                    to_index = 11
                    
                    # Lấy ra một số câu paraphrase trong vùng giới hạn từ from_index đến to_index
                    paraphrase_with_scores_sorted = [para for para, score in paraphrase_with_scores_sorted]
                    selected_para_questions = paraphrase_with_scores_sorted[from_index:to_index]
                    
                    para_questions.append(selected_para_questions)
                
                elif self.filter == 'knn':
                    question_paraphrases = row['question_paraphrase'] 
                    question_paraphrases = ast.literal_eval(question_paraphrases)   
                    collection = [question] + question_paraphrases
                    
                    vectorizer = TfidfVectorizer()
                    tfidf_matrix = vectorizer.fit_transform(collection)
                    scores = cosine_similarity(tfidf_matrix[:1], tfidf_matrix[1:]).flatten()
                    paraphrase_with_scores = list(zip(question_paraphrases, scores))
                    paraphrase_with_scores_sorted = sorted(paraphrase_with_scores, key=lambda x: x[1], reverse=True)
                    
                    # Đặt trước giới hạn
                    from_index = 1
                    to_index = 11
                    
                    # Lấy ra một số câu paraphrase trong vùng giới hạn từ from_index đến to_index
                    paraphrase_with_scores_sorted = [para for para, score in paraphrase_with_scores_sorted]
                    selected_para_questions = paraphrase_with_scores_sorted[from_index:to_index]
                    
                    para_questions.append(selected_para_questions)
                
                else:
                    question_paraphrases = row['question_paraphrase'] 
                    para_questions.append(question_paraphrases)
                    

            img_ids.append(f'{img_id}.jpg')
            img_name = f'{img_id:012}.jpg'
            img_path = os.path.join(self.img_dirpath, img_name)

            questions.append(question)
            answers.append(answer)
            img_paths.append(img_path)
            
        return questions, para_questions, img_paths, img_ids, answers 


    def __getitem__(self, idx):
        questions = self.questions[idx]
        answers = self.answers[idx]
        img_paths = self.img_paths[idx]
        img_ids = self.img_ids[idx]

        img_pils = Image.open(img_paths).convert('RGB')
        label = self.label_encoder[answers]

        img_inputs_lst = [self.img_encoder_dict['img_processor'](img_pils)]
        
        if self.data_mode == 'train' and self.is_img_augment:
            aug_imgs_path = os.path.join(self.aug_imgs_rootpath, str(img_ids))
            augmented_imgs_pil = augment_image_merge(aug_imgs_path)
            augmented_imgs = [self.img_encoder_dict['img_processor'](img) for img in augmented_imgs_pil]

            img_inputs_lst += augmented_imgs 

        text_inputs_lst = [self.text_encoder_dict['text_processor'](questions)]
        
        if self.data_mode == 'train' and self.is_text_augment:
            para_questions = self.para_questions[idx]
            selected_para_questions = random.sample(para_questions, self.n_text_paras)
            paraphrase_inputs_lst = [self.text_encoder_dict['text_processor'](text) for text in selected_para_questions]

            text_inputs_lst += paraphrase_inputs_lst 
        
        labels = torch.tensor(label, dtype=torch.long)

        data_outputs = {
            'text_inputs_lst': text_inputs_lst,
            'img_inputs': img_inputs_lst,
            'labels': labels,
            'img_ids': img_ids,
        }
        
        return data_outputs

    def __len__(self):
        return len(self.questions)