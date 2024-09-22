import torch
import torch.nn as nn

from torch.functional import F

class TextEncoder(nn.Module):
    def __init__(self, text_model, projection_dim, is_text_augment):
        super().__init__()
        for param in text_model.parameters():
            param.requires_grad = True
        self.is_text_augment = is_text_augment
        self.model = text_model
        self.linear = nn.Linear(self.model.config.hidden_size, projection_dim)

    def forward(self, text_inputs_lst, augment_thresh):
        r = torch.rand(1)
        if self.training and self.is_text_augment and r < augment_thresh:
            embed_lst = []
            for text_inputs in text_inputs_lst:
                x = self.model(**text_inputs)
                x = x['last_hidden_state'][:, 0, :]
                embed_lst.append(x)
        
            para_features_t = torch.stack(embed_lst[1:], dim=1)
            x = torch.sum(para_features_t, dim=1)
            x = x + embed_lst[0]
        else:
            text_inputs = text_inputs_lst[0]
            x = self.model(**text_inputs)
            x = x['last_hidden_state'][:, 0, :]

        x = self.linear(x)
        x = F.gelu(x)

        return x 

class ImageEncoder(nn.Module):
    def __init__(self, img_model, projection_dim, is_img_augment):
        super().__init__()
        for param in img_model.parameters():
            param.requires_grad = True
        self.is_img_augment = is_img_augment
        self.model = img_model
        self.linear = nn.Linear(self.model.num_features * 7 * 7, projection_dim)

    def forward(self, img_inputs_lst, augment_thresh):
        r = torch.rand(1)
        if self.training and self.is_img_augment and r < augment_thresh:
            embed_lst = [] 
            for img_inputs in img_inputs_lst:
                x = self.model.forward_features(img_inputs)
                x = x.view(x.size(0), -1)
                embed_lst.append(x)

            img_features_t = torch.stack(embed_lst[1:], dim=1)
            x = torch.sum(img_features_t, dim=1)
            x = x + embed_lst[0]
        else: 
            x = self.model.forward_features(img_inputs_lst[0])
            x = x.view(x.size(0), -1)
        
        x = self.linear(x)
        x = F.gelu(x)

        return x


class Classifier(nn.Module):
    def __init__(self, projection_dim, hidden_dim, answer_space):
        super().__init__()
        self.fc = nn.Linear(projection_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(0.4)
        self.classifier = nn.Linear(hidden_dim, answer_space)

    def forward(self, text_f, img_f):
        x = torch.cat((img_f, text_f), 1)
        x = self.fc(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.classifier(x)

        return x 


class ViVQAModel(nn.Module):
    def __init__(self, projection_dim, hidden_dim, answer_space_len, 
                 text_encoder_dict, img_encoder_dict,
                 is_text_augment=True, is_img_augment=False,
                 total_epochs=100, use_dynamic_thresh=True, 
                 text_para_thresh=0.6, img_augment_thresh=0.6):
        
        super().__init__()
        self.text_encoder = TextEncoder(text_model=text_encoder_dict['text_model'],
                                        projection_dim=projection_dim,
                                        is_text_augment=is_text_augment)
        self.img_encoder = ImageEncoder(img_model=img_encoder_dict['img_model'],
                                        projection_dim=projection_dim,
                                        is_img_augment=is_img_augment)
        self.classifier = Classifier(projection_dim=projection_dim,
                                     hidden_dim=hidden_dim,
                                     answer_space=answer_space_len)
    
        self.use_dynamic_thresh = use_dynamic_thresh
        self.text_para_thresh = text_para_thresh
        self.img_augment_thresh = img_augment_thresh
        self.total_epochs = total_epochs
        self.current_epoch = 0
        self.start_threshold = 1.0
        self.min_threshold = 0.0

    def get_threshold(self):
        if not self.use_dynamic_thresh:
            return self.text_para_thresh, self.img_augment_thresh
        
        decay = (self.start_threshold - self.min_threshold) * (self.current_epoch / self.total_epochs)
        updated_thresh = max(self.start_threshold - decay, self.min_threshold)

        return updated_thresh, updated_thresh

    def forward(self, text_inputs, img_inputs):
        text_thresh, img_thresh = self.get_threshold()
        text_f = self.text_encoder(text_inputs, text_thresh)
        img_f = self.img_encoder(img_inputs, img_thresh)

        logits = self.classifier(text_f, img_f)

        return logits

    def update_epoch(self, epoch):
        self.current_epoch = epoch