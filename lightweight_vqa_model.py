import torch
import torch.nn as nn
from torch.functional import F


class BottleneckBlock(nn.Module):
    def __init__(self, projection_dim, intermediate_dim):
        super().__init__()
        self.proj_in_ori = nn.Linear(projection_dim, intermediate_dim)
        self.proj_in_para = nn.Linear(projection_dim, intermediate_dim)
        self.proj_out = nn.Linear(intermediate_dim, projection_dim)
        self.relu = nn.ReLU()

    def forward(self, x_ori, x_paras):
        x = self.proj_in_ori(x_ori)
        for x_para in x_paras:
            x += self.proj_in_para(x_para)
        x = self.proj_out(x)
        x = self.relu(x) + x_ori
        return x


class TextEncoder(nn.Module):
    def __init__(self, text_model, projection_dim, is_text_augment):
        super().__init__()

        for param in text_model.parameters():
            param.requires_grad = True

        self.is_text_augment = is_text_augment
        self.model = text_model
        self.proj = nn.Sequential(
            nn.Linear(self.model.config.hidden_size, projection_dim),
            nn.ReLU()
        )

        if self.is_text_augment:
            self.BB = BottleneckBlock(
                self.model.config.hidden_size, self.model.config.hidden_size // 2)

    def forward(self, text_inputs_lst, augment_thresh):
        r = torch.rand(1)
        if self.training and self.is_text_augment and r < augment_thresh:
            embed_lst = []

            for text_inputs in text_inputs_lst:
                x = self.model(**text_inputs)
                embed_lst.append(x['last_hidden_state'][:, 0, :])

            ori_embed = embed_lst[0]
            para_embed = embed_lst[1:]

            ori_embed = self.BB(ori_embed, para_embed)

            x = self.proj(ori_embed)

        else:
            text_inputs = text_inputs_lst[0]
            x = self.model(**text_inputs)
            x = x['last_hidden_state'][:, 0, :]
            x = self.proj(x)

        return x


class ImageEncoder(nn.Module):
    def __init__(self, img_model, projection_dim):
        super().__init__()

        for param in img_model.parameters():
            param.requires_grad = True

        self.model = img_model

        self.proj = nn.Sequential(
            nn.Linear(self.model.num_features * 7 * 7, projection_dim),
            nn.ReLU()
        )

    def forward(self, img_inputs_lst):
        x = self.model.forward_features(img_inputs_lst[0])
        x = x.view(x.size(0), -1)
        x = self.proj(x)
        return x


class Classifier(nn.Module):
    def __init__(self, projection_dim, hidden_dim, answer_space):
        super().__init__()
        self.fc = nn.Linear(projection_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(hidden_dim, answer_space)

    def forward(self, text_f, img_f):
        x = torch.cat((img_f, text_f), 1)
        x = self.fc(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.classifier(x)

        return x


class ViVQAModel(nn.Module):
    def __init__(self, projection_dim, hidden_dim, answer_space_len,
                 text_encoder_dict, img_encoder_dict,
                 is_text_augment=True, text_para_thresh=0.6,
                 total_epochs=100, use_dynamic_thresh=True, start_threshold=0.6):

        super().__init__()

        self.text_encoder = TextEncoder(text_model=text_encoder_dict['text_model'],
                                        projection_dim=projection_dim,
                                        is_text_augment=is_text_augment)

        self.img_encoder = ImageEncoder(img_model=img_encoder_dict['img_model'],
                                        projection_dim=projection_dim)

        self.classifier = Classifier(projection_dim=projection_dim,
                                     hidden_dim=hidden_dim,
                                     answer_space=answer_space_len)

        self.use_dynamic_thresh = use_dynamic_thresh
        self.text_para_thresh = text_para_thresh
        self.total_epochs = total_epochs
        self.current_epoch = 0
        self.start_threshold = start_threshold
        self.min_threshold = 0.0

    def get_threshold(self):
        if not self.use_dynamic_thresh:
            return self.text_para_thresh

        decay = (self.start_threshold - self.min_threshold) * \
            (self.current_epoch / self.total_epochs)
        updated_thresh = max(self.start_threshold -
                             decay, self.min_threshold)

        return updated_thresh

    def forward(self, text_inputs, img_inputs):
        text_thresh = self.get_threshold()
        text_f = self.text_encoder(
            text_inputs, text_thresh)
        img_f = self.img_encoder(img_inputs)

        logits = self.classifier(text_f, img_f)

        return logits

    def update_epoch(self, epoch):
        self.current_epoch = epoch
