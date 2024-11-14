import torch
import torch.nn as nn
import numpy as np
from torch.functional import F
from cross_attention import *


class BottleneckBlock(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, input_dim),
        )
        # self.norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        x = self.proj(x) + x
        # x = self.norm(x)
        return x


# Define a Text Encoder class that handles the text input and projects it into a new dimension.
class TextEncoder(nn.Module):
    def __init__(self, text_model, projection_dim, is_text_augment):
        super().__init__()

        # Enable gradient updates for the text model
        for param in text_model.parameters():
            param.requires_grad = True

        self.is_text_augment = is_text_augment  # Flag for augmenting text data
        self.model = text_model  # Text model

        self.proj = nn.Sequential(
            nn.Linear(self.model.config.hidden_size, projection_dim),
            nn.ReLU()
        )

        # Bottleneck structure for augment_linear
        # if self.is_text_augment:
        #     self.augment_linear = BottleneckBlock(projection_dim)

        if self.is_text_augment:
            config = LanguageConfig()
            self.language_query_attn = CrossAttention(
                encoder_hidden_size=config.hidden_size,
                num_heads=config.self_attn_heads,
                hidden_size=config.hidden_size,
                dropout=config.attn_dropout
            )

            self.lang_cross_augment = CrossAugmentation(
                attn_layer=self.language_query_attn,
                mlp_layer=MLP(config),
                norm_layer=nn.LayerNorm(
                    config.hidden_size, eps=config.layer_norm_eps)
            )

    def forward(self, text_inputs_lst, augment_thresh):
        r = torch.rand(1)
        if self.training and self.is_text_augment and r < augment_thresh:

            embed_lst = []
            for text_inputs in text_inputs_lst:
                x = self.model(**text_inputs)
                # x = x['last_hidden_state'][:, 0, :]

                x = x['last_hidden_state']
                embed_lst.append(x)

            # para_features_t = torch.stack(embed_lst, dim=1)
            # Sum the embeddings along the new dimension
            # x = torch.sum(para_features_t, dim=1)
            # x = self.proj(x)
            # x = self.augment_linear(x)

            ori_embed = embed_lst[0]
            para_embed = embed_lst[1:]

            for _ in range(1):
                ori_embed = self.lang_cross_augment(ori_embed, para_embed)

            pooled_output = ori_embed[:, 0, :]
            x = self.proj(pooled_output)

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


# Main model class combining text, image encoders, and classifier for VQA (Visual Question Answering)
class ViVQAModel(nn.Module):
    def __init__(self, projection_dim, hidden_dim, answer_space_len,
                 text_encoder_dict, img_encoder_dict,
                 is_text_augment=True,
                 total_epochs=100, use_dynamic_thresh=True,
                 start_threshold=0.6, update_threshold_method='linear', init_learning_rate=1e-5,
                 text_para_thresh=0.6):

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
        self.first_loss = 0
        self.current_epoch = 0
        self.update_threshold_method = update_threshold_method
        self.start_threshold = start_threshold
        self.min_threshold = 0.01

        self.init_learning_rate = init_learning_rate

    def get_threshold(self):
        if not self.use_dynamic_thresh:
            return self.text_para_thresh

        decay = (self.start_threshold - self.min_threshold) * \
            (self.current_epoch / self.total_epochs)  # 0.6 * 1/30
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