import torch
import torch.nn as nn
from torch.functional import F
from lightweight_vqa_model import BottleneckBlock, TextEncoder, ImageEncoder, Classifier


class ViVQAGPT(nn.Module):
    def __init__(self, projection_dim, hidden_dim, answer_space_len,
                 text_encoder_dict, img_encoder_dict,
                 is_text_augment=True, text_para_thresh=0.6,
                 total_epochs=100, use_dynamic_thresh=True,
                 start_threshold=0.6, min_threshold=0.0):

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
        self.min_threshold = min_threshold

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
