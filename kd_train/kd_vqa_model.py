import torch
import torch.nn as nn

from torch.functional import F

# TextEncoder class to encode text inputs using a specified model
class TextEncoder(nn.Module):
    def __init__(self, text_model, projection_dim, is_text_augment):
        super().__init__()
        
        # Enable gradient computation for all parameters of the text model
        for param in text_model.parameters():
            param.requires_grad = True
        self.is_text_augment = is_text_augment  # Flag for text augmentation 
        self.model = text_model  # Initialize the text model
        self.linear = nn.Linear(self.model.config.hidden_size, projection_dim)


    def forward(self, text_inputs_lst):
        if self.training and self.is_text_augment:
            embed_lst = []
            
            # Loop through all but the last text input
            for text_inputs in text_inputs_lst[:-1]:
                x = self.model(**text_inputs)  # Get model output
                x = x['last_hidden_state'][:, 0, :]  # Extract the first token's hidden state
                embed_lst.append(x)

            # Stack embeddings and compute weighted sum
            para_features_t = torch.stack(embed_lst[1:], dim=1)
            x = torch.sum(para_features_t, dim=1) 
            x *= text_inputs_lst[-1]  # Apply weights from the last input
            x = x + embed_lst[0]  # Add first embedding

        else:
            # If not training or augmentation is off, process the first input only
            text_inputs = text_inputs_lst[0]
            x = self.model(**text_inputs)
            x = x['last_hidden_state'][:, 0, :]

        x = self.linear(x)
        x = F.gelu(x)

        return x 


# ImageEncoder class to encode image inputs using a specified model
class ImageEncoder(nn.Module):
    def __init__(self, img_model, projection_dim, is_img_augment):
        super().__init__()
        
        # Enable gradient computation for all parameters of the image model
        for param in img_model.parameters():
            param.requires_grad = True
            
        self.is_img_augment = is_img_augment  # Flag for image augmentation
        self.model = img_model  # Initialize the image model
        self.linear = nn.Linear(self.model.num_features * 7 * 7, projection_dim)  # Linear layer for projection


    def forward(self, img_inputs_lst):
        if self.training and self.is_img_augment:
            embed_lst = [] 
            
            # Loop through all but the last image input
            for img_inputs in img_inputs_lst[:-1]:
                x = self.model.forward_features(img_inputs)  # Get model output
                x = x.view(x.size(0), -1)  # Flatten the output
                embed_lst.append(x)
                
            # Stack embeddings and compute weighted sum                
            img_features_t = torch.stack(embed_lst[1:], dim=1)
            x = torch.sum(img_features_t, dim=1)
            x *= img_inputs_lst[-1]  # Apply weights from the last input
            x = x + embed_lst[0]  # Add first embedding
            
        else: 
            # If not training or augmentation is off, process the first input only
            x = self.model.forward_features(img_inputs_lst[0])
            x = x.view(x.size(0), -1)  # Flatten the output
        
        x = self.linear(x)
        x = F.gelu(x)

        return x


# Classifier class to classify combined features from text and image encoders
class Classifier(nn.Module):
    def __init__(self, projection_dim, hidden_dim, answer_space, is_kd):
        super().__init__()
        self.fc = nn.Linear(projection_dim * 2, hidden_dim)  # Linear layer for combining features
        self.dropout = nn.Dropout(0.4)  # Dropout layer to prevent overfitting
        self.classifier = nn.Linear(hidden_dim, answer_space)  # Linear layer for final classification
        self.is_kd = is_kd  # Flag for training with Knowledge Distillation method

    # Forward pass to combine text and image features and output the final prediction
    def forward(self, text_f, img_f):
        x = torch.cat((img_f, text_f), 1)  # Concatenate the text and image features
        x = self.fc(x)
        
        kd_logits = x  # Take logits of fused features to calculate Cosine loss minimization later
        
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.classifier(x)

        # If model is used for KD train, return kd_logits for calculate Cosine loss minimization in training
        if self.is_kd:
            return x, kd_logits
        else:
            return x


# Main model class combining text, image encoders, and classifier for VQA (Visual Question Answering)
class ViVQAModel(nn.Module):
    def __init__(self, projection_dim, hidden_dim, answer_space_len, 
                 text_encoder_dict, img_encoder_dict,
                 is_text_augment=False, is_img_augment=False,
                 is_kd=False):
        
        super().__init__()
        
        # Initialize the text encoder        
        self.text_encoder = TextEncoder(text_model=text_encoder_dict['text_model'],
                                        projection_dim=projection_dim,
                                        is_text_augment=is_text_augment)
        
        # Initialize the image encoder   
        self.img_encoder = ImageEncoder(img_model=img_encoder_dict['img_model'],
                                        projection_dim=projection_dim,
                                        is_img_augment=is_img_augment)
        
        # Initialize the classifier           
        self.classifier = Classifier(projection_dim=projection_dim,
                                     hidden_dim=hidden_dim,
                                     answer_space=answer_space_len,
                                     is_kd=is_kd)
        
        # Flag for Knowledge Distillation
        self.is_kd = is_kd


    # Forward pass for the entire model (combines text and image inputs)
    def forward(self, text_inputs, img_inputs):
        # Forward pass for both text and image inputs
        text_f = self.text_encoder(text_inputs)
        img_f = self.img_encoder(img_inputs)

        outputs = self.classifier(text_f, img_f)  # Classify based on combined features

        # If model's kd flag is True, outputs contain 2 values: logits and kd_logits
        # Else: return logits only
        # For more detail, read the class Classifier above
        return outputs
