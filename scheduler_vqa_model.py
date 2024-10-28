import torch
import torch.nn as nn

from torch.functional import F

# Define a Text Encoder class that handles the text input and projects it into a new dimension.
class TextEncoder(nn.Module):
    def __init__(self, text_model, projection_dim, is_text_augment):
        super().__init__()
        
        # Enable gradient updates for the text model
        for param in text_model.parameters():
            param.requires_grad = True
            
        self.is_text_augment = is_text_augment  # Flag for augmenting text data
        self.model = text_model  # Text model
        # self.layer_norm = nn.LayerNorm(self.model.config.hidden_size)
        self.linear = nn.Linear(self.model.config.hidden_size, projection_dim)

    def forward(self, text_inputs_lst, augment_thresh):
        r = torch.rand(1)  # Generate a random value to decide if augmentation should be applied
        if self.training and self.is_text_augment and r < augment_thresh:
            embed_lst = []
            for text_inputs in text_inputs_lst:
                x = self.model(**text_inputs)  # Forward pass through the text model
                x = x['last_hidden_state'][:, 0, :]  # Extract the embedding of the [CLS] token
                embed_lst.append(x)

            # Stack embeddings and sum them for augmented inputs
            para_features_t = torch.stack(embed_lst, dim=1)
            x = torch.sum(para_features_t, dim=1)  # Sum the embeddings along the new dimension
            # x = self.layer_norm(x)
        else:
            # Process a single text input if no augmentation is applied
            text_inputs = text_inputs_lst[0]
            x = self.model(**text_inputs)  # Forward pass through the text model
            x = x['last_hidden_state'][:, 0, :]  # Extract the embedding of the [CLS] token

        # Apply linear transformation and GELU activation
        x = self.linear(x) 
        x = F.gelu(x)

        return x 


# Define an Image Encoder class that handles the image input and projects it into a new dimension.
class ImageEncoder(nn.Module):
    def __init__(self, img_model, projection_dim, is_img_augment):
        super().__init__()
        
        # Enable gradient updates for the image model
        for param in img_model.parameters():
            param.requires_grad = True
        self.is_img_augment = is_img_augment  # Flag for image augmentation
        self.model = img_model  # Image model
        self.linear = nn.Linear(self.model.num_features * 7 * 7, projection_dim)  # Linear projection for flattened features

    def forward(self, img_inputs_lst, augment_thresh):
        r = torch.rand(1)  # Random value to decide if augmentation is applied
        if self.training and self.is_img_augment and r < augment_thresh:
            embed_lst = [] 
            for img_inputs in img_inputs_lst:
                x = self.model.forward_features(img_inputs)  # Extract features from the image model
                x = x.view(x.size(0), -1)  # Flatten the feature maps
                embed_lst.append(x)

            # Stack and sum embeddings for augmented inputs
            img_features_t = torch.stack(embed_lst, dim=1)
            x = torch.sum(img_features_t, dim=1)
        else: 
            # Process a single image input if no augmentation is applied
            x = self.model.forward_features(img_inputs_lst[0])
            x = x.view(x.size(0), -1)
        
        # Apply linear transformation and GELU activation
        x = self.linear(x)
        x = F.gelu(x)

        return x


# Define a Classifier class that combines text and image features and predicts the output
class Classifier(nn.Module):
    def __init__(self, projection_dim, hidden_dim, answer_space):
        super().__init__()
        
        # Fully connected layers for classification
        self.fc = nn.Linear(projection_dim * 2, hidden_dim)  # Linear layer for combining features
        self.dropout = nn.Dropout(0.4)  # Dropout layer to prevent overfitting
        self.classifier = nn.Linear(hidden_dim, answer_space)  # Linear layer for final classification

    # Forward pass to combine text and image features and output the final prediction
    def forward(self, text_f, img_f):
        x = torch.cat((img_f, text_f), 1)  # Concatenate the text and image features
        x = self.fc(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.classifier(x)

        return x 


# Main model class combining text, image encoders, and classifier for VQA (Visual Question Answering)
class ViVQAModel(nn.Module):
    def __init__(self, projection_dim, hidden_dim, answer_space_len, 
                 text_encoder_dict, img_encoder_dict,
                 is_text_augment=True, is_img_augment=False,
                 total_epochs=100, use_dynamic_thresh=True, 
                 text_para_thresh=0.6, img_augment_thresh=0.6):
        
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
                                     answer_space=answer_space_len)
    
        # Dynamic threshold settings for augmentation
        self.use_dynamic_thresh = use_dynamic_thresh
        self.text_para_thresh = text_para_thresh
        self.img_augment_thresh = img_augment_thresh
        self.total_epochs = total_epochs
        self.current_epoch = 0
        self.start_threshold = 0.6
        self.min_threshold = 0.0

    # Compute dynamic thresholds for augmentation based on the current epoch
    def get_threshold(self):
        if not self.use_dynamic_thresh:
            return self.text_para_thresh, self.img_augment_thresh
        
        decay = (self.start_threshold - self.min_threshold) * (self.current_epoch / self.total_epochs)
        updated_thresh = max(self.start_threshold - decay, self.min_threshold)

        return updated_thresh, updated_thresh

    # Forward pass for the entire model (combines text and image inputs)
    def forward(self, text_inputs, img_inputs):
        text_thresh, img_thresh = self.get_threshold()
        text_f = self.text_encoder(text_inputs, text_thresh)  # Encode text inputs
        img_f = self.img_encoder(img_inputs, img_thresh)  # Encode image inputs

        logits = self.classifier(text_f, img_f)  # Predict using classifier

        return logits

    # Update the current epoch for dynamic threshold adjustment
    def update_epoch(self, epoch):
        self.current_epoch = epoch