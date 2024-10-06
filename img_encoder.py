import timm
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_img_encoder(img_encoder_id):
    # Create an image encoder model using the timm library, loading a pretrained model
    img_model = timm.create_model(
        img_encoder_id,
        pretrained=True,
        num_classes=0 # remove the classifier nn.Linear
    ).to(device)
    
    # Retrieve the data configuration required for the model, 
    # which includes details like image size, normalization, etc.
    data_config = timm.data.resolve_model_data_config(img_model)
    
    # Create the image processing pipeline (transformation) from the data configuration.
    # is_training=False means this transformation is used in evaluation mode, not training.
    img_processor = timm.data.create_transform(**data_config, 
                                               is_training=False)

    # Return a dictionary
    return {
        'model_name': img_encoder_id,                       # Name of the image encoder model
        'img_processor': img_processor,                     # Image processing pipeline
        'features_dim': img_model.num_features * 7 * 7,     # Dimension of the features
        'img_model': img_model                              # The image encoder model itself
    }