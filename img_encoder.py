import timm
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_img_encoder(img_encoder_id):
    img_model = timm.create_model(
        img_encoder_id,
        pretrained=True,
        num_classes=0  
    ).to(device)

    data_config = timm.data.resolve_model_data_config(img_model)
    img_processor = timm.data.create_transform(**data_config, is_training=False)

    if 'resnet' in img_encoder_id:
        features_dim = img_model.num_features * 7 * 7
    elif 'beit' in img_encoder_id:
        features_dim = img_model.num_features  
    else:
        raise ValueError(f"Model {img_encoder_id} not supported for feature dimension detection")

    return {
        'model_name': img_encoder_id,
        'img_processor': img_processor,
        'features_dim': features_dim, 
        'img_model': img_model
    }


