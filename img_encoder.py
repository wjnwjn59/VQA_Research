import timm
import torch

from config import pipeline_config

IMG_MODEL_ID = pipeline_config.img_encoder_id

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# get model specific transforms (normalization, resize)
img_model = timm.create_model(
    IMG_MODEL_ID,
    pretrained=True,
    num_classes=0 # remove classifier nn.Linear
).to(device)
data_config = timm.data.resolve_model_data_config(img_model)
img_processor = timm.data.create_transform(**data_config, 
                                           is_training=False)
