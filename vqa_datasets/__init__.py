from typing import Dict, Type
from torch.utils.data import Dataset

from .openvivqa_dataset import OpenViVQADataset
from .vivqa_dataset import ViVQADataset

DATASET_MAPPING: Dict[str, Type[Dataset]] = {
    'openvivqa': OpenViVQADataset,
    'vivqa': ViVQADataset,
}


def get_dataset(text_encoder_dict,
                img_encoder_dict,
                label_encoder,
                is_train,
                **kwargs) -> Dataset:
    dataset_class = DATASET_MAPPING.get(kwargs['dataset_name'].lower())
    if not dataset_class:
        raise ValueError(
            f"Dataset '{kwargs['dataset_name']}' is not supported.")

    common_args = {
        'data_dir': kwargs['data_dir'],
        'text_encoder_dict': text_encoder_dict,
        'img_encoder_dict': img_encoder_dict,
        'label_encoder': label_encoder,
    }

    if is_train:
        return dataset_class(
            data_mode='train',
            is_text_augment=kwargs['is_text_augment'],
            n_text_paras=kwargs['n_text_paras'],
            text_para_thresh=kwargs['text_para_thresh'],
            n_para_pool=kwargs['n_text_para_pool'],
            **common_args
        )
    else:
        return dataset_class(
            data_mode='dev' if kwargs['dataset_name'].lower(
            ) == 'openvivqa' else 'val',
            is_text_augment=False,
            **common_args
        )
