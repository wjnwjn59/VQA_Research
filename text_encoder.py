import os
import py_vncorenlp
import torch

from pathlib import Path
from transformers import AutoModel, AutoTokenizer
from contextlib import contextmanager


dict_map = {
    "òa": "oà",
    "Òa": "Oà",
    "ÒA": "OÀ",
    "óa": "oá",
    "Óa": "Oá",
    "ÓA": "OÁ",
    "ỏa": "oả",
    "Ỏa": "Oả",
    "ỎA": "OẢ",
    "õa": "oã",
    "Õa": "Oã",
    "ÕA": "OÃ",
    "ọa": "oạ",
    "Ọa": "Oạ",
    "ỌA": "OẠ",
    "òe": "oè",
    "Òe": "Oè",
    "ÒE": "OÈ",
    "óe": "oé",
    "Óe": "Oé",
    "ÓE": "OÉ",
    "ỏe": "oẻ",
    "Ỏe": "Oẻ",
    "ỎE": "OẺ",
    "õe": "oẽ",
    "Õe": "Oẽ",
    "ÕE": "OẼ",
    "ọe": "oẹ",
    "Ọe": "Oẹ",
    "ỌE": "OẸ",
    "ùy": "uỳ",
    "Ùy": "Uỳ",
    "ÙY": "UỲ",
    "úy": "uý",
    "Úy": "Uý",
    "ÚY": "UÝ",
    "ủy": "uỷ",
    "Ủy": "Uỷ",
    "ỦY": "UỶ",
    "ũy": "uỹ",
    "Ũy": "Uỹ",
    "ŨY": "UỸ",
    "ụy": "uỵ",
    "Ụy": "Uỵ",
    "ỤY": "UỴ",
}


def text_tone_normalize(text, dict_map):
    for i, j in dict_map.items():
        text = text.replace(i, j)
    return text


@contextmanager
def temporary_directory_change(directory):
    original_directory = os.getcwd()
    os.chdir(directory)
    try:
        yield
    finally:
        os.chdir(original_directory)


VNCORENLP_PATH = Path('./models/VnCoreNLP')
ABS_VNCORENLP_PATH = VNCORENLP_PATH.resolve()
os.makedirs(VNCORENLP_PATH, exist_ok=True)


if not (ABS_VNCORENLP_PATH / 'models').exists():
    py_vncorenlp.download_model(save_dir=str(ABS_VNCORENLP_PATH))


with temporary_directory_change(ABS_VNCORENLP_PATH):
    rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"],
                                          save_dir=str(ABS_VNCORENLP_PATH))

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def text_processor(text, text_tokenizer):
    text = text_tone_normalize(text, dict_map)
    segmented_text = rdrsegmenter.word_segment(text)
    segmented_text = ' '.join(segmented_text)

    input_ids = text_tokenizer(segmented_text,
                               max_length=64,
                               padding='max_length',
                               truncation=True,
                               return_token_type_ids=False,
                               return_tensors='pt')

    # Ensure the output shape is (batch_size, len)
    # if input_ids['input_ids'].shape[0] == 1:
    #     print(input_ids['input_ids'].shape)
    #     print(input_ids['attention_mask'].shape)
    #     input_ids = {k: v.unsqueeze(0) for k, v in input_ids.items()}

    return input_ids


class TextProcessorWrapper:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, text):
        return text_processor(text, self.tokenizer)


def load_text_encoder(text_model_id):
    text_tokenizer = AutoTokenizer.from_pretrained(text_model_id)
    text_model = AutoModel.from_pretrained(text_model_id,
                                           device_map=device)

    return {
        'model_name': text_model_id,
        'text_processor': TextProcessorWrapper(text_tokenizer),
        'features_dim': text_model.config.hidden_size,
        'text_model': text_model
    }