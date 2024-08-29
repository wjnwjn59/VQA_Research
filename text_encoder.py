import os
import py_vncorenlp
import torch

from pathlib import Path
from transformers import AutoModel, AutoTokenizer
from contextlib import contextmanager
from config import pipeline_config

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

TEXT_MODEL_ID = pipeline_config.text_encoder_id
VNCORENLP_PATH = Path('./models/VnCoreNLP')
ABS_VNCORENLP_PATH = VNCORENLP_PATH.resolve()
os.makedirs(VNCORENLP_PATH, exist_ok=True)

if not (ABS_VNCORENLP_PATH / 'models').exists():
    py_vncorenlp.download_model(save_dir=str(ABS_VNCORENLP_PATH))

with temporary_directory_change(ABS_VNCORENLP_PATH):
    rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], 
                                        save_dir=str(ABS_VNCORENLP_PATH))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
text_encoder_tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_ID)
text_model = AutoModel.from_pretrained(TEXT_MODEL_ID,
                                       device_map=device)

def text_processor(text):
    text = text_tone_normalize(text, dict_map)
    segmented_text = rdrsegmenter.word_segment(text)
    segmented_text = ' '.join(segmented_text)

    input_ids = text_encoder_tokenizer(segmented_text,
                                       max_length=50,
                                       padding='max_length', 
                                       truncation=True,
                                       return_token_type_ids=False,
                                       return_tensors='pt').to(device)
    
    input_ids = {k: v.squeeze() for k, v in input_ids.items()}

    return input_ids