import os
import py_vncorenlp
import torch

from pathlib import Path
from transformers import AutoModel, AutoTokenizer
from contextlib import contextmanager

# A dictionary that maps various Vietnamese tones to their normalized form
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

# Mapping Vietnamese tones to normalized forms
def text_tone_normalize(text, dict_map):
    for i, j in dict_map.items():
        text = text.replace(i, j)
    return text


# Context manager to temporarily change the current working directory
@contextmanager
def temporary_directory_change(directory):
    original_directory = os.getcwd() # Store the original directory
    os.chdir(directory) # Change to the new directory
    try:
        yield # Allow code block execution within this context
    finally:
        os.chdir(original_directory) # Change back to the original directory after execution


# Define paths for VnCoreNLP model.
VNCORENLP_PATH = Path('./models/VnCoreNLP')
ABS_VNCORENLP_PATH = VNCORENLP_PATH.resolve() # Get absolute path of the model directory
os.makedirs(VNCORENLP_PATH, exist_ok=True)  # Create directory if it doesn't exist

# Download the model if it's not already present
if not (ABS_VNCORENLP_PATH / 'models').exists():
    py_vncorenlp.download_model(save_dir=str(ABS_VNCORENLP_PATH))

# Load the VnCoreNLP segmenter within the temporary directory context
with temporary_directory_change(ABS_VNCORENLP_PATH):
    rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], 
                                        save_dir=str(ABS_VNCORENLP_PATH))

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Process input text: normalize tones and tokenize.
def text_processor(text, text_tokenizer):
    text = text_tone_normalize(text, dict_map) # Normalize Vietnamese tones
    segmented_text = rdrsegmenter.word_segment(text) # Segment the text
    segmented_text = ' '.join(segmented_text) # Join the segmented words

    input_ids = text_tokenizer(segmented_text,
                               max_length=50, # Set maximum length of the tokenized text
                               padding='max_length', # Pad the text to the maximum length
                               truncation=True, # Truncate the text to the maximum length
                               return_token_type_ids=False, # Do not return token type IDs
                               return_tensors='pt') # Return PyTorch tensors

    return input_ids


# Wrapper class for text processing
class TextProcessorWrapper:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, text):
        return text_processor(text, self.tokenizer) # Allow function call behavior


# Load the text encoder model and its associated tokenizer and text processor
def load_text_encoder(text_model_id):
    text_tokenizer = AutoTokenizer.from_pretrained(text_model_id) # Load the tokenizer
    text_model = AutoModel.from_pretrained(text_model_id, # Load the model
                                           device_map=device)

    # Return a dictionary
    return {
        'model_name': text_model_id, # Model name
        'text_processor': TextProcessorWrapper(text_tokenizer), # Text processor
        'features_dim': text_model.config.hidden_size, # Features dimension
        'text_model': text_model # Text model
    }