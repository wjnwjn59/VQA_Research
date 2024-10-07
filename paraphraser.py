import torch
from transformers import MT5Tokenizer, MT5ForConditionalGeneration
from config import pipeline_config

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load paraphraser model configuration and tokenizer.
PARAPHRASER_ID = pipeline_config.paraphraser_id
paraphraser_tokenizer = MT5Tokenizer.from_pretrained(PARAPHRASER_ID)
paraphraser_model = MT5ForConditionalGeneration.from_pretrained(PARAPHRASER_ID).to(device)

def get_paraphrase(text, num_return_sequences):
    # Tokenize input text, apply padding, and move tensors to the selected device.
    inputs = paraphraser_tokenizer(text, 
                                   padding='longest', 
                                   max_length=64, 
                                   return_tensors='pt',
                                   return_token_type_ids=False).to(device)
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    
    # Generate paraphrased outputs using beam search.
    output = paraphraser_model.generate(input_ids, 
                            attention_mask=attention_mask, 
                            max_length=64,
                            num_beams=num_return_sequences,
                            early_stopping=True,
                            no_repeat_ngram_size=1,
                            num_return_sequences=num_return_sequences)
    
    # Decode the generated output tokens into text format.
    paraphrase_lst = []
    for beam_output in output:
        paraphrase_lst.append(paraphraser_tokenizer.decode(beam_output, skip_special_tokens=True))

    return paraphrase_lst