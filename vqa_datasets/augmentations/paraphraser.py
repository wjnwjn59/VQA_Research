import torch
from transformers import MT5Tokenizer, MT5ForConditionalGeneration
from config import pipeline_config
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

device = 'cuda' if torch.cuda.is_available() else 'cpu'

PARAPHRASER_ID = pipeline_config.paraphraser_id
paraphraser_tokenizer = MT5Tokenizer.from_pretrained(PARAPHRASER_ID)
paraphraser_model = MT5ForConditionalGeneration.from_pretrained(
    PARAPHRASER_ID).to(device)

sbert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')


def get_paraphrase(text, num_return_sequences):
    inputs = paraphraser_tokenizer(text,
                                   padding='longest',
                                   max_length=64,
                                   return_tensors='pt',
                                   return_token_type_ids=False).to(device)
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    output = paraphraser_model.generate(input_ids,
                                        attention_mask=attention_mask,
                                        max_length=64,
                                        num_beams=num_return_sequences,
                                        early_stopping=True,
                                        no_repeat_ngram_size=1,
                                        num_return_sequences=num_return_sequences)

    paraphrase_lst = []
    for beam_output in output:
        paraphrase_lst.append(paraphraser_tokenizer.decode(
            beam_output, skip_special_tokens=True))

    return paraphrase_lst


def knn_filter(origin, para_lst, from_index, ktop):
    collection = [origin] + para_lst

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(collection)

    scores = cosine_similarity(tfidf_matrix[:1], tfidf_matrix[1:]).flatten()

    paraphrase_with_scores = list(zip(para_lst, scores))
    paraphrase_with_scores_sorted = sorted(
        paraphrase_with_scores, key=lambda x: x[1], reverse=True)

    paraphrase_with_scores_sorted = [
        para for para, score in paraphrase_with_scores_sorted]
    filtered_para_questions = paraphrase_with_scores_sorted[from_index:(
        from_index + ktop)]

    return filtered_para_questions


def sbert_filter(origin, para_lst, from_index, ktop):
    paraphrase_with_scores = []

    ori_encode = sbert_model.encode(origin)
    paras_encode = sbert_model.encode(para_lst)

    for idx in range(len(paras_encode)):
        score = util.cos_sim(ori_encode, paras_encode[idx]).item()
        paraphrase_with_scores.append((para_lst[idx], score))

    paraphrase_with_scores_sorted = sorted(
        paraphrase_with_scores, key=lambda x: x[1], reverse=True)
    paraphrase_with_scores_sorted = [
        para for para, score in paraphrase_with_scores_sorted]
    filtered_para_questions = paraphrase_with_scores_sorted[from_index:(
        from_index + ktop)]

    return filtered_para_questions
