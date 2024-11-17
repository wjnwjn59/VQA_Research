from vqa_datasets.augmentations.paraphraser import get_paraphrase, knn_filter, sbert_filter
import ast
from tqdm import tqdm
import torch
import argparse
import pandas as pd
import json
import os


def set_seed(random_seed):
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(random_seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ['CUDNN_DETERMINISTIC'] = '1'


def generate_df_w_filter_paraphrases(data_filepath, is_paraphrased, num_paraphrase, filter_method, from_index, topk):
    df = pd.read_csv(data_filepath)
    filtered_question_paraphrases = []

    if not is_paraphrased:
        for idx, row in tqdm(df.iterrows(), desc='Generating paraphrases', total=len(df)):
            question_paraphrases = get_paraphrase(
                row['question'], num_paraphrase)

            if filter_method == 'knn':
                filtered_question_paraphrases.append(knn_filter(
                    row['question'], question_paraphrases, from_index, topk))
            elif filter_method == 'sbert':
                filtered_question_paraphrases.append(sbert_filter(
                    row['question'], question_paraphrases, from_index, topk))
            else:
                filtered_question_paraphrases.append(question_paraphrases)

    else:
        for idx, row in tqdm(df.iterrows(), desc='Generating paraphrases', total=len(df)):
            question_paraphrases = row['question_paraphrase']
            question_paraphrases = ast.literal_eval(question_paraphrases)

            if filter_method == 'knn':
                filtered_question_paraphrases.append(knn_filter(
                    row['question'], question_paraphrases, from_index, topk))
            elif filter_method == 'sbert':
                filtered_question_paraphrases.append(sbert_filter(
                    row['question'], question_paraphrases, from_index, topk))
            else:
                filtered_question_paraphrases.append(question_paraphrases)

    df['question_paraphrase'] = filtered_question_paraphrases
    return df


def generate_json_w_paraphrases(data_filepath, num_paraphrase, output_filepath=''):
    with open(data_filepath, 'r') as file:
        data = json.load(file)

    annotations = data.get('annotations', {})

    for annotation_id, annotation in annotations.items():
        question = annotation.get('question', "")
        if question:
            paraphrases = get_paraphrase(question, num_paraphrase)
            annotation['paraphrases'] = paraphrases

    if output_filepath:
        with open(output_filepath, 'w') as outfile:
            json.dump(data, outfile, ensure_ascii=False, indent=4)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate a new dataset")
    parser.add_argument("--train_filepath", type=str, required=True)
    parser.add_argument("--num_params", type=int,
                        default=20, help="Number of parameters")
    parser.add_argument("--random_seed", type=int, default=59)
    parser.add_argument("--save_filepath", type=str,
                        default="filtered_paraphrases_train.csv")
    parser.add_argument("--is_paraphrased", type=lambda x: (str(x).lower()
                        == 'true'), default=False, help="Is paraphrased dataset?")
    parser.add_argument("--filter_method", type=str, default='knn',
                        help="Select a filtering method: 'knn', 'sbert', 'no' (default: 'knn')")
    parser.add_argument("--from_index", type=int, default=0,
                        help="Select the start index to get paraphrases from after sorted in descending cosine similarity order (default: 0)")
    parser.add_argument("--topk", type=int, default=10,
                        help="Select the top-k results (default: 10)")

    return parser.parse_args()


def main():
    args = parse_arguments()
    train_filepath = args.train_filepath
    num_params = args.num_params
    random_seed = args.random_seed
    save_filepath = args.save_filepath
    is_paraphrased = args.is_paraphrased
    filter_method = args.filter_method
    from_index = args.from_index
    topk = args.topk

    set_seed(random_seed)

    if '.csv' in train_filepath:
        paraphrase_df = generate_df_w_filter_paraphrases(data_filepath=train_filepath,
                                                         is_paraphrased=is_paraphrased,
                                                         num_paraphrase=num_params,
                                                         filter_method=filter_method,
                                                         from_index=from_index,
                                                         topk=topk)

        if save_filepath:
            paraphrase_df.to_csv(save_filepath, index=False)
            print(f'Results saved to {save_filepath}')

    else:
        generate_json_w_paraphrases(data_filepath=train_filepath,
                                    num_paraphrase=num_params,
                                    output_filepath=save_filepath)

    print('Paraphrases generation completed!')


if __name__ == "__main__":
    main()
