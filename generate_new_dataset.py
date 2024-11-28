from vqa_datasets.augmentations import mt5_paraphraser
from config import pipeline_config
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


def generate_df_paraphrases_chatgpt(data_filepath, num_paraphrase, save_filepath):
    df = pd.read_csv(data_filepath)

    if not os.path.exists(save_filepath):
        df['question_paraphrase'] = ["[]"] * len(df)
        df.to_csv(save_filepath, index=False)

    df_output = pd.read_csv(save_filepath)

    if 'question_paraphrase' not in df_output.columns:
        df_output['question_paraphrase'] = ["[]"] * len(df_output)

    for idx, row in tqdm(df_output.iterrows(), desc='Generating paraphrases', total=len(df_output)):
        if row['question_paraphrase'] != "[]":
            continue

        try:
            question_paraphrases = gpt_paraphraser.get_gpt_paraphrase(
                row['question'], num_paraphrase)
        except Exception as e:
            print(f"Error at index {idx}: {e}")
            continue

        df_output.at[idx, 'question_paraphrase'] = question_paraphrases
        df_output.to_csv(save_filepath, index=False)

    missing_paraphrases = (df_output['question_paraphrase'] == "[]").sum()
    if missing_paraphrases == 0:
        print(f'The process went smoothly and saved to: {save_filepath}')
    else:
        print(f'There were {missing_paraphrases} missing paraphrases.')

    return df_output


def generate_df_paraphrases_mt5(data_filepath, num_paraphrase, save_filepath):
    df = pd.read_csv(data_filepath)
    question_paraphrases = []

    mt5_paraphraser.init_model(pipeline_config.paraphraser_id)
    for _, row in tqdm(df.iterrows(), desc='Generating paraphrases', total=len(df)):
        paraphrases = mt5_paraphraser.get_paraphrase(
            row['question'], num_paraphrase)
        question_paraphrases.append(paraphrases)

    df['question_paraphrase'] = question_paraphrases
    save_filepath.to_csv(save_filepath, index=False)
    print(f'The process went smoothly and saved to: {save_filepath}')

    return df


def generate_json_w_paraphrases(data_filepath, num_paraphrase, output_filepath=''):
    with open(data_filepath, 'r') as file:
        data = json.load(file)

    mt5_paraphraser.init_model(pipeline_config.paraphraser_id)
    annotations = data.get('annotations', {})
    for annotation_id, annotation in annotations.items():
        question = annotation.get('question', "")
        if question:
            paraphrases = mt5_paraphraser.get_paraphrase(
                question, num_paraphrase)
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
                        default="paraphrases.csv")
    parser.add_argument("--paraphrase_method", type=(lambda x: x.lower()), default='mt5',
                        help="Select a paraphrase method: 'mt5', 'gpt' (default: 'mt5)")

    return parser.parse_args()


def main():
    args = parse_arguments()
    train_filepath = args.train_filepath
    num_params = args.num_params
    random_seed = args.random_seed
    save_filepath = args.save_filepath
    paraphrase_method = args.paraphrase_method
    set_seed(random_seed)

    # if '.csv' in train_filepath:
    #     paraphrase_df = generate_df_w_filter_paraphrases(data_filepath=train_filepath,
    #                                                      is_paraphrased=is_paraphrased,
    #                                                      num_paraphrase=num_params,
    #                                                      filter_method=filter_method,
    #                                                      from_index=from_index,
    #                                                      topk=topk)

    #     if save_filepath:
    #         paraphrase_df.to_csv(save_filepath, index=False)
    #         print(f'Results saved to {save_filepath}')

    # else:
    #     generate_json_w_paraphrases(data_filepath=train_filepath,
    #                                 num_paraphrase=num_params,
    #                                 output_filepath=save_filepath)

    generate_json_w_paraphrases(data_filepath=train_filepath,
                                num_paraphrase=num_params,
                                output_filepath=save_filepath)

    print('Paraphrases generation completed!')


if __name__ == "__main__":
    main()
