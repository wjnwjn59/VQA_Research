import json
import shutil
import pandas as pd
import os


DATASET_NAME_MAP = {
    'vivqa': 'ViVQA',
    'openvivqa': 'OpenViVQA'
}


def generate_answer_space_file_vivqa(data_dir):
    """
    Generate an answer space file for the ViVQA dataset by combining 
    unique answers from the training and testing CSV files.

    Parameters:
        data_dir (str): The directory containing the dataset.
    """

    train_csv_path = os.path.join(data_dir, 'ViVQA', 'train.csv')
    test_csv_path = os.path.join(data_dir, 'ViVQA', 'test.csv')

    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)

    train_answers = train_df['answer'].unique().tolist()
    test_answers = test_df['answer'].unique().tolist()

    print(train_answers)
    print(test_answers)

    answer_space = set(list(train_answers + test_answers))

    save_path = os.path.join(data_dir, 'ViVQA', 'answer_space.txt')
    with open(save_path, 'w+') as f:
        f.write('\n'.join(answer_space))


def generate_answer_space_file_openvivqa(data_dir):
    """
    Generate an answer space file for the OpenViVQA dataset by extracting 
    unique answers from the JSON training and development files.

    Parameters:
        data_dir (str): The directory containing the dataset.
    """

    train_json_path = os.path.join(
        data_dir, 'OpenViVQA', 'vlsp2023_train_data.json')
    dev_json_path = os.path.join(
        data_dir, 'OpenViVQA', 'vlsp2023_dev_data.json')

    answer_space = set()

    for json_path in [train_json_path, dev_json_path]:
        with open(json_path, 'r') as file:
            data = json.load(file)

        annotations = data.get('annotations', {})

        for annotation in annotations.values():
            answer = annotation['answer']
            answer_space.add(answer)

    save_path = os.path.join(data_dir, 'OpenViVQA', 'answer_space.txt')
    with open(save_path, 'w+') as f:
        f.write('\n'.join(sorted(answer_space)))

    print(f"Answer space file generated at: {save_path}")
    print(f"Total unique answers: {len(answer_space)}")


def get_label_encoder(data_dir, dataset_name):
    """
    Retrieve a label encoder mapping from answers to indices and vice versa,
    as well as the length of the answer space.

    Parameters:
        data_dir (str): The directory containing the dataset.
        dataset_name (str): The name of the dataset (vivqa or openvivqa).

    Returns:
        tuple: A tuple containing the label-to-index mapping, 
               index-to-label mapping, and the length of the answer space.
    """

    save_path = os.path.join(
        data_dir, DATASET_NAME_MAP[dataset_name], 'answer_space.txt')
    with open(save_path, 'r') as f:
        lines = f.read().splitlines()

    label2idx = {label: idx for idx, label in enumerate(lines)}
    idx2label = {idx: label for idx, label in enumerate(lines)}
    answer_space_len = len(lines)

    return label2idx, idx2label, answer_space_len


def copy_and_rename_images(source_folder, destination_folder):
    """
    Copy images from the source folder to the destination folder, renaming them 
    in the process to only keep the last part of the filename after the last underscore.

    Parameters:
        source_folder (str): The folder to copy images from.
        destination_folder (str): The folder to copy images to.
    """

    os.makedirs(destination_folder, exist_ok=True)

    for filename in os.listdir(source_folder):
        source_path = os.path.join(source_folder, filename)

        if not os.path.isfile(source_path):
            continue

        new_filename = filename.split('_')[-1]

        destination_path = os.path.join(destination_folder, new_filename)

        shutil.copy2(source_path, destination_path)
        print(f"Copied and renamed: {filename} -> {new_filename}")


if __name__ == '__main__':
    generate_answer_space_file_openvivqa('/home/VLAI/datasets')
