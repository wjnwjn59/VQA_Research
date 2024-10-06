import json
import shutil
import pandas as pd
import os

# A mapping of dataset names to their formal titles
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
    
    # Construct paths to the training and testing CSV files
    train_csv_path = os.path.join(data_dir, 'ViVQA', 'train.csv')
    test_csv_path = os.path.join(data_dir, 'ViVQA', 'test.csv')

    # Read the CSV files into pandas DataFrames
    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)

    # Get unique answers from both datasets
    train_answers = train_df['answer'].unique().tolist()
    test_answers = test_df['answer'].unique().tolist()
    
    print(train_answers)
    print(test_answers)

    # Create a set of all unique answers
    answer_space = set(list(train_answers + test_answers))

    # Save the answer space to a text file
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
    
    # Construct paths to the training and development JSON files
    train_json_path = os.path.join(data_dir, 'OpenViVQA', 'vlsp2023_train_data.json')
    dev_json_path = os.path.join(data_dir, 'OpenViVQA', 'vlsp2023_dev_data.json')

    answer_space = set()

    # Loop through each JSON file to extract answers
    for json_path in [train_json_path, dev_json_path]:
        with open(json_path, 'r') as file:
            data = json.load(file)
        
        # Get annotations from the JSON data
        annotations = data.get('annotations', {})
        
        # Add each unique answer to the answer space
        for annotation in annotations.values():
            answer = annotation['answer']
            answer_space.add(answer)

    # Save the answer space to a text file, sorted alphabetically
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
    
    # Load the answer space from the text file
    save_path = os.path.join(data_dir, DATASET_NAME_MAP[dataset_name], 'answer_space.txt')
    with open(save_path, 'r') as f:
        lines = f.read().splitlines()

    # Create mappings from labels to indices and vice versa
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
    
    # Create the destination folder if it doesn't exist
    os.makedirs(destination_folder, exist_ok=True)

    # Loop through each file in the source folder
    for filename in os.listdir(source_folder):
        source_path = os.path.join(source_folder, filename)
        
        if not os.path.isfile(source_path):
            continue
        
        # Rename the file by keeping only the part after the last underscore
        new_filename = filename.split('_')[-1]
        
        # Create the destination path for the copied file
        destination_path = os.path.join(destination_folder, new_filename)

        # Copy and rename the image
        shutil.copy2(source_path, destination_path)
        print(f"Copied and renamed: {filename} -> {new_filename}")


if __name__ == '__main__':
    #generate_answer_space_file_vivqa('/home/VLAI/datasets')
    generate_answer_space_file_openvivqa('/home/VLAI/datasets')
    # source_folder = '/home/VLAI/datasets/COCO_Images/val2014'
    # destination_folder = '/home/VLAI/datasets/COCO_Images/merge'
    # copy_and_rename_images(source_folder, destination_folder)