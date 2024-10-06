import os

# Set the CUDA and world size environment variables for GPU settings.
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["WORLD_SIZE"] = '1'

import json
import pandas as pd
import argparse
import time
import torch
import random
import numpy as np

from .vqa_datasets.augmentations.paraphraser import get_paraphrase

# Set the random seed for reproducibility in all libraries (random, NumPy, PyTorch, CUDA).
def set_seed(random_seed):
    random.seed(random_seed)  # Set the random seed for the 'random' module
    np.random.seed(random_seed)  # Set the random seed for NumPy
    torch.manual_seed(random_seed)  # Set the seed for PyTorch CPU operations
    torch.cuda.manual_seed(random_seed)  # Set the seed for PyTorch CUDA operations (single GPU)
    torch.cuda.manual_seed_all(random_seed)  # Set the seed for PyTorch CUDA operations (multi-GPU)
    
    # Ensure deterministic behavior in cuDNN (for consistent training results)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    
    # Set environment variables for Python and CUDA to further enforce determinism
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ['CUDNN_DETERMINISTIC'] = '1'


# Generate a DataFrame with paraphrases based on a CSV input file
def generate_df_w_paraphrases(data_filepath, num_paraphrase):
    df = pd.read_csv(data_filepath)
    df['question_paraphrase'] = df['question'].apply(lambda x: get_paraphrase(x, num_paraphrase))
    
    return df


# Generate a JSON file with paraphrases based on a JSON input file
def generate_json_w_paraphrases(data_filepath, num_paraphrase, output_filepath=''):
    with open(data_filepath, 'r') as file:
        data = json.load(file)
    
    annotations = data.get('annotations', {})
    
    # Loop over all annotations and generate paraphrases for each question
    for annotation_id, annotation in annotations.items():
        question = annotation.get('question', "")
        if question:
            paraphrases = get_paraphrase(question, num_paraphrase)
            annotation['paraphrases'] = paraphrases
    
    # If an output file path is provided, save the updated data back to a new JSON file
    if output_filepath:
        with open(output_filepath, 'w') as outfile:
            json.dump(data, outfile, ensure_ascii=False, indent=4)
    

# Parse command line arguments for the script
def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate a new dataset")
    parser.add_argument("--train_filepath", type=str, required=True)
    parser.add_argument("--num_params", type=int, default=20, help="Number of parameters")
    parser.add_argument("--random_seed", type=int, default=59)
    parser.add_argument("--save_filepath", type=str, default="paraphrases_train.csv")

    return parser.parse_args()


# Main function to drive the paraphrase generation process
def main():
    args = parse_arguments()
    train_filepath = args.train_filepath
    num_params = args.num_params
    random_seed = args.random_seed
    save_filepath = args.save_filepath

    print('Start processing...')
    start_time = time.time()
    set_seed(random_seed)

    if '.csv' in train_filepath:
        paraphrase_df = generate_df_w_paraphrases(data_filepath=train_filepath,
                                                num_paraphrase=num_params)
        if save_filepath:
            paraphrase_df.to_csv(save_filepath, index=False)
            print(f'Results saved to {save_filepath}')
    else:
        generate_json_w_paraphrases(data_filepath=train_filepath,
                                    num_paraphrase=num_params,
                                    output_filepath=save_filepath)
        
    print('Paraphrases generation completed!')
    print(f'Processing time: {time.time() - start_time}')


if __name__ == "__main__":
    main()