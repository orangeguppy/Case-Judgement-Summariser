"""
Utility functions for dataset processing and loading
"""
import os
from random import shuffle, seed

def tokenize_data(texts, summaries, tokenizer):
    tokenized_data = tokenizer(
        texts,
        max_length=512,  # Adjust the max_length as needed
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    tokenized_data['labels'] = tokenizer(
        summaries,
        max_length=200,  # Adjust the max_length for summaries as needed
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        return_attention_mask=True,
    )['input_ids']
    return tokenized_data

# function to split Judgement and Summary to train and test sets. 
def split_data_train_test1(full_text_folder, summary_folder, test_ratio, random_seed=None):
    # Get list of files from both folders
    full_text_files = os.listdir(os.path.abspath(full_text_folder))
    summary_files = os.listdir(os.path.abspath(summary_folder))

    # Ensure the same files exist in both folders
    assert set(full_text_files) == set(summary_files), "Folders must contain the same files."

    # Combine and shuffle the files with optional random seed
    combined_files = list(zip(full_text_files, summary_files))
    if random_seed is not None:
        seed(random_seed)
    shuffle(combined_files)

    # Calculate split index based on test ratio
    split_index = round(len(combined_files) * test_ratio)

    # Separate the combined files back into full text and summary files
    shuffled_full_text_files, shuffled_summary_files = zip(*combined_files)

    # Create lists of file paths for training and testing sets
    training_full_text = [os.path.join(full_text_folder, file) for file in shuffled_full_text_files[:split_index]]
    testing_full_text = [os.path.join(full_text_folder, file) for file in shuffled_full_text_files[split_index:]]

    training_summary = [os.path.join(summary_folder, file) for file in shuffled_summary_files[:split_index]]
    testing_summary = [os.path.join(summary_folder, file) for file in shuffled_summary_files[split_index:]]

    # Returns lists of full file paths for training and test sets for both full text and summary
    return training_full_text, testing_full_text, training_summary, testing_summary

# Phase out?
# def split_data_train_test(folder_name, test_ratio, random_seed=None):
#     files = os.listdir(os.path.abspath(folder_name))
#     if random_seed != None:
#         shuffle(files, random_seed=random_seed)
#     else:
#         shuffle(files)
#     split_index = round(len(files) * test_ratio)
#     training = [os.path.join(folder_name, file) for file in files[:split_index]]
#     testing = [os.path.join(folder_name, file) for file in files[split_index:]]
#     # Returns list of full file paths for training and test set respectively
#     return training, testing