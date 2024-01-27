"""
Script for model training.
"""
from model import SummarizationModel
import os
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
from torch.utils.data import DataLoader, TensorDataset
import dataset_utils
import train_test_utils
import utils

# Set the model to use the GPU if available, else the CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Declare and instantiate model
model1 = SummarizationModel(device, model_name="t5-small")

# Declare country
country = "India"

# Declare filepaths for x and y (attrbutes and labels)
judgement_folder = f"dataset/{country}/judgement"
summary_folder = f"dataset/{country}/summary"

# Declare hyperparameters
num_epochs = 3
lr = 1e-5
batch_size = 3

# Create logger
logger = utils.setup_logging()

# Split filepaths into training and testing
train_X, test_X, train_y, test_y = dataset_utils.split_data_train_test1(judgement_folder, summary_folder, 0.2)

# Read contents and tokenize. 
train_data = dataset_utils.tokenize_data(
    [open(file, 'r').read() for file in train_X],
    [open(file, 'r').read() for file in train_y],
    model1.tokenizer
)

test_data = dataset_utils.tokenize_data(
    [open(file, 'r').read() for file in test_X],
    [open(file, 'r').read() for file in test_y],
    model1.tokenizer
)

# Initialize optimizer
optimizer = torch.optim.AdamW(model1.model.parameters(), lr=lr)

# Create Tensor Train/Test datasets
train_dataset = TensorDataset(
    train_data['input_ids'],
    train_data['attention_mask'],
    train_data['labels']
)

test_dataset = TensorDataset(
    test_data['input_ids'],
    test_data['attention_mask'],
    test_data['labels']
)

# Create Train/Test dataloaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# Train the model
for epoch in range(num_epochs):
    average_loss = model1.train_epoch1(train_loader, optimizer)
    print(f"Epoch {epoch+1}, Average Loss: {average_loss}") # Print t console
    logger.info(f"Epoch {epoch+1}, Average Loss: {average_loss}") # Save to log file

    # If the model performed best on a validation dataset, save the weights


# Save the trained model
model1.save("best_validation_weights.pt")

# Evaluate model performance
