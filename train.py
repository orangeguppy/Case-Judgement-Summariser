from model import SummarizationModel
import os
from random import shuffle, seed
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
from torch.utils.data import DataLoader, TensorDataset

#offical training code

#declare model
model1 = SummarizationModel()

#filepaths for x and y
judgement_folder = "your file path"
summary_folder = "your file path"

#split filepaths into training and testing
train_X, test_X, train_y, test_y = model1.split_data_train_test1(judgement_folder, summary_folder, 0.2)

#read contents and tokenize. 
train_data = model1.tokenize_data(
    [open(file, 'r').read() for file in train_X],
    [open(file, 'r').read() for file in train_y],
    model1.tokenizer
)

test_data = model1.tokenize_data(
    [open(file, 'r').read() for file in test_X],
    [open(file, 'r').read() for file in test_y],
    model1.tokenizer
)

# Initialize optimizer
optimizer = torch.optim.AdamW(model1.model.parameters(), lr=1e-5)

# Create DataLoader for training data
batch_size = 3
train_dataset = TensorDataset(
    train_data['input_ids'],
    train_data['attention_mask'],
    train_data['labels']
)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Train the model
num_epochs = 3

for epoch in range(num_epochs):
    average_loss = model1.train_epoch1(train_loader, optimizer)
    print(f"Epoch {epoch+1}, Average Loss: {average_loss}")

# Save the trained model
model1.save("a filepath")





        

