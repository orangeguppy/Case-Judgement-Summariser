"""
1) Loading/Preprocessing data. Includes tokenisation/anything we need to do to the data before using it to train the model.
2) Split dataset into train, validation, and test sets
3) Set hyperparameters (refer to Google Doc)
4) Instantiate model
5) Call the train() method to train the model. 
    We will need to test model performance at the end of each epoch on the Validation set as well during the training process.
    The Validation set is supposed to be like a proxy for the Testing dataset, and is used to select the best version of the model for 
    evaluating at the end.
    The number of epochs is the number of times the entire dataset is passed through the model.
6) Call the test() method to evaluate model performance

Add method parameters as needed, this is just a template/pseudocode. 
Don't need to follow the steps in this specific order either, it's just a list of things we need to do

Some helpful documentation:
https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html for training a generic classifier
"""
import model
import utils

# This line is to set the model to use the GPU if available, else the CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Generate Dataset
dataset = utils.generate_dataset() # Maybe can create a separate method for any additional preprocessing/tokenisation?
                                    # Just do whichever is easier, no hard/fast rules

# Specify hyperparameters, we probably will need to pass these hyperparameters into the train() method later on!

# Split dataset and create Dataloaders for each training set
# Need to create the train, validation, and testing dataset
# To load the data from each of the train, validation, and testing datasets, we need to create corresponding Dataloaders.
# train_loader, val_loader, test_loader =  # Do modify as needed, don't necessarily need to do it like this!!

# Instantiate model
# model = 
model.to(device) # Move the model to the GPU if available, else move it to the CPU

# Train the model
utils.train()

# Test the model
utils.test()