"""
This is the pipeline for model training.
1) Loading/Preprocessing data. Includes tokenisation/anything we need to do to the data before using it to train the model.
2) Split dataset into train, validation, and test sets
3) Set hyperparameters (refer to Google Doc). We can explore RandomSearch, GridSearch etc (these are methods for exploring the possible combinations of hyperparameters we can use)
4) Instantiate model
5) Training

Add method parameters as needed, this is just a template/pseudocode. 
Don't need to follow the steps in this specific order either, it's just a list of things we need to do

Clarifying hyperparameters vs parameters (excerpt from the internet):
Parameters allow the model to learn the rules from the data while hyperparameters control how the model is training. 
Parameters learn their own values from data. 
That's why learning rate/batch size etc are hyperparameters, while model weights are parameters

Some helpful documentation:
https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html for training a generic classifier
"""
import model
import utils
import mlflow

# This line is to set the model to use the GPU if available, else the CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Setup logger
logger = utils.setup_logging()

# Generate Dataset
dataset = utils.generate_dataset() # Maybe can create a separate method for any additional preprocessing/tokenisation?
                                    # Just do whichever is easier, no hard/fast rules

# Specify hyperparameters, we probably will need to pass these hyperparameters into the train() method later on!

# Split dataset and create Dataloaders for each training set
# Need to create the train, validation, and testing dataset
# To load the data from each of the train, validation, and testing datasets, we need to create corresponding Dataloaders.

# Instantiate model
# model = 
model.to(device) # Move the model to the GPU if available, else move it to the CPU

# Train the model
utils.train(model, device=device)

# Test the model
utils.test(model, device=device)