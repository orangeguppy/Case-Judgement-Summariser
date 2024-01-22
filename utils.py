"""
Utility functions for dataset/dataloader, training/testing/validation functions, and any other miscellaneous helper functions.
Add as needed!!
"""
import logging

def generate_dataset():
    pass

# This method should be called to train the model on the Training set. 
# Need to include code for saving the model weights when performance on the Validation set is best
def train(model, device):
    pass

# This method should be called for testing on both Validation and Test sets
# Make sure to include code to load the correct model weights
def test(model, device): 
    pass

def setup_logging():
    # Create a logger
    logger = logging.getLogger('train_test_logger')

    # Create a file handler
    file_handler = logging.FileHandler('output.log')

    # Define the log message format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)  # Set the formatter for the file handler

    # Add the file handler to the logger
    logger.addHandler(file_handler)

    # Set logging level
    logger.setLevel(logging.INFO)

    return logger