import logging

def setup_logging():
    # Create a logger
    logger = logging.getLogger('train_test_logger')  

    if not logger.handlers: #resolve the double logging issue
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

# Call setup_logging() once at the beginning of your script
logger = setup_logging()