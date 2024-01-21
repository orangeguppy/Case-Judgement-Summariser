"""
For specifying model architecture. Can add any other helper functions
"""
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
from random import shuffle
import os

class SummarizationModel:
    def __init__(self, model_name="t5-small"):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def split_data_train_test(folder_name, test_ratio, random_seed=None):
        files = os.listdir(os.path.abspath(folder_name))
        if random_seed != None:
            shuffle(files, random_seed=random_seed)
        else:
            shuffle(files)
        split_index = round(len(files) * test_ratio)
        training = [os.path.join(folder_name, file) for file in files[:split_index]]
        testing = [os.path.join(folder_name, file) for file in files[split_index:]]
        # Returns list of full file paths for training and test set respectively
        return training, testing

    def train_epoch(self, train_loader, optimizer):
        self.model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch['text_input_ids'].to(self.device)
            attention_mask = batch['text_attention_mask'].to(self.device)
            labels = batch['summary_input_ids'].to(self.device)

            # Clear previously calculated gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        return total_loss / len(train_loader)

    def evaluate(self, eval_loader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in eval_loader:
                input_ids = batch['text_input_ids'].to(self.device)
                attention_mask = batch['text_attention_mask'].to(self.device)
                labels = batch['summary_input_ids'].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

                total_loss += loss.item()

        return total_loss / len(eval_loader)

    def save(self, path):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def summarize(self, text, max_length=150, min_length=40):
        self.model.eval()
        input_ids = self.tokenizer.encode(text, return_tensors='pt').to(self.device)
        summary_ids = self.model.generate(input_ids, max_length=max_length, min_length=min_length, length_penalty=2.0, num_beams=4, early_stopping=True)
        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
