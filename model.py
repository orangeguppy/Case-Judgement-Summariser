"""
For specifying model architecture. Can add any other helper functions
"""
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
from random import shuffle, seed
import os

class SummarizationModel:
    def __init__(self, device, model_name="t5-small"):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.device = device
        self.model.to(device)

    def save(self, path):
        model_weights_path = "model_weights_" + path
        tokenizer_weights_path = "tokenizer_weights_" + path
        self.model.save_pretrained(model_weights_path)
        self.tokenizer.save_pretrained(tokenizer_weights_path)

    def summarize1(self, input_ids, max_length=150, min_length=40):
        self.model.eval()
        # input_ids = self.tokenizer.encode(text, return_tensors='pt').to(self.device)
        summary_ids = self.model.generate(input_ids, max_length=max_length, min_length=min_length, length_penalty=2.0, num_beams=4, early_stopping=True)
        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    def summarize(self, text, max_length=150, min_length=40):
        self.model.eval()
        input_ids = self.tokenizer.encode(text, return_tensors='pt').to(self.device)
        summary_ids = self.model.generate(input_ids, max_length=max_length, min_length=min_length, length_penalty=2.0, num_beams=4, early_stopping=True)
        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Phase out?
    # def train_epoch(self, train_loader, optimizer):
    #     self.model.train()
    #     total_loss = 0
    #     for batch in train_loader:
    #         input_ids = batch['text_input_ids'].to(self.device)
    #         attention_mask = batch['text_attention_mask'].to(self.device)
    #         labels = batch['summary_input_ids'].to(self.device)

    #         # Clear previously calculated gradients
    #         optimizer.zero_grad()

    #         # Forward pass
    #         outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    #         loss = outputs.loss

    #         # Backward pass and optimize
    #         loss.backward()
    #         optimizer.step()

    #         total_loss += loss.item()
        
    #     return total_loss / len(train_loader)