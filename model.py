from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
"""
For specifying model architecture. Can add any other helper functions
"""
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
from random import shuffle, seed
import os

class SummarizationModel:  #added option to continue training with trained weights
    def __init__(self, device, model_name="t5-small", model_weights_path=None, tokenizer_weights_path=None):
        if model_weights_path is not None:
            self.model = T5ForConditionalGeneration.from_pretrained(model_weights_path)
        else:
            self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        
        if tokenizer_weights_path is not None:
            self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_weights_path)
        else:
            self.tokenizer = T5Tokenizer.from_pretrained(model_name)
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

# country = "India"
# judgement_folder = f"dataset/{country}/judgement"
# for file_name in os.listdir(judgement_folder):
#     file_path = os.path.join(judgement_folder, file_name)
#     with open(file_path, 'r') as file:
#         text = file.read()
#     summary = model.summarize(text)
#     print(file_name, " summary: ", summary)

@app.route("/T5", methods=["POST"])
@cross_origin()
def get_summary():
    if request.is_json:
        data = request.json
        if "text" in data:
            input_text = data["text"]
            model = SummarizationModel(device="cpu", model_name="t5-small")
            model.model = T5ForConditionalGeneration.from_pretrained('weights/T5_india/model_weights_best_validation/')
            # model.load_state_dict(torch.load('weights/T5_india/model_weights_best_validation/model.safetensors'))
            output_summary = model.summarize(input_text)

            return jsonify({"summary": output_summary}), 200
        else:
            return jsonify({"error": "Missing 'text' field in JSON data"}), 400
    else:
        return jsonify({"error": "Request data must be in JSON format"}), 400

if __name__ == "__main__":
    app.run(debug=True)

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