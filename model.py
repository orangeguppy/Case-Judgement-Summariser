from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
"""
For specifying model architecture. Can add any other helper functions
"""
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import BartForConditionalGeneration, BartTokenizer
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
    
@app.route("/bart", methods=["POST"])
@cross_origin()
def get_bart_summary():
    if request.is_json:
        data = request.json
        if "text" in data:
            input_text = data["text"]
            model = SummarizationModel(device="cpu", model_name="t5-small")
            model.model = BartForConditionalGeneration.from_pretrained("facebook/bart-large", forced_bos_token_id=0)
            model.tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
            output_summary = model.summarize(input_text)

            return jsonify({"summary": output_summary}), 200
        else:
            return jsonify({"error": "Missing 'text' field in JSON data"}), 400
    else:
        return jsonify({"error": "Request data must be in JSON format"}), 400

if __name__ == "__main__":
    app.run(debug=True)