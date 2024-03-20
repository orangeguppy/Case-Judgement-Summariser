from model import SummarizationModel_Pegasus, SummarizationModel
import torch
from bert_score import score
import os
import dataset_utils
import random
import numpy as np
from transformers import BertTokenizer, BertModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = SummarizationModel_Pegasus(device, model_name="google/pegasus-large", model_weights_path=f"weights/Pegasus/model_weights_best_validation/", tokenizer_weights_path=f"weights/Pegasus/tokenizer_weights_best_validation/")
#model = SummarizationModel(device, model_name="google/pegasus-large", model_weights_path=f"weights/T5_india/model_weights_best_validation/", tokenizer_weights_path=f"weights/T5_india/tokenizer_weights_best_validation/")
print("model created")

def evaluate_bert(original_text, generated_summary):
    _, _, bert_scores = score([generated_summary], [original_text], lang="en", model_type="bert-base-uncased", nthreads=4)
    return bert_scores.numpy().mean()

def evaluate_bert1(original_text, generated_summary, tokenizer, bert_model):
    inputs1 = tokenizer(generated_summary, return_tensors="pt", padding=True, truncation=True)
    inputs2 = tokenizer(original_text, return_tensors="pt", padding=True, truncation=True)
    outputs1 = bert_model(**inputs1)
    outputs2 = bert_model(**inputs2)
    embeddings1 = outputs1.last_hidden_state.mean(dim=1).detach().numpy()
    embeddings2 = outputs2.last_hidden_state.mean(dim=1).detach().numpy()
    similarity = np.dot(embeddings1, embeddings2.T) / (np.linalg.norm(embeddings1) * np.linalg.norm(embeddings2))
    similarity = similarity[0][0]
    return similarity

country = "India"

# Declare filepaths for x and y (attrbutes and labels)
judgement_folder = f"dataset/{country}/judgement"
summary_folder = f"dataset/{country}/summary"

random.seed(42)
train_X, test_X, train_y, test_y = dataset_utils.split_data_train_test1(judgement_folder, summary_folder, 0.95)

bert_scores_list = []
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")
i = 0
for file in test_X[:10]:
    print(i," file")
    with open(file, "r", encoding="utf-8") as file:
        original_text = file.read()
    
    # Summarize text
    generated_summary = model.summarize(original_text)
    print("summary genereated")
    # Evaluate summary
    bert_score = evaluate_bert1(original_text, generated_summary, tokenizer, bert_model)
    print("evaluated")
    bert_scores_list.append(bert_score)
    i+=1
    
average_bert_score = sum(bert_scores_list) / len(bert_scores_list)

print(f"Average: {average_bert_score}")