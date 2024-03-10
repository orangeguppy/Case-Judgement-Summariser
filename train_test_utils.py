from nltk.translate import meteor_score
from nltk import word_tokenize
import nltk
nltk.download('punkt')
nltk.download('wordnet')
import torch
import utils
from bert_score import BERTScorer
from transformers import BertTokenizer, BertModel
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu
import numpy as np
logger = utils.setup_logging()

def train_epoch1(device, model, epoch, train_loader, val_loader, optimizer):
    # Log the current batch / max batches, and print the epoch
    # Save the model more frequently, probably every few hundred batches. Can be a bit safer at the start
    model.model.train()
    total_loss = 0
    i = 0
    best_validation_performance = 0
    batch_count = 0
    total_loss_30_batches = 0
    total_num_batches = len(train_loader)

    for batch in train_loader:

        print("currently training for batch: " + str(batch_count))

        input_ids = batch[0].to(device)  # Assuming input_ids is the first element in the batch
        attention_mask = batch[1].to(device)  # Assuming attention_mask is the second element in the batch
        labels = batch[2].to(device)  # Assuming labels is the third element in the batch

        #print("setting optimiser to zero gradient")
        optimizer.zero_grad()

        outputs = model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        loss.backward()
        print("done with backpropagation")

        optimizer.step()
        # print("done with optimiser")

        total_loss += loss.item()
        total_loss_30_batches += loss.item()

        print("calculating loss")

        if (i + 1) % 30 == 29:
            print(i)
            avg_loss = total_loss_30_batches / batch_count

            # Evaluate on the validation set
            meteor_score = evaluate_meteor(device, model, val_loader)
            print(f"{epoch + 1}, {i + 1}/{total_num_batches}, Average Loss: {avg_loss:.4f}, METEOR: {meteor_score}")
            logger.info(f"{epoch + 1}, {i + 1}/{total_num_batches}, Average Loss: {avg_loss:.4f}, METEOR: {meteor_score}")
            total_loss_30_batches = 0
            batch_count = 0

            model.save("checkpoint")

            if (meteor_score > best_validation_performance):
                model.save("best_validation")
                logger.info("Best validation performance. Model weights saved.")
        i += 1
        batch_count += 1

    meteor_score = evaluate_meteor(device, model, val_loader)
    return total_loss / len(train_loader), meteor_score

def evaluate(device, model, val_loader):
    model.model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['text_input_ids'].to(device)
            attention_mask = batch['text_attention_mask'].to(device)
            labels = batch['summary_input_ids'].to(device)

            outputs = model.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            total_loss += loss.item()

    return total_loss / len(val_loader)



def evaluate_meteor(device, model, val_loader): #added rouge and bleu
    model.model.eval()
    meteor_scores = []
    bert_scores = []
    rouge_scores = []
    bleu_scores = []
    batch_count = 0

    # Initialize tokenizer and BERT model once
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertModel.from_pretrained("bert-base-uncased")
    
    #Initialize rouge
    rouge = Rouge()

    smooth_func = SmoothingFunction().method1
    with torch.no_grad():
        for batch in val_loader:

            print("Evaluating for batch: " + str(batch_count))

            input_ids, attention_mask, labels = batch[0].to(device), batch[1].to(device), batch[2].to(device)

            # Generate summaries for the whole batch
            generated_summaries = [model.summarize1(ids.unsqueeze(0)) for ids in input_ids]

            print("Summaries generated")

            # Decode label summaries once per batch
            decoded_labels = [model.tokenizer.decode(label, skip_special_tokens=True) for label in labels]
            tokenised_labels = [word_tokenize(label) for label in decoded_labels]

            print("Labels labelled")

            for generated_summary, tokenised_label_summary in zip(generated_summaries, tokenised_labels):

                print("Calculating for this summary")

                # Ensure summaries are in string format
                generated_summary = ' '.join(generated_summary) if isinstance(generated_summary, list) else generated_summary
                tokenised_generated_summary = word_tokenize(generated_summary)

                # Calculate METEOR score
                meteor_score_value = meteor_score.single_meteor_score(tokenised_label_summary, tokenised_generated_summary)
                meteor_scores.append(meteor_score_value)

                # BERTScore
                inputs1 = tokenizer(generated_summary, return_tensors="pt", padding=True, truncation=True)
                inputs2 = tokenizer(' '.join(tokenised_label_summary), return_tensors="pt", padding=True, truncation=True)
                outputs1 = bert_model(**inputs1)
                outputs2 = bert_model(**inputs2)
                embeddings1 = outputs1.last_hidden_state.mean(dim=1).detach().numpy()
                embeddings2 = outputs2.last_hidden_state.mean(dim=1).detach().numpy()
                
                # Calculate BLEU score
                bleu_score_value = sentence_bleu(generated_summary, label_summary)
                bleu_scores.append(bleu_score_value)

                similarity = np.dot(embeddings1, embeddings2.T) / (np.linalg.norm(embeddings1) * np.linalg.norm(embeddings2))
                similarity = similarity[0][0]
                bert_scores.append(similarity)
                
                #rouge
                rouge_score_value = rouge.get_scores(generated_summary, ' '.join(tokenised_label_summary))
                rouge_scores.append(rouge_score_value[0]["rouge-1"]["f"])
                
                #BLEU score
                smooth_func = SmoothingFunction().method1
                bleu_score_value = sentence_bleu([tokenised_label_summary], generated_summary, smoothing_function=smooth_func)
                bleu_scores.append(bleu_score_value)

                print("batch: " + str(batch_count) + "\n" + 
                      " meteor: " + str(meteor_score_value) + "\n" + 
                      " bert: " + str(similarity) + "\n" + 
                      "rouge: " + str(rouge_score_value[0]["rouge-1"]["f"]) + "\n" + 
                      "bleu: " + str(bleu_score_value))

            batch_count += 1

    # Calculate and log average scores
    average_meteor_score = sum(meteor_scores) / len(meteor_scores)
    average_bert_score = sum(bert_scores) / len(bert_scores)
    average_rouge_score = sum(rouge_scores) / len(rouge_scores)
    average_bleu_score = sum(bleu_scores) / len(bleu_scores)

    logger.info(f"Average METEOR Score: {average_meteor_score}")
    print("Average METEOR Score: ", average_meteor_score)

    logger.info(f"Average BERT Score: {average_bert_score}")
    print("Average BERT Score: ", average_bert_score)

    logger.info(f"Average ROUGE-1 f1 Score: {average_rouge_score}")
    print("Average Rouge-1 f1 Score: ", average_rouge_score)

    logger.info(f"Average BLEU Score: {average_bleu_score}")
    print("Average BLEU Score: ", average_bleu_score)
    
    return average_meteor_score
