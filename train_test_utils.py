from nltk.translate import meteor_score
from nltk import word_tokenize
import nltk
nltk.download('punkt')
nltk.download('wordnet')
import torch
import utils

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

    for batch in train_loader:
        input_ids = batch[0].to(device)  # Assuming input_ids is the first element in the batch
        attention_mask = batch[1].to(device)  # Assuming attention_mask is the second element in the batch
        labels = batch[2].to(device)  # Assuming labels is the third element in the batch

        optimizer.zero_grad()

        outputs = model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_loss_30_batches += loss.item()

        if (i + 1) % 100 == 99:
            avg_loss = total_loss_30_batches / batch_count
            print(f"{epoch + 1}, {i + 1}, Average Loss: {avg_loss:.4f}")
            logger.info(f"{epoch + 1}, {i + 1}, Average Loss: {avg_loss:.4f}")
            total_loss_30_batches = 0
            batch_count = 0

            model.save("checkpoint")

            # Evaluate on the validation set
            meteor_score = evaluate_meteor(device, model, val_loader)
            if (meteor_score > best_validation_performance):
                model.save("best_validation")
                logger.info("Best validation performance. Model weights saved.")
        i += 1
        batch_count += 1

    return total_loss / len(train_loader)

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

def evaluate_meteor(device, model, val_loader):
    model.model.eval()
    meteor_scores = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch[0].to(device)  # Assuming input_ids is the first element in the batch
            attention_mask = batch[1].to(device)  # Assuming attention_mask is the second element in the batch
            labels = batch[2].to(device)  # Assuming labels is the third element in the batch

            # Generate summaries
            generated_summaries = []
            for ids in input_ids:
                summary = model.summarize1(ids.unsqueeze(0))
                generated_summaries.append(summary)

            # Calculate METEOR score for each generated summary
            for generated_summary, label_summary in zip(generated_summaries, labels):
                generated_summary = word_tokenize(generated_summary)
                label_summary = model.tokenizer.decode(label_summary) # FIrst decode to plaintext
                label_summary = word_tokenize(label_summary)
                meteor_score_value = meteor_score.single_meteor_score(label_summary, generated_summary)
                meteor_scores.append(meteor_score_value)

    average_meteor_score = sum(meteor_scores) / len(meteor_scores)
    logger.info("Average METEOR Score: ", average_meteor_score)
    print("Average METEOR Score: ", average_meteor_score)
    return average_meteor_score