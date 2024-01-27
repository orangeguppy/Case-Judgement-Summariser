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

def evaluate_meteor(self, eval_loader, references):
    self.model.eval()
    meteor_scores = []
    with torch.no_grad():
        for batch, reference_summary in zip(eval_loader, references):
            input_ids = batch['text_input_ids'].to(self.device)
            attention_mask = batch['text_attention_mask'].to(self.device)

            generated_summary = self.summarize(input_ids)
            meteor_score_value = meteor_score.single_meteor_score(reference_summary, generated_summary)
            meteor_scores.append(meteor_score_value)

    average_meteor_score = sum(meteor_scores) / len(meteor_scores)
    return average_meteor_score