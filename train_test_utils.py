def train_epoch1(self, device, model, train_loader, optimizer):
    model.train()
    total_loss = 0

    for batch in train_loader:
        input_ids = batch[0].to(device)  # Assuming input_ids is the first element in the batch
        attention_mask = batch[1].to(device)  # Assuming attention_mask is the second element in the batch
        labels = batch[2].to(device)  # Assuming labels is the third element in the batch

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)

def evaluate(self, device, model, eval_loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch['text_input_ids'].to(device)
            attention_mask = batch['text_attention_mask'].to(device)
            labels = batch['summary_input_ids'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            total_loss += loss.item()

    return total_loss / len(eval_loader)

# def evaluate_meteor(self, eval_loader, references):
#     self.model.eval()
#     meteor_scores = []
#     with torch.no_grad():
#         for batch, reference_summary in zip(eval_loader, references):
#             input_ids = batch['text_input_ids'].to(self.device)
#             attention_mask = batch['text_attention_mask'].to(self.device)

#             generated_summary = self.summarize(input_ids)
#             meteor_score_value = meteor_score.single_meteor_score(reference_summary, generated_summary)
#             meteor_scores.append(meteor_score_value)

#     average_meteor_score = sum(meteor_scores) / len(meteor_scores)
#     return average_meteor_score