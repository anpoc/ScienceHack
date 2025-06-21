from dataset import InvoiceBatchDataset
from src.text_based_split.utils.utils import pdf_to_embeds
from src.text_based_split.model import Classifier
from src.text_based_split.predict import predict
from evaluation import evaluate_during_training
import torch
from tqdm import tqdm
from torch import nn
import shutil
import os
from copy import deepcopy


model_save_dir = "./src/text_based_split/ckpts"

model = Classifier()
criterion = nn.BCELoss(reduction="none")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

best_accu = 0
best_chunk = 0
for epoch in range(1):
    ds = InvoiceBatchDataset("data", split="train", min_n=10, max_n=20, size=200)
    model.train()
    for idy, batch in tqdm(enumerate(ds)):
        pdf_path, y_true = batch
        file2embeds=pdf_to_embeds(pdf_path)
        embeddings_batch = []
        for idx, y_single in enumerate(y_true):
            embeddings_batch.append(file2embeds.callback(idx))
        y_tensor = torch.tensor(y_true)
        embeddings_batch = torch.stack(embeddings_batch).squeeze(dim=1)
        outputs = model(torch.tensor(embeddings_batch))
        weights = torch.where(y_tensor == 1, torch.tensor(1), torch.tensor(1))
        loss = (criterion(outputs.float(), y_tensor.float())*weights).mean()
        loss.backward()
        optimizer.step()

        # Calculate predictions (threshold at 0.5)
        preds = (outputs >= 0.5).float()
        correct = (preds == y_tensor).sum().item()
        samples = len(y_tensor)
        print(f"Loss = {loss.item():.4f}, Accuracy = {correct/samples:.4f}")
        
        torch.save(model.state_dict(), os.path.join(model_save_dir, f"model_last.pt"))
        
        if idy %10 == 0:
            acc, chunk_score = evaluate_during_training(predict, "test", 10, deepcopy(model))
            if acc > best_accu:
                best_accu = acc
                torch.save(model.state_dict(), os.path.join(model_save_dir, f"model_best_acc.pt"))
            if chunk_score> best_chunk:
                best_chunk = chunk_score
                torch.save(model.state_dict(), os.path.join(model_save_dir, f"model_best_chunk.pt"))



shutil.rmtree('/tmp/invoice_batch_dataset')