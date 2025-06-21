from test_embeddings import JSONDataset
from torch.utils.data import Dataset, DataLoader
from model import Classifier
import torch
from torch import nn

dataset = JSONDataset()
dataloader = DataLoader(dataset, batch_size=5, shuffle=True, drop_last=True)

# 6. Initialize model and optimizer
model = Classifier()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCELoss(reduction="none")


w1 = 0.1
w0 = 0.5
# 7. Training loop
for epoch in range(10):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    for batch_embeds, batch_labels in dataloader:
        embeddings = batch_embeds
        labels_tensor = batch_labels.squeeze()

        optimizer.zero_grad()
        outputs = model(embeddings)
        
        #print(outputs, labels_tensor)
        # Since BCELoss expects outputs in [0,1], ensure model outputs are sigmoid probabilities
        outputs = outputs.float()
        labels_tensor = labels_tensor.float()
        weights = torch.where(labels_tensor == 1, torch.tensor(w1), torch.tensor(w0))
        loss = (criterion(outputs, labels_tensor)*weights).mean()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Calculate predictions (threshold at 0.5)
        preds = (outputs >= 0.5).float()
        total_correct += (preds == labels_tensor).sum().item()
        total_samples += len(labels_tensor)

    accuracy = total_correct / total_samples
    print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}, Accuracy = {accuracy:.4f}")
