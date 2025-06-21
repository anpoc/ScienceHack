from dataset import InvoiceBatchDataset
from src.manuel.model import Classifier
import torch
from tqdm import tqdm
from torch import nn
import shutil
import os
from src.manuel.manual_pdf import *

model_save_dir = "./src/manuel/ckpts"

num_inputs = len(list(main_page_types.keys()))

model = Classifier(input_dim=num_inputs, hidden_dim=num_inputs//2)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

best_accu = 0
for epoch in range(1):
    ds = InvoiceBatchDataset("data", split="train", min_n=10, max_n=20, size=200)
    model.train()
    for idy, batch in enumerate(ds):
        pdf_path, y_true = batch
        y_tensor = torch.tensor(y_true)
        
        tuple_lists = manual_4training_readPDF(pdf_path)

        list_parameters = tuple_lists[1]

        outputs = model(torch.tensor(list_parameters, dtype=torch.float))
        loss = (criterion(outputs.float(), y_tensor.float()))
        loss.backward()
        optimizer.step()

        # Calculate predictions (threshold at 0.5)
        preds = (outputs >= 0.5).float()
        correct = (preds == y_tensor).sum().item()
        samples = len(y_true)
        
        torch.save(model.state_dict(), os.path.join(model_save_dir, f"model_last.pt"))

        if idy % 10 == 0:
            dt = InvoiceBatchDataset("data", split="test", min_n=10, max_n=20, size=10)
            model.eval()
            total_correct = 0
            total_samples = 0
            for idz, batch_val in tqdm(enumerate(dt)):
                pdf_path_val, y_true_val = batch_val
                y_tensor_val = torch.tensor(y_true_val)
                tuple_lists_val = manual_4training_readPDF(pdf_path_val)

                list_parameters_val = tuple_lists_val[1]

                outputs_val = model(torch.tensor(list_parameters_val, dtype=torch.float))
                preds_val = (outputs_val >= 0.5).float()
                total_correct += (preds_val == y_tensor_val).sum().item()
                total_samples += len(y_true)
            
            iter_accu = total_correct/total_samples
            print(f"iter: {idy} accuracy val: {iter_accu} \n")
            if best_accu < iter_accu:
                best_accu = iter_accu
                torch.save(model.state_dict(), os.path.join(model_save_dir, f"model_best_acc.pt"))


        
                


shutil.rmtree('/tmp/invoice_batch_dataset')