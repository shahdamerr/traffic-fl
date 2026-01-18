import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np


def train_one_node(model, train_loader, val_loader, epochs=2, lr=1e-3, device="cpu"):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    for ep in range(epochs):
        model.train()
        total = 0.0
        for x, y in tqdm(train_loader, desc=f"train ep {ep+1}", leave=False):
            x = x.to(device)              # [B, L, 1]
            y = y.to(device)              # [B, H]
            pred = model(x)               # [B, H]
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total += loss.item() * x.size(0)

        train_loss = total / len(train_loader.dataset)

        # quick validation
        model.eval()
        vtotal = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                pred = model(x)
                loss = loss_fn(pred, y)
                vtotal += loss.item() * x.size(0)
        val_loss = vtotal / len(val_loader.dataset)

        print(f"  epoch {ep+1}: train_mse={train_loss:.6f} val_mse={val_loss:.6f}")

    return model



def evaluate_model(
    model,
    data_loader,
    mean,
    std,
    device="cpu"
):
    model.eval()
    preds, trues = [], []

    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x)

            preds.append(pred.cpu().numpy())
            trues.append(y.cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)

    # Denormalize
    preds = preds * std + mean
    trues = trues * std + mean

    return preds, trues
