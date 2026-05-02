import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import copy


def train_one_node(
    model,
    train_loader,
    val_loader,
    epochs=100,
    lr=1e-3,
    weight_decay=1e-5,
    patience=15,
    max_grad_norm=5.0,
    device="cpu",
    seq2seq=False,
    tf_start=0.5,
    tf_end=0.0,
):
    """Train a single-node forecasting model with modern best practices.

    Includes:
        - MSE loss
        - Adam with weight decay
        - ReduceLROnPlateau scheduler
        - Gradient clipping
        - Early stopping with best-checkpoint restoration
        - Optional: teacher forcing for Seq2Seq models

    Args:
        seq2seq: if True, passes y to model with teacher forcing
        tf_start: initial teacher forcing ratio
        tf_end: final teacher forcing ratio (decays linearly)
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    for ep in range(epochs):
        # Teacher forcing ratio: linear decay
        if seq2seq:
            tf_ratio = tf_start - (tf_start - tf_end) * (ep / max(epochs - 1, 1))
        else:
            tf_ratio = 0.0

        # ---- training ----
        model.train()
        total = 0.0
        for x, y in tqdm(train_loader, desc=f"train ep {ep+1}", leave=False):
            x = x.to(device)              # [B, L, 1]
            y = y.to(device)              # [B, H]

            if seq2seq:
                pred = model(x, y=y, teacher_forcing_ratio=tf_ratio)
            else:
                pred = model(x)

            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            total += loss.item() * x.size(0)

        train_loss = total / len(train_loader.dataset)

        # ---- validation (always autoregressive, no teacher forcing) ----
        model.eval()
        vtotal = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)

                if seq2seq:
                    pred = model(x, y=None, teacher_forcing_ratio=0.0)
                else:
                    pred = model(x)

                loss = loss_fn(pred, y)
                vtotal += loss.item() * x.size(0)
        val_loss = vtotal / len(val_loader.dataset)

        # step scheduler
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        tf_str = f"  tf={tf_ratio:.2f}" if seq2seq else ""
        print(
            f"  epoch {ep+1:3d}: train_loss={train_loss:.6f}  "
            f"val_loss={val_loss:.6f}  lr={current_lr:.2e}{tf_str}"
        )

        # ---- early stopping ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  Early stopping at epoch {ep+1} (patience={patience})")
                break

    # restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"  Restored best checkpoint (val_loss={best_val_loss:.6f})")

    return model


def evaluate_model(model, data_loader, mean, std, device="cpu", seq2seq=False):
    """Evaluate model and return denormalized predictions and ground truth.

    Args:
        model: trained model
        data_loader: DataLoader for the evaluation split
        mean: scalar mean for this sensor
        std: scalar std for this sensor
        device: torch device
        seq2seq: if True, call model with teacher_forcing_ratio=0.0

    Returns:
        preds: [N, H] denormalized predictions
        trues: [N, H] denormalized ground truth
    """
    model.eval()
    preds, trues = [], []

    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device)

            if seq2seq:
                pred = model(x, y=None, teacher_forcing_ratio=0.0)
            else:
                pred = model(x)

            preds.append(pred.cpu().numpy())
            trues.append(y.cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)

    # Denormalize
    preds = preds * std + mean
    trues = trues * std + mean

    return preds, trues
