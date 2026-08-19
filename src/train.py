"""Training loop and CLI entry point.

Wraps the train/eval step functions used across every experiment in this
project, plus a small CLI so a model can be trained end-to-end with:

    python -m src.train --model resnet18 --epochs 5

Fixes carried over from the notebook's debugging history:
- `train()` evaluates against the `val_dataloader` argument it was actually
  given, not a same-named global (see README "Lessons learned").
- Every tensor is moved to `device` explicitly, so this also works on GPU,
  not just the CPU-only setup used during development.
"""

import argparse
import os
from pathlib import Path

import torch
from torch import nn
from torchvision import transforms
from tqdm.auto import tqdm

from src.data import build_patient_split, get_dataloaders
from src.model import TinyVGG, build_resnet

TINYVGG_TRANSFORM = transforms.Compose([
    transforms.Resize(size=(64, 64)),
    transforms.ToTensor(),
])

RESNET_TRANSFORM = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def train_step(model, dataloader, loss_fn, optimizer, device):
    model.train()
    train_loss, train_acc = 0.0, 0.0

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)

        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)

    return train_loss / len(dataloader), train_acc / len(dataloader)


def eval_step(model, dataloader, loss_fn, device):
    model.eval()
    loss_total, acc_total = 0.0, 0.0

    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            logits = model(X)
            loss = loss_fn(logits, y)
            loss_total += loss.item()

            preds = logits.argmax(dim=1)
            acc_total += (preds == y).sum().item() / len(preds)

    return loss_total / len(dataloader), acc_total / len(dataloader)


def train(model, train_dataloader, val_dataloader, optimizer, loss_fn, epochs, device):
    model.to(device)
    results = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model, train_dataloader, loss_fn, optimizer, device)
        val_loss, val_acc = eval_step(model, val_dataloader, loss_fn, device)

        print(
            f"Epoch: {epoch} | Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f} "
            f"| Val loss: {val_loss:.4f} | Val acc: {val_acc:.4f}"
        )

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc)

    return results


def compute_class_weights(new_train_dir: Path, class_names: tuple) -> torch.Tensor:
    """Inverse-frequency class weights, computed from the actual train split counts."""
    counts = [len(list((new_train_dir / cls).glob("*.jpeg"))) for cls in class_names]
    total = sum(counts)
    weights = [total / (len(class_names) * c) for c in counts]
    return torch.tensor(weights, dtype=torch.float32)


def main():
    parser = argparse.ArgumentParser(description="Train a pneumonia classifier.")
    parser.add_argument("--model", choices=["tinyvgg", "resnet18", "resnet34"], default="resnet18")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--data-dir", type=Path, default=Path("chest_xray"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_dir = args.data_dir / "train"
    test_dir = args.data_dir / "test"
    new_train_dir = args.data_dir / "new_train"
    new_val_dir = args.data_dir / "new_val"
    class_names = ("NORMAL", "PNEUMONIA")

    build_patient_split(train_dir, new_train_dir, new_val_dir)

    is_tinyvgg = args.model == "tinyvgg"
    transform = TINYVGG_TRANSFORM if is_tinyvgg else RESNET_TRANSFORM

    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(
        new_train_dir, new_val_dir, test_dir, transform, batch_size=args.batch_size
    )

    if is_tinyvgg:
        model = TinyVGG(input_shape=3, hidden_units=10, output_shape=len(class_names))
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        model = build_resnet(args.model, num_classes=len(class_names), freeze_backbone=True)
        optimizer = torch.optim.Adam(model.fc.parameters(), lr=args.lr)

    class_weights = compute_class_weights(new_train_dir, class_names).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    train(model, train_dataloader, val_dataloader, optimizer, loss_fn, args.epochs, device)

    test_loss, test_acc = eval_step(model, test_dataloader, loss_fn, device)
    print(f"Final test loss: {test_loss:.4f} | Final test acc: {test_acc:.4f}")

    checkpoint_names = {
        "tinyvgg": "tinyvgg_baseline.pth",
        "resnet18": "resnet18_frozen.pth",
        "resnet34": "resnet34_frozen.pth",
    }

    args.checkpoint_dir.mkdir(exist_ok=True)
    checkpoint_path = args.checkpoint_dir / checkpoint_names[args.model]
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
