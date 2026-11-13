"""
Training and independent test-set evaluation pipeline for AirNox Bioaerosol Classifier.
Automatically uses Apple Silicon GPU (MPS) on Mac and exports metrics + confusion matrix.
"""

import json
import time
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from src.models.classifier import build_classifier, get_device

PROCESSED_DIR = Path("data/processed")
CHECKPOINT_DIR = Path("models/checkpoints")
RESEARCH_DIR = Path("research")
SEED = 42


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_and_evaluate(arch: str = "resnet18", epochs: int = 10, batch_size: int = 16, lr: float = 3e-4):
    set_seed(SEED)
    device = get_device()
    print(f"Using compute device: {device}")

    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.ImageFolder(PROCESSED_DIR / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(PROCESSED_DIR / "val", transform=eval_tf)
    test_ds = datasets.ImageFolder(PROCESSED_DIR / "test", transform=eval_tf)

    class_names = train_ds.classes
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = build_classifier(arch=arch, num_classes=len(class_names), pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0
    best_ckpt_path = CHECKPOINT_DIR / f"airnox_{arch}_best.pt"

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total

        # Validation pass
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = model(imgs).argmax(1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        val_acc = val_correct / val_total

        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Acc: {train_acc*100:5.1f}% | Val Acc: {val_acc*100:5.1f}%")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "arch": arch,
                "classes": class_names,
                "state_dict": model.state_dict(),
                "val_acc": best_val_acc,
            }, best_ckpt_path)

    elapsed = time.time() - start_time
    print(f"\nTraining finished in {elapsed:.1f}s. Best Val Acc: {best_val_acc*100:.1f}%")

    # Independent Test Set Evaluation
    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            preds = model(imgs).argmax(1).cpu().numpy()
            y_true.extend(labels.numpy())
            y_pred.extend(preds)

    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    test_acc = report["accuracy"]
    print(f"Independent Test Set Accuracy: {test_acc*100:.1f}%")

    # Save Confusion Matrix Figure
    RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 5.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(f"AirNox Test Confusion Matrix ({arch}, Acc={test_acc*100:.1f}%)")
    plt.ylabel("True Taxon / Class")
    plt.xlabel("Predicted Class")
    plt.tight_layout()
    plt.savefig(RESEARCH_DIR / "confusion_matrix.png", dpi=180)
    plt.close()

    # Log Experiment Result
    log_md = f"""# Experimental Results Log

## Experiment EXP-001: Baseline {arch} Transfer Learning
- **Compute Device:** `{device}`
- **Random Seed:** `{SEED}`
- **Epochs / Batch Size / LR:** `{epochs}` / `{batch_size}` / `{lr}`
- **Validation Accuracy:** `{best_val_acc*100:.2f}%`
- **Independent Test Accuracy:** `{test_acc*100:.2f}%`
- **Macro F1-Score:** `{report['macro avg']['f1-score']:.3f}`
- **Artifact Saved:** `{best_ckpt_path}` & `research/confusion_matrix.png`
"""
    (RESEARCH_DIR / "experimental_results.md").write_text(log_md)
    print("Saved evaluation report to research/experimental_results.md and research/confusion_matrix.png")


if __name__ == "__main__":
    train_and_evaluate(arch="resnet18", epochs=10)
