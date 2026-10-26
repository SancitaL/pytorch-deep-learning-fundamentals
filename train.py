
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from models.fashion_mnist_mlp import FashionMNISTMLP
from training.fashion_mnist_train import evaluate


SEED = 42
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.01

CHECKPOINT_PATH = Path("checkpoints/best_model.pt")
RESULTS_DIR = Path("results")


def save_history(history):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = RESULTS_DIR / "training_history.csv"

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(history[0].keys()),
        )
        writer.writeheader()
        writer.writerows(history)

    epochs = [row["epoch"] for row in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(
        epochs,
        [row["train_loss"] for row in history],
        label="Train Loss",
    )
    axes[0].plot(
        epochs,
        [row["val_loss"] for row in history],
        label="Validation Loss",
    )
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(
        epochs,
        [row["train_accuracy"] * 100 for row in history],
        label="Train Accuracy",
    )
    axes[1].plot(
        epochs,
        [row["val_accuracy"] * 100 for row in history],
        label="Validation Accuracy",
    )
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()

    for ax in axes:
        ax.grid(alpha=0.3)
        ax.set_xticks(range(1, EPOCHS + 1))

    fig.tight_layout()
    fig.savefig(
        RESULTS_DIR / "training_curves.png",
        dpi=150,
    )
    plt.close(fig)

    print("Training history saved:", output_path)


def train():
    torch.manual_seed(SEED)

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    print("Device:", device)

    dataset = datasets.FashionMNIST(
        root="./data",
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )

    train_dataset, val_dataset = random_split(
        dataset,
        [50_000, 10_000],
        generator=torch.Generator().manual_seed(SEED),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = FashionMNISTMLP(
        hidden1=256,
        hidden2=128,
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    CHECKPOINT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_val_loss = float("inf")
    best_epoch = 0
    history = []

    print("\n=== Final Model Training ===")

    for epoch in range(EPOCHS):
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size

            predictions = logits.argmax(dim=1)
            correct += (
                (predictions == labels).sum().item()
            )
            total += batch_size

        train_loss = running_loss / total
        train_acc = correct / total

        val_loss, val_acc = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
        })

        print(
            f"Epoch {epoch + 1:2d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc * 100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc * 100:.2f}%"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1

            checkpoint = {
                "epoch": best_epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "hidden1": 256,
                "hidden2": 128,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "batch_size": BATCH_SIZE,
                "seed": SEED,
            }

            torch.save(
                checkpoint,
                CHECKPOINT_PATH,
            )

            print(
                f"  Best checkpoint saved "
                f"(Epoch {best_epoch})"
            )

        # Preserve metrics from completed epochs
        save_history(history)

    print("\n=== Training Finished ===")
    print("Best Epoch:", best_epoch)
    print("Best Val Loss:", round(best_val_loss, 4))
    print("Checkpoint:", CHECKPOINT_PATH)


if __name__ == "__main__":
    train()
