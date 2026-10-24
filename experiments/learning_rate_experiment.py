
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
WEIGHT_DECAY = 0.01

LEARNING_RATES = [0.0001, 0.001, 0.01]


def run_experiment(
    learning_rate,
    train_dataset,
    val_loader,
    device,
):
    # Reset random state for a fair comparison
    torch.manual_seed(SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    # Always use the baseline architecture
    model = FashionMNISTMLP(
        hidden1=256,
        hidden2=128,
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=WEIGHT_DECAY,
    )

    print(f"\n=== Learning Rate: {learning_rate} ===")

    history = []

    for epoch in range(EPOCHS):
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        # Training
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)

            running_loss += loss.item() * batch_size

            predictions = outputs.argmax(dim=1)
            correct += (
                (predictions == labels).sum().item()
            )
            total += batch_size

        train_loss = running_loss / total
        train_acc = correct / total

        # Validation
        val_loss, val_acc = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        row = {
            "learning_rate": learning_rate,
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
        }

        history.append(row)

        print(
            f"Epoch {epoch + 1:2d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc * 100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc * 100:.2f}%"
        )

    return history


def save_csv(rows, output_path):
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_losses(all_history, output_path):
    plt.figure(figsize=(9, 5))

    for lr in LEARNING_RATES:
        rows = [
            row for row in all_history
            if row["learning_rate"] == lr
        ]

        epochs = [row["epoch"] for row in rows]
        losses = [row["val_loss"] for row in rows]

        plt.plot(
            epochs,
            losses,
            marker="o",
            label=f"LR={lr}",
        )

    plt.title("Learning Rate vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.xticks(range(1, EPOCHS + 1))
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    print("Device:", device)

    # Use the same dataset split as Day 16 and Day 17
    full_dataset = datasets.FashionMNIST(
        root="./data",
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )

    train_dataset, val_dataset = random_split(
        full_dataset,
        [50000, 10000],
        generator=torch.Generator().manual_seed(SEED),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    print("Training samples:", len(train_dataset))
    print("Validation samples:", len(val_dataset))

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    all_history = []
    summary = []

    for lr in LEARNING_RATES:
        history = run_experiment(
            lr,
            train_dataset,
            val_loader,
            device,
        )

        all_history.extend(history)

        final = history[-1]

        summary.append({
            "learning_rate": lr,
            "train_loss": round(final["train_loss"], 4),
            "train_accuracy": round(final["train_accuracy"], 4),
            "val_loss": round(final["val_loss"], 4),
            "val_accuracy": round(final["val_accuracy"], 4),
        })

        # Save after each run to avoid losing completed results
        save_csv(
            all_history,
            output_dir / "learning_rate_history.csv",
        )
        save_csv(
            summary,
            output_dir / "learning_rate_results.csv",
        )

    plot_losses(
        all_history,
        output_dir / "learning_rate_loss.png",
    )

    print("\n=== Final Comparison ===")

    for row in summary:
        print(
            f"LR: {row['learning_rate']:<7} | "
            f"Train Loss: {row['train_loss']:.4f} | "
            f"Val Loss: {row['val_loss']:.4f} | "
            f"Val Acc: {row['val_accuracy'] * 100:.2f}%"
        )

    print("\nResults saved to:", output_dir)


if __name__ == "__main__":
    main()
