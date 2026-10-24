import csv
from pathlib import Path

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

CONFIGS = [
    ("Small", 128, 64),
    ("Baseline", 256, 128),
    ("Large", 512, 256),
]

def run_experiment(
        name, 
        hidden1, 
        hidden2, 
        train_dataset, 
        val_loader, 
        device
    ):
    
    torch.manual_seed(SEED)

    # create a new dataloader
    train_loader = DataLoader(train_dataset,BATCH_SIZE,shuffle = True)

    # create a new model for each config
    model = FashionMNISTMLP(
        hidden1=hidden1,
        hidden2=hidden2
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr = LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    total_parameters = sum(p.numel() for p in model.parameters())

    print(f"\n==={name}===")
    print(f"Hidden size: {hidden1}, {hidden2}")
    print(f"Parameters: {total_parameters}")

    final_train_loss = None
    final_train_acc = None
    final_val_loss = None
    final_val_acc = None

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

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

        val_loss, val_acc = evaluate(
            model,
            val_loader,
            criterion,
            device
        )

        print(
            f"Epoch {epoch + 1:2d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc * 100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc * 100:.2f}%"
        )

        final_train_loss = train_loss
        final_train_acc = train_acc
        final_val_loss = val_loss
        final_val_acc = val_acc

    return {
        "model" : name,
        "hidden1" : hidden1,
        "hidden2" : hidden2,
        "parameters" : total_parameters,
        "train_loss" : round(final_train_loss, 4),
        "train_accuracy" : round(final_train_acc, 4),
        "val_loss" : round(final_val_loss,4),
        "val_accuracy": round(final_val_acc,4)
    }

def main():
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    print("Device:", device)

    transform = transforms.ToTensor()

    full_train_dataset = datasets.FashionMNIST(
        root = "./data",
        train = True,
        download = True,
        transform = transform
    )

    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [50_000,10_000],
        generator=torch.Generator().manual_seed(SEED)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    print("Training samples:", len(train_dataset))
    print("Validation samples:", len(val_dataset))

    results = []

    for name, hidden1, hidden2, in CONFIGS:
        result = run_experiment(
            name,
            hidden1,
            hidden2,
            train_dataset,
            val_loader,
            device
        )

        results.append(result)

    # Save experimental results
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / "hidden_size_results.csv"

    with output_path.open("w",newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(results[0].keys())
        )

        writer.writeheader()
        writer.writerows(results)

    print("\n=== Final Comparison ===")

    for result in results:
        print(
            f"{result['model']:<10} | "
            f"Params: {result['parameters']:>7,} | "
            f"Val Loss: {result['val_loss']:.4f} | "
            f"Val Acc: {result['val_accuracy'] * 100:.2f}%"
        )

    print("\nResults saved to:", output_path)


if __name__ == "__main__":
    main()
