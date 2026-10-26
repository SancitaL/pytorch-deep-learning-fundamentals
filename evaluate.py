import csv
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.fashion_mnist_mlp import FashionMNISTMLP
from training.fashion_mnist_train import evaluate

CHECKPOINT_PATH = Path("checkpoints/best_model.pt")
RESULTS_PATH = Path("results/test_results.csv")

BATCH_SIZE = 64

def main():
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    print("Device", device)

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            "Checkpoint not found. Run python train.py first"
        )

    # Loadcheckpoint
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
    )

    # build model with saved weights
    model = FashionMNISTMLP(
        hidden1=checkpoint["hidden1"],
        hidden2=checkpoint["hidden2"]
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)
    model.eval()

    # load the independent test set
    test_dataset = datasets.FashionMNIST(
        root = "./data",
        train = False,
        download=True,
        transform=transforms.ToTensor()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    print("Test samples:", len(test_dataset))
    print("Loaded checkpoint epoch:", checkpoint["epoch"])

    criterion = nn.CrossEntropyLoss()

    test_loss, test_acc = evaluate(
        model,
        test_loader,
        criterion,
        device,
    )

    print("\n=== Final Test Results ===")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc * 100:.2f}%")

    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with RESULTS_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "checkpoint_epoch",
                "test_samples",
                "test_loss",
                "test_accuracy",
            ],
        )

        writer.writeheader()

        writer.writerow({
            "checkpoint_epoch": checkpoint["epoch"],
            "test_samples": len(test_dataset),
            "test_loss": round(test_loss, 4),
            "test_accuracy": round(test_acc, 4),
        })

    # Show predictions for the first ten samples
    images, labels = next(iter(test_loader))

    images = images.to(device)

    with torch.no_grad():
        logits = model(images)
        predictions = logits.argmax(dim=1)

    print("\n=== Sample Predictions ===")

    for i in range(10):
        print(
            f"Sample {i:2d} | "
            f"Predicted: {predictions[i].item()} | "
            f"Actual: {labels[i].item()}"
        )

    print("\nTest results saved:", RESULTS_PATH)


if __name__ == "__main__":
    main()

