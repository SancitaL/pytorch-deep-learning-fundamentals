import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.fashion_mnist_mlp import FashionMNISTMLP

def train():
    # reproductibility
    torch.manual_seed(42)

    # device
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    # dataset
    transform = transforms.ToTensor()

    train_dataset = datasets.FashionMNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    #dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=0
    )

    print("Training samples:", len(train_dataset))
    print("Batches per epoch:", len(train_loader))

    # model
    model = FashionMNISTMLP().to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
    )

    epochs = 10
    print("\n=== Fashion-MNIST Training ===")

    # training loop:
    for epoch in range(epochs):
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, lables in train_loader:
            # Move data to device
            images = images.to(device)
            lables = lables.to(device)

            # clear gradient
            optimizer.zero_grad()

            # forward
            outputs = model(images)

            # loss
            loss  = criterion(outputs, lables)

            # backward
            loss.backward()

            # update
            optimizer.step()

            # Accumulate metrics

            batch_size = lables.size(0)
            running_loss += loss.item() * batch_size
            predictions = outputs.argmax(dim=1)

            correct += (
                (predictions == lables).sum().item()
            )
            total += batch_size

        epoch_loss = running_loss / total
        epoch_accuracy = correct / total

        print(
            f"Epoch {epoch + 1:2d}/{epochs} | "
            f"Train Loss: {epoch_loss:.4f} | "
            f"Train Accuracy: {epoch_accuracy * 100:.2f}%"
        )

    print("\n=== Training Complete ===")


if __name__ == "__main__":
    train()
    