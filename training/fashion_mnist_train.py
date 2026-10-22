import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from models.fashion_mnist_mlp import FashionMNISTMLP

def evaluate(model, data_loader, criterion, device):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            # foward only
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Accumulate loss
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size

            # accuracy
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += batch_size
        avg_loss = total_loss / total
        accuracy = correct / total

    return avg_loss, accuracy


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

    full_train_dataset = datasets.FashionMNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    # train and validation split
    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [50_000,10_000],
        generator=torch.Generator().manual_seed(42)

    )

    # dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False
    )

    print("Training samples:", len(train_dataset))
    print("Validation samples:", len(val_dataset))
    print("Training batches:", len(train_loader))
    print("Validation batches:", len(val_loader))

    # model
    model = FashionMNISTMLP().to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=0.01
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

        train_loss = running_loss / total
        train_acc = correct / total

        # Validation phase
        val_loss, val_acc = evaluate(model, val_loader, criterion,device)
        # compare
        accuracy_gap = train_acc - val_acc

        print(
            f"Epoch {epoch + 1:2d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc * 100:.2f}% |"
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc * 100:.2f}% | "
            f"Gap: {accuracy_gap * 100:+.2f} pp"
        )

    print("\n=== Training Complete ===")


if __name__ == "__main__":
    train()
    