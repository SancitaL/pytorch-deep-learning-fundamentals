from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from models.fashion_mnist_mlp import FashionMNISTMLP
from training.fashion_mnist_train import evaluate


SEED = 42
EPOCHS = 10
BATCH_SIZE = 64

CHECKPOINT_PATH = Path("checkpoints/best_model.pt")

def train_and_save():
    torch.manual_seed(SEED)

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    print("Device:", device)

    dataset = datasets.FashionMNIST(
        root="./data",
        train = True,
        download=True,
        transform=transforms.ToTensor()
    )

    train_dataset, val_dataset = random_split(
        dataset,
        [50_000,10_000],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle = True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size = BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    model = FashionMNISTMLP(
        hidden1=256,
        hidden2=128,
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr = 0.001,
        weight_decay=0.01
    )

    CHECKPOINT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    best_val_loss = float("inf")
    best_epoch = 0

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

            correct += (predictions == labels).sum().item()

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

        # save only if validation loss improves
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
            }

            torch.save(
                checkpoint,
                CHECKPOINT_PATH,
            )

            print(
                f"  Best checkpoint saved "
                f"(Epoch {best_epoch}, "
                f"Val Loss {best_val_loss:.4f})"
            )

    print("\nTraining finished.")
    print("Best Epoch:", best_epoch)
    print("Best Validation Loss:", round(best_val_loss, 4))

    return model, val_loader, device


def load_and_verify(last_model, val_loader, device):
    # load checkpoint from disk
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=True
    )

    print("\n=== Loaded Checkpoint ===")
    print("Saved Epoch:", checkpoint["epoch"])
    print("Saved Val Loss:", checkpoint["val_loss"])

    # Create a NEW model with the saved architecture
    loaded_model = FashionMNISTMLP(
        hidden1=checkpoint["hidden1"],
        hidden2=checkpoint["hidden2"]
    )

    # replace the initial weights by pre saved weights
    loaded_model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    loaded_model = loaded_model.to(device)
    loaded_model.eval()

    criterion = nn.CrossEntropyLoss()

    loaded_val_loss, loaded_val_acc = evaluate(
        loaded_model,
        val_loader,
        criterion,
        device
    )

    print("Reloaded Val Loss:", round(loaded_val_loss, 4))
    print(
        "Reloaded Val Accuracy:",
        f"{loaded_val_acc * 100:.2f}%",
    )

    # Compare model predictions on one fixed batch
    images,_ = next(iter(val_loader))

    images = images.to(device)

    last_model.eval()

    with torch.no_grad():
        last_outputs = last_model(images)
        loaded_outputs = loaded_model(images)

    print("\n=== Model Comparison ===")
    print(
        "Last model vs loaded best model:",
        torch.allclose(
            last_outputs,
            loaded_outputs,
            atol=1e-6,
        ),
    )

    # Compare the loaded model with the saved validation metric
    print(
        "Reloaded loss matches checkpoint:",
        abs(loaded_val_loss - checkpoint["val_loss"]) < 1e-6,
    )


if __name__ == "__main__":
    last_model, val_loader, device = train_and_save()

    load_and_verify(
        last_model,
        val_loader,
        device,
    )