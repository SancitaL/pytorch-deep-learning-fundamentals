import torch
from torch import nn

from models.simple_mlp import SimpleMLP

def train():
    torch.manual_seed(42)
    
    # 1.Generate synthetic classification data
    num_samples = 200

    x = torch.randn(num_samples, 4)

    # define labels using input features
    labels = (x[:,0] + x[:,1] > 0).long()

    # 2. Create a model instance
    model = SimpleMLP()

    # 3.Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr = 0.01,
        weight_decay=0.0
    )

    epochs = 100
    print("=== Training Start ===")
    print("Input Shape:", x.shape)
    print("Labels Shape:", labels.shape)

    for epoch in range(epochs):
        # training mode:
        model.train()

        # clear zero gradients
        optimizer.zero_grad()

        # forward pass
        outputs = model(x)

        # loss calculation
        loss = criterion(outputs, labels)

        # backward pass
        loss.backward()

        # update parameters
        optimizer.step()

        # caculate accuracy after update
        model.eval()

        with torch.no_grad():
            current_outputs = model(x)
            current_loss = criterion(current_outputs, labels)
            predictions = current_outputs.argmax(dim=1)
            accuracy = (
                (predictions == labels)
                .float()
                .mean()
                .item()
            )

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Epoch {epoch + 1:3d} | "
                f"Loss: {current_loss.item():.6f} | "
                f"Accuracy: {accuracy * 100:.2f}%"
            )

    print("\n=== Training Complete ===")

    # Evaluate on unseen synthetic data
    test_x = torch.randn(100, 4)

    test_labels = (
        test_x[:, 0] + test_x[:, 1] > 0
    ).long()

    model.eval()

    with torch.no_grad():
        test_outputs = model(test_x)

        test_loss = criterion(
            test_outputs,
            test_labels,
        )

        test_predictions = test_outputs.argmax(dim=1)

        test_accuracy = (
            (test_predictions == test_labels)
            .float()
            .mean()
            .item()
        )

    print("\n=== Test Results ===")
    print(f"Test Loss: {test_loss.item():.6f}")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")


    print("\n=== Sample Predictions ===")

    for i in range(10):
        print(
            f"Sample {i + 1:2d} | "
            f"Input: {test_x[i].tolist()} | "
            f"Predicted: {test_predictions[i].item()} | "
            f"Actual: {test_labels[i].item()}"
        )

    custom_x = torch.tensor([
        [2.0, 1.0, 0.5, -0.2],
        [-2.0, -1.0, 0.3, 0.8],
        [1.5, -0.5, 0.0, 0.0],
    ])
    model.eval()

    with torch.no_grad():
        custom_outputs = model(custom_x)
        custom_predictions = custom_outputs.argmax(dim=1)

    print("\n=== Custom Predictions ===")
    print("Predictions:", custom_predictions.tolist())
    print("Custom Outputs:", custom_outputs.tolist())

if __name__ == "__main__":
    train()