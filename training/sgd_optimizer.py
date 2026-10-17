
import torch


def single_step_sgd():
    # Input and target
    x = torch.tensor(2.0)
    y = torch.tensor(3.0)

    # Learnable parameters
    w = torch.tensor(1.0, requires_grad=True)
    b = torch.tensor(-1.0, requires_grad=True)

    learning_rate = 0.1

    # Create SGD optimizer
    optimizer = torch.optim.SGD(
        [w, b],
        lr=learning_rate
    )

    # Forward propagation
    z = w * x + b
    prediction = torch.relu(z)
    loss = (prediction - y) ** 2

    print("=== Before Update ===")
    print("w:", w.item())
    print("b:", b.item())
    print("Loss:", loss.item())

    # Clear gradients
    optimizer.zero_grad()

    # Backpropagation
    loss.backward()

    assert w.grad is not None
    assert b.grad is not None

    print("\n=== Gradients ===")
    print("w.grad:", w.grad.item())
    print("b.grad:", b.grad.item())

    # Update parameters
    optimizer.step()

    print("\n=== After SGD Update ===")
    print("w:", w.item())
    print("b:", b.item())

    # Verify new loss
    new_prediction = torch.relu(w * x + b)
    new_loss = (new_prediction - y) ** 2

    print("New Prediction:", new_prediction.item())
    print("New Loss:", new_loss.item())

def train_with_sgd():
    x = torch.tensor(2.0)
    y = torch.tensor(3.0)

    w = torch.tensor(0.5, requires_grad=True)
    b = torch.tensor(0.0, requires_grad=True)

    learning_rate = 0.01
    epochs = 20

    optimizer = torch.optim.SGD(
        [w,b],
        lr=learning_rate
    )

    print("\n=== SGD Training ===")

    for epoch in range(epochs):
        optimizer.zero_grad()

        # Forward
        prediction = torch.relu(w*x + b)
        loss = (prediction - y) ** 2

        # backward
        loss.backward()

        # Update
        optimizer.step()

        print(
            f"Epoch {epoch + 1:2d} | "
            f"Loss: {loss.item():.6f} | "
            f"w: {w.item():.4f} | "
            f"b: {b.item():.4f}"
        )


if __name__ == "__main__":
    single_step_sgd()
    train_with_sgd()
