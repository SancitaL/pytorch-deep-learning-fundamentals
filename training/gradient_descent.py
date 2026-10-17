import torch

def main():
    x = torch.tensor(2.0)
    y = torch.tensor(3.0)

    w = torch.tensor(1.0, requires_grad=True)
    b = torch.tensor(-1.0, requires_grad=True)

    learning_rate = 0.1

    # Forward
    z = w*x+b
    prediction = torch.relu(z)
    loss = (prediction - y) ** 2

    print("=== Before Update ===")
    print("w:", w.item())
    print("b:", b.item())
    print("Prediction:", prediction.item())
    print("Loss:", loss.item())

    # Backpropagation
    loss.backward()

    assert w.grad is not None
    assert b.grad is not None

    print("\n=== Gradients ===")
    print("w.grad:", w.grad.item())
    print("b.grad:", b.grad.item())

    # Gradient descent
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad

    print("\n=== After Update ===")
    print("w:", w.item())
    print("b:", b.item())

    # New forward propagation
    z_new = w * x + b
    prediction_new = torch.relu(z_new)
    loss_new = (prediction_new - y) ** 2

    print("Prediction:", prediction_new.item())
    print("Loss:", loss_new.item())

def train_gradient_descent():
    x = torch.tensor(2.0)
    y = torch.tensor(3.0)

    w = torch.tensor(0.5, requires_grad=True)
    b = torch.tensor(0.0, requires_grad=True)

    learning_rate = 0.001
    epochs = 20

    print("\n=== Gradient Descent Training ===")

    for epoch in range(epochs):
        # Forward
        prediction = torch.relu(w*x+b)
        loss = (prediction - y)**2

        # clear previous gradients
        if w.grad is not None:
            w.grad.zero_()
        if b.grad is not None:
            b.grad.zero_()

        # Backward
        loss.backward()

        assert w.grad is not None
        assert b.grad is not None

        with torch.no_grad():
            w -= learning_rate * w.grad
            b -= learning_rate * b.grad

        print(
            f"Epoch {epoch + 1:2d} | "
            f"Loss: {loss.item():.6f} | "
            f"w: {w.item():.4f} | "
            f"b: {b.item():.4f}"
        )


if __name__ == "__main__":
    main()

    train_gradient_descent()

