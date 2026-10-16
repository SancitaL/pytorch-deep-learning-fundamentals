import torch

if __name__ == "__main__":
    x = torch.tensor(3.0)
    y = torch.tensor(3.0)
    w = torch.tensor(1.0, requires_grad=True)
    b = torch.tensor(-1.0, requires_grad=True)

    # Forward propagation
    z = w * x + b
    a = torch.relu(z)

    loss = (a - y) ** 2

    print("=== Forward Propagation ===")
    print("z:", z.item())
    print("a:", a.item())
    print("Loss:", loss.item())

    # Backpropagation
    loss.backward()

    assert w.grad is not None
    assert b.grad is not None

    print("\n=== Backpropagation ===")
    print("w gradient:", w.grad.item())
    print("b gradient:", b.grad.item())

    print("\n=== Gradient Accumulation ===")
    # new forward graph
    z2= w*x + b
    a2 = torch.relu(z2)

    loss2 = (a2 - y)** 2
    loss2.backward()

    print("Accumulated w gradient: ", w.grad.item())
    print("Accumulated b gradient: ", b.grad.item())

    # Clear gradients
    w.grad.zero_()
    b.grad.zero_()

    print("Cleared w gradient:", w.grad.item())
    print("Cleared b gradient:", b.grad.item())  