import torch

def single_step_optimizer(optimizer_name):
    x = torch.tensor(2.0)
    y = torch.tensor(3.0)

    w = torch.tensor(1.0, requires_grad=True)
    b = torch.tensor(-1.0, requires_grad=True)

    learning_rate = 0.1

    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(
            [w,b],
            lr=learning_rate,
            weight_decay=0.0
        )

    elif optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(
            [w,b],
            lr=learning_rate,
            weight_decay=0.0
        )

    else:
        raise ValueError("Unknown optimizer")

    optimizer.zero_grad()

    prediction = torch.relu(w*x + b)
    loss = (prediction - y)**2

    print(f"\n=== {optimizer_name} ===")
    print("Before w:", w.item())
    print("Before b:", b.item())
    print("Before Loss:", loss.item())

    loss.backward()

    assert w.grad is not None
    assert b.grad is not None

    print("w.grad:", w.grad.item())
    print("b.grad:", b.grad.item())   

    optimizer.step()

    new_prediction = torch.relu(w*x + b) 
    new_loss = (new_prediction - y) ** 2

    print("After w:", w.item())
    print("After b:", b.item())
    print("After Loss:", new_loss.item())

def train_optimizer(optimizer_name, learning_rate = 0.01):
    x = torch.tensor(2.0)
    y = torch.tensor(3.0)

    w = torch.tensor(0.5, requires_grad=True)
    b = torch.tensor(0.0, requires_grad=True)

    epochs = 20

    if optimizer_name == "SGD":
        optimizer = torch.optim.SGD(
            [w,b],
            lr=learning_rate
        )
    elif optimizer_name == "Adam":
        optimizer = torch.optim.Adam(
            [w,b],
            lr = learning_rate,
            weight_decay=0.0
        )
    elif optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(
            [w, b],
            lr=learning_rate,
            weight_decay=0.0,
        )
    else:
        raise ValueError("Unknow optimizer")

    print(f"\n=== {optimizer_name} Training ===")

    for epoch in range(epochs):
        optimizer.zero_grad()

        prediction = torch.relu(w*x + b)
        loss = (prediction - y) ** 2

        loss.backward()
        optimizer.step()

        print(
            f"Epoch {epoch + 1:2d} | "
            f"Loss: {loss.item():.6f} | "
            f"w: {w.item():.4f} | "
            f"b: {b.item():.4f}"
        )      






if __name__ == "__main__":
    single_step_optimizer("Adam")
    single_step_optimizer("AdamW")

    train_optimizer("SGD")
    train_optimizer("Adam")
    train_optimizer("AdamW")