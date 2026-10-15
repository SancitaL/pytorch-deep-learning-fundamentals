import torch

def mse_demo():
    predictions = torch.tensor(
        [2.0, 4.0, 6.0]
    )

    targets = torch.tensor(
        [1.0, 5.0, 5.0]
    )

    # 1. predictions - targets
    error = predictions - targets

    # 2. error sqeure
    squared_error = error ** 2

    # 3.mse
    manual_mse = squared_error.mean()

    # 4.use torch.nn.mseloss
    criterion = torch.nn.MSELoss()
    pytorch_mse = criterion(predictions, targets)

    # 5.compare
    print(
        f"Manual MSE: {manual_mse}\n"
        f"PyTorch MSE: {pytorch_mse}\n"
        f"Match: {torch.allclose(manual_mse,pytorch_mse)}"
    )

def cross_entropy_demo():
    logits = torch.tensor([
        [2.0, 1.0, 0.1]
    ])

    labels = torch.tensor([0])

    # 1.do softmax to logits with dim=1
    after_softmax_logits = torch.softmax(logits, dim=1)

    # 2.print probability:
    print("Probabilities: ", after_softmax_logits)

    # 3.create torch.nn.crossentrypoyloss
    criterion = torch.nn.CrossEntropyLoss()

    # 4.compute loss by original logits and lables
    loss = criterion(logits, labels)

    # 5.print loss and shape
    print("Cross Entropy Loss:", loss)
    print("Loss Shape:", loss.shape)

def loss_autograd_demo():
    logits = torch.tensor(
        [[2.0,1.0,0.1]],
        requires_grad=True
        )

    lables = torch.tensor([0])

    # TODO 1: 创建 CrossEntropyLoss
    criterion = torch.nn.CrossEntropyLoss()

    # TODO 2: 计算 loss
    loss = criterion(logits, lables)

    # TODO 3: 打印 backward 前的 logits.grad
    print(f"Before backward: {logits.grad}")

    # TODO 4: 调用 loss.backward()
    loss.backward()

    # TODO 5: 打印 backward 后的 logits.grad
    print(f"After backward: {logits.grad}")

    # TODO 6: 打印 logits.grad.shape
    if logits.grad is not None:
        print(f"Gradient Shape: {logits.grad.shape}")

    
if __name__ == "__main__":
    mse_demo()
    cross_entropy_demo()
    loss_autograd_demo()