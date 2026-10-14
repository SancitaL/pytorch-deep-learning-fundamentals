import torch
import matplotlib.pyplot as plt
from pathlib import Path

def manual_relu(x):
    rs = torch.clamp(x,0)
    return rs

def activation_demo():
    x = torch.tensor([
        -3.0, -1.0, 0.0, 1.0, 3.0
    ])

    # TODO 1: 计算 ReLU
    relu = torch.relu(x)

    # TODO 2: 计算 Sigmoid
    sigmoid = torch.sigmoid(x)
    
    # TODO 3: 计算 Tanh
    tanh = torch.tanh(x)

    # TODO 4: 打印各个函数的结果
    print(
        f"ReLU: {relu}\n"
        f"Sigmoid: {sigmoid}\n"
        f"Tanh: {tanh}\n"
    )

    # TODO 5: 打印三个输出的 Shape
    print(
        f"Shape:-----\n"
        f"ReLU : {relu.shape}\n"
        f"Sigmoid: {sigmoid.shape}\n"
        f"Tanh: {tanh.shape}"
    )

    # Compare manual rs with API rs
    manual = manual_relu(x)
    print(torch.allclose(manual_relu(x), relu))

def plot_activation_fucntions():
    x = torch.linspace(-5,5,200)

    y_relu = torch.relu(x)
    y_sigmoid = torch.sigmoid(x)
    y_tanh = torch.tanh(x)

    plt.plot(x.numpy(),y_relu.numpy(),label = "ReLU")
    plt.plot(x.numpy(),y_sigmoid.numpy(),label = "Sigmoid")
    plt.plot(x.numpy(),y_tanh.numpy(),label = "Tanh")

    plt.title("Activation Functions")
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.legend()
    plt.grid(True)

    output_dir = Path("results/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        output_dir / "activation_functions.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.show()

if __name__ == "__main__":
    activation_demo()

    x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])

    manual_output = manual_relu(x)
    pytorch_output = torch.relu(x)

    print("Manual ReLU:", manual_output)
    print("PyTorch ReLU:", pytorch_output)

    print(
        "Match:",
        torch.allclose(manual_output, pytorch_output)
    )

    plot_activation_fucntions()