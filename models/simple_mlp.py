import torch
from torch import nn

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(4,8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8,2)

    def forward(self,x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x

if __name__ == "__main__":
    torch.manual_seed(42)
    model = SimpleMLP()
    x = torch.randn(3,4)
    output = model(x)

    print("=== Simple MLP ===")
    print(model)

    print("\nInput Shape:", x.shape)
    print("Output Shape:", output.shape)

    print("\nModel Output:")
    print(output)

    # adding model parameters

    print("\n=== Model Parameters ===")

    total_parameters = 0

    for name, parameter in model.named_parameters():
        print(
            f"{name:<15} "
            f"Shape: {str(tuple(parameter.shape)):<10} "
            f"Count: {parameter.numel()}"
        )

        total_parameters += parameter.numel()

    print("\nTotal Parameters:", total_parameters)

    print("\nState Dict Keys:")
    print(list(model.state_dict().keys()))

