import torch
from torch import nn

class FashionMNISTMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(784, 256)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

if __name__ == "__main__":
    torch.manual_seed(42)

    model = FashionMNISTMLP()

    sample = torch.randn(64,1,28,28)  # Batch of 64 images, 1 channel, 28x28 pixels
    output = model(sample)

    total_params = sum(
        p.numel() for p in model.parameters()

    )

    print(model)
    print("Input Shape:", sample.shape)
    print("Output Shape:", output.shape)
    print("Total Parameters:", total_params)