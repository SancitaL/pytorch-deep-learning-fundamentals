import torch
from torch import nn

class FashionMNISTMLP(nn.Module):
    def __init__(self,hidden1=256,hidden2=128):
        super().__init__()
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(784, hidden1)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(hidden1, hidden2)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(hidden2, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

if __name__ == "__main__":
    for hidden1, hidden2 in [
        (128,64),
        (256,128),
        (512,256)
    ]:
        model = FashionMNISTMLP(hidden1,hidden2)
        sample = torch.randn(64,1,28,28)
        output = model(sample)

        total_p = sum(p.numel() for p in model.parameters())

        print(
        f"Hidden: ({hidden1}, {hidden2}) | "
        f"Output: {tuple(output.shape)} | "
        f"Parameters: {total_p:,}"
    )