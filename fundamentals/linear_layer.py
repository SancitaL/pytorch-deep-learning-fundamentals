import torch

def manual_linear():
    X = torch.tensor([
        [1.0,2.0,3.0]
    ])

    W = torch.tensor([
        [1.0, 0.0, 2.0],
        [0.0, 1.0, 1.0]
    ])

    b = torch.tensor([1.0, -1.0])

    # 1.XWb shape
    print(f"X shape:",X.shape)
    print(f"W shape:",W.shape)
    print(f"b shape:",b.shape)

    # 2.W.T
    W_T = W.T

    # 3.X @ W.T + b
    Y = X @ W_T + b

    print(f"Y: ", Y)
    print(f"Y shape: ", Y.shape)


def nn_linear_demo():
    X = torch.tensor([
        [1.0, 2.0, 3.0]
    ])

    W = torch.tensor([
        [1.0, 0.0, 2.0],
        [0.0, 1.0, 1.0]
    ])

    b = torch.tensor([1.0, -1.0])

    # 1.create nn.Linear(3,2)
    linear = torch.nn.Linear(3,2,True)

    # 2.use no_grad and copy_, copy W,b on Linear Layer
    with torch.no_grad():
        linear.weight.copy_(W)
        linear.bias.copy_(b)

    # 3.Call linear(X)
    Output = linear(X)

    # 4.print Y and Y.shape
    print(f"Output: ", Output)
    print(f"Output shape: ",Output.shape)

    # 5.check output are equan
    print(torch.allclose(Output,X@W.T + b))


def batch_linear_demo():
    X = torch.tensor([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]
    ])

    W = torch.tensor([
        [1.0, 0.0, 2.0],
        [0.0, 1.0, 1.0]
    ])

    b = torch.tensor([1.0, -1.0])

    # manual calculation
    Y = X@W.T + b

    # craete Linear(3,2)
    linear = torch.nn.Linear(3,2,True)

    # copy W and b into linear
    with torch.no_grad():
        linear.weight.copy_(W)
        linear.bias.copy_(b)

    # calculation
    output = linear(X)

    # rs comparing
    rs = torch.allclose(output, Y)

    # print Shape
    print(f"Shape: ", output.shape)
    print(rs)

if __name__ == "__main__":
    manual_linear()
    nn_linear_demo()
    batch_linear_demo()