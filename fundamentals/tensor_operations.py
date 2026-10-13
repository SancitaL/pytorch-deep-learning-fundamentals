import torch

def reshape_demo():
    x = torch.arange(1,13)
    print("Original: ", x)
    print("Shape: ", x.shape)

    print("(3,4):", x.reshape(3,4))
    print("(2,6):", x.reshape(2,6))
    print("(4,-1:)", x.reshape(4,-1))
    matrix = x.reshape(3,4)
    print("Transpose:",matrix.T)

def broadingcasting_demo():
    A = torch.tensor([
        [1,2,3],
        [4,5,6]
    ])

    B = torch.tensor([10,20,30])

    # 1. A + B
    C1 = A + B
    # 2.Print shape
    print(C1)
    print("Shape:",C1.shape)
    # 3.create tensor with (2,1)
    C2 = torch.randn(2,1)
    # 4.A + C2
    C3 = A + C2
    print(C3)
    print("Shape:",C3.shape)

def shape_reasoning_demo():
    A = torch.randn(3, 4)
    B = torch.randn(4, 5)
    x1 = A@B
    print(x1.shape)

    C = torch.randn(5, 1)
    D = torch.randn(1, 4)
    x2 = C + D
    print(x2.shape)

if __name__ == "__main__":
    reshape_demo()

    print("---------")
    A = torch.tensor([
        [1,2,3],
        [4,5,6]
    ])

    B = torch.tensor([
        [1,2],
        [3,4],
        [5,6]
    ])

    C1 = torch.matmul(A,B)
    C2 = torch.tensor([
        [22, 28],
        [49, 64]
    ])
    print(torch.equal(C1,C2))

    broadingcasting_demo()

    shape_reasoning_demo()