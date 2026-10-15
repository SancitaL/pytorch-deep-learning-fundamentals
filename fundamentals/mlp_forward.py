import torch

def mini_mlp_forward():
    torch.manual_seed(42)

    # 1. Create input tensor
    x = torch.randn(3, 4)

    # 2. Create first Linear Layer
    # TODO: in_features=4, out_features=8
    linear1 = torch.nn.Linear(4,8)

    # 3. Create second Linear Layer
    # TODO: in_features=8, out_features=2
    linear2 = torch.nn.Linear(8,2)

    # 4. First Linear Forward
    # TODO
    hidden_before_relu = linear1(x)

    # 5. Apply ReLU
    # TODO
    hidden_after_relu = torch.relu(hidden_before_relu)

    # 6. Second Linear Forward
    # TODO
    output = linear2(hidden_after_relu)

    # 7. Print Tensor Shapes
    # x.shape
    # hidden_before_relu.shape
    # hidden_after_relu.shape
    # output.shape
    print(
        f"x.shape: {x.shape}\n"
        f"hidden_before_relu.shpae: {hidden_before_relu.shape}\n"
        f"hidden_after_relu.shape: {hidden_after_relu.shape}\n"
        f"output shape: {output.shape}"
        )

    # Parameters Num
    print("Linear 1 Weight:", linear1.weight.shape)
    print("Linear 1 Bias:", linear1.bias.shape)

    print("Linear 2 Weight:", linear2.weight.shape)
    print("Linear 2 Bias:", linear2.bias.shape)

    # 1.linear1.weight
    linear1_weight = linear1.weight.numel()

    # 2.lienar1.bias
    linear1_bias = linear1.bias.numel()

    # 3.lienar2.weight and bis
    linear2_weight = linear2.weight.numel()
    linear2_bias = linear2.bias.numel()

    # 4.total num of para
    total = linear1_bias + linear1_weight + linear2_bias + linear2_weight 

    # 5.print
    print(f"Total: {total}")

    # mini MLP Gradient
    S = output.sum()
    S.backward()

    if linear1.weight.grad is not None:
        print(f"linear1.weight.grad: {linear1.weight.grad}")
    if linear1.bias.grad is not None:
        print(f"linear1.bias.grad: {linear1.bias.grad}")
    if linear2.weight.grad is not None:
        print(f"linear2.weight.grad: {linear2.weight.grad}")
    if linear2.bias.grad is not None:
        print(f"linear2.bias.grad: {linear2.bias.grad}")


if __name__ == "__main__":
    mini_mlp_forward()