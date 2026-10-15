import torch

def computational_graph_demo():
    x = torch.tensor(
        2.0,
        requires_grad=True
    )

    a = x * 3
    y = a + 1

    # TODO 1: 输出 x、a、y 的数值
    print(
        f"x: {x}\n"
        f"a: {a}\n"
        f"y: {y}\n"
    )
    # TODO 2: 输出三个 Tensor 的 requires_grad
    print(
        f"requires grad:\n"
        f"x: {x.requires_grad}\n"
        f"a: {a.requires_grad}\n"
        f"y: {y.requires_grad}\n"
    )
    # TODO 3: 输出三个 Tensor 的 is_leaf
    print(
        f"is leaf:\n"
        f"x: {x.is_leaf}\n"
        f"a: {a.is_leaf}\n"
        f"y: {y.is_leaf}\n"
    )
        
    # TODO 4: 输出三个 Tensor 的 grad_fn
    print(
        f"grad_fn:\n"
        f"x: {x.grad_fn}\n"
        f"y: {y.grad_fn}\n"
        f"a: {a.grad_fn}\n"
    )

def gradient_tracking_demo():
    x = torch.tensor(
        3.0,
        requires_grad=True
    )

    # a = x**2
    a = x ** 2

    with torch.no_grad():
        b = x**2

    print(f"a: {a}")
    print(f"b: {b}")

    # compare a.requires_grad and b.requires_grad
    print(f"\na.requires_grad: {a.requires_grad}")
    print(f"b.requires_grad: {b.requires_grad}")

    # a.grad_fn and b.grad_fn
    print(f"\na.grad_fn:{a.grad_fn}")
    print(f"b.grad_fn: {b.grad_fn}")

    # a,b is leaf:
    print(f"\nIs a a leaf: {a.is_leaf}")
    print(f"Is b a leaf: {b.is_leaf}")

def linear_graph_demo():
    linear = torch.nn.Linear(3,2)
    x = torch.tensor([
        [1.0,2.0,3.0]
    ])

    y = linear(x)

    # 1. print linear.weight.requires_grad
    print(f"weight.requires_grad: {linear.weight.requires_grad}")

    # 2.print linear.bias.requires_grad
    print(f"bias.requires_grad: {linear.bias.requires_grad}")

    # 3.print linear.weight.is_leaf
    print(f"weight.is_leaf: {linear.weight.is_leaf}")

    # 4.print x.requires_grad
    print(f"x.requires_grad: {x.requires_grad}")

    # 5.print y.requries_grad
    print(f"y.requires_grad: {y.requires_grad}")

    # 6.print y.grad_fn
    print(f"y.grad_fn: {y.grad_fn}")

if __name__ == "__main__":
    computational_graph_demo()
    gradient_tracking_demo()
    linear_graph_demo()