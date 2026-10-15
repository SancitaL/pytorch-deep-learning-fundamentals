import torch

def simple_autograd_demo():
    x = torch.tensor(
        3.0,
        requires_grad=True
    )

    y = x ** 2

    # print x and y
    print(
        f"x: {x}"
        f"\ny: {y}"
    )
    # print x.grad before backward
    print(f"x grad: {x.grad}")

    # call backward
    y.backward()

    # reprint x.grad
    print(f"x grad after backward: {x.grad}")

    # calculate 2*x and compare
    z = 2 * x
    if x.grad is not None:
        print(torch.allclose(z, x.grad))

def chain_rule_demo():
    x = torch.tensor(
        2.0,
        requires_grad=True
    )

    a = x**2
    y = 3*a+1

    # 1.calculate dy / da
    dy_da = 3

    # 2.da/dx
    da_dx = 2 * x

    # 3.chain rule calculate dydx
    dy_dx = dy_da * da_dx

    # 4.autograd and compare
    y.backward()
    if x.grad is not None:
        print("Manual Gradient:", dy_dx)
        print("Autograd Gradient:", x.grad)

        print("Match:", torch.allclose(dy_dx, x.grad))

def gradient_accumulation_demo():
    x = torch.tensor(
        2.0,
        requires_grad=True
    )
    # 1.compute y = x**2
    y = x **2

    # 2.call backward on y
    y.backward()

    # 3.print first x.grad
    print(f"First Gradient: {x.grad}")

    # 4. compute z
    z = x*3

    # 5.z.backward and print second x.grad
    z.backward()
    print(f"Accumulated Gradient: {x.grad}")

    # 6.use x.grad = None clear grad and print
    x.grad = None
    print(f"After Clearing: {x.grad}")

def vector_autograd_demo():
    x = torch.tensor(
        [1.0,2.0,3.0],
        requires_grad=True
    )

    y = x ** 2
    # 1.get scalar L by summing y
    L = torch.sum(y)
    print(f"Loss: {L}")
    L.backward()
    
    # print x.grad
    print(f"Autograd Gradient: {x.grad}")
    # 2.manually compute gradient 2*x
    L1 = 2*x
    print(f"Manual Gradient: {L1}")

    # compare
    if x.grad is not None:
        print(f"Match: {torch.allclose(x.grad,L1)}")


if __name__ == "__main__":
    simple_autograd_demo()
    chain_rule_demo()
    gradient_accumulation_demo()
    vector_autograd_demo()