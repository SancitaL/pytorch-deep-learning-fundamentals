import torch

def inspect_tensor(tensor):
    print(f"Tensor: {tensor}")
    print(f"Shape: {tensor.shape}")
    print(f"ndim: {tensor.ndim}")
    print(f"numel: {tensor.numel()}")
    print(f"type: {tensor.dtype}")
    print(f"device: {tensor.device}")

if __name__ == "__main__":
    float_tensor = torch.tensor(
        [1,2,3,4],
        dtype=torch.float32
    )
    inspect_tensor(float_tensor)

    int_tensor = torch.tensor(
        [1,2,3,4],
        dtype=torch.int64
    )
    inspect_tensor(int_tensor)

    fp32_tensor = torch.ones(
        1000,
        dtype=torch.float32
    )

    fp16_tensor = torch.ones(
        1000,
        dtype=torch.float16
    )

    fp32_memory = (
        fp32_tensor.numel()
        * fp32_tensor.element_size()
    )

    fp16_memory = (
        fp16_tensor.numel()
        * fp16_tensor.element_size()
    )

    print(f"FP32 Memory: {fp32_memory} Bytes")
    print(f"FP16 Memory: {fp16_memory} Bytes")

    