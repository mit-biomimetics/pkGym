import torch


def compare_tensors(file1, file2):
    tensor1 = torch.load(file1)
    tensor2 = torch.load(file2)
    assert torch.all(
        torch.eq(tensor1, tensor2)
    ), f"Tensors in {file1} and {file2} are not equal."


def test_tensor_comparison():
    file1 = "main_output.pt"
    file2 = "pr_output.pt"
    compare_tensors(file1, file2)
