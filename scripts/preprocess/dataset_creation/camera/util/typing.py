import torch


def to_base_tensor(tensor: torch.Tensor):
    # TODO: Not sure if this always works
    tensor.__class__ = torch.Tensor
    return tensor
    # cast = torch.DoubleTensor if tensor.dtype == torch.float64 else torch.FloatTensor
    # return cast(tensor)