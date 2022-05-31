import torch


def to_device(tensor, device):
    if '__getitem__' in dir(tensor):
        assert isinstance(tensor[0], torch.Tensor)
        # 如果传进来是个列表或元组，每个元素是一个tensor，就把每个元素转换到指定的device上
        return [to_device(t, device) for t in tensor]
    elif isinstance(tensor, torch.Tensor):
        return tensor.to(device)
