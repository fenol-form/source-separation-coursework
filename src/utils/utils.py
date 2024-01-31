import torch


def loadObj(obj, device):
    """
    Offload tensor object in obj to cuda device
    """

    def cuda(obj):
        return obj.to(device) if isinstance(obj, torch.Tensor) else obj

    if isinstance(obj, dict):
        return {key: loadObj(obj[key], device) for key in obj}
    elif isinstance(obj, list) or isinstance(obj, tuple):
        return [loadObj(val, device) for val in obj]
    else:
        return cuda(obj)
