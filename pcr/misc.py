import torch


def onnx_minimum(x1, x2):
    return torch.where(x2 < x1, x2, x1)


def fp_sampling(points, num: int):
    batch_size = points.shape[0]
    # TODO use onnx_cdists just to export to onnx, otherwise use torch.cdist
    # D = onnx_cdists(points, points)
    D = torch.cdist(points, points)
    # By default, takes the first point in the list to be the
    # first point in the permutation, but could be random
    res = torch.zeros((batch_size, 1), dtype=torch.int32, device=points.device)
    ds = D[:, 0, :]
    for i in range(1, num):
        idx = ds.max(dim=1)[1]
        res = torch.cat([res, idx.unsqueeze(1).to(torch.int32)], dim=1)
        ds = onnx_minimum(ds, D[torch.arange(batch_size), idx, :])

    return res
