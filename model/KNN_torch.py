import torch

class KNN:
    def __init__(self, k: int):
        self.k = k
    def __call__(self, y: torch.Tensor, x: torch.Tensor = None):

        if y is None:
            y = x
        B, N1, C = x.shape
        _, N2, _ = y.shape

        # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 * x·y
        x_norm = (x ** 2).sum(dim=2, keepdim=True)  # [B, N1, 1]
        y_norm = (y ** 2).sum(dim=2, keepdim=True)  # [B, N2, 1]
        y_norm = y_norm.permute(0, 2, 1)            # [B, 1, N2]

        inner_prod = torch.bmm(x, y.transpose(1, 2))  # [B, N1, N2]
        dists = x_norm + y_norm - 2 * inner_prod      # [B, N1, N2]

        dists = torch.clamp(dists, min=0)

        dists_k, idx = torch.topk(dists, self.k, dim=-1, largest=False, sorted=True)  # [B, N1, k]
        return dists_k, idx

