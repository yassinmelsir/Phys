import torch
from torch import vmap
from torch.func import jacrev, hessian


def compute_residual(u, txy):
    hess = vmap(hessian(u))(txy)
    laplacian = hess.diagonal(dim1=-2, dim2=-1)
    dtt, dxx, dyy = laplacian[:, 0], laplacian[:, 1], laplacian[:, 2]

    c = 1
    residual = dtt - (c ** 2) * (dxx + dyy)

    return residual

def generate_2d_coords():
    t = torch.arange(0, 100, 1, dtype=torch.float)
    x = torch.arange(0, 100, 1, dtype=torch.float)
    y = torch.arange(0, 100, 1, dtype=torch.float)

    T, X, Y = torch.meshgrid(t, x, y, indexing="ij")

    txy = torch.stack((T.flatten(), X.flatten(), Y.flatten()), dim=1).requires_grad_(True)

    return txy, T, X, Y