import torch
from torch import vmap
from torch.func import jacrev, hessian


def compute_residual(u, txyz):
    batch_size = 50000
    num_batches = txyz.shape[0] // batch_size
    hessians = []
    for i in range(num_batches):
        print(f"batch: {i}/{num_batches}.")
        batch = txyz[i * batch_size:(i + 1) * batch_size]
        hessians.append(vmap(hessian(u))(batch))

    hess = torch.cat(hessians)
    laplacian = hess.diagonal(dim1=-2, dim2=-1)

    dtt, dxx, dyy, dzz = laplacian[:, 0], laplacian[:, 1], laplacian[:, 2], laplacian[:, 3]

    c = 1
    residual = dtt - (c ** 2) * (dxx + dyy + dzz)

    return residual

def generate_2d_coords():
    t = torch.arange(0, 100, 1, dtype=torch.float)
    x = torch.arange(0, 100, 1, dtype=torch.float)
    y = torch.arange(0, 100, 1, dtype=torch.float)
    z = torch.arange(0, 100, 1, dtype=torch.float)

    T, X, Y, Z = torch.meshgrid(t, x, y, z, indexing="ij")
    txyz = torch.stack((T.flatten(), X.flatten(), Y.flatten(), Z.flatten()), dim=1).requires_grad_(True)

    return txyz, T, X, Y, Z

breakpoint()