import numpy as np
import torch
import torch.autograd
from torch import nn, vmap
from torch.func import jacrev, hessian
import torch.optim as optim
from u_1d import U_1d


def compute_residual(u, tx):
    hess = vmap(hessian(u))(tx)
    laplacian = hess.diagonal(dim1=-2, dim2=-1)
    dtt, dxx = laplacian[:, 0], laplacian[:, 1]

    c = 1
    residual = dtt - (c ** 2) * dxx

    return residual

t = torch.linspace(0, 1, 100, dtype=torch.float)
x = torch.linspace(0, 1, 100, dtype=torch.float)
T, X = torch.meshgrid(t, x, indexing="ij")
tx = torch.stack((T.flatten(), X.flatten()), dim=1).requires_grad_(True)

u = U_1d()
crit = nn.MSELoss()

optimizer = optim.Adam(u.parameters())
epochs = 1000

g = 0
L = 1

for ep in range(epochs):
    optimizer.zero_grad()

    res = compute_residual(u, tx)
    ut_o = u(tx[tx[:, 0] == 0])
    ux_o =  u(tx[tx[:, 1] == 0])
    ux_L = u(tx[tx[:, 1] == L])

    l_bc = torch.mean(ux_o)**2 + torch.mean(ux_L)**2
    l_ic = torch.mean((ut_o - g)**2)
    l_res = torch.mean(res**2)


    loss = l_res + l_ic + l_bc

    loss.backward()
    optimizer.step()
    if (ep + 1) % 50 == 0:
        print(f"Epoch [{ep + 1}/{epochs}], Loss: {loss.item()}")

torch.save(u.state_dict(), './cwave_1d.pt')