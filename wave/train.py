import torch
import torch.autograd
from torch import nn, vmap
from torch.func import jacrev, hessian
import torch.optim as optim

from wave.u import U


def compute_residual(u, tx):
    hess = vmap(hessian(u))(tx)
    laplacian = hess.diagonal(dim1=-2, dim2=-1)
    dtt, dxx = laplacian[:, 0], laplacian[:, 1]

    c = 1
    residual = dtt - (c ** 2) * dxx

    return residual

t = torch.arange(0, 100, 1, dtype=torch.float)
x = torch.arange(0, 100, 1, dtype=torch.float)
T, X = torch.meshgrid(t, x, indexing="ij")
tx = torch.stack((T.flatten(), X.flatten()), dim=1).requires_grad_(True)

u = U()
crit = nn.MSELoss()

optimizer = optim.Adam(u.parameters())
epochs = 1000

for ep in range(epochs):
    optimizer.zero_grad()

    res = compute_residual(u, tx)


    loss = torch.mean(res**2)

    loss.backward()
    optimizer.step()
    if (ep + 1) % 50 == 0:
        print(f"Epoch [{ep + 1}/{epochs}], Loss: {loss.item()}")

torch.save(u.state_dict(), './wave/classic_wave.pt')