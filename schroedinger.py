import torch
import torch.autograd
from torch import nn, vmap
from torch.func import jacrev, hessian
import torch.optim as optim

class U(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(2, 10,),
            nn.Tanh(),
            nn.Linear(10,5),
            nn.Tanh(),
            nn.Linear(5, 1)
        )

    def forward(self, tx):
        return self.seq(tx).squeeze()

def compute_residual(u, tx):
    hess = vmap(hessian(u))(tx)
    laplacian = hess.diagonal(dim1=-2, dim2=-1)
    dtt, dxx = laplacian[:, 0], laplacian[:, 1]

    c = 1
    residual = dtt - (c ** 2) * dxx

    return residual

t = torch.linspace(0, 10, 100)
x = torch.linspace(0, 10, 100)

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
        print(f"residual: {res}")
        print(f"Epoch [{ep + 1}/{epochs}], Loss: {loss.item()}")

