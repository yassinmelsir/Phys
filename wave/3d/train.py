import torch.autograd
from torch import nn
import torch.optim as optim
from u_3d import U_3d

from helpers import generate_2d_coords, compute_residual

txyz, T, X, Y, Z = generate_2d_coords()

u = U_3d()
crit = nn.MSELoss()

optimizer = optim.Adam(u.parameters())
epochs = 1000

for ep in range(epochs):
    optimizer.zero_grad()

    res = compute_residual(u, txyz)
    loss = torch.mean(res**2)

    loss.backward()
    optimizer.step()

    print(f"Epoch [{ep + 1}/{epochs}], Loss: {loss.item()}")

torch.save(u.state_dict(), './cwave_3d.pt')