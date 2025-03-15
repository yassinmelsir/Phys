import torch.autograd
from torch import nn
import torch.optim as optim
from u_2d import U_2d

from helpers import generate_2d_coords, compute_residual

txy, T, X, Y = generate_2d_coords()

u = U_2d()
crit = nn.MSELoss()

optimizer = optim.Adam(u.parameters())
epochs = 1000

for ep in range(epochs):
    optimizer.zero_grad()

    res = compute_residual(u, txy)

    loss = torch.mean(res**2)

    loss.backward()
    optimizer.step()

    print(f"Epoch [{ep + 1}/{epochs}], Loss: {loss.item()}")

torch.save(u.state_dict(), './classic_wave.pt')