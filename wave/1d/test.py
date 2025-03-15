import torch
from u_1d import U_1d
import matplotlib.pyplot as plt
import matplotlib.animation as animation
u = U_1d()
u.load_state_dict(torch.load('/Users/yme/Code/Phys/wave/1d/cwave_1d.pt', weights_only=True))

t = torch.arange(0, 100, 1, dtype=torch.float)
x = torch.arange(0, 100, 1, dtype=torch.float)
T, X = torch.meshgrid(t, x, indexing="ij")
tx = torch.stack((T.flatten(), X.flatten()), dim=1).requires_grad_(True)

y = u(tx).detach()
Y = y.reshape(T.shape)
txy = torch.cat((tx, y.unsqueeze(1)), dim=1)

breakpoint()