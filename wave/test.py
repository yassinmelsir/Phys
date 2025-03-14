import torch
from wave.u import U
import matplotlib.pyplot as plt
import matplotlib.animation as animation


u = U()
u.load_state_dict(torch.load('/Users/yme/Code/Phys/wave/classic_wave.pt', weights_only=True))

t = torch.arange(0, 100, 1, dtype=torch.float)
x = torch.arange(0, 100, 1, dtype=torch.float)
T, X = torch.meshgrid(t, x, indexing="ij")
tx = torch.stack((T.flatten(), X.flatten()), dim=1).requires_grad_(True)

y = u(tx).detach()
xo = tx[tx[:, 0] == 0][:, 1].detach()
yo = y[tx[:, 0] == 0].detach()

fig, ax = plt.subplots()
line,  = ax.plot(xo, yo)

ax.set(xlim=[X.min(), X.max()], ylim=[y.min(), y.max()])

def update(i):
    xi = tx[tx[:, 0] == i][:, 1].detach()
    yi = y[tx[:, 0] == i].detach()
    line.set_data(xi, yi)
    return line,


ani = animation.FuncAnimation(fig, update, frames=T.shape[0])

ani.save('./wave.mp4')

plt.show()

