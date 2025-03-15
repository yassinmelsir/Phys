import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from u_2d import U_2d
from helpers import generate_2d_coords

u = U_2d()

u.load_state_dict(torch.load('./classic_wave.pt', weights_only=True))

txy, T, X, Y = generate_2d_coords()

xo = txy[txy[:, 0] == 0][:, 1].detach()
yo = txy[txy[:, 0] == 0][:, 2].detach()

z = u(txy).detach()
zo = z[txy[:, 0] == 0].detach()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_trisurf(xo, yo, zo, cmap="viridis")

ax.set(xlim=[X.min(), X.max()], ylim=[Y.min(), Y.max()], zlim=[z.min(), z.max()])

def update(i):
    xi = txy[txy[:, 0] == i][:, 1].detach()
    yi = txy[txy[:, 0] == i][:, 2].detach()
    zi = z[txy[:, 0] == i].detach()

    ax.plot_trisurf(xi, yi, zi, cmap="viridis")

    print(f"Rendering frame {i}/{T.shape[0]}")

    return ax,

ani = animation.FuncAnimation(fig, update, frames=T.shape[0])

ani.save('./classic_wave.mp4')

plt.show()

