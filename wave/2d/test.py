import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from u_2d import U_2d
from helpers import generate_2d_coords

u = U_2d()

u.load_state_dict(torch.load('/Users/yme/Code/Phys/wave/2d/cwave_2d.pt', weights_only=True))

txy, T, X, Y = generate_2d_coords()

z = u(txy).detach()
Z = z.reshape(T.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(X[0], Y[0], Z[0], cmap="viridis")

ax.set(xlim=[X.min(), X.max()], ylim=[Y.min(), Y.max()], zlim=[Z.min(), Z.max()])

def update(i):
    ax.plot_surface(X[i], Y[i], Z[i], cmap="viridis")
    print(f"Rendering frame {i}/{T.shape[0]}")

    return ax,

ani = animation.FuncAnimation(fig, update, frames=100, blit=False)

ani.save('/Users/yme/Code/Phys/wave/2d/cwave_2d.mp4')

plt.show()

