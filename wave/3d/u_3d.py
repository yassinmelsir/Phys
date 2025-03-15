from torch import nn


class U_3d(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(4, 10,),
            nn.Tanh(),
            nn.Linear(10,5),
            nn.Tanh(),
            nn.Linear(5, 1)
        )

    def forward(self, txy):
        return self.seq(txy).squeeze()