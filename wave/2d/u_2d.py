from torch import nn


class U_2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(3, 10,),
            nn.Tanh(),
            nn.Linear(10,5),
            nn.Tanh(),
            nn.Linear(5, 1)
        )

    def forward(self, txy):
        return self.seq(txy).squeeze()