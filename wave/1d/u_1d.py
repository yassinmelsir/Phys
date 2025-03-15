from torch import nn


class U_1d(nn.Module):
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