import torch.nn as nn
from torch_scatter import scatter_add

class SetModel(nn.Module):

    def __init__(
        self,
        x: int,
        y: int,
    ):
        super().__init__()
        
        self.x = x
        self.y = y

        self.emb = nn.Embedding(100, x)
        self.lin = nn.Sequential(nn.Linear(x, x), nn.ReLU(), nn.Linear(x, y))
        self.lin_out = nn.Sequential(nn.Linear(y, y), nn.ReLU(), nn.Linear(y, 1))

    def forward(self, x, batch):
        
        x = self.emb(x)
        x = self.lin(x)
        out = scatter_add(x, batch, dim=0)
        out = self.lin_out(out)
        return out
    
    def forward2(self, x, edge_index, edge_attr, batch, label = None):

        x = self.emb(x)
        x = self.lin(x)
        out = scatter_add(x, batch, dim=0)
        return out