import torch
import torch.nn as nn

class ParetoSetModel(nn.Module):
    def __init__(self, n_dim, n_obj, n_heads):
        super(ParetoSetModel, self).__init__()
        self.n_dim = n_dim
        self.n_obj = n_obj
        self.n_heads = n_heads
        self.n_node = 1024

        # Shared layers
        self.shared_fc1 = nn.Linear(n_obj, self.n_node)
        self.shared_fc2 = nn.Linear(self.n_node, self.n_node)

        # -------- 修改：每个 head 变成两层 --------
        self.output_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.n_node, self.n_node),
                nn.ReLU(),
                nn.Linear(self.n_node, n_dim)
            )
            for _ in range(n_heads)
        ])

    def forward(self, pref, ps_id):
        # Shared trunk
        x = torch.relu(self.shared_fc1(pref))
        x = torch.relu(self.shared_fc2(x))

        output = torch.zeros(pref.size(0), self.n_dim,
                             dtype=torch.float64, device=pref.device)

        # Head-specific prediction
        for i in range(self.n_heads):
            idx = (ps_id == i)
            if idx.any():
                out_i = self.output_heads[i](x[idx])
                output[idx] = torch.sigmoid(out_i).to(torch.float64)

        return output