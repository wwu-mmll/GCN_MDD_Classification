import torch
import torch.nn.functional as F
from torch_geometric.nn import ChebConv, global_mean_pool


class GCN(torch.nn.Module):
    """GCN model(network architecture can be modified)"""

    def __init__(self,
                 num_features,
                 num_classes,
                 k_order,
                 dropout=0.):
        super(GCN, self).__init__()

        num_features = int(num_features)

        self.conv1 = ChebConv(num_features, 32, K=k_order)
        self.conv2 = ChebConv(32, 32, K=k_order)
        self.conv3 = ChebConv(32, 64, K=k_order)
        self.conv4 = ChebConv(64, 64, K=k_order)
        self.conv5 = ChebConv(64, 128, K=k_order)

        self.lin1 = torch.nn.Linear(128, int(num_classes))

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv3(x, edge_index))
        x = F.relu(self.conv4(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv5(x, edge_index))

        x = global_mean_pool(x, batch)
        x = self.lin1(x)

        return F.log_softmax(x, dim=1)
