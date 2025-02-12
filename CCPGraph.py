import torch
from torch import nn
from torch.nn import Linear
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing, GraphNorm
from torch_geometric.utils import softmax
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool
from torch_scatter import scatter
from torch_scatter import scatter_add


class Conv(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_dim=6, aggr='max', dropout_rate=0.2):
        super().__init__(aggr=aggr)
        self.lin_neg = nn.Linear(in_channels + edge_dim, out_channels)
        self.lin_root = nn.Linear(in_channels, out_channels)
        self.dropout = nn.Dropout(p=dropout_rate)

        if in_channels != out_channels:
            self.residual = nn.Linear(in_channels, out_channels)
            self.skip = nn.Linear(in_channels, out_channels)
        else:
            self.residual = nn.Identity()
            self.skip = nn.Identity()  # Skip connection if dimensions are the same

    def forward(self, x, edge_index, edge_attr):
        # Original input for skip connection
        skip_input = self.skip(x)
        # Process edges
        x_adj = torch.cat([x[edge_index[1]], edge_attr], dim=1)
        x_adj = torch.relu(self.lin_neg(x_adj))
        x_adj = self.dropout(x_adj)
        neg_sum = scatter(x_adj, edge_index[0], dim=0, reduce=self.aggr, dim_size=x.size(0))
        # Process root nodes
        x_root = torch.relu(self.lin_root(x))
        x_root = self.dropout(x_root)
        # Combine features
        out = self.residual(x) + neg_sum + x_root + skip_input  # Add skip connection
        return out


class DiffPoolLayer(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.sage = DenseSAGEConv(in_channels, hidden_channels)
        self.assign_mat = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, adj, mask=None):
        x = F.relu(self.sage(x, adj, mask))
        s = self.assign_mat(x)
        x, adj, l, e = dense_diff_pool(x, adj, s, mask)
        return x, adj, l, e


def reset(nn):  

    def _reset(item):  
        if hasattr(item, 'reset_parameters'):  
            item.reset_parameters()

    if nn is not None: 
        if hasattr(nn, 'children') and len(list(nn.children())) > 0: 
            for item in nn.children():  
                _reset(item)
        else:  
            _reset(nn)


class GlobalAttention(torch.nn.Module):

    def __init__(self, gate_nn, nn=None):
        super().__init__()
        self.gate_nn = gate_nn 
        self.nn = nn  
        self.reset_parameters() 

    def reset_parameters(self):
        reset(self.gate_nn) 
        reset(self.nn)

    def forward(self, x, batch, size=None): 
        x = x.unsqueeze(-1) if x.dim() == 1 else x  
        size = batch[-1].item() + 1 if size is None else size 
        gate = self.gate_nn(x).view(-1, 1)  
        x = self.nn(x) if self.nn is not None else x  
        assert gate.dim() == x.dim() and gate.size(0) == x.size(0)  
        gate = softmax(gate, batch, num_nodes=size)  
        out = scatter_add(gate * x, batch, dim=0, dim_size=size)  
        return out, gate  

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(gate_nn={self.gate_nn}, nn={self.nn})')


class MultiHeadGlobalAttention(torch.nn.Module):
    def __init__(self, gate_nn, nn_module=None, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.heads = nn_module.ModuleList([GlobalAttention(gate_nn) for _ in range(num_heads)])

    def forward(self, x, batch, size=None):
        heads_output = [head(x, batch, size) for head in self.heads]
        embeddings = torch.cat([out[0].unsqueeze(0) for out in heads_output], dim=0)
        attention_scores = torch.cat([out[1].unsqueeze(0) for out in heads_output], dim=0)

        embeddings = embeddings.mean(dim=0)
        attention_scores = attention_scores.mean(dim=0)

        return embeddings, attention_scores


class CCPGraph(nn.Module):
    def __init__(self, use_global_features=False):
        super(CCPGraph, self).__init__()
        self.use_global_features = use_global_features

        self.conv1 = Conv(38, 64)
        self.gn1 = GraphNorm(64)
        self.conv2 = Conv(64, 128)
        self.gn2 = GraphNorm(128)
        self.conv3 = Conv(128, 256)
        self.gn3 = GraphNorm(256)
        self.conv4 = Conv(256, 256)
        self.gn4 = GraphNorm(256)
        self.conv5 = Conv(256, 128)
        self.gn5 = GraphNorm(128)
        self.conv6 = Conv(128, 16)
        self.gn6 = GraphNorm(16)

        gate_nn = nn.Sequential(
            nn.Linear(16, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.readout = MultiHeadGlobalAttention(gate_nn, nn_module=nn, num_heads=4)

        if self.use_global_features:
            self.lin1 = nn.Linear(16 + 612, 1024)
        else:
            self.lin1 = nn.Linear(16, 1024)

        self.bn1 = nn.BatchNorm1d(1024)
        self.dp1 = nn.Dropout(p=0.5)
        self.lin2 = nn.Linear(1024, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.dp2 = nn.Dropout(p=0.4)
        self.lin3 = nn.Linear(1024, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.dp3 = nn.Dropout(p=0.3)
        self.lin4 = nn.Linear(256, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.dp4 = nn.Dropout(p=0.2)
        self.lin_final = nn.Linear(128, 1)

    def forward(self, data):
        x = self.conv1(data.x, data.edge_index, data.edge_attr)
        x = self.gn1(x)
        x = self.conv2(x, data.edge_index, data.edge_attr)
        x = self.gn2(x)
        x = self.conv3(x, data.edge_index, data.edge_attr)
        x = self.gn3(x)
        x = self.conv4(x, data.edge_index, data.edge_attr)
        x = self.gn4(x)
        x = self.conv5(x, data.edge_index, data.edge_attr)
        x = self.gn5(x)
        x = self.conv6(x, data.edge_index, data.edge_attr)
        x = self.gn6(x)
        embedding, att = self.readout(x, data.batch)

        if self.use_global_features:
            if not hasattr(data, 'global_feature'):
                raise AttributeError("Input data lacks 'global_feature'")
            global_features = data.global_feature.view(-1, 612)
            if embedding.size(0) != global_features.size(0):
                raise ValueError(
                    f"Size mismatch: embedding size {embedding.size(0)}, global features size {global_features.size(0)}")
            combined_features = torch.cat([embedding, global_features], dim=1)
        else:
            combined_features = embedding

        out = self.dp1(self.bn1(F.relu(self.lin1(combined_features))))
        out = self.dp2(self.bn2(F.relu(self.lin2(out))))
        out = self.dp3(self.bn3(F.relu(self.lin3(out))))
        out = self.dp4(self.bn4(F.relu(self.lin4(out))))
        out = self.lin_final(out)

        return out.view(-1), att


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    return {"total_params": total_params, "trainable_params": trainable_params,
            "non_trainable_params": non_trainable_params}


def print_model_parameters(model):
    params = count_parameters(model)
    print(f"Total Parameters: {params['total_params']:,}")
    print(f"Trainable Parameters: {params['trainable_params']:,}")
    print(f"Non-Trainable Parameters: {params['non_trainable_params']:,}")


if __name__ == "__main__":
    model = CCPGraph()
    print(model)
    print_model_parameters(model)

    print("PyTorch version:", torch.__version__)
    print("CUDA version:", torch.version.cuda)
    print("cuDNN version:", torch.backends.cudnn.version())
    print("Is CUDA available:", torch.cuda.is_available())
    print("PyTorch Geometric version:", torch_geometric.__version__)
