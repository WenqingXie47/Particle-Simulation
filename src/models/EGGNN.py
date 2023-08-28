
import torch
from typing import Optional, Tuple
from torch import Tensor
from torch_geometric.utils import scatter

class EquivariantMetaLayer(torch.nn.Module):
   
    def __init__(
        self,
        edge_model: Optional[torch.nn.Module] = None,
        node_model: Optional[torch.nn.Module] = None,
        global_model: Optional[torch.nn.Module] = None,
        n_dimensions=3,
    ):
        super().__init__()
        self.edge_model = edge_model
        self.node_model = node_model
        self.global_model = global_model
        self.n_dimensions = n_dimensions

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        for item in [self.node_model, self.edge_model, self.global_model]:
            if hasattr(item, 'reset_parameters'):
                item.reset_parameters()


    def forward(
        self,
        node_attr: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        u: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        r"""
        Args:
            x (torch.Tensor): The node features.
            edge_index (torch.Tensor): The edge indices.
            edge_attr (torch.Tensor, optional): The edge features.
                (default: :obj:`None`)
            u (torch.Tensor, optional): The global graph features.
                (default: :obj:`None`)
            batch (torch.Tensor, optional): The batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
                each node to a specific graph. (default: :obj:`None`)
        """
        

        
        node_attr_scalar = node_attr[:,2*self.n_dimensions:]
        node_attr_vector = node_attr[:,0:2*self.n_dimensions]
        pos = node_attr_vector[:,0:self.n_dimensions]
        vel = node_attr_vector[:,self.n_dimensions:]

        radial, pos_diff = self.vec2radial(edge_index, pos, vel)
        edge_attr_scalar = torch.cat([edge_attr, radial], dim=1)
        edge_attr_vector = pos_diff

        row, col = edge_index
        edge_feature_scalar = self.edge_model(node_attr_scalar[row], node_attr_scalar[col], 
                                            edge_attr_scalar , u,
                                            batch if batch is None else batch[row])

        acc = self.node_model(node_attr_scalar, node_attr_vector,
                               edge_index, edge_feature_scalar, edge_attr_vector, 
                               u, batch)
        u = self.global_model(torch.cat([acc,node_attr],dim=1), edge_index, edge_attr, u, batch)
        pos = pos + vel
        vel = vel + acc
        
        new_node_attr = torch.cat([pos,vel,node_attr_scalar],dim=1)

        return node_attr, edge_feature_scalar, u


    def edge_model(self, src, dst, edge_attr, u, batch):
        out = torch.cat([src, dst, edge_attr], 1)
        out = self.edge_mlp(out)
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        return out

    def node_model(self, node_attr_scalar, node_attr_vector,
                    edge_index, edge_attr_scalar, edge_attr_vector, 
                    u, batch):

        row, col = edge_index

        attr_scalar = torch.cat([node_attr_scalar[row], edge_attr_scalar], dim=1)
        out_scalar = self.node_mlp_1(attr_scalar)
        out_vector = out_scalar * edge_attr_vector
        out_graph = scatter(out_vector, col, dim=0, dim_size=node_attr_scalar.size(0),
                      reduce='mean')
        

        feature_global = torch.cat([node_attr_scalar, node_attr_vector, u[batch]], dim=1)
        out_global = self.node_mlp_2(feature_global)

        # Add global part and graph part
        out = out_global + out_graph
        return out

    def global_model(self, node_attr, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        out = torch.cat([
            u,
            scatter(node_attr, batch, dim=0, reduce='mean'),
        ], dim=1)
        return self.global_mlp(out)


    def vec2radial(self, edge_index, pos, vel):
        row, col = edge_index
        pos_diff = pos[row] - pos[col]
        vel_diff = vel[row] - vel[col]

        f_x = torch.sum((pos_diff)**2, 1).sqrt().unsqueeze(1)
        f_v = torch.sum((vel_diff)**2, 1).sqrt().unsqueeze(1)
        f_xv = torch.sum(pos_diff * vel_diff, 1).unsqueeze(1)

        radial = torch.cat((f_x, f_v, f_xv), dim = 1)
        return radial, pos_diff

class EdgeModel(torch.nn.Module):
    def __init__(self, node_nf, edge_nf, global_nf, hidden_nf):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(node_nf*2 + edge_nf + global_nf, hidden_nf),
            nn.ReLU(),
            nn.Linear(hidden_nf, edge_nf),
        )

    def forward(self, src, dst, edge_attr, u, batch):
        # src, dst: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: [E] with max entry B - 1.
        out = torch.cat([src, dst, edge_attr, u[batch]], 1)
        return self.edge_mlp(out)
