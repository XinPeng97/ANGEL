import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pygnn
from torch_geometric.utils import *
from torch_geometric.nn.pool.select import SelectOutput
from torch_geometric.nn.pool.connect import FilterEdges
from torch import Tensor

def get_conv(conv_type, in_dim, out_dim, args=None):
    if conv_type == 'gcn':
        return pygnn.GCNConv(in_dim, out_dim)
    else:
        raise NotImplementedError


def get_pool(pool_type, in_dim, pool_rate, args=None):
    if pool_type == 'topk':
        return pygnn.TopKPooling(in_dim, ratio=pool_rate)
    else:
        raise NotImplementedError

def get_activation(activation):
    if activation == 'relu':
        return torch.nn.ReLU()
    elif activation == 'prelu':
        return torch.nn.PReLU()
    else:
        raise NotImplementedError



class Hpool(torch.nn.Module):
    def __init__(self, in_channels, out_channels, i, args):
        super().__init__()

        self.act = get_activation(args.pool_act)
        self.drop = nn.Dropout(p=args.pool_drop)
        self.args = args
        self.poollist = args.poollist[i]
        self.pool = get_pool(args.pool_pool_type, out_channels, args.hierarchical_pool_rate[i], args)
        self.lin = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            get_activation(args.pool_act),
            nn.Linear(out_channels, out_channels)
        )
        self.appnp = pygnn.APPNP(K=8, alpha=0.2)
        
        

    def forward(self, x, edge_index, edge_attr, batch):
        
        num_nodes = scatter(batch.new_ones(x.size(0)), batch, reduce='sum') # 返回每个batch的节点数
        select_sample = num_nodes > self.poollist # size batch   
        if torch.sum(select_sample) == 0:
            return x, edge_index, edge_attr, batch, 0
        perm = torch.cat([select_sample[i].repeat(num_nodes[i]) for i in range(len(select_sample))]) #perm广播
        node_index = torch.arange(x.size()[0], device=x.device)
        select_node_index = node_index[perm]
        select_output = SelectOutput(
            node_index=select_node_index,                                
            num_nodes=x.size()[0],                                   
            cluster_index=torch.arange(len(select_node_index), device=x.device),        
            num_clusters=len(select_node_index),                       
        )
        connect = FilterEdges()
        out = connect(select_output, edge_index, edge_attr, batch)
        edge_index_pool_before = out.edge_index
        batch_pool_before = out.batch
        x_pool_before = x[perm]                      
        x_unpool = x[~perm]
        x_pool_before = self.lin(x_pool_before)
        if self.args.pool_drop != 0:
            x_pool_before = self.drop(x_pool_before)
        x_pool_before = self.appnp(x_pool_before, edge_index_pool_before)
        x_pool_after, edge_index_pool_after, edge_attr_pool_after, batch_pool_after, perm_pool_after, score_pool_after = self.pool(x_pool_before, edge_index_pool_before, None, batch_pool_before)
        num_nodes_pool_after = scatter(batch_pool_after.new_ones(x_pool_after.size(0)), batch_pool_after, reduce='sum') 
        if len(num_nodes_pool_after) != len(num_nodes):
            for i in range((len(num_nodes)-len(num_nodes_pool_after))):
                num_nodes_pool_after = torch.cat([num_nodes_pool_after, torch.tensor([0], device=x.device)], dim=0)
        cat_num_node = torch.where(num_nodes_pool_after == 0, num_nodes, num_nodes_pool_after)
        perm_cat_pool = torch.cat([select_sample[i].repeat(cat_num_node[i]) for i in range(len(select_sample))])
        perm_cat_unpool = ~perm_cat_pool
        x_num = x_pool_after.size()[0] + (~perm).sum()
    
        x_cat = torch.zeros(size=(x_num, x_pool_after.size()[1]), device=x.device)
        x_cat[perm_cat_pool] = x_pool_after
        x_cat[perm_cat_unpool] = x_unpool
        num_graph = batch.max().item() + 1
        batch_index = torch.arange(num_graph, device=x.device)
        batch_cat_pool = torch.cat([batch_index[i].repeat(cat_num_node[i]) for i in range(len(select_sample))])
        node_pool_index = node_index[perm]  
        node_unpool_index = node_index[~perm]
        subgraph_index = torch.cat([node_unpool_index, node_pool_index[perm_pool_after]])
        subgraph_index, _ = torch.sort(subgraph_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        cat_edge_index, cat_edge_attr = subgraph(subgraph_index, edge_index, edge_attr, relabel_nodes=True)
        cat_edge_index, _ = remove_self_loops(cat_edge_index)
        return x_cat, cat_edge_index, cat_edge_attr, batch_cat_pool, select_sample.unsqueeze(0)
    
class GNNLayer(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.conv=get_conv(args.conv_type, args.node_hidden_dim, args.node_hidden_dim, args)
        self.act=get_activation(args.GNNact)
        self.drop=nn.Dropout(p=args.GNNdrop)

    def forward(self, x, edge_index, edge_attr, batch, y):
        
        x = self.conv(x, edge_index)
        x = self.act(x)
        if self.args.GNNdrop != 0:
            x = self.drop(x)
        return x


class MultiHeadAttention(nn.Module): 
    def __init__(self, hidden_dim: int, num_heads: int, 
            dropout_ratio: float = 0.0): 
        super().__init__() 
        self.hidden_dim = hidden_dim 
        self.num_heads = num_heads 
        self.dropout = nn.Dropout(dropout_ratio) 
        self.linear_q = nn.Linear(hidden_dim, hidden_dim) 
        self.linear_k = nn.Linear(hidden_dim, hidden_dim) 
        self.linear_v = nn.Linear(hidden_dim, hidden_dim) 
        self.linear_attn_out = nn.Linear(hidden_dim, hidden_dim) 
    def forward(
        self,
        x_q: Tensor,
        x_k: Tensor, 
        x_v: Tensor, 
        mask: Tensor 
    ) -> Tensor: # B x N x F -> B x N x F 

        Q = self.linear_q(x_q) 
        K = self.linear_k(x_k) 
        V = self.linear_v(x_v) 

        dim_split = self.hidden_dim // self.num_heads
        Q_heads = torch.cat(Q.split(dim_split, 2), dim=0)
        K_heads = torch.cat(K.split(dim_split, 2), dim=0)
        V_heads = torch.cat(V.split(dim_split, 2), dim=0)
        
        attention_score = Q_heads.bmm(K_heads.transpose(1, 2)) 
        attention_score = attention_score / math.sqrt(self.hidden_dim // self.num_heads) 

        inf_mask = (~mask).unsqueeze(1).to(dtype=torch.float) * -1e9
        inf_mask = torch.cat([inf_mask for _ in range(self.num_heads)], 0) 
        A = torch.softmax(attention_score + inf_mask, -1) 
        
        A = self.dropout(A) 
        out = torch.cat((A.bmm(V_heads)).split(Q.size(0), 0), 2) 
        out = self.linear_attn_out(out) 
        
        return out


class NodeSelfAttention(MultiHeadAttention): 

    def __init__(self, hidden_dim: int, num_heads: int, 
            dropout_ratio: float = 0.):
        super().__init__(hidden_dim, num_heads, 
                            dropout_ratio) 

    def forward(
        self,
        x: Tensor, 
        mask: Tensor 
    ) -> Tensor: 

        return super().forward(x, x, x, mask) 
    
class GraphTransLayer(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.hidden_dim = args.node_hidden_dim
        self.num_heads = args.num_heads

        if args.GTnorm == 'ln':
            norm_class = nn.LayerNorm 
        elif args.GTnorm == 'bn': 
            norm_class = nn.BatchNorm1d 

        self.dropout1 = nn.Dropout(args.GTdrop) 
        self.dropout2 = nn.Dropout(args.GTdrop) 
        self.dropout3 = nn.Dropout(args.GTdrop) 
        
        self.node_self_attention = NodeSelfAttention( 
                                    args.node_hidden_dim, args.num_heads, 
                                    args.attn_drop)
        ffn_hidden_times = 2
        self.linear_out1 = nn.Linear(args.node_hidden_dim, ffn_hidden_times * args.node_hidden_dim) 
        self.linear_out2 = nn.Linear(ffn_hidden_times * args.node_hidden_dim, args.node_hidden_dim) 

        self.norm0 = norm_class(args.node_hidden_dim) 
        self.norm1 = norm_class(args.node_hidden_dim) 

        
    def forward(self, graph): 

        # Sparse -> Dense 
        x, batch, batch_CLS = graph
        x_dense, mask = to_dense_batch(x, batch) 
        attention_mask = mask

        if batch_CLS is not None: 
            x_dense = torch.cat([x_dense, batch_CLS], dim=1) 
            CLS_mask = torch.full((batch_CLS.shape[0], 1), True, device=batch_CLS.device) 
            attention_mask = torch.cat([mask, CLS_mask], dim=1) 

        attention_out = self.node_self_attention(x_dense, attention_mask) 
        attention_out = self.dropout1(attention_out) 
        attention_out = attention_out + x_dense 
        if batch_CLS is not None: 
            batch_CLS = attention_out[:, -1, :] 
            attention_out = attention_out[:, :-1, :] 
            attention_out = attention_out[mask] 
            attention_out = torch.cat((attention_out, batch_CLS), dim=0) 
        else: 
            attention_out = attention_out[mask] 

        attention_out = self.norm0(attention_out) 
        
        out = self.linear_out1(attention_out) 
        out = torch.relu(out) 
        out = self.dropout2(out) 
        out = self.linear_out2(out) 
        out = self.dropout3(out) 

        out = out + attention_out 
        out = self.norm1(out) 
        return out 


class ANGEL(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        in_channels, hidden_channels = args.num_node_features, args.node_hidden_dim
        
        self.args = args
        self.lin = nn.GRU(input_size=in_channels, hidden_size=hidden_channels, num_layers = 2, batch_first=True)

        self.hppool_list = nn.ModuleList()
        for i in range(len(args.poollist)):
            self.hppool_list.append(Hpool(hidden_channels, hidden_channels, i, args))

        self.GNNLayers = nn.ModuleList()
        self.GraphTransLayers = nn.ModuleList()
        self.middle_layers = nn.ModuleList()
        self.GNNdrops = nn.ModuleList()
        for i in range(args.num_layers):
            self.GNNLayers.append(GNNLayer(args))
            self.GNNdrops.append(nn.Dropout(args.GNNdrop))
            self.GraphTransLayers.append(GraphTransLayer(args))
            if self.args.middle_layer_type == 'bn':
                self.middle_layers.append(nn.BatchNorm1d(hidden_channels))
            if self.args.middle_layer_type == 'ln':
                self.middle_layers.append(nn.LayerNorm(hidden_channels)) 
            if self.args.middle_layer_type == 'ident':
                self.middle_layers.append(nn.Identity(hidden_channels)) 

        predict_head_modules = [] 
        last_layer_dim = hidden_channels
        for i in range(args.out_layer-1): 
            predict_head_modules.append(nn.Linear(last_layer_dim, args.out_hidden_times * hidden_channels)) 
            predict_head_modules.append(nn.ReLU()) 
            last_layer_dim = args.out_hidden_times * hidden_channels
        predict_head_modules.append(nn.Linear(last_layer_dim, 2)) 
        self.cls = nn.Sequential(*predict_head_modules) 

    def forward(self, x, edge_index, edge_attr, batch, y):
        edge_attr = None
        num_graph = batch.max().item() + 1
        x = torch.unsqueeze(x, dim=1)
        x, _ = self.lin(x)
        x = torch.flatten(x, 1)
        select_sample_list = []
        for i in range(len(self.args.poollist)):
            x, edge_index, edge_attr, batch, select_sample = self.hppool_list[i](x, edge_index, edge_attr, batch)
            select_sample_list.append(select_sample)
        if self.args.readout == 'cls': 
            batch_cls = self.CLS.expand(num_graph, 1, -1) 
        else: 
            batch_cls = None 
        if self.args.middle_layer_type != 'none': 
            out = x
        else: 
            out = 0

        for i in range(self.args.num_layers):
            x = self.GNNLayers[i](x, edge_index, edge_attr, batch, y)
            x = self.GNNdrops[i](x)
            if self.args.middle_layer_type != 'none': 
                x = out + x 
                x = self.middle_layers[i](x) 
                out = x 
            graph = (x, batch, batch_cls)
            x = self.GraphTransLayers[i](graph)
            if self.args.skip_connection == 'none': 
                out = x
            elif self.args.skip_connection == 'short': 
                out = out + x 
                x = out
        x = pygnn.pool.global_mean_pool(x, batch)
        out = self.cls(x) 
        return out
    




