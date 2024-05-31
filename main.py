import argparse
import numpy as np
import torch
from model import ANGEL
from data_loader.dataset import *
from trainer import train
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="Big_Vul")
parser.add_argument("--method", type=str, default="ANGEL")
parser.add_argument("--root", type=str, default="ANGEL")
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--num_node_features', type=int, default=100)
parser.add_argument('--node_hidden_dim', type=int, default=64)
parser.add_argument('--poollist', type=list, default=[40,40,40,40,40])
parser.add_argument('--hierarchical_pool_rate', type=list, default=[0.1,0.2,0.3,0.4,0.5])
parser.add_argument('--pool_pool_type', type=str, default='topk')
parser.add_argument('--pool_act', type=str, default='relu')
parser.add_argument('--conv_type', type=str, default='gcn')
parser.add_argument('--GNNact', type=str, default='relu')
parser.add_argument('--GTnorm', type=str, default='ln')
parser.add_argument('--num_heads', type=int, default=4)
parser.add_argument('--GTdrop', type=float, default=0.5)
parser.add_argument('--attn_drop', type=float, default=0.0)
parser.add_argument('--num_layers', type=int, default=5)
parser.add_argument('--readout', type=str, default='mean')
parser.add_argument('--middle_layer_type', type=str, default='ident')
parser.add_argument('--skip_connection', type=str, default='short')
parser.add_argument('--out_layer', type=int, default=2)
parser.add_argument('--out_hidden_times', type=int, default=1)
parser.add_argument('--max_iters', type=int, default=3000)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--weight_decay', type=int, default=0.001)
parser.add_argument('--weight', type=float, default=1.0)

def main(args):
    device = 'cuda'
    dataset = PYGDataset(root=f'./data/{args.dataset}')
    train_size = int(0.7 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size)
    nodefeat = dataset[0].x.size()[-1]
    args.num_node_features = nodefeat
    model = ANGEL(args)
    model = model.to(device)
    train(args, train_loader, val_loader, test_loader, device, model)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
