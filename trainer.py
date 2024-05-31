import datetime
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import torch.distributed as dist
import random
import nni

def ce_loss(logit, label):
    weight = torch.FloatTensor([1, 1.2]).to(logit.device)
    f = torch.nn.CrossEntropyLoss(weight=weight)
    loss = f(logit, label)
    return loss

def evaluate(dataloader, device, model):
    all_predictions, all_targets = [], []
    with torch.no_grad():
        model.eval()
        for batch, data in enumerate(dataloader):
            data = data.to(device)
            logits = model(data.x, data.edge_index, data.edge_attr, data.batch, data.y)
            predictions = logits.detach().cpu()
            all_predictions.extend(
                np.argmax(predictions.numpy(), axis=-1).tolist())
            all_targets.extend(data.y.detach().cpu().numpy().tolist())

    model.train()
    return accuracy_score(all_targets, all_predictions) * 100, \
        precision_score(all_targets, all_predictions) * 100, \
        recall_score(all_targets, all_predictions) * 100, \
        f1_score(all_targets, all_predictions) * 100



def train(args, train_loader, val_loader, test_loader, device, model):

    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)

    if args.scheduler == 'cosine': 
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    elif args.scheduler == 'linear': 
        scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=
                                                1000)
    elif args.scheduler == 'none': 
        scheduler = None 

    train_loader_iter = iter(train_loader)
    total_loss = 0
    epochs = 0
    for i in range(1, args.max_iters+1):
        model.train()
        try:
            data = next(train_loader_iter)
        except StopIteration:
            train_loader_iter = iter(train_loader)
            epochs += 1
            data = next(train_loader_iter)
        data = data.detach()
        data = data.to(device)
        loss = model(data.x, data.edge_index, data.edge_attr, data.batch, data.y)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        if args.scheduler != 'none':
            scheduler.step()

        total_loss += loss.item()
        train_loss = total_loss / (args.eval_step + 1)
        val_acc, val_pr, val_rc, val_f1 = evaluate(val_loader, device, model)
        test_acc, test_pr, test_rc, test_f1 = evaluate(test_loader, device, model)




