# !/usr/bin/env python
from __future__ import print_function  # Ensures compatibility with Python 2/3 for print function usage.
import argparse  # For parsing command-line arguments.
import os  # For file and directory operations.
import torch  # Core PyTorch library.
import torch.nn.functional as F  # Common functions for neural networks.
import torch.optim as optim  # Optimization algorithms like SGD, Adam.
from torch.utils.data import DataLoader  # For creating DataLoader for datasets.
import numpy as np  # For numerical operations.
import time  # For tracking execution time.
from tensorboardX import SummaryWriter  # For logging to TensorBoard.
from math import ceil  # For ceiling division.
from deeplesion.Dataset import MARTrainDataset  # Custom dataset loader (replace with your dataset path if different).
from network.indudonet import InDuDoNet  # Custom network (replace with your model's module if different).
import matplotlib.pyplot as plt  # For visualization (optional but included here).
from math import ceil
from deeplesion.Dataset import MARTrainDataset
from network.indudonet import InDuDoNet

if __name__ == '__main__':
    def print_network(name, net):
        num_params = 0
        for param in net.parameters():
            num_params += param.numel()
        print('name={:s}, Total number={:d}'.format(name, num_params))

    device = (
        "mps" if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else
        "cpu"
    )
    print(f"Using device: {device}")

    net = InDuDoNet(opt).to(device)
    print_network("InDuDoNet:", net)

    optimizer = optim.Adam(net.parameters(), betas=(0.5, 0.999), lr=opt.lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.milestone, gamma=0.5)

    for _ in range(opt.resume):
        scheduler.step()

    if opt.resume:
        net.load_state_dict(torch.load(os.path.join(opt.model_dir, f'InDuDoNet_{opt.resume}.pt')))
        print(f'Loaded checkpoints, epoch {opt.resume}')

    train_mask = np.load(os.path.join(opt.data_path, 'trainmask.npy'))
    train_dataset = MARTrainDataset(opt.data_path, opt.patchSize, train_mask)

    train_model(net, optimizer, scheduler, train_dataset)

def train_model(net, optimizer, scheduler, datasets):
    data_loader = DataLoader(
        datasets, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers), pin_memory=True
    )
    writer = SummaryWriter(opt.log_dir)
    step = 0

    for epoch in range(opt.resume, opt.niter):
        mse_per_epoch = 0
        tic = time.time()
        net.train()

        for ii, data in enumerate(data_loader):
            Xma, XLI, Xgt, mask, Sma, SLI, Sgt, Tr = [x.to(device) for x in data]
            optimizer.zero_grad()

            ListX, ListS, ListYS = net(Xma, XLI, mask, Sma, SLI, Tr)
            loss_l2YSmid = 0.1 * F.mse_loss(ListYS[opt.S - 2], Sgt)
            loss_l2Xmid = 0.1 * F.mse_loss(ListX[opt.S - 2] * (1 - mask), Xgt * (1 - mask))
            loss_l2YSf = F.mse_loss(ListYS[-1], Sgt)
            loss_l2Xf = F.mse_loss(ListX[-1] * (1 - mask), Xgt * (1 - mask))
            loss_l2YS = loss_l2YSf + loss_l2YSmid
            loss_l2X = loss_l2Xf + loss_l2Xmid
            loss = opt.gamma * loss_l2YS + loss_l2X
            loss.backward()
            optimizer.step()

            mse_iter = loss.item()
            mse_per_epoch += mse_iter
            writer.add_scalar('Loss', loss, step)
            step += 1

        scheduler.step()
        torch.save(net.state_dict(), os.path.join(opt.model_dir, 'InDuDoNet_latest.pt'))

    writer.close()