from __future__ import print_function
import sys
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import dataset
import random
import math
import os
from model import YoLoNet
from yolo_loss import *
from config import Config
from utils import *

#Parameters
use_cuda = torch.cuda.is_available()
eps = 1e-9

def adjust_learning_rate(optimizer, batch, steps, scales, lr):
    for i in range(len(steps)):
        scale = scales[i] if i < len(scales) else 1
        if batch >= steps[i]:
            lr = lr * scale
            if batch == steps[i]:
                break
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr / batch_size
    return lr

def train_epoch(epoch, train_loader, yolo_config, device, writer=None):
    global processed_batches
    t0 = time.time()
    if yolo_config.solver == 'sgd':
        learning_rate = adjust_learning_rate(optimizer, processed_batches, yolo_config.steps, yolo_config.scales, yolo_config.learning_rate)
        logging('epoch %d, processed %d samples, lr %f' % (epoch, epoch * len(train_loader.dataset), learning_rate))
    elif yolo_config.solver == 'adam':
        logging('epoch %d, processed %d samples, lr %f' % (epoch, epoch * len(train_loader.dataset), yolo_config.learning_rate))

    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        if yolo_config.solver == 'sgd':
            adjust_learning_rate(optimizer, processed_batches, yolo_config.steps, yolo_config.scales, yolo_config.learning_rate)
        processed_batches = processed_batches + 1

        data = data.to(device)
        optimizer.zero_grad()

        output1, output2, output3 = model(data)
        Outputs = [output1, output2, output3]

        loss, nGT, nCorrect, nProposals = YoloLoss(Outputs, target, yolo_config.num_classes, anchors, 2,
                                                   processed_batches * batch_size, device)

        total_loss = loss[0] + loss[1] + loss[2]

        if yolo_config.tensorboard:
            writer.add_scalars('PR_1', {'recall': float(nCorrect[0] / (nGT + eps)), 'precision': float(nCorrect[0] / (nProposals[0] + eps))}, processed_batches)
            writer.add_scalars('PR_2', {'recall': float(nCorrect[1] / (nGT + eps)), 'precision': float(nCorrect[1] / (nProposals[1] + eps))}, processed_batches)
            writer.add_scalars('PR_3', {'recall': float(nCorrect[2] / (nGT + eps)), 'precision': float(nCorrect[2] / (nProposals[2] + eps))}, processed_batches)
            writer.add_scalar('tra_loss_1', loss[0].item(), processed_batches)
            writer.add_scalar('tra_loss_2', loss[1].item(), processed_batches)
            writer.add_scalar('tra_loss_3', loss[2].item(), processed_batches)
            writer.add_scalar('total_tra_loss', total_loss.item(), processed_batches)

        total_loss.backward()
        optimizer.step()

    print('')
    t1 = time.time()
    logging('training with %f samples/s' % (len(train_loader.dataset) / (t1 - t0)))
    if (epoch + 1) % yolo_config.save_interval == 0:
        logging('save weights to %s/%06d.pkl' % (yolo_config.backupDir, epoch + 1))
        save_model_filename = '%s/%06d.pkl' % (yolo_config.backupDir, epoch + 1)
        model.seen = (epoch + 1) * len(train_loader.dataset)
        model.save_weights(save_model_filename)

if __name__ == '__main__':

    # Training settings
    yolo_config = Config()

    batch_size = yolo_config.batch_size
    if not os.path.exists(yolo_config.backupDir):
        os.mkdir(yolo_config.backupDir)
    kwargs = {'num_workers': yolo_config.num_workers, 'pin_memory': True} if use_cuda else {}
    device = torch.device("cuda:%s" % yolo_config.gpus if use_cuda else "cpu")

    model = YoLoNet(yolo_config.num_classes, yolo_config.in_channels)
    model.width = yolo_config.init_width
    model.height = yolo_config.init_height
    model = model.to(device)

    writer = None
    if yolo_config.tensorboard:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(yolo_config.logsDir)

    if yolo_config.solver == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=yolo_config.learning_rate/batch_size, momentum=yolo_config.momentum, weight_decay=yolo_config.decay*batch_size)
    elif yolo_config.solver == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=yolo_config.learning_rate, betas=yolo_config.betas, weight_decay=yolo_config.decay, amsgrad=True)
    else:
        print('No %s solver! Please check your config file!' % (yolo_config.solver))
        exit()
    nsamples = file_lines(yolo_config.trainList)
    ################################
    #         load weights         #
    ################################
    if yolo_config.weightFile != 'none':
        model.load_weights(yolo_config.weightFile)
    model.seen = 0
    processed_batches = model.seen / batch_size
    init_epoch = int(model.seen / nsamples)

    train_loader = DataLoader(
        dataset.listDataset(yolo_config.trainList, shape=(model.width, model.height),
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                            ]),
                            train=True,
                            imgdirpath=yolo_config.imgDirPath,
                            labdirpath=yolo_config.labelDirPath),
        batch_size=yolo_config.batch_size, shuffle=True, **kwargs)
    anchors = yolo_config.get_anchors()
    for epoch in range(init_epoch, yolo_config.max_epochs):
        train_epoch(epoch, train_loader, yolo_config, device, writer)

    if yolo_config.tensorboard:
        writer.close()

    print('Done!')