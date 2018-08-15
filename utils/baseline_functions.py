import os
from sklearn.datasets import fetch_mldata
import random
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torch.nn as nn
import torch.nn.parallel
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2
import numpy as np
import torchvision.models as models
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import time

from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

def get_loader(tensor_list, label):
    X_tensor = torch.cat(tensor_list)
    X_label = torch.LongTensor(label)
    tensor_dataset = torch.utils.data.TensorDataset(X_tensor,X_label)
    tensor_loader = torch.utils.data.DataLoader(tensor_dataset, shuffle=True, batch_size=1)
    return tensor_loader

def get_tensor_list(image_list):
    image_list = [torchvision.transforms.Scale(224)(n) for n in image_list]
    image_list = [torchvision.transforms.CenterCrop(224)(n) for n in image_list]
    tensor_list = [torchvision.transforms.ToTensor()(n) for n in image_list]
    tensor_list = [torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])(n).view(-1,3,224,224) for n in tensor_list]
    return tensor_list

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train(train_loader, model, criterion, optimizer, epoch):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        target_var = Variable(target.cuda(async=True))
        input_var = Variable(input.cuda())

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target_var, topk=(1,))[0]
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
       
        if i%29 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      loss=losses, top1=top1))
     
    # log to TensorBoard

def validate(val_loader, model, criterion, epoch):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    true_label = []
    predict = []
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target_var = Variable(target.cuda(async=True))#label
        input_var = Variable(input.cuda())

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)
        true_label.extend([target_var.data.cpu().numpy()[n] for n in range(0, target_var.data.cpu().numpy().shape[0])])
        predict.extend([np.argmax(output.data.cpu().numpy()[n, :]) for n in range(0, output.data.cpu().numpy().shape[0])])
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target_var, topk=(1,))[0]
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i%100 == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      i, len(val_loader), batch_time=batch_time, loss=losses,
                      top1=top1))
    print('f1_score:', f1_score(true_label, predict, average='weighted'),
      'recall:', recall_score(true_label, predict, average='weighted'),
      'precision:', precision_score(true_label, predict, average='weighted'))

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
    # log to TensorBoard
  #  if args.tensorboard:
  #      log_value('val_loss', losses.avg, epoch)
  #      log_value('val_acc', top1.avg, epoch)
    return top1.avg

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.data.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class resnet(nn.Module):
    def __init__(self,num_classes=2):
        super(resnet, self).__init__()
        self.model = models.resnet50(pretrained=True)
        self.fc = nn.Linear(1000, num_classes,bias=False)
    def forward(self, x):
        out = self.fc(self.model(x))
        return out
