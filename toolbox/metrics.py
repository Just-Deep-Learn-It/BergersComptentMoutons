
'''
Code snippets for keeping track of evaluation metrics
'''

import numpy as np
import json


'''
                                .
                              .o8
ooo. .oo.  .oo.    .ooooo.  .o888oo  .ooooo.  oooo d8b  .oooo.o
`888P"Y88bP"Y88b  d88' `88b   888   d88' `88b `888""8P d88(  "8
 888   888   888  888ooo888   888   888ooo888  888     `"Y88b.
 888   888   888  888    .o   888 . 888    .o  888     o.  )88b
o888o o888o o888o `Y8bod8P'   "888" `Y8bod8P' d888b    8""888P'
'''

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

    def value(self):
        return self.avg

class SumMeter(object):
    """Computes and stores the sum and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    def value(self):
        return self.sum


class ValueMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0

    def update(self, val):
        self.val = val

    def value(self):
        return self.val





def make_meters():
    meters_dict = {
        'loss': AverageMeter(),
        'squared_mse': AverageMeter(),
        'mae':AverageMeter()
        'acc_class': ValueMeter(),
        'fwavacc': ValueMeter(),
        'batch_time': AverageMeter(),
        'data_time': AverageMeter(),
        'epoch_time': SumMeter(),
    }
    return meters_dict



def save_meters(meters, fn, epoch=0):

    logged = {}
    for name, meter in meters.items():
        logged[name] = meter.value()

    if epoch > 0:
        logged['epoch'] = epoch

    print(f'Saving meters to {fn}')
    with open(fn, 'w') as f:
        json.dump(logged, f)


'''
 .oooo.o  .ooooo.   .ooooo.  oooo d8b  .ooooo.   .oooo.o
d88(  "8 d88' `"Y8 d88' `88b `888""8P d88' `88b d88(  "8
`"Y88b.  888       888   888  888     888ooo888 `"Y88b.
o.  )88b 888   .o8 888   888  888     888    .o o.  )88b
8""888P' `Y8bod8P' `Y8bod8P' d888b    `Y8bod8P' 8""888P'
'''

def evaluate(hist):
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)+1e-10)
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / (hist.sum()+ 1e-10)
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc


def accuracy_regression(output, target):
    mae_mean = (output - target).abs().mean()
    mse_mean = (output - target).pow(2).mean()
    rmse = (output - target).pow(2).mean().sqrt()
    return mae_mean.item(), mse_mean.item(), rmse_mean.item()


def fast_hist(pred, label, n):
    k = (label >= 0) & (label < n)
    return np.bincount(
        n * label[k].astype(int) + pred[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist)+ 1e-10)


