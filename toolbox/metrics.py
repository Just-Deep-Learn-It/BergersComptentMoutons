
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
        'mae':AverageMeter(),
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



def accuracy_regression(output, target):
    mae = (output - target).sum((1,2,3)).abs().mean() 
    squared_mse = (output - target).sum((1,2,3)).pow(2).mean()
    #mse = (output - target).sum().pow(2).mean().sqrt()
    count = output.sum()
    return mae.item(), squared_mse.item(), count.item()

