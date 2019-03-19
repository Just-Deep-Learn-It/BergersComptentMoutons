'''
Custom loss functions
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_criterion(args):
    
    return {
        'crossentropy': nn.CrossEntropyLoss(ignore_index=255),
        'nll': nn.NLLLoss(ignore_index=255),
        'bce': nn.BCELoss(),
        'mse': nn.MSELoss(reduction='mean'), 
        'l1': nn.L1Loss(reduction='mean'), 
    }[args.criterion]

        

''' 
 oooo
 `888
  888   .ooooo.   .oooo.o  .oooo.o  .ooooo.   .oooo.o
  888  d88' `88b d88(  "8 d88(  "8 d88' `88b d88(  "8
  888  888   888 `"Y88b.  `"Y88b.  888ooo888 `"Y88b.
  888  888   888 o.  )88b o.  )88b 888    .o o.  )88b
 o888o `Y8bod8P' 8""888P' 8""888P' `Y8bod8P' 8""888P'
''' 

