import sys
import argparse

import torch


def parse_args():
    parser = argparse.ArgumentParser(description='')

    #  experiment settings
    # name of the experiment
    parser.add_argument('--name', default='csrnet', type=str,
                        help='name of experiment')
    # name of dataset used in the experiment, e.g. gtsrd
    parser.add_argument('--dataset', default='shanghaitech', type=str,
                        help='name of dataset to train upon')
    parser.add_argument('--test', dest='test', action='store_true', default=False,
                        help='To only run inference on test set')

    # main folder for data storage
    parser.add_argument('--root-dir', type=str, default='/home/aymen/Desktop/EA/ShanghaiTech/part_A/')
   
    # model settings
    parser.add_argument('--arch', type=str, default='csrnet',
                        help='type of architecture to be used, e.g. csrnet')
    parser.add_argument('--model-name', type=str, default='csrnet',
                        help='type of model to be used. Particular instance of a given architecture, e.g. csrnet')    
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='which checkpoint to resume from. possible values["latest", "best", epoch]')
    parser.add_argument('--pretrained', action='store_true',
                        default=False, help='use pre-trained model')

    # data settings
    # number of workers for the dataloader
    parser.add_argument('-j', '--workers', type=int, default=4)

    # training settings
    parser.add_argument('--start-epoch', type=int, default=1)
    parser.add_argument('--step', type=int, default=20, help='frequency of updating learning rate, given in epochs')
    parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                        help='input batch size for training (default: 4)')
    parser.add_argument('--epochs', type=int, default=70, metavar='N',
                        help='number of epochs to train (default: 70)')
    parser.add_argument('--optimizer', default='adam', type=str,
                        help='name of the optimizer')
    parser.add_argument('--scheduler', default='StepLR', type=str,
                        help='name of the learning rate scheduler')
    parser.add_argument('--lr', type=float, default=1e-6, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='sgd momentum (default: 0.9)')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--lr_decay', default=0.995, type=float,
                        metavar='lrd', help='learning rate decay (default: 0.995)')
    parser.add_argument('--criterion', default='mse', type=str,
                        help='criterion to optimize')
    # misc settings
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--short_run', action='store_true',
                        default=False, help='running only over few mini-batches for debugging purposes')
    parser.add_argument('--disable_cuda', action='store_true', default=False,
                        help='disables CUDA training / using only CPU')
    parser.add_argument('--tensorboard', dest='tensorboard', action='store_true',default=False,
                        help='Use tensorboard to track and plot')

    args = parser.parse_args()

    # update args
    args.data_dir = args.root_dir #'{}/{}'.format(args.root_dir, args.dataset) #'/home/aymen/Desktop/EA/ShanghaiTech/part_A/'
    args.log_dir = '{}/runs/{}/'.format(args.data_dir, args.name)
    args.res_dir = '%s/runs/%s/res' % (args.data_dir, args.name)
    args.out_pred_dir = '%s/runs/%s/pred' % (args.data_dir, args.name)
    
    args.cuda = not args.disable_cuda and torch.cuda.is_available()
    args.device = 'cuda' if args.cuda else 'cpu'

    assert args.data_dir is not None

    print(' '.join(sys.argv))
    print(args)

    return args
