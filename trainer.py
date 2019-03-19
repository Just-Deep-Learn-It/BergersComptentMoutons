import time
import os
import numpy as np
import torch
from PIL import Image
import imageio

from toolbox import utils, metrics

'''
    .                       o8o
  .o8                       `"'
.o888oo oooo d8b  .oooo.   oooo  ooo. .oo.
  888   `888""8P `P  )88b  `888  `888P"Y88b
  888    888      .oP"888   888   888   888
  888 .  888     d8(  888   888   888   888
  "888" d888b    `Y888""8o o888o o888o o888o
'''

def train(args, train_loader, model, criterion, optimizer, logger, epoch,
          eval_score=None, print_freq=10, tb_writer=None):
    
    # switch to train mode
    model.train()
    meters = logger.reset_meters('train')
    meters_params = logger.reset_meters('hyperparams')
    meters_params['learning_rate'].update(optimizer.param_groups[0]['lr'])
    end = time.time()

    for i, (input, target) in enumerate(train_loader):
        # print(f'{i} - {input.size()} - {target.size()}')
        batch_size = input.size(0)
        #count=target[0]
        target=target[1]
        # measure data loading time
        meters['data_time'].update(time.time() - end, n=batch_size)
        #print("DEBUGG=",target)
        input, target = input.to(args.device).requires_grad_(), target.to(args.device)
        output = model(input)

        loss = criterion(output, target)
        
        meters['loss'].update(loss.data.item(), n=batch_size)
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        if eval_score is not None:
            mae, squared_mse, count = eval_score(output, target)
            meters['mae'].update(mae, n=batch_size)
            meters['squared_mse'].update(squared_mse,n=batch_size)


        # measure elapsed time
        meters['batch_time'].update(time.time() - end, n=batch_size)
        end = time.time()

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'LR {lr.val:.2e}\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'mae {mae.val:.3f} ({mae.avg:.3f})\t'
                  'squared_mse {squared_mse.val:.3f} ({squared_mse.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=meters['batch_time'],
                   data_time=meters['data_time'], lr=meters_params['learning_rate'], loss=meters['loss'], mae=meters['mae'], squared_mse=meters['squared_mse']))


        if True == args.short_run:
            if 12 == i:
                print(' --- running in short-run mode: leaving epoch earlier ---')
                break    

   
    if args.tensorboard:
        tb_writer.add_scalar('mae/train', meters['mae'].avg, epoch)
        tb_writer.add_scalar('squared_mse/train', meters['squared_mse'].avg, epoch)
        tb_writer.add_scalar('loss/train', meters['loss'].avg, epoch)
        tb_writer.add_scalar('learning rate', meters_params['learning_rate'].val, epoch)
       
    logger.log_meters('train', n=epoch)
    logger.log_meters('hyperparams', n=epoch)


   
'''
                      oooo
                      `888
oooo    ooo  .oooo.    888
 `88.  .8'  `P  )88b   888
  `88..8'    .oP"888   888
   `888'    d8(  888   888
    `8'     `Y888""8o o888o
'''

def validate(args, val_loader, model, criterion, logger, epoch, eval_score=None, print_freq=10, tb_writer=None):

    # switch to evaluate mode
    model.eval()
    meters = logger.reset_meters('val')
    end = time.time()
    grid_pred = None
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            batch_size = input.size(0)

            meters['data_time'].update(time.time()-end, n=batch_size)
        
            input, target = input.to(args.device).requires_grad_(), target.to(args.device)
            
            output = model(input)

            loss = criterion(output, target)
            meters['loss'].update(loss.data.item(), n=batch_size)

            # measure accuracy and record loss
            if eval_score is not None:
                mae, squared_mse, count = eval_score(output, target)
                meters['mae'].update(mae, n=batch_size)
                meters['squared_mse'].update(squared_mse,n=batch_size)



            # measure elapsed time
            end = time.time()
            meters['batch_time'].update(time.time() - end, n=batch_size)

          
            if i % print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'LR {lr.val:.2e}\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'MAE {mae.val:.3f} ({mae.avg:.3f})\t'
                      'Squared MSE {squared_mse.val:.3f} ({squared_mse.avg:.3f})'.format(
                       epoch, i, len(train_loader), batch_time=meters['batch_time'],
                       data_time=meters['data_time'], lr=meters_params['learning_rate'], loss=meters['loss'], mae=meters['mae'], squared_mse=meters['squared_mse']))


            if True == args.short_run:
                if 12 == i:
                    print(' --- running in short-run mode: leaving epoch earlier ---')
                    break    



    print(' * Validation set: Average loss {:.4f}, MAE {:.3f}%, Squared MSE {:.3f}% \n'.format(meters['loss'].avg, meters['mae'].avg, meters['squared_mse'].avg))

    logger.log_meters('val', n=epoch)
        
    if args.tensorboard:
        tb_writer.add_scalar('mae/val', meters['mae'].avg, epoch)
        tb_writer.add_scalar('squared_mse/val', meters['squared_mse'].avg, epoch)
        tb_writer.add_scalar('loss/val', meters['loss'].avg, epoch)
    return meters['mae'].val, meters['squared_mse'].val, meters['loss'].avg


'''
    .                          .
  .o8                        .o8
888oo  .ooooo.   .oooo.o .o888oo
  888   d88' `88b d88(  "8   888
  888   888ooo888 `"Y88b.    888
  888 . 888    .o o.  )88b   888 .
  "888" `Y8bod8P' 8""888P'   "888"
'''

def test(args, eval_data_loader, model, criterion, epoch, eval_score=None,
         output_dir='pred', has_gt=True, print_freq=10):

    model.eval()
    meters = metrics.make_meters()
    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(eval_data_loader):
            # print(input.size())
            batch_size = input.size(0)
            meters['data_time'].update(time.time()-end, n=batch_size)
           
            input, target = input.to(args.device).requires_grad_(), target.to(args.device)
            
            output = model(input)

            loss = criterion(output, target)
            
            meters['loss'].update(loss.data.item(), n=batch_size)

            # measure accuracy and record loss
            if eval_score is not None:
                mae, squared_mse, count = eval_score(output, target)
                meters['mae'].update(mae, n=batch_size)
                meters['squared_mse'].update(squared_mse, n=batch_size)

            end = time.time()
            meters['batch_time'].update(time.time() - end, n=batch_size)

            end = time.time()
            print('Testing: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'MAE {mae.val:.3f} ({mae.avg:.3f})\t'
                      'Squared MSE {squared_mse.val:.3f} ({squared_mse.avg:.3f})'.format(
                      i, len(eval_data_loader), batch_time=meters['batch_time'], loss=meters['loss'],
                      mae=meters['mae'],squared_mse=meters['squared_mse']), flush=True)

            if True == args.short_run:
                if 12 == i:
                    print(' --- running in short-run mode: leaving epoch earlier ---')
                    break    


        print(' * Test set: Average loss {:.4f}, MAE {:.3f}%, Squared MSE {:.3f}% \n'.format(meters['loss'].avg, meters['mae'].avg, meters['squared_mse'].avg))

    metrics.save_meters(meters, os.path.join(args.log_dir, 'test_results_ep{}.json'.format(epoch)), epoch)    
    utils.save_res_list(res_list, os.path.join(args.res_dir, 'test_results_list_ep{}.json'.format(epoch)))    


