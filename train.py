import os
import sys
import torch
import visdom
import argparse
import numpy as np
import time
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.optim.lr_scheduler import StepLR
from tensorboardX import SummaryWriter

from torch.autograd import Variable
from torch.utils import data
from tqdm import tqdm

from ptsemseg.models import get_model
from ptsemseg.loader import get_loader, get_data_path
from ptsemseg.metrics import runningScore
from ptsemseg.loss import *
from ptsemseg.augmentations import *

def train(args):

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    assert len(args.gpus) // args.batch_size == 3, 'Each gpu must have 3'

    # Setup Augmentations
    data_aug= Compose([RandomRotate(10),                                        
                       RandomHorizontallyFlip()])

    # Setup Dataloader
    data_loader = get_loader('semi_cityscapes')
    data_path = get_data_path(args.dataset)
    if args.subsample:
      city_names = '[a-h]*'
    else:
      city_names = '*'
    t_loader = data_loader(data_path, is_transform=True,
                           img_size=(args.img_rows, args.img_cols),
                           augmentations=data_aug, gamma_augmentation=args.gamma,
                           city_names=city_names, real_synthetic=args.real_synthetic)
    # Full val dataset
    v_loader = data_loader(data_path, is_transform=True, split='val', img_size=(args.img_rows, args.img_cols),
                           city_names='*')
    print("Training dataset size: {}".format(len(t_loader)))

    n_classes = t_loader.n_classes
    trainloader = data.DataLoader(t_loader, batch_size=args.batch_size, num_workers=8, shuffle=True)
    valloader = data.DataLoader(v_loader, batch_size=args.batch_size, num_workers=8)

    # Checkpoint
    ckpt_dir = os.path.join(args.ckpt_dir, args.name)
    os.makedirs(ckpt_dir, exist_ok=True)
    tb_path = os.path.join(ckpt_dir, 'tb')
    if os.path.exists(tb_path):
      os.system('rm -r {}'.format(tb_path))
    writer = SummaryWriter(tb_path)
    log_path = os.path.join(ckpt_dir, 'train.log')
    with open(log_path, 'w+') as f:
      args_dict = vars(args)
      for k in sorted(args_dict.keys()):
        f.write('{}: {}\n'.format(k, str(args_dict[k])))

    # Setup Metrics
    running_metrics = runningScore(n_classes)
        
    # Setup visdom for visualization
    if args.visdom:
        vis = visdom.Visdom()

        loss_window = vis.line(X=torch.zeros((1,)).cpu(),
                           Y=torch.zeros((1)).cpu(),
                           opts=dict(xlabel='minibatches',
                                     ylabel='Loss',
                                     title='Training Loss',
                                     legend=['Loss']))

    # Setup Model
    model = get_model(args.arch, n_classes)
    
    # model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    # model.cuda()
    model = torch.nn.DataParallel(model.cuda())
    
    # Check if model has custom optimizer / loss
    if hasattr(model.module, 'optimizer'):
        optimizer = model.module.optimizer
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.l_rate, momentum=0.95, weight_decay=5e-4)
    scheduler = StepLR(optimizer, step_size=args.lr_step_size, gamma=0.1)

    if hasattr(model.module, 'loss'):
        print('Using custom loss')
        loss_fn = model.module.loss
    else:
        loss_fn = cross_entropy2d

    if args.resume is not None:                                         
        if os.path.isfile(args.resume):
            print("Loading model and optimizer from checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            print("Loaded checkpoint '{}' (epoch {})"                    
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("No checkpoint found at '{}'".format(args.resume)) 

    best_iou = -100.0 
    for epoch in range(args.n_epoch):
        epoch_start_time = time.time()
        scheduler.step()
        model.train()
        for i, (images, labels) in enumerate(trainloader):
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())

            optimizer.zero_grad()
            outputs = model(images)

            loss = loss_fn(input=outputs, target=labels)

            loss.backward()
            optimizer.step()

            if args.visdom:
                vis.line(
                    X=torch.ones((1, 1)).cpu() * i,
                    Y=torch.Tensor([loss.data[0]]).unsqueeze(0).cpu(),
                    win=loss_window,
                    update='append')

            if (i+1) % 20 == 0:
                print("Epoch [%d/%d] Loss: %.4f" % (epoch+1, args.n_epoch, loss.data[0]))

        print("Epoch [{}/{}] done ({} sec)".format(epoch+1, args.n_epoch, int(time.time() - epoch_start_time)))

        model.eval()
        for i_val, (images_val, labels_val) in tqdm(enumerate(valloader)):
            images_val = Variable(images_val.cuda(), volatile=True)
            labels_val = Variable(labels_val.cuda(), volatile=True)

            outputs = model(images_val)
            pred = outputs.data.max(1)[1].cpu().numpy()
            gt = labels_val.data.cpu().numpy()
            running_metrics.update(gt, pred)

        score, class_iou = running_metrics.get_scores()
        for k, v in score.items():
            print(k, v)
        running_metrics.reset()

        mean_iou = score['Mean IoU : \t']
        writer.add_scalar('mean IoU', mean_iou, epoch)
        if mean_iou >= best_iou:
            best_iou = mean_iou
            state = {'epoch': epoch+1,
                     'model_state': model.state_dict(),
                     'optimizer_state' : optimizer.state_dict(),}
            torch.save(state, "{}/best_model.pkl".format(ckpt_dir))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', nargs='?', type=str, default='fcn8s', 
                        help='Architecture to use [\'fcn8s, unet, segnet etc\']')
    parser.add_argument('--dataset', nargs='?', type=str, default='pascal', 
                        help='Dataset to use [\'pascal, camvid, ade20k etc\']')
    parser.add_argument('--ckpt_dir', nargs='?', type=str, default='ckpt',
                        help='Checkpoint directory')
    parser.add_argument('--name', nargs='?', type=str, default='',
                        help='Name')
    parser.add_argument('--gpus', nargs='?', type=str, default='0', help='GPUs')
    parser.add_argument('--img_rows', nargs='?', type=int, default=256, 
                        help='Height of the input image')
    parser.add_argument('--img_cols', nargs='?', type=int, default=256, 
                        help='Height of the input image')
    parser.add_argument('--n_epoch', nargs='?', type=int, default=100, 
                        help='# of the epochs')
    parser.add_argument('--batch_size', nargs='?', type=int, default=1, 
                        help='Batch Size')
    parser.add_argument('--l_rate', nargs='?', type=float, default=1e-5, 
                        help='Learning Rate')
    parser.add_argument('--feature_scale', nargs='?', type=int, default=1, 
                        help='Divider for # of features to use')
    parser.add_argument('--resume', nargs='?', type=str, default=None,    
                        help='Path to previous saved model to restart from')
    parser.add_argument('--visdom', nargs='?', type=bool, default=False, 
                        help='Show visualization(s) on visdom | False by  default')
    parser.add_argument('--lr_step_size', nargs='?', type=int, default=35,
                        help='Learning rate decay step size')
    parser.add_argument('--gamma', nargs='?', type=float, default=0,
                        help='Gamma augmentation')
    parser.add_argument('--real_synthetic', nargs='?', type=str, default='real',
                        choices=['real', 'synthetic', 'real+synthetic'],
                        help='Real, synthetic, or real+synthetic')
    parser.add_argument('--subsample', nargs='?', type=int, default=0,
                        choices=[0, 1], help='Subsample training set')
    args = parser.parse_args()
    train(args)
