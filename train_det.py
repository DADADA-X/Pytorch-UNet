import sys
import os
import argparse
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim

from eval import eval_net
from unet import UNet
from det_unet import DetNet
from utils import get_ids, split_ids, split_train_val, get_imgs_and_masks, batch

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
def train_net(net,
              epochs=5,
              batch_size=1,
              lr=1e-3,
              val_percent=0.05,
              save_cp=True,
              gpu=False,
              img_scale=0.5):

    dir_img = '/home/xyj/data/spacenet/vegas/train/images/'
    dir_mask = '/home/xyj/data/spacenet/vegas/train/centerlines/'
    dir_checkpoint = 'checkpoints_cen/'
    
    if not os.path.exists(dir_checkpoint):
        os.mkdir(dir_checkpoint)

    ids = get_ids(dir_img)  # 返回train文件夹下文件的名字列表，生成器（except last 4 character，.jpg这样的）
    ids = split_ids(ids)    # 返回(id, i), id属于ids，i属于range(n)，相当于把train的图✖️了n倍多张，是tuple的生成器

    iddataset = split_train_val(ids, val_percent)   # validation percentage，是dict = {"train": ___(一个list), "val"：___(一个list)}

    print('''
    Starting training:
        Epochs: {}
        Batch size: {}
        Learning rate: {}
        Training size: {}
        Validation size: {}
        Checkpoints: {}
        CUDA: {}
    '''.format(epochs, batch_size, lr, len(iddataset['train']),
               len(iddataset['val']), str(save_cp), str(gpu)))

    N_train = len(iddataset['train'])

#     optimizer = optim.SGD(net.parameters(),
#                           lr=lr,
#                           momentum=0.9,
#                           weight_decay=0.0005)
    optimizer = optim.Adam(net.parameters(),
                         lr=lr,
                         betas=(0.9, 0.999),
                         eps=1e-3)
#     scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=40,gamma = 0.3)
#     weight = torch.FloatTensor([10, 1])
#     criterion = nn.BCEWithLogitsLoss(pos_weight=weight)
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
        net.train()

        # reset the generators
        train = get_imgs_and_masks(iddataset['train'], dir_img, dir_mask, img_scale)
        val = get_imgs_and_masks(iddataset['val'], dir_img, dir_mask, img_scale)
        epoch_loss = 0

        for i, b in enumerate(batch(train, batch_size)):
            imgs = np.array([i[0] for i in b]).astype(np.float32)
            true_masks = np.array([i[1]//200 for i in b])

            imgs = torch.from_numpy(imgs)
            true_masks = torch.from_numpy(true_masks)

            if gpu:
                imgs = imgs.cuda()
                true_masks = true_masks.cuda()

            masks_pred = net(imgs)
            masks_probs_flat = masks_pred.view(-1)

            true_masks_flat = true_masks.view(-1)

            loss = criterion(masks_probs_flat, true_masks_flat)
            epoch_loss += loss.item()

            print('{0:.4f} --- loss: {1:.6f}'.format(i * batch_size / N_train, loss.item()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
#             scheduler.step()

        print('Epoch finished ! Loss: {}'.format(epoch_loss / i))

        if 1:
            val_dice = eval_net(net, val, gpu)
            print('Validation Dice Coeff: {}'.format(val_dice))

        if save_cp:
            torch.save(net.state_dict(),
                       dir_checkpoint + 'CP{}.pth'.format(epoch + 1))
            print('Checkpoint {} saved !'.format(epoch + 1))



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', dest='epochs', default=5, type=int,
                        help='number of epochs')
    parser.add_argument('-b', '--batch-size', dest='batchsize', default=3,
                        type=int, help='batch size')
    parser.add_argument('-l', '--learning-rate', dest='lr', default=0.001,
                        type=float, help='learning rate')
    parser.add_argument('-g', '--gpu', action='store_true', dest='gpu',
                        default=False, help='use cuda')
    parser.add_argument('-c', '--load', dest='load',
                        default=False, help='load file model')
    parser.add_argument('-s', '--scale', dest='scale', type=float,
                        default=0.5, help='downscaling factor of the images')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    # net = UNet(n_channels=3, n_classes=1)
    net = DetNet()

    if args.load:
        net.load_state_dict(torch.load(args.load))
        print('Model loaded from {}'.format(args.load))

    if args.gpu:
        net.cuda()
        # cudnn.benchmark = True # faster convolutions, but more memory

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  gpu=args.gpu,
                  img_scale=args.scale)
    except KeyboardInterrupt:           # 异常处理👍，出错的话就会save下来
        torch.save(net.state_dict(), dir_checkpoint + 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
