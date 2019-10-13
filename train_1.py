import sys
import os
from optparse import OptionParser
import numpy as np
from PIL import Image

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim

from eval import eval_net
from unet import UNet
from utils import get_ids, split_ids, split_train_val, get_imgs_and_masks, batch


def train_net(net,
              epochs=5,
              batch_size=1,
              lr=0.1,
              val_percent=0.05,
              save_cp=True,
              gpu=False,
              img_scale=0.5):
    # dir_img = 'data/train/'
    # dir_mask = 'data/train_masks/'
    # dir_checkpoint = 'checkpoints/'
    #
    # ids = get_ids(dir_img)
    # ids = split_ids(ids)
    #
    # iddataset = split_train_val(ids, val_percent)

    # print('''
    # Starting training:
    #     Epochs: {}
    #     Batch size: {}
    #     Learning rate: {}
    #     Training size: {}
    #     Validation size: {}
    #     Checkpoints: {}
    #     CUDA: {}
    # '''.format(epochs, batch_size, lr, len(iddataset['train']),
    #            len(iddataset['val']), str(save_cp), str(gpu)))
    #
    # N_train = len(iddataset['train'])

    optimizer = optim.SGD(net.parameters(),
                          lr=lr,
                          momentum=0.9,
                          weight_decay=0.0005)

    criterion = nn.BCELoss()  # 交叉熵损失，用于二分类，前面要加上 Sigmoid 函数

    # for epoch in range(epochs):
    #     print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
    net.train()

    # reset the generators
    # train = get_imgs_and_masks(iddataset['train'], dir_img, dir_mask, img_scale)
    # val = get_imgs_and_masks(iddataset['val'], dir_img, dir_mask, img_scale)

    imgs = np.zeros((20, 3, 50, 50), dtype=np.float32)
    # img = Image.fromarray(img)
    masks = np.zeros((20, 50, 50), dtype=np.float32)
    # mask = Image.fromarray(mask)
    train = zip(imgs, masks)
    val = zip(imgs, masks)

    epoch_loss = 0

    for i, b in enumerate(batch(train, batch_size)):
        imgs = np.array([i[0] for i in b]).astype(np.float32)
        true_masks = np.array([i[1] for i in b])

        imgs = torch.from_numpy(imgs)
        true_masks = torch.from_numpy(true_masks)

        if gpu:
            imgs = imgs.cuda()
            true_masks = true_masks.cuda()

        masks_pred = net(imgs)
        masks_probs_flat = masks_pred.view(-1)

        true_masks_flat = true_masks.view(-1).float()

        loss = criterion(masks_probs_flat, true_masks_flat)
        epoch_loss += loss.item()

        print('{0:.4f} --- loss: {1:.6f}'.format(i * batch_size / 20, loss.item())) # 算了多少数据，loss是多少

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Epoch finished ! Loss: {}'.format(epoch_loss / i))

    if 1:
        val_dice = eval_net(net, val, gpu)
        print('Validation Dice Coeff: {}'.format(val_dice))

    # if save_cp:
    #     torch.save(net.state_dict(),
    #                dir_checkpoint + 'CP{}.pth'.format(epoch + 1))
    #     print('Checkpoint {} saved !'.format(epoch + 1))


def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=5, type='int',
                      help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batchsize', default=10,
                      type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.1,
                      type='float', help='learning rate')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=False, help='use cuda')
    parser.add_option('-c', '--load', dest='load',
                      default=False, help='load file model')
    parser.add_option('-s', '--scale', dest='scale', type='float',
                      default=0.5, help='downscaling factor of the images')

    (options, args) = parser.parse_args()
    return options


if __name__ == '__main__':
    args = get_args()

    net = UNet(n_channels=3, n_classes=1)

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
    except KeyboardInterrupt:  # 异常处理👍，出错的话就会save下来
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
