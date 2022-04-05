import os
import argparse
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter

from src.dataset import ImageDataset
from src.generator import PUNet
from src.utils import VGG16FeatureExtractor
from src.loss import generator_loss
from src import config


# 各项参数的初始化，训练时可以在终端中调整
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
parser = argparse.ArgumentParser()
parser.add_argument('--num_workers', type=int, default=0, help='workers for dataloader')
parser.add_argument('--image_root', type=str, default='')
parser.add_argument('--mask_root', type=str, default='')
parser.add_argument('--save_dir', type=str, default='checkpoint', help='path for saving models')
parser.add_argument('--log_dir', type=str, default='runs/PairsStreetView', help='log with tensorboardX')
parser.add_argument('--pre_trained', type=str, default='', help='the path of checkpoint')
parser.add_argument('--save_interval', type=int, default=5, help='interval between model save')
parser.add_argument('--gen_lr', type=float, default=0.0002)
parser.add_argument('--D2G_lr', type=float, default=0.1)
parser.add_argument('--lr_finetune', type=float, default=0.00005)
parser.add_argument('--finetune', type=int, default=0)
parser.add_argument('--b1', type=float, default=0.0)
parser.add_argument('--b2', type=float, default=0.9)
parser.add_argument('--batch_size', type=int, default=6)
parser.add_argument('--load_size', type=int, default=350, help='image loading size')
parser.add_argument('--crop_size', type=int, default=256, help='image training size')
parser.add_argument('--start_iter', type=int, default=1, help='start iter')
parser.add_argument('--train_epochs', type=int, default=80, help='training epochs')
args = parser.parse_args()

# 模型输出路径初始化
if not os.path.exists(args.save_dir):
    os.makedirs('{:s}/ckpt'.format(args.save_dir))

# 数据加载
load_size = (args.load_size, args.load_size)
crop_size = (args.crop_size, args.crop_size)
image_dataset = ImageDataset(args.image_root, args.mask_root, load_size, crop_size)
data_loader = DataLoader(
    image_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    drop_last=False,
    pin_memory=True
)

# 模型初始化
generator = PUNet(in_channels=3, out_channels=3)
extractor = VGG16FeatureExtractor()
start_iter = args.start_iter
if args.pre_trained != '':

    ckpt_dict_load = torch.load(args.pre_trained)
    start_iter = ckpt_dict_load['n_iter']
    generator.load_state_dict(ckpt_dict_load['generator'])
    print('Starting from iter ', start_iter)

# 微调（迁移学习用）
if args.finetune != 0:
    print('Fine tune...')
    lr = args.lr_finetune
    generator.freeze_ec_bn = True
else:
    lr = args.gen_lr

# 优化器
G_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(args.b1, args.b2))
writer = SummaryWriter(args.log_dir)

# 训练开始
print('Start train...')
count = 0
for epoch in range(start_iter, args.train_epochs + 1):
    generator.train()
    for _, (input_images, ground_truths, masks) in enumerate(data_loader):

        count = count + 1
        start_time = time.time()
        G_optimizer.zero_grad()

        # 损失函数
        outputs = generator(input_images, masks)
        G_loss, G_loss_list = generator_loss(input_images, masks, outputs, ground_truths, extractor)
        writer.add_scalar('G hole loss', G_loss_list[0].item(), count)
        writer.add_scalar('G valid loss', G_loss_list[1].item(), count)
        writer.add_scalar('G perceptual loss', G_loss_list[2].item(), count)
        writer.add_scalar('G style loss', G_loss_list[3].item(), count)
        writer.add_scalar('G tv loss', G_loss_list[4].item(), count)
        writer.add_scalar('G total loss', G_loss.item(), count)

        G_loss.backward()
        G_optimizer.step()

        end_time = time.time()
        sum_time = end_time - start_time

        print('[Epoch %d/%d] [Batch %d/%d] [count %d] [G_loss %f] [time %f]' %
              (epoch, args.train_epochs, _ + 1, len(data_loader), count, G_loss, sum_time))

    # 模型保存（每五个保存一个）
    if epoch % args.save_interval == 0:

        ckpt_dict_save = {'n_iter': epoch + 1}
        ckpt_dict_save['generator'] = generator.state_dict()

        torch.save(ckpt_dict_save, '{:s}/ckpt/model_{:d}.pth'.format(args.save_dir, epoch))

writer.close()

