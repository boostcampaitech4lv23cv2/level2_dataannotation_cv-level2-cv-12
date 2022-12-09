import os
import os.path as osp
import time
import math
from datetime import timedelta
from argparse import ArgumentParser
from collections import defaultdict

import torch
from torch import cuda
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm
import wandb

from east_dataset import EASTDataset
from dataset import SceneTextDataset
from model import EAST
from utils import set_seed
from set_wandb import wandb_init

def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument('--data_dir', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN', '../input/data/ICDAR17_Korean'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR',
                                                                        'trained_models'))

    parser.add_argument('--device', default='cuda' if cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--image_size', type=int, default=1024)
    parser.add_argument('--input_size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--save_interval', type=int, default=5)

    # Custom args
    parser.add_argument("--experiment_name",type=str,default="test")
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--warmup_epoch", type=int, default=5) 
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    return args


def do_training(args,model):
    print("\n### TRAINING ###")
    dataset = SceneTextDataset(args.data_dir, split='train', image_size=args.image_size, crop_size=args.input_size)
    dataset = EASTDataset(dataset)
    num_batches = math.ceil(len(dataset) / args.batch_size)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #model = EAST()
    #model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[args.max_epoch // 2], gamma=0.1)

    model.train()
    for epoch in range(args.max_epoch):
        epoch_loss, epoch_start = 0, time.time()
        train_dict = defaultdict(int)
        with tqdm(total=num_batches) as pbar:
            for img, gt_score_map, gt_geo_map, roi_mask in train_loader:
                pbar.set_description('[Epoch {}]'.format(epoch + 1))

                loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_val = loss.item()
                epoch_loss += loss_val

                pbar.update(1)
                tmp_dict = {
                    'Cls loss': extra_info['cls_loss'], 'Angle loss': extra_info['angle_loss'],
                    'IoU loss': extra_info['iou_loss']
                }
                pbar.set_postfix(tmp_dict)

                train_dict['train_total_loss'] += loss.item() / len(train_loader)
                train_dict['train_cls_loss'] += extra_info['cls_loss'] / len(train_loader)
                train_dict['train_angle_loss'] += extra_info['angle_loss'] / len(train_loader)
                train_dict['train_iou_loss'] += extra_info['iou_loss'] / len(train_loader)

        scheduler.step()
        train_dict['epoch'] = epoch
        train_dict['learning_rate'] = optimizer.param_groups[0]['lr']
        wandb.log(train_dict)

        print('Mean loss: {:.4f} | Elapsed time: {}'.format(
            mean_loss, timedelta(seconds=time.time() - epoch_start)))

        if (epoch + 1) % args.save_interval == 0:
            if not osp.exists(args.model_dir):
                os.makedirs(args.model_dir)
            ckpt_fpath = osp.join(args.model_dir, args.experiment_name, 'latest.pth')
            torch.save(model.state_dict(), ckpt_fpath)

def do_warmup(args, model):
    dataset = SceneTextDataset(args.data_dir, split='train', image_size=args.image_size, crop_size=args.input_size)
    dataset = EASTDataset(dataset)
    num_batches = math.ceil(len(dataset) / args.batch_size)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    #scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[args.max_epoch // 2], gamma=0.1)
    #scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=args.learning_rate, total_steps = args.warmup_epoch*len(train_loader), pct_start=1.0)

    model.train()
    print("\n### WARMING UP ###")
    for name, param in model.named_parameters():
        if 'extractor' in name:
            param.requires_grad = False

    for epoch in range(args.warmup_epoch):
        epoch_loss, epoch_start = 0, time.time()
        train_dict = defaultdict(int)
        with tqdm(total=num_batches) as pbar:
            for img, gt_score_map, gt_geo_map, roi_mask in train_loader:
                pbar.set_description('[Warmup Epoch {}]'.format(epoch + 1))

                loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_val = loss.item()
                epoch_loss += loss_val

                pbar.update(1)
                tmp_dict = {
                    'Cls loss': extra_info['cls_loss'], 'Angle loss': extra_info['angle_loss'],
                    'IoU loss': extra_info['iou_loss']
                }
                pbar.set_postfix(tmp_dict)

                train_dict['warmup_total_loss'] += loss.item() / len(train_loader)
                train_dict['warmup_cls_loss'] += extra_info['cls_loss'] / len(train_loader)
                train_dict['warmup_angle_loss'] += extra_info['angle_loss'] / len(train_loader)
                train_dict['warmup_iou_loss'] += extra_info['iou_loss'] / len(train_loader)

        #scheduler.step()
        train_dict['warmup_epoch'] = epoch
        train_dict['learning_rate'] = optimizer.param_groups[0]['lr']
        wandb.log(train_dict)

        print('Mean loss: {:.4f} | Elapsed time: {}'.format(
            epoch_loss / num_batches, timedelta(seconds=time.time() - epoch_start)))

        # if (epoch + 1) % args.save_interval == 0:
        #     if not osp.exists(args.model_dir):
        #         os.makedirs(args.model_dir)
        #     ckpt_fpath = osp.join(args.model_dir, args.experiment_name, 'latest.pth')
        #     torch.save(model.state_dict(), ckpt_fpath)

    for name, param in model.named_parameters():
        if 'extractor' in name:
            param.requires_grad = True
    
def main(args):
    do_training(**args.__dict__)


if __name__ == '__main__':
    args = parse_args()
    wandb_init(args)
    main(args)
