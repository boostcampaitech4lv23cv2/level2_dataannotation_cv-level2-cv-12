import os
import os.path as osp
import time
import math
from datetime import timedelta
from argparse import ArgumentParser
from collections import defaultdict
import json

import torch
from torch import cuda
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm
import wandb

from east_dataset import EASTDataset
from dataset import SceneTextDataset, ValidationDataset
from model import EAST
from utils.seed import set_seed
#from validation import do_valdation
from logger.set_wandb import wandb_init

import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.augmentations.geometric.resize import LongestMaxSize

import multiprocessing
from multiprocessing import Pool

def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument('--train_dir', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN', '../input/data/ICDAR17_Korean'))
    parser.add_argument('--train_name', type=str, default='val_3')
    #parser.add_argument('--val_dir', type=str,
    #                    default='../input/data/dataset')
    #parser.add_argument('--val_name', type=str, default='val_3')
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR',
                                                                        'trained_models'))

    parser.add_argument('--device', default='cuda' if cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--image_size', type=int, default=1024)
    parser.add_argument('--input_size', type=int, default=512)
    #parser.add_argument('--val_input_size', type=int, default=1024)

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

def do_training(args,model):#,process_cnt,process_pool):
    print("\n##### TRAINING #####")
    dataset = SceneTextDataset(args.train_dir, split=args.train_name, image_size=args.image_size, crop_size=args.input_size)
    dataset = EASTDataset(dataset)
    num_batches = math.ceil(len(dataset) / args.batch_size)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    #with open(osp.join(args.val_dir, 'ufo/{}.json'.format(args.val_name)), 'r') as f:
    #    val_gt_dict = json.load(f)['images']
    #val_image_dir = osp.join(args.val_dir,'images')

    #val_illegibility_dict = {}
    #for image_fname in val_gt_dict:
    #    val_illegibility_dict[image_fname] = [val_gt_dict[image_fname]['words'][i]['illegibility'] for i in val_gt_dict[image_fname]['words']]
    #    val_gt_dict[image_fname] = [val_gt_dict[image_fname]['words'][i]['points'] for i in val_gt_dict[image_fname]['words']]

    #val_dataset = ValidationDataset(image_fnames=list(val_gt_dict.keys()),image_dir=val_image_dir,input_size=args.val_input_size)
    #val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=args.num_workers)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[args.max_epoch // 2], gamma=0.1)
    best_score, best_epoch  = 0, 0

    model.train()
    for epoch in range(1, args.max_epoch+1):
        print('\n ### epoch {} ###'.format(epoch))
        epoch_loss, start = 0, time.time()
        train_dict = defaultdict(int)
        #model.train()
        with tqdm(total=num_batches) as pbar:
            for img, gt_score_map, gt_geo_map, roi_mask in train_loader:
                #pbar.set_description('[Epoch {}]'.format(epoch + 1))

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

        print('[train] loss: {:.4f} | Elapsed time: {}'.format(
            epoch_loss / num_batches, timedelta(seconds=time.time() - start)))

        if (epoch + 1) % args.save_interval == 0:
            if not osp.exists(args.model_dir):
                os.makedirs(args.model_dir)

            ckpt_fpath = osp.join(args.model_dir, 'ep' + str(epoch+1) + '.pth')
            torch.save(model.state_dict(), ckpt_fpath)


        """
        ### validation ###
        start = time.time()
        res_dict = do_valdation(model=model, loader=val_loader, gt_bboxes_dict=val_gt_dict, transcriptions_dict=val_illegibility_dict, input_size=args.val_input_size, process_cnt=process_cnt, process_pool=process_pool)
        val_dict = res_dict['total']
        val_dict['epoch'] = epoch
        wandb.log(val_dict)
        print('[val] f1 : {:.4f} | precision : {:.4f} | recall : {:.4f} | Elapsed time: {}'.format(
            val_dict['hmean'],val_dict['precision'],val_dict['recall'], timedelta(seconds=time.time() - start)))

        if args.save_interval !=-1 and epoch % args.save_interval == 0:
            if not osp.exists(args.model_dir):
                os.makedirs(args.model_dir)
            ckpt_fpath = osp.join(args.model_dir, args.experiment_name, 'latest.pth')
            torch.save(model.state_dict(), ckpt_fpath)
        
        if best_score<val_dict['hmean']:
            best_score = val_dict['hmean']
            best_epoch = epoch
            if not osp.exists(args.model_dir):
                os.makedirs(args.model_dir)

            ckpt_fpath = osp.join(args.model_dir, args.experiment_name, 'best.pth')
            torch.save(model.state_dict(), ckpt_fpath)

            with open(osp.join(args.model_dir, args.experiment_name, 'best_result.json'), 'w') as f:
                json.dump(res_dict, f, indent=4)
            print('@@@ best model&result are saved!! @@@')
        print('[best] epoch : {} | score : {:.4f}'.format(best_epoch,best_score))
        """

def do_warmup(args, model):
    dataset = SceneTextDataset(args.train_dir, split=args.train_name, image_size=args.image_size, crop_size=args.input_size)
    dataset = EASTDataset(dataset)
    num_batches = math.ceil(len(dataset) / args.batch_size)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    model.train()
    print("\n##### WARMING UP #####")
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

        train_dict['warmup_epoch'] = epoch
        train_dict['learning_rate'] = optimizer.param_groups[0]['lr']
        wandb.log(train_dict)

        print('Mean loss: {:.4f} | Elapsed time: {}'.format(
            epoch_loss / num_batches, timedelta(seconds=time.time() - epoch_start)))
        
    for name, param in model.named_parameters():
        if 'extractor' in name:
            param.requires_grad = True
    
def main(args):
    set_seed(args.seed)
    #process_cnt = multiprocessing.cpu_count()
    #process_pool = Pool(process_cnt)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST()
    model.to(device)

    #if args.warmup:
    do_warmup(args, model)
    do_training(args, model)#, process_cnt, process_pool)


if __name__ == '__main__':
    args = parse_args()
    wandb_init(args)
    main(args)