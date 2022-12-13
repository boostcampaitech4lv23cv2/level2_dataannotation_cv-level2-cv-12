import os
import os.path as osp
import json
from argparse import ArgumentParser
from glob import glob

import torch
import cv2
from torch import cuda
from model import EAST
from tqdm import tqdm

from detect import detect
from deteval import calc_deteval_metrics
from time import time

import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.augmentations.geometric.resize import LongestMaxSize
from detect import *

CHECKPOINT_EXTENSIONS = ['.pth', '.ckpt']


def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument('--data_dir', default=os.environ.get('SM_CHANNEL_EVAL'))
    parser.add_argument('--model_dir', default=os.environ.get('SM_CHANNEL_MODEL', 'trained_models'))
    parser.add_argument('--output_dir', default=os.environ.get('SM_OUTPUT_DATA_DIR', 'predictions'))

    parser.add_argument('--device', default='cuda' if cuda.is_available() else 'cpu')
    parser.add_argument('--input_size', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--mode', type=str, default='save_pred')


    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    return args


def val_detect(model, images, input_size):
    prep_fn = A.Compose([
        LongestMaxSize(input_size), A.PadIfNeeded(min_height=input_size, min_width=input_size,
                                                  position=A.PadIfNeeded.PositionType.TOP_LEFT),
        A.Normalize(), ToTensorV2()])
    device = list(model.parameters())[0].device

    start = time()

    batch, orig_sizes = [], []
    for image in images:
        orig_sizes.append(image.shape[:2])
        batch.append(prep_fn(image=image)['image'])
    batch = torch.stack(batch, dim=0).to(device)

    print('0 :', time()-start)

    start = time()

    with torch.no_grad():
        score_maps, geo_maps = model(batch)
    score_maps, geo_maps = score_maps.cpu().numpy(), geo_maps.cpu().numpy()

    print('1 :', time()-start)

    start = time()

    by_sample_bboxes = []
    for score_map, geo_map, orig_size in zip(score_maps, geo_maps, orig_sizes):
        map_margin = int(abs(orig_size[0] - orig_size[1]) * 0.25 * input_size / max(orig_size))
        if orig_size[0] > orig_size[1]:
            score_map, geo_map = score_map[:, :, :-map_margin], geo_map[:, :, :-map_margin]
        else:
            score_map, geo_map = score_map[:, :-map_margin, :], geo_map[:, :-map_margin, :]

        bboxes = get_bboxes(score_map, geo_map)
        if bboxes is None:
            bboxes = np.zeros((0, 4, 2), dtype=np.float32)
        else:
            bboxes = bboxes[:, :8].reshape(-1, 4, 2)
            bboxes *= max(orig_size) / input_size

        by_sample_bboxes.append(bboxes)

    print('2 :', time()-start)
    return by_sample_bboxes


def save_prediction(model, ckpt_fpath, root_dir, input_size, batch_size, split='train', output_fname = 'valid_pred.csv'):
    with open(osp.join(root_dir, 'ufo/{}.json'.format(split)), 'r') as f:
            anno = json.load(f)

    image_dir = osp.join(root_dir,'images')

    model.load_state_dict(torch.load(ckpt_fpath))
    model.eval()
    image_fnames, by_sample_bboxes = [], []
    images = []
    for image_fname in tqdm(sorted(anno['images'].keys())):
        image_fpath = osp.join(image_dir, image_fname)
        #image_fnames.append(osp.basename(image_fpath))
        image_fnames.append(image_fname)

        images.append(cv2.imread(image_fpath)[:, :, ::-1])
        if len(images) == batch_size:
            by_sample_bboxes.extend(detect(model, images, input_size))
            images = []

    if len(images):
        by_sample_bboxes.extend(detect(model, images, input_size))

    ufo_result = dict(images=dict())
    for image_fname, bboxes in zip(image_fnames, by_sample_bboxes):
        words_info = {idx: dict(points=bbox.tolist()) for idx, bbox in enumerate(bboxes)}
        ufo_result['images'][image_fname] = dict(words=words_info)

    result = dict(images=dict()) 
    result['images'].update(ufo_result['images']) # images key 추가하는거?
    with open(osp.join('.', output_fname), 'w') as f:
        json.dump(result, f, indent=4)




def do_valdation(model, gt_bboxes_dict,transcriptions_dict, image_dir, input_size, batch_size, output_fname=None):
    model.eval()
    image_fnames, by_sample_bboxes = [], []

    images = []
    for image_fname in tqdm(gt_bboxes_dict.keys()):
        image_fpath = osp.join(image_dir, image_fname)
        image_fnames.append(image_fname)

        images.append(cv2.imread(image_fpath)[:, :, ::-1])
        if len(images) == batch_size:
            by_sample_bboxes.extend(detect(model, images, input_size))
            images = []

    if len(images):
        by_sample_bboxes.extend(detect(model, images, input_size))

    # for image_fname, image, orig_size in loader:
    #     print(type(image_fname),type(image),type(orig_size))
    #     print(image_fname.shape)
    #     print(image.shape)
    #     print(orig_size.shape)
    #     break

    pred_bboxes_dict = dict(zip(image_fnames, by_sample_bboxes))
    resDict = calc_deteval_metrics(pred_bboxes_dict, gt_bboxes_dict,transcriptions_dict)

    if output_fname is not None:
        with open(output_fname, 'w') as f:
            json.dump(resDict, f, indent=4)

    return resDict


def main(args):
    # Initialize model
    import os.path as osp
    from dataset import ValidationDataset
    from torch.utils.data import DataLoader

    model = EAST(pretrained=False).to(args.device)
    if args.mode == 'save_pred':
        save_prediction(model=model, ckpt_fpath=ckpt_fpath, root_dir='../input/data/total_data',input_size=args.input_size,
                                batch_size=args.batch_size, split='train', output_fname = 'valid_text.csv')
    elif args.mode == 'run':
        ckpt_fpath = 'trained_models/baseline/latest.pth'
        model.load_state_dict(torch.load(ckpt_fpath))
        root_dir = '../input/data/dataset'
        with open(osp.join(root_dir, 'ufo/{}.json'.format('annotation')), 'r') as f:
            gt_bboxes_dict = json.load(f)['images']
        val_image_dir = osp.join(root_dir,'images')

        transcriptions_dict = {}
        for image_fname in gt_bboxes_dict:
            #transcriptions_dict[image_fname] = [gt_bboxes_dict[image_fname]['words'][i]['transcription'] for i in gt_bboxes_dict[image_fname]['words']]
            transcriptions_dict[image_fname] = [gt_bboxes_dict[image_fname]['words'][i]['illegibility'] for i in gt_bboxes_dict[image_fname]['words']]
            gt_bboxes_dict[image_fname] = [gt_bboxes_dict[image_fname]['words'][i]['points'] for i in gt_bboxes_dict[image_fname]['words']]

        # val_dataset = ValidationDataset(image_fnames=list(gt_bboxes_dict.keys()),image_dir=val_image_dir,input_size=args.input_size)
        # val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

        resDict = do_valdation(model=model,gt_bboxes_dict=gt_bboxes_dict, transcriptions_dict=transcriptions_dict, image_dir=val_image_dir, input_size=args.input_size,
                                batch_size=args.batch_size)
        print(resDict['total'])
    
    # transcriptions_dict = {}
    # for image_fname in gt_bboxes_dict:
    #     transcriptions_dict[image_fname] = [gt_bboxes_dict[image_fname]['words'][i]['transcription'] for i in gt_bboxes_dict[image_fname]['words']]
    #     gt_bboxes_dict[image_fname] = [gt_bboxes_dict[image_fname]['words'][i]['points'] for i in gt_bboxes_dict[image_fname]['words']]
    

if __name__ == '__main__':
    args = parse_args()
    main(args)