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

    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    return args


def do_valdation(model, ckpt_fpath, root_dir, input_size, batch_size, split='train'):
    #model.load_state_dict(torch.load(ckpt_fpath, map_location='cpu'))

    with open(osp.join(root_dir, 'ufo/{}.json'.format(split)), 'r') as f:
            anno = json.load(f)

    image_dir = osp.join(root_dir,'images')

    model.load_state_dict(torch.load(ckpt_fpath))
    model.eval()

    image_fnames, by_sample_bboxes = [], []

    images = []
    num = 0
    for image_fname in tqdm(sorted(anno['images'].keys())):
        image_fpath = osp.join(image_dir, image_fname)
        #image_fnames.append(osp.basename(image_fpath))
        image_fnames.append(image_fname)

        images.append(cv2.imread(image_fpath)[:, :, ::-1])
        if len(images) == batch_size:
            by_sample_bboxes.extend(detect(model, images, input_size))
            images = []
        num+=1
        if num == 10:
            break

    if len(images):
        by_sample_bboxes.extend(detect(model, images, input_size))

    ufo_result = dict(images=dict())
    for image_fname, bboxes in zip(image_fnames, by_sample_bboxes):
        words_info = {idx: dict(points=bbox.tolist()) for idx, bbox in enumerate(bboxes)}
        ufo_result['images'][image_fname] = dict(words=words_info)
    return ufo_result

def main(args):
    # Initialize model
    model = EAST(pretrained=False).to(args.device)

    # Get paths to checkpoint files
    # ckpt_fpath = osp.join(args.model_dir, 'latest.pth')
    ckpt_fpath = 'trained_models/test_400_c/latest.pth'

    # if not osp.exists(args.output_dir):
    #     os.makedirs(args.output_dir)

    print('valdation in progress')

    ufo_result = dict(images=dict())

    split_result = do_valdation(model, ckpt_fpath,'../input/data/total_data', args.input_size,
                                args.batch_size, split='train')
    ufo_result['images'].update(split_result['images'])

    output_fname = 'valid_text.csv'
    with open(osp.join('.', output_fname), 'w') as f:
        json.dump(ufo_result, f, indent=4)

if __name__ == '__main__':
    args = parse_args()
    main(args)


