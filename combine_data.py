import os.path as osp
import json
import os
import copy
from tqdm import tqdm
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('--data_path_list', type=str, nargs='+',
                        default=['../input/data/dataset/ufo/annotation.json', '../input/data/ICDAR17_Korean/ufo/train.json'])
    parser.add_argument('--save_path', type=str, default='../input/data/total_data/ufo/')

    args = parser.parse_args()

    if not osp.exists(args.save_path):
        os.makedirs(args.save_path)

    return args

# --이미지 파일은 수작업으로 sava_path의 상위 폴더에서 images 폴더를 만들어 복사해서 넣음!
def main(args):
    data_path = args.data_path_list
    json_data_list = list()

    for l in data_path:
        json_data_list.append(l)

    read_json = {'images':{}}

    for json_ in json_data_list:
        with open(json_, 'r') as f:
            json_data = json.load(f)
        print(len(json_data['images']))
        read_json['images'].update(json_data['images'])
    print(len(read_json['images']))

    anno = read_json['images']

    anno_temp = copy.deepcopy(anno)

    count = 0
    count_normal = 0
    count_none_anno = 0

    for img_name, img_info in tqdm(anno.items()):
        if img_info['words'] == {}:
            del(anno_temp[img_name])
            count_none_anno += 1
            continue
        
        for obj, obj_info in img_info['words'].items():
            # 점이 4개인 box의 경우 정상
            if len(img_info['words'][obj]['points']) == 4:
                count_normal += 1
                continue
            
            # 다각형인 모양에서 점이 4개 미만인 경우 삭제
            elif len(img_info['words'][obj]['points']) < 4:
                del (anno_temp[img_name]['words'][obj])
                if anno_temp[img_name]['words'] == {}:
                    del (anno_temp[img_name])
                    count_none_anno += 1
                    continue
            
            elif len(img_info['words'][obj]['points']) > 4:
                del (anno_temp[img_name]['words'][obj])
                if anno_temp[img_name]['words'] == {}:
                    del (anno_temp[img_name])
                    count_none_anno += 1
                    continue

    print(f'normal polygon count : {count_normal}')

    anno = {'images': anno_temp}
    save_path_name = args.save_path + 'train.json'

    with open(save_path_name, 'w') as f:
        json.dump(anno, f, indent='\t')

if __name__ == '__main__':
    args = parse_args()
    main(args)