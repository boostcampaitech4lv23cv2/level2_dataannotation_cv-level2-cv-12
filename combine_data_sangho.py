import os.path as osp
import json
import os
import copy
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
import random

def parse_args():
    parser = ArgumentParser()

    parser.add_argument('--anno_path_list', type=str, nargs='+',
                        #default=['/opt/ml/input/data/ufo/Boostcamp_val_3.json'])
                        default=['/opt/ml/input/data/ufo/AIHub_Docs.json', 
                                 '/opt/ml/input/data/ufo/Boostcamp_train_3.json',
                                '/opt/ml/input/data/ufo/ICDAR17_Kor.json',
                                 '/opt/ml/input/data/ufo/KAIST_SceneText.json'])
    parser.add_argument('--image_midpath_list', type=str, nargs='+',
                        default=['AIHub_Docs', 'Boostcamp', 'ICDAR17_Kor', 'KAIST_SceneText'])#['Boostcamp'])#
    parser.add_argument('--file_name', type=str, default='set2_Boostcamp3_AIDoc_ICDAR17Kor_KAIST.json')#'Boostcamp_val_3_adddir.json')#
    parser.add_argument('--save_path', type=str, default='../input/data/ufo/')

    args = parser.parse_args()

    if not osp.exists(args.save_path):
        os.makedirs(args.save_path)

    return args

# --이미지 파일은 수작업으로 sava_path의 상위 폴더에서 images 폴더를 만들어 복사해서 넣음!
def main(args):
    get_ratio = [1., 1., 1., 0.3]

    data_path = args.anno_path_list  #data_path_list
    image_midpath_list = args.image_midpath_list
    json_data_list = list()

    for l in data_path:
        json_data_list.append(l)

    read_json = {'images':{}}
    json_cnt = 0
    for json_, image_midpath, ratio in zip(json_data_list, image_midpath_list, get_ratio):
        json_cnt += 1
        with open(json_, 'r') as f:
            json_data = json.load(f)
        print(f'file number {json_cnt} images :', len(json_data['images']))
        for k in list(json_data['images'].keys()):
            if random.random() 
            read_json['images'][osp.join(image_midpath, k)] = json_data['images'][k]
    print('total images :', len(read_json['images']))
    print()

    anno = read_json['images']

    anno_temp = copy.deepcopy(anno)

    count_normal = 0
    count_polygon = 0
    count_none_anno = 0

    for img_name, img_info in tqdm(anno.items()):
        if img_info['words'] == {}:
            del(anno_temp[img_name])
            count_none_anno += 1
            continue
        
        for obj, _ in img_info['words'].items():
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
            
            # 다각형인 모양에서 점이 4개 이상인경우 외접하는 큰 사각형으로 변경
            elif len(img_info['words'][obj]['points']) > 4:
                a = np.array(img_info['words'][obj]['points'])
                img_info['words'][obj]['points'] = [[a[:, 0].min(), a[:, 1].max()], 
                                                    [a[:, 0].max(), a[:, 1].max()],
                                                    [a[:, 0].max(), a[:, 1].min()],
                                                    [a[:, 0].min(), a[:, 1].min()]]
                anno_temp[img_name]['words'][obj]['points'] = img_info['words'][obj]['points']
                count_polygon += 1

    print()
    print(f'normal rect count : {count_normal}')
    print(f'polygon count : {count_polygon}')
    print(f'less 4 point : {count_none_anno}')
    print(f'total anno : {count_normal + count_polygon}')

    anno = {'images': anno_temp}
    save_path_name = args.save_path + args.file_name

    with open(save_path_name, 'w') as f:
        json.dump(anno, f, indent='\t')

    print(f'save the file at {save_path_name}')

if __name__ == '__main__':
    args = parse_args()
    main(args)