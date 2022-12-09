import json
import os
import argparse
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from seed import set_seed

parser = argparse.ArgumentParser()
parser.add_argument('-a', '--annotation-path', default='/opt/ml/input/data/ICDAR17_Korean/kfold/annotations.json', help='input annotation json path')
parser.add_argument('-i', '--input-path', default='/opt/ml/input/data/ICDAR17_Korean/ufo/train.json', help='input train json path')
parser.add_argument('-o', '--output-path', default='/opt/ml/input/data/ICDAR17_Korean/kfold', help='output dir path')
parser.add_argument('-v', '--val-ratio', default=0.2, help='validation split ratio')
parser.add_argument('-s', '--seed', default=42, help='random seed')
args = parser.parse_args()
    
def stratified_group_kfold_dataset(args):    
    with open(args.annotation_path, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    
    with open(args.input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)['images']

    X = np.ones((len(annotations), 1))
    y = np.array([0 if info['language'] == "ko" else 1 for info in annotations])
    groups = np.array([info['file_name'] for info in annotations])

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    for idx, (train_ids, val_ids) in enumerate(cv.split(X, y, groups)):
        train_file_names = list(set(groups[train_ids]))
        val_file_names = list(set(groups[val_ids]))

        train_info = {}
        for file_name in train_file_names:
            train_info[file_name] = data[file_name]
            
        val_info = {}
        for file_name in val_file_names:
            val_info[file_name] = data[file_name]
            
        train_data = {
            'images': train_info
        }

        val_data = {
            'images': val_info
        }
        
        output_seed_dir = os.path.join(args.output_path, f'seed{args.seed}')
        os.makedirs(output_seed_dir, exist_ok=True)
        output_train_json = os.path.join(output_seed_dir, f'train_{idx}.json')
        output_val_json = os.path.join(output_seed_dir, f'val_{idx}.json')
        
        with open(output_train_json, 'w') as train_writer:
            json.dump(train_data, train_writer, ensure_ascii=False, indent=4)
        print(f'done. {output_train_json}')
        
        with open(output_val_json, 'w') as val_writer:
            json.dump(val_data, val_writer, ensure_ascii=False, indent=4)
        print(f'done. {output_val_json}')

if __name__ == "__main__":
    set_seed(args.seed)    
    stratified_group_kfold_dataset(args)