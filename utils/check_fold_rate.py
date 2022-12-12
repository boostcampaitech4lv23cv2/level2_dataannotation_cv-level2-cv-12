import json
import pandas as pd
from collections import defaultdict

def check_fold_info(orientation_info : dict, split_num : int):
    orientations = [orientation for orientation in orientation_info]

    fold_names, rates = [], []
    train_infos, val_infos = [], []
    
    for i in range(split_num):
        with open(f'/opt/ml/input/data/total_data/kfold/seed42/train_{i}.json', 'r', encoding='utf-8') as file:
            trains = json.load(file)['images']
            
        with open(f'/opt/ml/input/data/total_data/kfold/seed42/val_{i}.json', 'r', encoding='utf-8') as file:
            vals = json.load(file)['images']
        
        train_info = defaultdict(int)
        val_info = defaultdict(int)
        
        for train in trains:
            words = trains[train]['words']
            for word_key in words:
                orientation = words[word_key]['orientation']
                train_info[orientation] += 1

        for val in vals:
            words = vals[val]['words']
            for word_key in words:
                    orientation = words[word_key]['orientation']
                    val_info[orientation] += 1
        
        train_infos.append(train_info)
        val_infos.append(val_info)

    for idx, (train_info, val_info) in enumerate(zip(train_infos, val_infos)):
        fold_names.append(f'train_fold_{idx}')
        fold_names.append(f'val_fold_{idx}')
        rates.append([f'{train_info[orientation]/orientation_info[orientation]*100:.2f}%' for orientation in orientations])
        rates.append([f'{val_info[orientation]/orientation_info[orientation]*100:.2f}%' for orientation in orientations])
        
    df = pd.DataFrame(rates, index=fold_names, columns=orientations)
    print(df)
    
def check_raw_info():
    with open('../../input/data/total_data/ufo/train.json', 'r', encoding='utf-8') as file:
        data = json.load(file)['images']

    orientation_info = defaultdict(int)

    for key in data:
        words = data[key]['words']
        for word_key in words:
            orientation = words[word_key]['orientation']
            orientation_info[orientation] += 1
            
    return dict(orientation_info)
    
if __name__ == "__main__":
    orientation_info = check_raw_info()
    print()
    print(orientation_info)
    print()
    check_fold_info(orientation_info, split_num=5)