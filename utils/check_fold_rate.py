import json
import pandas as pd

def check_fold_info():
    folds = []

    for i in range(5):
        with open(f'/opt/ml/input/data/ICDAR17_Korean/kfold/seed42/train_{i}.json', 'r', encoding='utf-8') as file:
            trains = json.load(file)['images']
            
        with open(f'/opt/ml/input/data/ICDAR17_Korean/kfold/seed42/val_{i}.json', 'r', encoding='utf-8') as file:
            vals = json.load(file)['images']
            
        en_cnt, en_val_cnt = 0, 0
        ko_cnt, ko_val_cnt = 0, 0

        for train in trains:
            words = trains[train]['words']
            for word in words:
                if words[word]['language'] == ['en']:
                    en_cnt += 1
                else:
                    ko_cnt += 1

        for val in vals:
            words = vals[val]['words']
            for word in words:
                if words[word]['language'] == ['en']:
                    en_val_cnt += 1
                else:
                    ko_val_cnt += 1
        
        folds.append([en_cnt, ko_cnt, en_val_cnt, ko_val_cnt])
        
    index = []
    dist = []

    for idx, fold in enumerate(folds):
        en_total = fold[0] + fold[2]
        ko_total = fold[1] + fold[3]
        
        dist.append([f'{fold[0]/en_total*100:.2f}%', f'{fold[1]/ko_total*100:.2f}%'])
        dist.append([f'{fold[2]/en_total*100:.2f}%', f'{fold[3]/ko_total*100:.2f}%'])
        index.append(f'train_fold_{idx}')
        index.append(f'val_fold_{idx}')
        
    df = pd.DataFrame(dist, index=index, columns=['en', 'ko'])
    print(df)
    
if __name__ == "__main__": 
    check_fold_info()