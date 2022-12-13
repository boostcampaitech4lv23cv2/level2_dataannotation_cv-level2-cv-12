import json
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input-path', default='/opt/ml/input/data/total_data/ufo/train.json', help='input train json path')
parser.add_argument('-o', '--output-path', default='/opt/ml/input/data/total_data/kfold', help='output dir path')
args = parser.parse_args()

def split_annotation(args):
    with open(args.input_path, 'r', encoding='utf-8') as file:
        data = json.load(file)['images']

    annotations = []

    for idx, file_name in enumerate(data):
        words = data[file_name]['words']
        
        for word_key in words:
            annotations.append({
                'id': idx,
                'file_name': file_name,
                'language': words[word_key]['language'],
                'orientation': words[word_key]['orientation'],
                'points': words[word_key]['points'],
                'transcription': words[word_key]['transcription']
            })
    
    os.makedirs(args.output_path, exist_ok=True)
    
    with open(os.path.join(args.output_path, 'annotations.json'), 'w', encoding='utf-8') as file:
        json.dump(annotations, file, ensure_ascii=False, indent=4)
    
if __name__ == "__main__": 
    split_annotation(args)