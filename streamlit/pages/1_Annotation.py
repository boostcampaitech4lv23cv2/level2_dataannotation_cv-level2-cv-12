import streamlit as st
import os
import json
import argparse
import numpy as np
import pandas as pd
import cv2 as cv
import numpy as np
import math
from copy import deepcopy

# SETTING PAGE CONFIG TO WIDE MODE
st.set_page_config(page_title="annotation", layout="wide")

## ex) streamlit run app.py --server.port=30001 -- --submission_csv <path> --gt_json <path> --dataset_path <path>
parser = argparse.ArgumentParser(description='basic Argparse')
parser.add_argument('--valid_json', type=str, default='/opt/ml/dataset_val(ill).json', help='Infered된 json 파일의 경로 ex)~/submission.json')
parser.add_argument('--gt_json', type=str, default='/opt/ml/input/data/dataset/ufo/annotation.json', help='Ground Truth 데이터의 json 파일 경로 ex)/opt/ml/dataset/train.json')
parser.add_argument('--dataset_path', type=str, default='/opt/ml/input/data/dataset/images', help='데이터셋 폴더 경로 ex)/opt/ml/dataset/')
args = parser.parse_args()

def draw_bbox(img, word_idx, points, color, thickness):
    points = [list(map(int, x)) for x in points]

    font = cv.FONT_HERSHEY_SIMPLEX
    h, w, _ = img.shape
    FONT_SCALE= 0.5e-3  # Adjust for larger font size in all images
    THICKNESS_SCALE = 1e-3  # Adjust for larger thickness in all images
    TEXT_Y_OFFSET_SCALE = 1e-2  # Adjust for larger Y-offset of text and bounding box
    fontscale = min(h, w) * FONT_SCALE
    textX = points[0][0]
    textY = points[0][1] - int(h * TEXT_Y_OFFSET_SCALE)
    text_thickness = math.ceil(min(w, h) * THICKNESS_SCALE)

    text_pos = [textX, textY]

    cv.putText(img, word_idx, text_pos, fontFace=font, fontScale=fontscale, color=color, thickness=text_thickness)
    for i in range(len(points) - 1):
        cv.arrowedLine(img, points[i], points[i + 1], color, thickness, tipLength = 0.1)
    cv.arrowedLine(img, points[i + 1], points[0], color, thickness, tipLength = 0.1)
    

    cv.circle(img, points[0], radius=7, color=(255, 255, 0), thickness=-1) # start point

    return img

def draw_illegibility(img, points, color):
    points = [list(map(int, x)) for x in points]
    vertices = np.array(points)
    alpha = 0.5
    overlay = img.copy()
    cv.fillPoly(overlay, [vertices], color)
    cv.addWeighted(overlay, alpha, img, 1-alpha, 0, img)
    return img

def IoU(box1, box2):
    # box = (x1, y1, x2, y2)
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # obtain x1, y1, x2, y2 of the intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # compute the width and height of the intersection
    w = max(0, x2 - x1 + 1)
    h = max(0, y2 - y1 + 1)

    inter = w * h
    iou = inter / (box1_area + box2_area - inter)
    return iou

def check_language(word_lang, select_lang):
    if not word_lang:
        if "null" in select_lang:
            return True
        else:
            return False
    for lang in word_lang:
        if lang not in select_lang:
            return False
    return True

def check_orientation(word_ori, select_ori):
    if not word_ori:
        if "null" in select_ori:
            return True
        else:
            return False
    if word_ori not in select_ori:
        return False
    return True


def main():
    st.title("Annotation Visualization")
    ##TODO
    ## csv파일로 박스 그리기 
    # data = pd.read_csv(args.submission_csv, index_col=False)
    orientation_lst = ["Horizontal", "Vertical", "Irregular", "null"]
    language_lst = ["ko", "en", "others", "null"]

    with open(args.gt_json) as f:
        gt_data = json.load(f)
    with open(args.valid_json) as f:
        pred_data = json.load(f)
    
    #image index 설정
    img_filenames = list(gt_data['images'].keys())
    img_len = len(img_filenames)
    
    if "img_idx" not in st.session_state:
        st.session_state["img_idx"] = 0
    side_col1, side_col2, side_col3 = st.sidebar.columns([1,1,3])
    if side_col1.button('Prev'):
        st.session_state["img_idx"] -= 1
    if side_col2.button('Next'):
        st.session_state["img_idx"] += 1
    st.session_state["img_idx"] = st.sidebar.selectbox('Selcet Image', range(len(img_filenames)), format_func=lambda x:img_filenames[x], index=st.session_state["img_idx"])
    
    img_idx = st.session_state["img_idx"]
    img_filename = img_filenames[img_idx]

    dataset_path = args.dataset_path
    img_path = os.path.join(dataset_path, img_filename)
    img = cv.imread(img_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    inferred_img = deepcopy(img)
    
    ori = st.sidebar.multiselect("Orientation", orientation_lst, default=orientation_lst)
    lang = st.sidebar.multiselect("Language", language_lst, default=language_lst)

    gt_words = gt_data['images'][img_filename]['words']
    st.sidebar.text('GT')

    if "gt_checkbox" not in st.session_state:
        st.session_state["gt_checkbox"] = True
    if st.sidebar.button("Select All"):
        st.session_state["gt_checkbox"] = True
    if st.sidebar.button("Empty All"):
        st.session_state["gt_checkbox"] = False
    
    for k, v in gt_words.items():
        if v['illegibility']:
            draw_illegibility(img, v['points'], (0,0,0))
            continue
        if check_orientation(v['orientation'], ori) and check_language(v['language'], lang):
            check = st.sidebar.checkbox(f"{k} {v['transcription']}", value=st.session_state["gt_checkbox"])
            if check:
                draw_bbox(img, k, v['points'], (255, 0, 0), 3)

    st.header(img_filename)
    st.text('Ground Truth')
    st.image(img)
    data = []
    indices = []
    for k, v in gt_words.items():
        if v['illegibility']:
            continue
        indices.append(k)
        data.append({
            'transcription' : v['transcription'],
            'language' : ", ".join(v['language']) if v['language'] is not None else None,
            'orientation': v['orientation'],
            'word_tags': ", ".join(v['tags'])})
    df = pd.DataFrame(data, index=indices)
    st.table(df)

main()