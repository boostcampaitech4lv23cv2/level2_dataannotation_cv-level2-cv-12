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
st.set_page_config(layout="wide")

## ex) streamlit run app.py --server.port=30001 -- --submission_csv <path> --gt_json <path> --dataset_path <path>
parser = argparse.ArgumentParser(description='basic Argparse')
parser.add_argument('--submission_csv', type=str, default=None, help='Infered된 csv 파일의 경로 ex)~/submission.csv')
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
    cv.arrowedLine(img, points[0], points[1], color, thickness, tipLength = 0.1)
    cv.arrowedLine(img, points[1], points[2], color, thickness, tipLength = 0.1)
    cv.arrowedLine(img, points[2], points[3], color, thickness, tipLength = 0.1)
    cv.arrowedLine(img, points[3], points[0], color, thickness, tipLength = 0.1)
    cv.circle(img, points[0], radius=7, color=(255, 255, 0), thickness=-1) # start point

    return img

def draw_illegibility(img, points, color):
    points = [list(map(int, x)) for x in points]
    vertices = np.array(points)
    alpha = 0.7
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
    st.title("Textbox Visualization")
    ##TODO
    ## csv파일로 박스 그리기 
    # data = pd.read_csv(args.submission_csv, index_col=False)
    orientation_lst = ["Horizontal", "Vertical", "Irregular", "null"]
    language_lst = ["ko", "en", "others", "null"]

    with open(args.gt_json) as f:
        gt_data = json.load(f)
    
    #image index 설정
    img_filenames = list(gt_data['images'].keys())
    img_len = len(img_filenames)
    
    # img_idx = int(st.sidebar.number_input(f'보고싶은 이미지의 인덱스 (max {img_len - 1})', value=0))
    if "img_idx" not in st.session_state:
        st.session_state["img_idx"] = 0
    
    if st.sidebar.button('Next'):
        st.session_state["img_idx"] += 1
    st.session_state["img_idx"] = st.sidebar.selectbox('Selcet Image', range(len(img_filenames)), format_func=lambda x:img_filenames[x], index=st.session_state["img_idx"])
    
    img_idx = st.session_state["img_idx"]
    img_filename = img_filenames[img_idx]

    dataset_path = args.dataset_path
    img_path = os.path.join(dataset_path, img_filename)
    img = cv.imread(img_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    infered_img = deepcopy(img)
    
    ori = st.sidebar.multiselect("Orientation", orientation_lst, default=orientation_lst)
    lang = st.sidebar.multiselect("Language", language_lst, default=language_lst)

    words = gt_data['images'][img_filename]['words']
    side_col1, side_col2 = st.sidebar.columns(2)
    side_col1.text('GT')
    side_col2.text('Inferred')


    if "gt_checkbox" not in st.session_state:
        st.session_state["gt_checkbox"] = True
    if "infer_checkbox" not in st.session_state:
        st.session_state["infer_checkbox"] = True
    if side_col1.button("Select All"):
        st.session_state["gt_checkbox"] = True
    if side_col1.button("Empty All"):
        st.session_state["gt_checkbox"] = False
    if side_col2.button("Select All", key='col2 select'):
        st.session_state["infer_checkbox"] = True
    if side_col2.button("Empty All", key='col2 empty'):
        st.session_state["infer_checkbox"] = False
    
    for k, v in words.items():
        with side_col1:
            check = st.checkbox(f"{k} {v['transcription']}", value=st.session_state["gt_checkbox"])
        if check:
            if v['illegibility']:
                draw_illegibility(img, v['points'], (0,0,0))
            elif check_orientation(v['orientation'], ori) and check_language(v['language'], lang):
                draw_bbox(img, k, v['points'], (255, 0, 0), 3)
    
    ## Infer 결과로 수정 예정
    for k, v in words.items():
        with side_col2:
            check2 = st.checkbox(f"infer {k} {v['transcription']}", value=st.session_state["infer_checkbox"])
    
    ## TODO
    ## metadata 기반 박스 on/off 기능


    st.header(img_filename)
    col1, col2 = st.columns(2)
    col1.text('Ground Truth')
    col1.image(img)
    col2.text('Inferred')
    col2.image(img)

main()