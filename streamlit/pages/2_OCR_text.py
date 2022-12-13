import streamlit as st
import os
import json
import numpy as np
import pandas as pd
import argparse
import cv2 as cv
import numpy as np
import math
import matplotlib
from copy import deepcopy

# SETTING PAGE CONFIG TO WIDE MODE
st.set_page_config(page_title="OCR", layout="wide")

# ex) streamlit run app.py --server.port=30001 -- --submission_csv <path> --gt_json <path> --dataset_path <path>
parser = argparse.ArgumentParser(description='basic Argparse')
parser.add_argument('--valid_json', type=str, default='/opt/ml/dataset_val(ill).json', help='Infered된 json 파일의 경로 ex)~/submission.json')
parser.add_argument('--gt_json', type=str, default='/opt/ml/input/data/dataset/ufo/annotation.json', help='Ground Truth 데이터의 json 파일 경로 ex)/opt/ml/dataset/train.json')
parser.add_argument('--dataset_path', type=str, default='/opt/ml/input/data/dataset/images', help='데이터셋 폴더 경로 ex)/opt/ml/dataset/')
args = parser.parse_args()

cmap = matplotlib.cm.get_cmap('Set1')

def draw_bbox(img, word_idx, points, color, thickness, pair_type=None):
    points = list(map(int, points))
    font = cv.FONT_HERSHEY_SIMPLEX
    h, w, _ = img.shape
    FONT_SCALE= 0.5e-3  # Adjust for larger font size in all images
    THICKNESS_SCALE = 1e-3  # Adjust for larger thickness in all images
    TEXT_Y_OFFSET_SCALE = 1e-2  # Adjust for larger Y-offset of text and bounding box

    fontscale = min(h, w) * FONT_SCALE
    textX = points[0]
    textY = points[1] - int(h * TEXT_Y_OFFSET_SCALE)
    text_thickness = math.ceil(min(w, h) * THICKNESS_SCALE)

    text_pos = [textX, textY]
    cv.putText(img, word_idx, text_pos, fontFace=font, fontScale=fontscale, color=color, thickness=text_thickness)
    cv.rectangle(img, (points[0], points[1]), (points[2], points[3]), color, thickness=thickness)

    return img

def draw_illegibility(img, points, color):
    points = list(map(int, points))
    alpha = 0.5
    overlay = img.copy()
    cv.rectangle(img, (points[0], points[1]), (points[2], points[3]), color,-1)
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
    st.title("OCR detector result")

    with open(args.valid_json) as f:
        pred_data = json.load(f)
    
    recall_range = st.sidebar.slider(f'이미지 recall 범위', min_value=0., max_value=1., value=(0., 1.), step=0.01, key='recall')
    precision_range = st.sidebar.slider(f'이미지 precision 범위', min_value=0., max_value=1., value=(0., 1.), step=0.01, key='precision')
    hmean_range = st.sidebar.slider(f'이미지 hmean 범위', min_value=0., max_value=1., value=(0., 1.), step=0.01, key='hmean')
    #image index 설정
    img_filenames = [k for k, v in pred_data['per_sample'].items()
                    if v['recall'] >= recall_range[0] and v['recall'] <= recall_range[1]
                    and v['precision'] >= precision_range[0] and v['precision'] <= precision_range[1]
                    and v['hmean'] >= hmean_range[0] and v['hmean'] <= hmean_range[1]]
    img_len = len(img_filenames)
    
    if "img_idx2" not in st.session_state:
        st.session_state["img_idx2"] = 0

    st.session_state["img_idx2"] = min(st.session_state["img_idx2"], img_len - 1)
    side_col1, side_col2, side_col3 = st.sidebar.columns([1,1,3])
    if side_col1.button('Prev'):
        st.session_state["img_idx2"] -= 1
    if side_col2.button('Next'):
        st.session_state["img_idx2"] += 1
    st.session_state["img_idx2"] = st.sidebar.selectbox('Selcet Image', range(len(img_filenames)), format_func=lambda x:img_filenames[x], index=st.session_state["img_idx2"])
    
    img_idx = st.session_state["img_idx2"]
    img_filename = img_filenames[img_idx]

    dataset_path = args.dataset_path
    img_path = os.path.join(dataset_path, img_filename)
    img = cv.imread(img_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    inferred_img = deepcopy(img)

    gt_bboxes = pred_data['per_sample'][img_filename]['gt_bboxes']
    infer_bboxes = pred_data['per_sample'][img_filename]['det_bboxes']
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


    pairs = pred_data['per_sample'][img_filename]['pairs']
    for i, bbox in enumerate(gt_bboxes):
        if i in pred_data['per_sample'][img_filename]['gt_dont_care']:
            draw_illegibility(img, bbox, (0,0,0))
            continue
        with side_col1:
            check = st.checkbox(str(i), value=st.session_state["gt_checkbox"], key=f'gt check {i}')
        if check:
            gt_bboxes = pred_data['per_sample'][img_filename]['gt_bboxes']
            pair_type = pred_data['per_sample'][img_filename]['pairs']
            draw_bbox(img, str(i), bbox, (255, 0, 0), 3, pair_type)
    
    for i, bbox in enumerate(infer_bboxes):
        if i in pred_data['per_sample'][img_filename]['det_dont_care']:
            draw_illegibility(inferred_img, bbox, (0,0,0))
            continue
        with side_col2:
            check2 = st.checkbox(str(i), value=st.session_state["infer_checkbox"], key=f'infer check {i}')
        if check2:
            infer_bboxes = pred_data['per_sample'][img_filename]['det_bboxes']
            draw_bbox(inferred_img, str(i), bbox, (255, 0, 0), 3)

    # gt_bboxes = pred_data['per_sample'][img_filename]['gt_bboxes']
    # det_bboxes = pred_data['per_sample'][img_filename]['det_bboxes']
    # for idx, pair in enumerate(pairs):
    #     # # with side_col1:
    #     # #     check = st.checkbox(str(i), value=st.session_state["gt_checkbox"], key=f'gt check {i}')
    #     # if check:
    #     if isinstance(pair['gt'], int):
    #         gt_lst = [pair['gt']]
    #     else:
    #         gt_lst = pair['gt']
    #     if isinstance(pair['det'], int):
    #         det_lst = [pair['det']]
    #     else:
    #         det_lst = pair['det']
    #     pair_type = pair['type']
    #     for i in gt_lst:
    #         draw_bbox(img, str(i-1) + pair_type, gt_bboxes[i], cmap(idx), 3, pair_type)
    #     for j in det_lst:
    #         draw_bbox(inferred_img, str(j) + pair_type, det_bboxes[j], cmap(idx), 3, pair_type)

    st.header(img_filename)
    col1, col2 = st.columns(2)
    col1.text('Ground Truth')
    col1.image(img)
    col2.text('Inferred')
    col2.image(inferred_img)
    
    df = pd.DataFrame(
        [[pred_data['per_sample'][img_filename]['precision'],
        pred_data['per_sample'][img_filename]['recall'],
        pred_data['per_sample'][img_filename]['hmean']]],
        columns=["precision", "recall", "hmean"])
    st.table(df)
    # st.metric("precision", pred_data['per_sample'][img_filename]['precision'])
    # st.metric("recall", pred_data['per_sample'][img_filename]['recall'])
    # st.metric("hmean", pred_data['per_sample'][img_filename]['hmean'])
main()