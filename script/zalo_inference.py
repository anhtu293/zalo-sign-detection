import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from pycocotools.coco import COCO
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv
import json
import argparse
from tqdm import tqdm
from post_processing import nms_interclass


def init_model(config, checkpoint):
    model = init_detector(config, checkpoint, device='cuda:0')
    return model


def load_img(file):
    org_img = cv2.imread(file)
    return org_img


def argsparse():
    argsparser = argparse.ArgumentParser()
    argsparser.add_argument('config', help='path to config file')
    argsparser.add_argument('checkpoint', help='path to checkpoint')
    args = argsparser.parse_args()
    return args


def format_result(result, idx_img, idx_cls):
    formated_pred = {}
    formated_pred['image_id'] = idx_img
    formated_pred['category_id'] = idx_cls
    formated_pred['bbox'] = [
        result[0],
        result[1],
        result[2] - result[0],
        result[3] - result[1]
    ]
    formated_pred['score'] = result[4]
    return formated_pred


def main(args):
    model = init_model(args.config, args.checkpoint)
    data_dir = '/workspace/zalo_sign_detection/za_traffic_2020/traffic_public_test/images'
    img_files = os.listdir(data_dir)
    final_result = []
    for idx in tqdm(range(len(img_files))):
        file = img_files[idx]
        img = load_img(os.path.join(data_dir, file))
        results = inference_detector(model, img)
        results = nms_interclass(results)
        idx_img = int(file.split('.')[0])
        for idx_cls, res in enumerate(results):
            if res.shape[0] == 0:
                continue
            for i in range(res.shape[0]):
                final_result.append(format_result(res[i, :].tolist(), idx_img, idx_cls))

    with open('result.json', 'w') as f:
        json.dump(final_result, f)
    print('Inference completed !')


if __name__ == '__main__':
    args = argsparse()
    main(args)
