import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from pycocotools.coco import COCO
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv
import json
import argparse
from tqdm import tqdm
from mmdet.classification.network import NetConv


def init_model(config, checkpoint):
    model = init_detector(config, checkpoint, device='cuda:0')
    return model


def init_classifier(path):
    device = torch.device("cuda")
    model = NetConv()
    model.load_state_dict(torch.load(path))
    model.to(device)
    return model


def load_img(file):
    org_img = cv2.imread(file)
    return org_img


def argsparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='path to config file')
    parser.add_argument('checkpoint', help='path to checkpoint')
    parser.add_argument('classifier', help='path to classifier checkpoint')
    args = parser.parse_args()
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
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    detector = init_model(args.config, args.checkpoint)
    classifier = init_classifier(args.classifier)
    data_dir = '/workspace/zalo_sign_detection/za_traffic_2020/traffic_public_test/images'
    img_files = os.listdir(data_dir)
    final_result = []
    for idx in tqdm(range(len(img_files))):
        file = img_files[idx]
        img = load_img(os.path.join(data_dir, file))
        results = inference_detector(detector, img)
        idx_img = int(file.split('.')[0])
        for res in enumerate(results):
            if res.shape[0] == 0:
                continue
            for i in range(res.shape[0]):
                sign = img[int(res[i,1]):int(res[i,3]),
                           int(res[i,0]):int(res[i,2])]
                sign = cv2.cvtColor(sign, cv2.COLOR_BGR2GRAY)
                sign = cv2.resize(sign, (128,128))
                sign_tensor = torch.tensor(sign).to(device)
                sign_tensor = sign_tensor.float().unsqueeze(0)
                output_cls = classifier(sign_tensor)
                idx_cls = output_cls.argmax(dim=1)
                if idx_cls == 7:
                    continue
                idx_cls += 1
                final_result.append(format_result(res[i, :].tolist(), idx_img, idx_cls))

    with open('result.json', 'w') as f:
        json.dump(final_result, f)
    print('Inference completed !')


if __name__ == '__main__':
    args = argsparse()
    main(args)
