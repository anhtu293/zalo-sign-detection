from pycocotools.coco import COCO
import json
import argparse
from tqdm import tqdm


def argsparser():
    args = argparse.ArgumentParser()
    args.add_argument('ann', help='path to coco annotation file')
    args.add_argument('output', help='path to new coco annotation file')
    args = args.parse_args()
    return args


def main():
    args = argsparser()
    with open(args.ann, 'r') as f:
        coco = json.load(f)

    coco['categories'] = [
        {'supercategory': 'traffic sign',
            'id': 1,
            'name': 'traffic sign'
        }
    ]
    for ann in coco['annotations']:
        ann['category_id'] = 1

    with open(args.output, 'w') as f:
        json.dump(coco, f)
    print('Converted !')


if __name__ == '__main__':
    main()