import torch
import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
from random import randint
import json

ANNOTATION_GENERATED = 5000
class SignDataset(torch.utils.data.Dataset):


    def __init__(self, image_dir, annotation_dir):
        self.image_dir = image_dir
        self.annotations = self.load_annotations(annotation_dir, image_dir)


    def __len__(self):
        return len(self.annotations)


    def __getitem__(self, index):
        img_id, bbox, category = self.annotations[index]
        img = np.float32(cv2.imread(self.image_dir + str(img_id) + ".png"))
        img_height, img_width = img.shape[0], img.shape[1]
        x, y, w, h = bbox
        #shift bbox
        if category != 7:
            shift = randint(0,1)
            if shift == 1:
                min_len = min(w,h)
                for i in range(len(bbox)):
                    change = randint(0,2)
                    save = bbox[i]
                    if change == 1:
                        bbox[i] += randint(min_len // 5, min_len // 3)
                    elif change == 2:
                        bbox[i] -= randint(min_len // 5, min_len // 3)
                    if bbox[i] <= 0:
                        bbox[i] = save
            if bbox[0] + bbox[2] > img_width or bbox[1] + bbox[3] > img_height:
                 bbox = [x,y,w,h]
        #crop and resize
        x, y, w, h = bbox
        sign = img[y:y+h, x:x+w]
        sign /= 255.
        # plt.imshow(sign)
        # plt.show()
        try:
            sign_resized = cv2.resize(sign, (128, 128))
        except:
            print(sign.shape)
            exit()
        return sign_resized, category 
        

    @classmethod
    def interval_overlap(self, interval1, interval2):
        x1, x2 = interval1
        x3, x4 = interval2
        if (x1 >= x3 and x1 <= x4) or (x2 >= x3 and x2 <= x4):
            return True
        return False

    @classmethod
    def bbox_overlap(self, bbox1, bbox2):
        x1min, y1min, w1, h1 = bbox1
        x2min, y2min, w2, h2 = bbox2
        x1max, y1max, x2max, y2max = x1min + w1, y1min + h1, x2min + w2, y2min + h2
        #projections
        if (
            self.interval_overlap((x1min, x1max), (x2min, x2max)) and
            self.interval_overlap((y1min, y1max), (y2min, y2max))
        ):
            return True
        return False


    @classmethod
    def generate_background(self, image_dir, image_names, image_bboxes):
        while True:
            #choose a random image
            random_image_name = image_names[randint(0, len(image_names)-1)]
            img = cv2.imread("../../../za_traffic_2020/traffic_train/images/" + random_image_name, 1)
            img_height, img_width = img.shape[0], img.shape[1]
            img_id = int(random_image_name.split(".")[0])
            img_bboxes = image_bboxes[img_id]
            #random a bounding box
            side = randint(10, 150)
            random_bbox = [randint(0, img_height), randint(0, img_width), side, side]
            valid = True
            for img_bbox in img_bboxes:
                if (
                    random_bbox[0] + random_bbox[2] > img_width or
                    random_bbox[1] + random_bbox[3] > img_height or 
                    self.bbox_overlap(random_bbox, img_bbox)
                ):
                    valid = False
                    break
            if valid == True:
                return [img_id, random_bbox, 7]
            else:
                continue


    @classmethod
    def load_annotations(self, annotation_dir, image_dir):
        #load from file
        annotation_dict = {}
        with open(annotation_dir) as f:
            annotation_dict = json.load(f)
        sign_annotations = annotation_dict["annotations"]

        #Get bboxes by image
        image_bboxes = {}
        all_annotations = []
        for sign in sign_annotations:
            image_id = sign["image_id"]
            bbox = sign["bbox"]
            category = sign["category_id"] - 1
            #save in final annotations
            all_annotations.append([image_id, bbox, category])
            #save bboxes for each image
            if image_id not in image_bboxes:
                image_bboxes[image_id] = [bbox]
            else:
                image_bboxes[image_id].append(bbox)

        #multiply annotations
        final_annotations = {}
        index = 0
        for anno in all_annotations:
            multiply = randint(3,5)
            for _ in range(multiply):
                final_annotations[index] = anno
                index += 1


        #generate background annotations
        original_len = len(final_annotations)
        generated_len = original_len // 7
        image_names = sorted(os.listdir(image_dir))
        while len(final_annotations) != original_len + generated_len:        
            new_background = self.generate_background(image_dir, image_names, image_bboxes)
            image_bboxes[new_background[0]].append(new_background[1])
            final_annotations[index] = new_background
            index += 1

        return final_annotations



# if __name__ == "__main__":
#     dataset = SignDataset("../../../za_traffic_2020/traffic_train/images/", "../../../za_traffic_2020/traffic_train/train_traffic_sign_dataset.json")