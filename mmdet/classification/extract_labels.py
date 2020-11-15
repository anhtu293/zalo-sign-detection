import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import json

def extract_labels():
	with open("../../../za_traffic_2020/traffic_train/train_traffic_sign_dataset.json") as f:
		label_dict = json.load(f)
	annotations = label_dict["annotations"]

	if not os.path.exists("../../../za_traffic_2020/signs"):
		os.makedirs("../../../za_traffic_2020/signs")

	for annotation in annotations:
		annotation_id = annotation["id"]
		image_id = annotation["image_id"]
		bbox = annotation["bbox"]
		category = annotation["category_id"]
		#Read image
		img = cv2.imread("../../../za_traffic_2020/traffic_train/images/" + str(image_id) +".png", 1)
		#Crop sign
		x, y, w, h = bbox
		sign_img = img[y:y+h, x:x+w]
		#Write to disk
		sign_filename = str(annotation_id).zfill(5) + "_" + str(category) + ".png"
		print(sign_filename)
		cv2.imwrite("../../../za_traffic_2020/signs/" + sign_filename, sign_img)
		
if __name__ == "__main__":
	extract_labels()