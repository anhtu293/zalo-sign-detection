import torch
import os
import cv2
from matplotlib import pyplot as plt
import numpy as np


class SignDataset(torch.utils.data.Dataset):
  def __init__(self, sign_dir):
        self.sign_names = os.listdir(sign_dir)
        self.dir = sign_dir

  def __len__(self):
        return len(self.sign_names)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        sign_name = self.sign_names[index]

        # Load data and get label
        sign = np.float32(cv2.imread(self.dir + sign_name,0))
        sign_resized = cv2.resize(sign, (128, 128))
        sign_resized /= 255.
        label = int(sign_name.split(".")[0].split("_")[1]) - 1

        return sign_resized, label