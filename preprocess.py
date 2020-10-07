import os
import boto3
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import time
import json
import shutil
from IPython.display import clear_output
import numpy as np 
from tqdm import tqdm
import cv2
import os
import shutil
import itertools
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import preprocess_input
import sagemaker.amazon.common as smac
from sagemaker import get_execution_role


role = get_execution_role()


def load_data(dir_path, img_size=(100,100)):
    """
    Load resized images as np.arrays to workspace
    """
    X = []
    y = []
    i = 0
    labels = dict()
    for path in tqdm(sorted(os.listdir(dir_path))):
        if not path.startswith('.'):
            labels[i] = path
            for file in os.listdir(dir_path + path):
                if not file.startswith('.'):
                    img = cv2.imread(dir_path + path + '/' + file)
                    X.append(img)
                    y.append(i)
            i += 1
    X = np.array(X)
    y = np.array(y)
    print(f'{len(X)} images loaded from {dir_path} directory.')
    return X, y, labels


def split(img_path, train_dir, test_dir, val_dir):
    # split the data by train/val/test
    for CLASS in os.listdir(img_path):
        if not CLASS.startswith('.'):
            IMG_NUM = len(os.listdir(img_path + CLASS))
            for (n, FILE_NAME) in enumerate(os.listdir(img_path + CLASS)):
                img = img_path + CLASS + '/' + FILE_NAME
                if n < 2:
                    shutil.copy(img, test_dir + CLASS.upper() + '/' + FILE_NAME)
                elif n < 0.8*IMG_NUM:
                    shutil.copy(img, train_dir + CLASS.upper() + '/' + FILE_NAME)
                else:
                    shutil.copy(img, val_dir + CLASS.upper() + '/' + FILE_NAME)
    


def save_new_images(x_set, y_set, folder_name):
    i = 0
    for (img, imclass) in zip(x_set, y_set):
        if imclass == 0:
            cv2.imwrite(folder_name+'DOG/'+str(i)+'.jpg', img)
        else:
            cv2.imwrite(folder_name+'CAT/'+str(i)+'.jpg', img)
        i += 1

def preprocess_imgs(set_name, img_size):
    """
    Resize and apply VGG-15 preprocessing
    """
    set_new = []
    for img in set_name:
        img = cv2.resize(
            img,
            dsize=img_size,
            interpolation=cv2.INTER_CUBIC
        )
        set_new.append(preprocess_input(img))
    return np.array(set_new)


